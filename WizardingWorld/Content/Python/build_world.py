"""WizardingWorld - one-step scene builder for Unreal Engine 5 (Python Editor Script Plugin).

The castle on its granite crag above the Black Lake, the river gorge, the ravine and its waterfall, the grounds, the
forests and the mountains - every mesh procedurally generated (see ../../generator), imported here as Nanite meshes.

Run inside the editor, either way:
    Output Log (Cmd box set to Python):   import build_world; build_world.run()
    or  Tools > Execute Python Script...  and pick this file.
Then, to look through a preset in the editor:   build_world.apply_preset("night")   (mist, day, sunset, dusk, night)

What it does (every stage is isolated; a failure is logged and the rest still runs):
    1. imports the textures and the FBX meshes from <Project>/SourceAssets (Nanite for every opaque mesh)
    2. verifies scale / axes of the imported meshes against the manifest (self-calibrating)
    3. builds the material parameter collection MPC_World and the materials: world-aligned (triplanar) granite with a
       height-blended moss layer, terrain layers driven by baked masks, castle stone with grime / moss, course-aligned
       slate, leaded windows that glow at night, two-sided foliage, single-layer water, the waterfall, the moon, the stars
    4. builds the level /Game/WizardingWorld/Maps/WizardingWorld: meshes, ~620 000 instances (trees, boulders, moss,
       ferns, grass, boats) as instanced static meshes, sun + moon, sky atmosphere, volumetric clouds, height fog with
       volumetric fog, local fog volumes (gorge / lake / ravine mist), sky light, post-process, boat lanterns
    5. creates a cine camera, a Level Sequence (camera cut + the lighting preset keyed in) and a Movie Render Queue job
       for every shot in SourceAssets/scene.json
Re-running is safe: assets are replaced and previously spawned actors are removed first.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import unreal

# --------------------------------------------------------------------------- config
CFG = {
    "game_root": "/Game/WizardingWorld",
    "level_name": "WizardingWorld",
    "actor_tag": "WizardingWorld",
    "legacy_fbx_import": True,        # use the classic FBX importer (the FbxImportUI options below); False = Interchange
    "start_preset": "mist",           # preset applied to the editor level at the end of the build
    # instanced detail
    "instances": True,                # False: skip every instance set (fast look at the bare world)
    "instance_density": 1.0,          # 0..1 thins every set evenly (try 0.5 on GPUs with < 12 GB)
    "instance_sets": None,            # e.g. ["spruce", "moss"]; None = all
    "instance_batch": 20000,
    # look
    "moss_gain": 1.0,
    "window_glow": 22.0,              # emissive strength of a lit window at night (cd/m2-ish, with auto exposure)
    "lantern_intensity": 1200.0,      # candela-ish, boat lanterns at night
    "water_absorption": (0.0105, 0.0042, 0.0034),          # per cm (peaty highland loch: brown-green, dark)
    "water_scattering": (0.0006, 0.0011, 0.0010),
    "water_phase_g": 0.3,
    # render queue
    "spatial_samples": 4,
    "temporal_samples": 8,
    "warmup_frames": 48,
    "output_exr": False,
}

PATHS = {k: f"{CFG['game_root']}/{k}" for k in ("Textures", "Meshes", "Materials", "Maps", "Cinematics")}


def to_ue(p):
    """Blender metres (x, y, z) -> Unreal centimetres (100x, -100y, 100z): see generator/export_world.py."""
    return unreal.Vector(100.0 * p[0], -100.0 * p[1], 100.0 * p[2])


# --------------------------------------------------------------------------- logging / reporting
class Report:
    def __init__(self):
        self.ok = []
        self.fail = []
        self.warn = []
        self.t0 = time.time()

    def step(self, name, fn, *a, **k):
        t = time.time()
        unreal.log(f"[WW] >>> {name}")
        try:
            r = fn(*a, **k)
            self.ok.append(name)
            unreal.log(f"[WW] <<< {name} done ({time.time() - t:.1f}s)")
            return r
        except Exception as e:                                     # noqa: BLE001 - we want to continue
            self.fail.append((name, str(e)))
            unreal.log_error(f"[WW] !!! {name} FAILED: {e}")
            unreal.log_error(traceback.format_exc())
            return None

    def note(self, msg):
        self.warn.append(msg)
        unreal.log_warning(f"[WW] {msg}")


REPORT = Report()


def log(msg):
    unreal.log(f"[WW] {msg}")


def sp(cls, obj, **props):
    """set_editor_property for several properties; `cls` documents (and lets the verifier check) the owning class."""
    bad = []
    for k, v in props.items():
        try:
            obj.set_editor_property(k, v)
        except Exception as e:                                     # noqa: BLE001
            bad.append(k)
            REPORT.note(f"could not set {getattr(cls, '__name__', cls)}.{k}: {str(e)[:120]}")
    return bad


def find_source_dir():
    cands = []
    env = os.environ.get("WW_SOURCE")
    if env:
        cands.append(env)
    try:
        cands.append(os.path.join(unreal.Paths.project_dir(), "SourceAssets"))
    except Exception:                                              # noqa: BLE001
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cands += [os.path.join(here, "..", "..", "SourceAssets"), os.path.join(here, "SourceAssets")]
    except NameError:
        pass
    for c in cands:
        if os.path.isfile(os.path.join(c, "manifest.json")):
            return os.path.normpath(c)
    raise RuntimeError("SourceAssets/manifest.json not found. Looked in: " + "; ".join(cands))


def asset_exists(path):
    return unreal.EditorAssetLibrary.does_asset_exist(path)


def load(path):
    return unreal.EditorAssetLibrary.load_asset(path)


def console(cmd):
    try:
        unreal.SystemLibrary.execute_console_command(None, cmd)
    except Exception as e:                                         # noqa: BLE001
        REPORT.note(f"console command '{cmd}' failed: {e}")


# --------------------------------------------------------------------------- 1. textures
MASK_TEXTURES = {"T_Macro", "T_Foam"}


def tex_kind(name):
    if name.endswith("_N"):
        return "normal"
    if name.endswith(("_ORH", "_A")) or name in MASK_TEXTURES:
        return "mask"
    return "color"


def import_textures(src):
    tdir = os.path.join(src, "textures")
    files = sorted(f for f in os.listdir(tdir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    tasks = []
    for f in files:
        t = unreal.AssetImportTask()
        sp(unreal.AssetImportTask, t, filename=os.path.join(tdir, f), destination_path=PATHS["Textures"],
           destination_name=os.path.splitext(f)[0], automated=True, replace_existing=True, replace_existing_settings=True, save=False)
        tasks.append(t)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    n = 0
    for f in files:
        name = os.path.splitext(f)[0]
        tex = load(f"{PATHS['Textures']}/{name}")
        if tex is None:
            REPORT.note(f"texture {name} was not imported")
            continue
        kind = tex_kind(name)
        if kind == "normal":
            sp(unreal.Texture2D, tex, compression_settings=unreal.TextureCompressionSettings.TC_NORMALMAP, srgb=False,
               flip_green_channel=True, lod_group=unreal.TextureGroup.TEXTUREGROUP_WORLD_NORMAL_MAP)
        elif kind == "mask":
            sp(unreal.Texture2D, tex, compression_settings=unreal.TextureCompressionSettings.TC_MASKS, srgb=False)
        else:
            sp(unreal.Texture2D, tex, srgb=True)
        sp(unreal.Texture2D, tex, address_x=unreal.TextureAddress.TA_WRAP, address_y=unreal.TextureAddress.TA_WRAP)
        if name == "T_Moon":
            sp(unreal.Texture2D, tex, address_x=unreal.TextureAddress.TA_CLAMP, address_y=unreal.TextureAddress.TA_CLAMP)
        n += 1
    log(f"{n}/{len(files)} textures imported and configured")


# --------------------------------------------------------------------------- 2. meshes
def make_fbx_ui(nanite):
    ui = unreal.FbxImportUI()
    sp(unreal.FbxImportUI, ui, import_mesh=True, import_textures=False, import_materials=False, import_as_skeletal=False,
       import_animations=False, mesh_type_to_import=unreal.FBXImportType.FBXIT_STATIC_MESH, automated_import_should_detect_type=False)
    smd = ui.get_editor_property("static_mesh_import_data")
    sp(unreal.FbxStaticMeshImportData, smd, combine_meshes=True, auto_generate_collision=False, generate_lightmap_u_vs=False,
       build_nanite=bool(nanite), normal_import_method=unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS,
       normal_generation_method=unreal.FBXNormalGenerationMethod.MIKK_T_SPACE, remove_degenerates=True,
       reorder_material_to_fbx_order=True, vertex_color_import_option=unreal.VertexColorImportOption.IGNORE,
       import_uniform_scale=1.0, convert_scene=True, convert_scene_unit=False, force_front_x_axis=False)
    return ui


def import_meshes(src, manifest):
    if CFG.get("legacy_fbx_import", True):
        console("Interchange.FeatureFlags.Import.FBX 0")
    fdir = os.path.join(src, "fbx")
    entries = manifest["meshes"]
    tasks = []
    for e in entries:
        t = unreal.AssetImportTask()
        sp(unreal.AssetImportTask, t, filename=os.path.join(fdir, e["file"]), destination_path=PATHS["Meshes"],
           destination_name=e["name"], automated=True, replace_existing=True, replace_existing_settings=True, save=False,
           options=make_fbx_ui(e.get("nanite", False)))
        tasks.append(t)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    batch = 3
    for i in range(0, len(tasks), batch):
        tools.import_asset_tasks(tasks[i:i + batch])
        log(f"meshes imported: {min(i + batch, len(tasks))}/{len(tasks)}")
    meshes = {}
    for e in entries:
        m = load(f"{PATHS['Meshes']}/{e['name']}")
        if m is None:
            REPORT.note(f"mesh {e['name']} did not import")
            continue
        meshes[e["name"]] = m
        tune_mesh(m, e)
        if e.get("nanite"):
            ensure_nanite(m, e)
    return meshes


def tune_mesh(mesh, entry):
    """Full-precision UVs (terrain UVs are world metres, masks live in UV1-UV3); no distance fields for thin foliage."""
    try:
        sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
        bs = sub.get_lod_build_settings(mesh, 0)
        sp(unreal.MeshBuildSettings, bs, use_full_precision_u_vs=True, use_high_precision_tangent_basis=True, recompute_normals=False,
           recompute_tangents=True, use_mikk_t_space=True, generate_lightmap_u_vs=False)
        sub.set_lod_build_settings(mesh, 0, bs)
    except Exception as e:                                         # noqa: BLE001
        REPORT.note(f"build settings not changed on {mesh.get_name()}: {str(e)[:100]}")
    if entry.get("foliage") or entry.get("kind") in ("water", "waterfall"):
        sp(unreal.StaticMesh, mesh, generate_mesh_distance_field=False)


def ensure_nanite(mesh, entry=None):
    """Import already requested Nanite; this is the safety net.  Foliage keeps its area when Nanite simplifies far away."""
    try:
        sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
        ns = sub.get_nanite_settings(mesh)
        changed = False
        if not ns.get_editor_property("enabled"):
            ns.set_editor_property("enabled", True)
            changed = True
        if entry and entry.get("foliage") and not ns.get_editor_property("preserve_area"):
            ns.set_editor_property("preserve_area", True)
            changed = True
        if changed:
            sub.set_nanite_settings(mesh, ns, True)
    except Exception as e:                                         # noqa: BLE001
        REPORT.note(f"Nanite could not be enabled on {mesh.get_name()}: {str(e)[:100]}")


# --------------------------------------------------------------------------- 3. calibration (scale / axes)
def _xy_matrices():
    out = []
    for perm in ((0, 1), (1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                m = [[0, 0], [0, 0]]
                m[0][perm[0]] = sx
                m[1][perm[1]] = sy
                out.append(m)
    return out


def calibrate(manifest, meshes):
    """Find (scale s, matrix Mx) with imported_bounds = s * Mx * (100 * blender_bounds) for all meshes and return the actor
    correction (yaw degrees, scale vector) mapping imported coordinates to the canonical placement.  Identity expected."""
    samples = []
    for e in manifest["meshes"]:
        m = meshes.get(e["name"])
        if m is None or e.get("library"):
            continue
        b = m.get_bounds()
        o, ex = b.origin, b.box_extent
        samples.append(((e["bbox_center_m"]), (e["bbox_extent_m"]), (o.x, o.y, o.z), (ex.x, ex.y, ex.z)))
    if not samples:
        raise RuntimeError("no meshes to calibrate with")
    best = None
    for s in (1.0, 0.01, 100.0):
        for M in _xy_matrices():
            err = 0.0
            for c, e, o, ex in samples:
                px = s * (M[0][0] * c[0] + M[0][1] * c[1]) * 100.0
                py = s * (M[1][0] * c[0] + M[1][1] * c[1]) * 100.0
                pz = s * c[2] * 100.0
                ax = s * (abs(M[0][0]) * e[0] + abs(M[0][1]) * e[1]) * 100.0
                ay = s * (abs(M[1][0]) * e[0] + abs(M[1][1]) * e[1]) * 100.0
                az = s * e[2] * 100.0
                err += abs(px - o[0]) + abs(py - o[1]) + abs(pz - o[2]) + abs(ax - ex[0]) + abs(ay - ex[1]) + abs(az - ex[2])
            if best is None or err < best[0]:
                best = (err, s, M)
    err, s, M = best
    per_mesh = err / (6 * len(samples))
    log(f"calibration: scale={s}, matrix={M}, mean error {per_mesh:.2f} cm over {len(samples)} meshes")
    if per_mesh > 50.0:
        REPORT.note("calibration error is large - imported meshes do not match the manifest; leaving transforms unchanged")
        return 0.0, unreal.Vector(1, 1, 1), 1.0, True
    MT = [[M[0][0], M[1][0]], [M[0][1], M[1][1]]]
    A = [[1, 0], [0, -1]]
    P = [[sum(A[i][k] * MT[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    det = P[0][0] * P[1][1] - P[0][1] * P[1][0]
    d = 1.0 if det > 0 else -1.0
    R = [[P[0][0], P[0][1] * d], [P[1][0], P[1][1] * d]]
    theta = math.degrees(math.atan2(R[1][0], R[0][0]))
    k = 1.0 / s
    ok = abs(theta) < 1e-6 and d > 0 and abs(k - 1.0) < 1e-9
    if not ok:
        REPORT.note(f"imported meshes need a correction: yaw={theta:.0f} deg, mirror={d < 0}, scale={k}")
    return theta, unreal.Vector(k, k * d, k), k, ok


# --------------------------------------------------------------------------- 4. materials
MEL = unreal.MaterialEditingLibrary
MP = unreal.MaterialProperty
S_COLOR = unreal.MaterialSamplerType.SAMPLERTYPE_COLOR
S_NORMAL = unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL
S_MASKS = unreal.MaterialSamplerType.SAMPLERTYPE_MASKS
S_LINEAR = unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR

OUT_ALTS = {"RGB": ["RGB", ""], "R": ["R"], "G": ["G"], "B": ["B"], "A": ["A"], "": ["", "Output", "Result"], "XYZ": ["XYZ", ""]}
IN_ALTS = {"": ["", "Input", "A"], "A": ["A"], "B": ["B"], "Alpha": ["Alpha"], "UVs": ["UVs", "UV"], "Coordinate": ["Coordinate", "UV"],
           "Min": ["Min"], "Max": ["Max"], "Value": ["Value", "Input"], "Base": ["Base"], "Exp": ["Exp", "Exponent"], "Time": ["Time"]}

MPC = {"asset": None}


class MatBuilder:
    def __init__(self, name):
        self.name = name
        self.fails = 0
        self.conns = 0
        full = f"{PATHS['Materials']}/{name}"
        if asset_exists(full):
            unreal.EditorAssetLibrary.delete_asset(full)
        self.mat = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, PATHS["Materials"], unreal.Material, unreal.MaterialFactoryNew())
        if self.mat is None:
            raise RuntimeError(f"could not create material {name}")
        sp(unreal.Material, self.mat, used_with_instanced_static_meshes=True, used_with_nanite=True)
        self._x = -1600
        self._y = 0
        self._cache = {}

    def node(self, cls, props=None, x=None, y=None):
        self._y += 120
        e = MEL.create_material_expression(self.mat, cls, self._x if x is None else x, self._y if y is None else y)
        if props:
            sp(cls, e, **props)
        return e

    def tex(self, texture, uv=None, sampler=None, param=None):
        props = {"texture": texture, "sampler_source": unreal.SamplerSourceMode.SSM_WRAP_WORLD_GROUP_SETTINGS}
        if sampler is not None:
            props["sampler_type"] = sampler
        if param:
            props["parameter_name"] = param
            e = self.node(unreal.MaterialExpressionTextureSampleParameter2D, props)
        else:
            e = self.node(unreal.MaterialExpressionTextureSample, props)
        if uv is not None:
            self.conn(uv, "", e, "UVs")
        return e

    def uv(self, index=0, tiling=1.0):
        return self.node(unreal.MaterialExpressionTextureCoordinate, {"coordinate_index": index, "u_tiling": tiling, "v_tiling": tiling})

    def mask(self, src, src_out, r=False, g=False, b=False, a=False):
        e = self.node(unreal.MaterialExpressionComponentMask, {"r": r, "g": g, "b": b, "a": a})
        self.conn(src, src_out, e, "")
        return e

    def const(self, v):
        return self.node(unreal.MaterialExpressionConstant, {"r": float(v)})

    def const3(self, rgb):
        return self.node(unreal.MaterialExpressionConstant3Vector, {"constant": unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0)})

    def scalar_param(self, name, default):
        return self.node(unreal.MaterialExpressionScalarParameter, {"parameter_name": name, "default_value": float(default)})

    def vec_param(self, name, rgb):
        return self.node(unreal.MaterialExpressionVectorParameter, {"parameter_name": name, "default_value": unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0)})

    def mpc(self, name):
        """A scalar from the world parameter collection (night, window glow, moss gain); falls back to a constant."""
        if MPC["asset"] is None:
            return self.const({"Night": 0.0, "WindowGlow": 0.0, "MossGain": 1.0, "Wetness": 0.0}.get(name, 0.0))
        return self.node(unreal.MaterialExpressionCollectionParameter, {"collection": MPC["asset"], "parameter_name": name})

    def _bin(self, cls, a, a_out, b, b_out, k):
        e = self.node(cls, {"const_b": float(k)} if k is not None else None)
        self.conn(a, a_out, e, "A")
        if b is not None:
            self.conn(b, b_out, e, "B")
        return e

    def mul(self, a, a_out, b=None, b_out="", k=None):
        return self._bin(unreal.MaterialExpressionMultiply, a, a_out, b, b_out, k)

    def add(self, a, a_out, b=None, b_out="", k=None):
        return self._bin(unreal.MaterialExpressionAdd, a, a_out, b, b_out, k)

    def sub(self, a, a_out, b=None, b_out="", k=None):
        return self._bin(unreal.MaterialExpressionSubtract, a, a_out, b, b_out, k)

    def div(self, a, a_out, b=None, b_out="", k=None):
        return self._bin(unreal.MaterialExpressionDivide, a, a_out, b, b_out, k)

    def lerp(self, a, a_out, b, b_out, alpha, alpha_out=""):
        e = self.node(unreal.MaterialExpressionLinearInterpolate)
        self.conn(a, a_out, e, "A")
        self.conn(b, b_out, e, "B")
        self.conn(alpha, alpha_out, e, "Alpha")
        return e

    def saturate(self, a, a_out=""):
        e = self.node(unreal.MaterialExpressionSaturate)
        self.conn(a, a_out, e, "")
        return e

    def smoothstep(self, value, v_out, lo, hi):
        e = self.node(unreal.MaterialExpressionSmoothStep, {"const_min": float(lo), "const_max": float(hi)})
        self.conn(value, v_out, e, "Value")
        return e

    def unary(self, cls, a, a_out="", props=None):
        e = self.node(cls, props)
        self.conn(a, a_out, e, "")
        return e

    def append(self, a, a_out, b, b_out=""):
        e = self.node(unreal.MaterialExpressionAppendVector)
        self.conn(a, a_out, e, "A")
        self.conn(b, b_out, e, "B")
        return e

    def conn(self, a, a_out, b, b_in):
        self.conns += 1
        for o in OUT_ALTS.get(a_out, [a_out]):
            for i in IN_ALTS.get(b_in, [b_in]):
                try:
                    if MEL.connect_material_expressions(a, o, b, i):
                        return True
                except Exception:                                  # noqa: BLE001
                    pass
        self.fails += 1
        REPORT.note(f"{self.name}: could not connect {a.get_class().get_name() if a else '?'}[{a_out}] -> {b.get_class().get_name() if b else '?'}[{b_in}]")
        return False

    def out(self, expr, expr_out, prop):
        self.conns += 1
        for o in OUT_ALTS.get(expr_out, [expr_out]):
            try:
                if MEL.connect_material_property(expr, o, prop):
                    return True
            except Exception:                                      # noqa: BLE001
                pass
        self.fails += 1
        REPORT.note(f"{self.name}: could not connect to material output {prop}")
        return False

    # -- world-aligned helpers (cached per material)
    def wp(self):
        if "wp" not in self._cache:
            self._cache["wp"] = self.node(unreal.MaterialExpressionWorldPosition)
        return self._cache["wp"]

    def nws(self):
        if "n" not in self._cache:
            self._cache["n"] = self.node(unreal.MaterialExpressionVertexNormalWS)
        return self._cache["n"]

    def up(self):
        if "up" not in self._cache:
            self._cache["up"] = self.mask(self.nws(), "XYZ", b=True)
        return self._cache["up"]

    def tri_weights(self):
        if "tw" not in self._cache:
            an = self.unary(unreal.MaterialExpressionAbs, self.nws(), "XYZ")
            p = self.node(unreal.MaterialExpressionPower, {"const_exponent": 4.0})
            self.conn(an, "", p, "Base")
            s = self.add(self.add(self.mask(p, "", r=True), "", self.mask(p, "", g=True), ""), "", self.mask(p, "", b=True), "")
            w = self.div(p, "", s, "")
            self._cache["tw"] = (self.mask(w, "", r=True), self.mask(w, "", g=True), self.mask(w, "", b=True))
        return self._cache["tw"]

    def proj_uvs(self, tile_cm):
        key = ("uv", tile_cm)
        if key not in self._cache:
            wp = self.wp()
            x, y, z = (self.mask(wp, "XYZ", **{c: True}) for c in ("r", "g", "b"))
            k = 1.0 / tile_cm
            ux = self.append(self.mul(y, "", k=k), "", self.mul(z, "", k=-k), "")
            uy = self.append(self.mul(x, "", k=k), "", self.mul(z, "", k=-k), "")
            uz = self.append(self.mul(x, "", k=k), "", self.mul(y, "", k=-k), "")
            self._cache[key] = (ux, uy, uz)
        return self._cache[key]

    def triplanar(self, texture, tile_cm, sampler, param=None):
        """World-aligned blend of three planar projections (weights from the vertex normal)."""
        wx, wy, wz = self.tri_weights()
        ux, uy, uz = self.proj_uvs(tile_cm)
        sx, sy, sz = (self.tex(texture, u, sampler, param) for u in (ux, uy, uz))
        out = self.mul(sx, "RGB", wx, "")
        out = self.add(out, "", self.mul(sy, "RGB", wy, ""), "")
        return self.add(out, "", self.mul(sz, "RGB", wz, ""), "")

    def triplanar_perturb(self, texture, tile_cm, strength=1.0, param=None):
        """World-space normal perturbation from a tangent-space normal map sampled triplanar:
        x projection -> (0, a, b), y -> (a, 0, b), z -> (a, b, 0) (UDN blend), weighted."""
        wx, wy, wz = self.tri_weights()
        ux, uy, uz = self.proj_uvs(tile_cm)
        sx, sy, sz = (self.tex(texture, u, S_NORMAL, param) for u in (ux, uy, uz))
        zero = self.const(0.0)
        px = self.append(self.append(zero, "", self.mask(sx, "RGB", r=True), ""), "", self.mask(sx, "RGB", g=True), "")
        py = self.append(self.append(self.mask(sy, "RGB", r=True), "", zero, ""), "", self.mask(sy, "RGB", g=True), "")
        pz = self.append(self.mask(sz, "RGB", r=True, g=True), "", zero, "")
        acc = self.add(self.add(self.mul(px, "", wx, ""), "", self.mul(py, "", wy, ""), ""), "", self.mul(pz, "", wz, ""), "")
        return self.mul(acc, "", k=strength)

    def planar_perturb(self, nsample, strength=1.0):
        """Perturbation of a top-projected (world XY) normal map sample: (a, b, 0)."""
        return self.mul(self.append(self.mask(nsample, "RGB", r=True, g=True), "", self.const(0.0), ""), "", k=strength)

    def world_normal(self, perturb):
        return self.unary(unreal.MaterialExpressionNormalize, self.add(self.nws(), "", perturb, ""))

    def finish(self):
        try:
            MEL.layout_material_expressions(self.mat)
        except Exception:                                          # noqa: BLE001
            pass
        MEL.recompile_material(self.mat)
        unreal.EditorAssetLibrary.save_loaded_asset(self.mat)
        log(f"material {self.name}: {self.conns - self.fails}/{self.conns} connections ok")
        return self.mat


def T(name):
    t = load(f"{PATHS['Textures']}/{name}")
    if t is None:
        raise RuntimeError(f"texture {name} missing")
    return t


def build_mpc():
    """MPC_World: Night (0/1), WindowGlow (emissive strength), MossGain, Wetness - driven by the presets and sequences."""
    path = f"{PATHS['Materials']}/MPC_World"
    if asset_exists(path):
        mpc = load(path)
    else:
        mpc = unreal.AssetToolsHelpers.get_asset_tools().create_asset("MPC_World", PATHS["Materials"], unreal.MaterialParameterCollection,
                                                                      unreal.MaterialParameterCollectionFactoryNew())
    params = []
    for name, val in (("Night", 0.0), ("WindowGlow", 0.0), ("MossGain", CFG["moss_gain"]), ("Wetness", 0.0)):
        p = unreal.CollectionScalarParameter()
        sp(unreal.CollectionScalarParameter, p, parameter_name=name, default_value=float(val))
        params.append(p)
    sp(unreal.MaterialParameterCollection, mpc, scalar_parameters=params)
    unreal.EditorAssetLibrary.save_loaded_asset(mpc)
    MPC["asset"] = mpc
    return mpc


def set_mpc(name, value):
    mpc = MPC["asset"]
    if mpc is None:
        return
    try:
        params = list(mpc.get_editor_property("scalar_parameters"))
        for p in params:
            if str(p.get_editor_property("parameter_name")) == name:
                p.set_editor_property("default_value", float(value))
        mpc.set_editor_property("scalar_parameters", params)
    except Exception as e:                                         # noqa: BLE001
        REPORT.note(f"MPC {name} not set: {e}")


def _layer(b, prefix, uvs_top=None, tile_cm=200.0, triplanar=False, param=True):
    """(colour, orh, perturbation) of a texture set, top-projected (world XY) or triplanar."""
    if triplanar:
        c = b.triplanar(T(f"T_{prefix}_BC"), tile_cm, S_COLOR, f"{prefix}_BC" if param else None)
        o = b.triplanar(T(f"T_{prefix}_ORH"), tile_cm, S_MASKS, f"{prefix}_ORH" if param else None)
        n = b.triplanar_perturb(T(f"T_{prefix}_N"), tile_cm, 1.0, f"{prefix}_N" if param else None)
        return c, o, n
    uv = uvs_top[2] if uvs_top else b.proj_uvs(tile_cm)[2]
    c = b.tex(T(f"T_{prefix}_BC"), uv, S_COLOR, f"{prefix}_BC" if param else None)
    o = b.tex(T(f"T_{prefix}_ORH"), uv, S_MASKS, f"{prefix}_ORH" if param else None)
    n = b.tex(T(f"T_{prefix}_N"), uv, S_NORMAL, f"{prefix}_N" if param else None)
    return c, o, b.planar_perturb(n)


def build_terrain(name="M_Terrain"):
    """Terrain: grass / forest floor / shore / dirt from the top, granite (triplanar) on rock and anything steep, a
    height-blended moss layer on the rock, baked ambient occlusion.  Masks: UV1 = (rock, moss), UV2 = (forest, shore),
    UV3 = (road, ao).  World-space normals."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, tangent_space_normal=False)
    uv1, uv2, uv3 = b.uv(1), b.uv(2), b.uv(3)
    rock_m, moss_m = b.mask(uv1, "", r=True), b.mask(uv1, "", g=True)
    forest_m, shore_m = b.mask(uv2, "", r=True), b.mask(uv2, "", g=True)
    road_m, ao_m = b.mask(uv3, "", r=True), b.mask(uv3, "", g=True)
    g_c, g_o, g_n = _layer(b, "Grass", tile_cm=200.0)
    f_c, f_o, f_n = _layer(b, "ForestFloor", tile_cm=200.0)
    s_c, s_o, s_n = _layer(b, "Shore", tile_cm=200.0)
    d_c, d_o, d_n = _layer(b, "Dirt", tile_cm=200.0)
    r_c, r_o, r_n = _layer(b, "Granite", tile_cm=400.0, triplanar=True)
    m_c, m_o, m_n = _layer(b, "Moss", tile_cm=110.0, triplanar=True)
    mac = b.tex(T("T_Macro"), b.proj_uvs(3200.0)[2], S_MASKS)
    mac_r, mac_g, mac_b = (b.mask(mac, "RGB", **{c: True}) for c in ("r", "g", "b"))
    up = b.up()
    # ground layers
    fw = b.smoothstep(forest_m, "", 0.25, 0.75)
    sw = b.smoothstep(shore_m, "", 0.2, 0.8)
    dw = b.smoothstep(road_m, "", 0.3, 0.8)
    col = b.lerp(b.lerp(b.lerp(g_c, "RGB", f_c, "RGB", fw, ""), "", s_c, "RGB", sw, ""), "", d_c, "RGB", dw, "")
    orh = b.lerp(b.lerp(b.lerp(g_o, "RGB", f_o, "RGB", fw, ""), "", s_o, "RGB", sw, ""), "", d_o, "RGB", dw, "")
    nrm = b.lerp(b.lerp(b.lerp(g_n, "", f_n, "", fw, ""), "", s_n, "", sw, ""), "", d_n, "", dw, "")
    # granite where the mask says so and on anything steep
    steep = b.unary(unreal.MaterialExpressionOneMinus, b.smoothstep(up, "", 0.55, 0.8))
    rf = b.node(unreal.MaterialExpressionMax)
    b.conn(b.smoothstep(rock_m, "", 0.35, 0.7), "", rf, "A")
    b.conn(steep, "", rf, "B")
    rock_c = b.mul(b.mul(r_c, "", b.vec_param("RockTint", (0.88, 0.92, 0.97)), ""), "", b.add(b.mul(mac_r, "", k=0.30), "", k=0.85), "")
    col = b.lerp(col, "", rock_c, "", rf, "")
    orh = b.lerp(orh, "", r_o, "", rf, "")
    nrm = b.lerp(nrm, "", r_n, "", rf, "")
    # moss: baked cover + macro bias + (moss canopy height - rock height), only on up-facing rock
    m_h = b.mask(m_o, "", b=True)
    r_h = b.mask(r_o, "", b=True)
    cover = b.add(b.mul(b.mul(moss_m, "", b.mpc("MossGain"), ""), "", k=1.6), "", b.mul(b.sub(mac_b, "", k=0.5), "", k=0.5), "")
    mv = b.add(cover, "", b.mul(b.sub(m_h, "", r_h, ""), "", k=0.6), "")
    mf = b.mul(b.smoothstep(mv, "", 0.40, 0.75), "", b.smoothstep(up, "", 0.05, 0.5), "")
    col = b.lerp(col, "", b.mul(m_c, "", b.add(b.mul(mac_g, "", k=0.3), "", k=0.85), ""), "", mf, "")
    orh = b.lerp(orh, "", m_o, "", mf, "")
    nrm = b.lerp(nrm, "", m_n, "", mf, "")
    # baked occlusion + wet shore
    ao = b.mul(ao_m, "", ao_m, "")
    col = b.mul(col, "", b.lerp(b.const(1.0), "", b.const(0.62), "", sw, ""), "")
    b.out(b.mul(col, "", b.add(b.mul(ao, "", k=0.75), "", k=0.25), ""), "", MP.MP_BASE_COLOR)
    b.out(b.lerp(b.mask(orh, "", r=True), "", b.const(0.25), "", b.mul(sw, "", k=0.6), ""), "", MP.MP_ROUGHNESS)
    b.out(b.mul(b.mask(orh, "", g=True), "", ao, ""), "", MP.MP_AMBIENT_OCCLUSION)
    b.out(b.world_normal(nrm), "", MP.MP_NORMAL)
    b.out(b.const(0.35), "", MP.MP_SPECULAR)
    return b.finish(), b


def build_boulder(name="M_Boulder"):
    """Instanced granite boulders: triplanar granite, moss on top, a little per-instance colour drift."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, tangent_space_normal=False)
    r_c, r_o, r_n = _layer(b, "Granite", tile_cm=300.0, triplanar=True, param=False)
    m_c, m_o, m_n = _layer(b, "Moss", tile_cm=90.0, triplanar=True, param=False)
    rnd = b.node(unreal.MaterialExpressionPerInstanceRandom)
    tint = b.lerp(b.const3((0.82, 0.86, 0.92)), "", b.const3((1.05, 1.0, 0.95)), "", rnd, "")
    mf = b.mul(b.smoothstep(b.up(), "", 0.35, 0.85), "", b.mpc("MossGain"), "")
    col = b.lerp(b.mul(r_c, "", tint, ""), "", m_c, "", mf, "")
    b.out(col, "", MP.MP_BASE_COLOR)
    b.out(b.lerp(b.mask(r_o, "", r=True), "", b.mask(m_o, "", r=True), "", mf, ""), "", MP.MP_ROUGHNESS)
    b.out(b.lerp(b.mask(r_o, "", g=True), "", b.mask(m_o, "", g=True), "", mf, ""), "", MP.MP_AMBIENT_OCCLUSION)
    b.out(b.world_normal(b.lerp(r_n, "", m_n, "", mf, "")), "", MP.MP_NORMAL)
    return b.finish(), b


def build_masonry(name, prefix, tile_cm=200.0, moss_k=1.0):
    """Castle stone / dressed trim: triplanar ashlar, grime (UV1.x) and moss (UV1.y) blended in, macro colour drift."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, tangent_space_normal=False)
    c, o, n = _layer(b, prefix, tile_cm=tile_cm, triplanar=True)
    m_c, m_o, m_n = _layer(b, "Moss", tile_cm=100.0, triplanar=True, param=False)
    uv1 = b.uv(1)
    grime, moss = b.mask(uv1, "", r=True), b.mask(uv1, "", g=True)
    mac = b.tex(T("T_Macro"), b.proj_uvs(2400.0)[1], S_MASKS)
    mac_r = b.mask(mac, "RGB", r=True)
    stone = b.mul(c, "", b.add(b.mul(mac_r, "", k=0.25), "", k=0.86), "")
    stone = b.lerp(stone, "", b.mul(stone, "", b.const3((0.42, 0.40, 0.34)), ""), "", b.mul(grime, "", k=0.65), "")
    mf = b.mul(b.mul(b.smoothstep(moss, "", 0.55, 0.95), "", k=moss_k), "", b.mpc("MossGain"), "")
    col = b.lerp(stone, "", m_c, "", mf, "")
    b.out(col, "", MP.MP_BASE_COLOR)
    b.out(b.lerp(b.mask(o, "", r=True), "", b.mask(m_o, "", r=True), "", mf, ""), "", MP.MP_ROUGHNESS)
    b.out(b.mask(o, "", g=True), "", MP.MP_AMBIENT_OCCLUSION)
    b.out(b.world_normal(b.lerp(n, "", m_n, "", mf, "")), "", MP.MP_NORMAL)
    b.out(b.const(0.4), "", MP.MP_SPECULAR)
    return b.finish(), b


def build_slate(name="M_Slate"):
    """Slate roofs: UV0 is metric along the eaves / up the slope; one texture tile = 6 courses of 0.42 m."""
    b = MatBuilder(name)
    uv = b.uv(0, 1.0 / 2.52)
    c = b.tex(T("T_Slate_BC"), uv, S_COLOR, "Slate_BC")
    o = b.tex(T("T_Slate_ORH"), uv, S_MASKS, "Slate_ORH")
    n = b.tex(T("T_Slate_N"), uv, S_NORMAL, "Slate_N")
    m_c = b.tex(T("T_Moss_BC"), b.uv(0, 1.0 / 1.1), S_COLOR)
    moss = b.mask(b.uv(1), "", g=True)
    mf = b.mul(b.smoothstep(moss, "", 0.5, 0.9), "", b.mpc("MossGain"), "")
    b.out(b.lerp(c, "RGB", m_c, "RGB", mf, ""), "", MP.MP_BASE_COLOR)
    b.out(b.add(b.mul(b.mask(o, "RGB", r=True), "", k=0.45), "", k=0.5), "", MP.MP_ROUGHNESS)
    b.out(b.mask(o, "RGB", g=True), "", MP.MP_AMBIENT_OCCLUSION)
    b.out(n, "RGB", MP.MP_NORMAL)
    b.out(b.const(0.4), "", MP.MP_SPECULAR)
    return b.finish(), b


def build_simple(name, prefix=None, color=(0.5, 0.5, 0.5), rough=0.8, metal=0.0, uv_tile=0.5, tint=None):
    b = MatBuilder(name)
    if prefix:
        uv = b.uv(0, uv_tile)
        c = b.tex(T(f"T_{prefix}_BC"), uv, S_COLOR)
        o = b.tex(T(f"T_{prefix}_ORH"), uv, S_MASKS)
        n = b.tex(T(f"T_{prefix}_N"), uv, S_NORMAL)
        col = b.mul(c, "RGB", b.const3(tint), "") if tint else c
        b.out(col, "" if tint else "RGB", MP.MP_BASE_COLOR)
        b.out(b.mask(o, "RGB", r=True), "", MP.MP_ROUGHNESS)
        b.out(b.mask(o, "RGB", g=True), "", MP.MP_AMBIENT_OCCLUSION)
        b.out(n, "RGB", MP.MP_NORMAL)
    else:
        b.out(b.const3(color), "", MP.MP_BASE_COLOR)
        b.out(b.const(rough), "", MP.MP_ROUGHNESS)
    if metal:
        b.out(b.const(metal), "", MP.MP_METALLIC)
    return b.finish(), b


def build_glass(name="M_Glass"):
    """Leaded diamond-pane windows.  Emissive = warm glow x lit flag (UV2.y) x pane (not the lead cames) x MPC WindowGlow,
    with a per-window intensity / warmth variation from UV2.x."""
    b = MatBuilder(name)
    uv = b.uv(0)
    c = b.tex(T("T_Glass_BC"), uv, S_COLOR)
    o = b.tex(T("T_Glass_ORH"), uv, S_MASKS)
    came = b.mask(o, "RGB", b=True)
    uv2 = b.uv(2)
    rnd, lit = b.mask(uv2, "", r=True), b.mask(uv2, "", g=True)
    warm = b.lerp(b.const3((1.0, 0.50, 0.18)), "", b.const3((1.0, 0.68, 0.36)), "", rnd, "")
    glow = b.mul(b.mul(lit, "", b.unary(unreal.MaterialExpressionOneMinus, came), ""), "", b.mpc("WindowGlow"), "")
    glow = b.mul(glow, "", b.add(b.mul(rnd, "", k=0.6), "", k=0.6), "")
    b.out(c, "RGB", MP.MP_BASE_COLOR)
    b.out(b.lerp(b.const(0.06), "", b.const(0.5), "", came, ""), "", MP.MP_ROUGHNESS)
    b.out(b.const(0.6), "", MP.MP_SPECULAR)
    b.out(b.mul(warm, "", glow, ""), "", MP.MP_EMISSIVE_COLOR)
    b.out(b.tex(T("T_Glass_N"), uv, S_NORMAL), "RGB", MP.MP_NORMAL)
    return b.finish(), b


def build_cloth(name="M_Cloth"):
    """Stand canopies: four house colours picked by UV2.x (0, .25, .5, .75)."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, two_sided=True)
    c = b.tex(T("T_Cloth_BC"), b.uv(0, 0.5), S_COLOR)
    pick = b.mask(b.uv(2), "", r=True)
    pal = [b.vec_param("House1", (0.42, 0.03, 0.025)), b.vec_param("House2", (0.03, 0.20, 0.07)), b.vec_param("House3", (0.035, 0.07, 0.32)),
           b.vec_param("House4", (0.60, 0.42, 0.03))]
    col = pal[0]
    for k, p in enumerate(pal[1:], 1):
        col = b.lerp(col, "", p, "", b.smoothstep(pick, "", 0.25 * k - 0.05, 0.25 * k - 0.04), "")
    b.out(b.mul(c, "RGB", col, ""), "", MP.MP_BASE_COLOR)
    b.out(b.const(0.9), "", MP.MP_ROUGHNESS)
    return b.finish(), b


def build_emissive(name, color, strength, night=True):
    b = MatBuilder(name)
    e = b.mul(b.const3(color), "", k=strength)
    if night:
        e = b.mul(e, "", b.add(b.mpc("Night"), "", k=0.02), "")
    b.out(b.const3(color), "", MP.MP_BASE_COLOR)
    b.out(e, "", MP.MP_EMISSIVE_COLOR)
    return b.finish(), b


def build_foliage(name, tex, nrm=None, tint_a=(0.80, 0.90, 0.74), tint_b=(1.12, 1.05, 0.84), rough=0.5, spec=0.3, sss=(0.9, 1.0, 0.35),
                  brightness=1.0):
    """Opaque two-sided foliage for the instanced high-poly plants (the mesh carries the outline, so no alpha test).
    UV1.x = per-spray tint, UV1.y = outer crown (sun) vs inner shade; PerInstanceRandom varies plant to plant."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, shading_model=unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE, blend_mode=unreal.BlendMode.BLEND_OPAQUE,
       two_sided=True)
    uv0 = b.uv(0)
    t = b.tex(T(tex), uv0, S_COLOR)
    uv1 = b.uv(1)
    tint_m, sun = b.mask(uv1, "", r=True), b.mask(uv1, "", g=True)
    rnd = b.node(unreal.MaterialExpressionPerInstanceRandom)
    mix = b.add(b.mul(tint_m, "", k=0.6), "", b.mul(rnd, "", k=0.4), "")
    tc = b.lerp(b.const3(tint_a), "", b.const3(tint_b), "", mix, "")
    shade = b.add(b.mul(sun, "", k=0.55), "", k=0.45)
    col = b.mul(b.mul(b.mul(t, "RGB", tc, ""), "", shade, ""), "", k=brightness)
    b.out(col, "", MP.MP_BASE_COLOR)
    b.out(b.mul(col, "", b.const3(sss), ""), "", MP.MP_SUBSURFACE_COLOR)
    if nrm:
        b.out(b.tex(T(nrm), uv0, S_NORMAL), "RGB", MP.MP_NORMAL)
    b.out(b.const(rough), "", MP.MP_ROUGHNESS)
    b.out(b.const(spec), "", MP.MP_SPECULAR)
    return b.finish(), b


def build_bark(name, prefix, tint=None, uv_tile=1.0 / 1.5):
    return build_simple(name, prefix, uv_tile=uv_tile, tint=tint)


def build_moss_clump(name="M_MossClump"):
    b = MatBuilder(name)
    uv0 = b.uv(0, 4.0)
    c = b.tex(T("T_Moss_BC"), uv0, S_COLOR)
    n = b.tex(T("T_Moss_N"), uv0, S_NORMAL)
    o = b.tex(T("T_Moss_ORH"), uv0, S_MASKS)
    rnd = b.node(unreal.MaterialExpressionPerInstanceRandom)
    inst = b.lerp(b.const3((0.78, 0.88, 0.70)), "", b.const3((1.15, 1.08, 0.74)), "", rnd, "")
    ao = b.mask(o, "RGB", g=True)
    col = b.mul(b.mul(c, "RGB", inst, ""), "", b.lerp(b.const(0.6), "", b.const(1.0), "", ao, ""), "")
    b.out(col, "", MP.MP_BASE_COLOR)
    b.out(n, "RGB", MP.MP_NORMAL)
    b.out(b.mask(o, "RGB", r=True), "", MP.MP_ROUGHNESS)
    b.out(ao, "", MP.MP_AMBIENT_OCCLUSION)
    b.out(b.const(0.25), "", MP.MP_SPECULAR)
    return b.finish(), b


def _water_pin(b, wo, names, src):
    b.conns += 1
    for nme in names:
        for o in ("", "RGB", "Output"):
            try:
                if MEL.connect_material_expressions(src, o, wo, nme):
                    return True
            except Exception:                                      # noqa: BLE001
                pass
    b.fails += 1
    REPORT.note(f"{b.name}: could not connect water pin {names[0]} (water will use default volume values)")
    return False


def build_water(name="M_Water", flow=False):
    """Single Layer Water.  Lake: two drifting ripple scales in world space.  River: ripples streaming along UV0.v
    (the flow direction) with foam bands; peaty Highland water absorption / scattering from CFG."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, shading_model=unreal.MaterialShadingModel.MSM_SINGLE_LAYER_WATER, blend_mode=unreal.BlendMode.BLEND_OPAQUE)
    if flow:
        base = b.uv(0)
        pa = b.node(unreal.MaterialExpressionPanner, {"speed_x": 0.0, "speed_y": -0.35})
        b.conn(base, "", pa, "Coordinate")
        pb = b.node(unreal.MaterialExpressionPanner, {"speed_x": 0.02, "speed_y": -0.9})
        b.conn(b.mul(base, "", k=3.0), "", pb, "Coordinate")
    else:
        xy = b.mask(b.wp(), "XYZ", r=True, g=True)
        pa = b.node(unreal.MaterialExpressionPanner, {"speed_x": 0.0022, "speed_y": 0.0011})
        b.conn(b.mul(xy, "", k=1.0 / 900.0), "", pa, "Coordinate")
        pb = b.node(unreal.MaterialExpressionPanner, {"speed_x": -0.0031, "speed_y": 0.0019})
        b.conn(b.mul(xy, "", k=1.0 / 230.0), "", pb, "Coordinate")
    na = b.tex(T("T_Water_N"), pa, S_NORMAL)
    nb = b.tex(T("T_Water_Ripple_N"), pb, S_NORMAL)
    flat = b.const3((0.0, 0.0, 1.0))
    n = b.unary(unreal.MaterialExpressionNormalize, b.add(b.lerp(flat, "", na, "RGB", b.const(0.45 if not flow else 0.7), ""), "",
                                                           b.mul(b.mask(nb, "RGB", r=True, g=True), "", k=0.25), ""))
    foam = b.const(0.0)
    if flow:
        f = b.mask(b.tex(T("T_Foam"), pb, S_MASKS), "RGB", r=True)
        foam = b.mul(b.smoothstep(f, "", 0.55, 0.9), "", k=0.7)
    b.out(b.lerp(b.const3((0.02, 0.03, 0.03)), "", b.const3((0.75, 0.78, 0.78)), "", foam, ""), "", MP.MP_BASE_COLOR)
    b.out(foam, "", MP.MP_OPACITY)
    b.out(b.lerp(b.const(0.03 if not flow else 0.08), "", b.const(0.6), "", foam, ""), "", MP.MP_ROUGHNESS)
    b.out(n, "", MP.MP_NORMAL)
    b.out(b.const(0.5), "", MP.MP_SPECULAR)
    wo = b.node(unreal.MaterialExpressionSingleLayerWaterMaterialOutput)
    _water_pin(b, wo, ("Scattering Coefficients", "ScatteringCoefficients"), b.const3(CFG["water_scattering"]))
    _water_pin(b, wo, ("Absorption Coefficients", "AbsorptionCoefficients"), b.const3(CFG["water_absorption"]))
    _water_pin(b, wo, ("Phase G", "PhaseG"), b.const(CFG["water_phase_g"]))
    _water_pin(b, wo, ("Color Scale Behind Water", "ColorScaleBehindWater"), b.const3((1.0, 1.0, 1.0)))
    return b.finish(), b


def build_fall(name="M_Fall"):
    """Waterfall sheet: translucent, streaks pouring down UV0.v, brighter where aerated."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT, two_sided=True,
       translucency_lighting_mode=unreal.TranslucencyLightingMode.TLM_SURFACE)
    base = b.uv(0)
    pa = b.node(unreal.MaterialExpressionPanner, {"speed_x": 0.0, "speed_y": 1.4})
    b.conn(base, "", pa, "Coordinate")
    pb = b.node(unreal.MaterialExpressionPanner, {"speed_x": 0.0, "speed_y": 0.9})
    b.conn(b.add(b.mul(base, "", k=1.7), "", k=0.37), "", pb, "Coordinate")
    c = b.tex(T("T_Fall_BC"), pa, S_COLOR)
    a1 = b.mask(b.tex(T("T_Fall_A"), pa, S_MASKS), "RGB", r=True)
    a2 = b.mask(b.tex(T("T_Fall_A"), pb, S_MASKS), "RGB", r=True)
    edge = b.mask(base, "", r=True)
    side = b.mul(b.smoothstep(edge, "", 0.0, 0.18), "", b.smoothstep(b.unary(unreal.MaterialExpressionOneMinus, edge), "", 0.0, 0.18), "")
    op = b.saturate(b.mul(b.mul(b.add(a1, "", a2, ""), "", k=0.55), "", side, ""))
    b.out(c, "RGB", MP.MP_BASE_COLOR)
    b.out(op, "", MP.MP_OPACITY)
    b.out(b.const(0.3), "", MP.MP_ROUGHNESS)
    b.out(b.mul(c, "RGB", k=0.04), "", MP.MP_EMISSIVE_COLOR)
    return b.finish(), b


def build_mist(name="M_Mist"):
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, blend_mode=unreal.BlendMode.BLEND_TRANSLUCENT, two_sided=True,
       translucency_lighting_mode=unreal.TranslucencyLightingMode.TLM_VOLUMETRIC_NON_DIRECTIONAL)
    f = b.mask(b.tex(T("T_Foam"), b.proj_uvs(1800.0)[1], S_MASKS), "RGB", r=True)
    # the dome's silhouette fades out, so it reads as a cloud of spray rather than a shell: x (1 - fresnel)
    edge = b.unary(unreal.MaterialExpressionOneMinus, b.node(unreal.MaterialExpressionFresnel, {"exponent": 1.5, "base_reflect_fraction": 0.0}))
    df = b.node(unreal.MaterialExpressionDepthFade, {"fade_distance_default": 400.0})
    b.conn(b.mul(b.mul(f, "", k=0.22), "", edge, ""), "", df, "")
    b.out(b.const3((0.85, 0.88, 0.9)), "", MP.MP_BASE_COLOR)
    b.out(df, "", MP.MP_OPACITY)
    return b.finish(), b


def build_sky_layer(name, tex, strength, uv_from_mesh=True):
    """Additive emissive (stars on the sky dome, the moon disc): invisible in daylight (x MPC Night), so the sky
    atmosphere and the clouds stay untouched."""
    b = MatBuilder(name)
    sp(unreal.Material, b.mat, shading_model=unreal.MaterialShadingModel.MSM_UNLIT, blend_mode=unreal.BlendMode.BLEND_ADDITIVE,
       two_sided=True)
    t = b.tex(T(tex), b.uv(0), S_COLOR)
    b.out(b.mul(b.mul(t, "RGB", k=strength), "", b.mpc("Night"), ""), "", MP.MP_EMISSIVE_COLOR)
    return b.finish(), b


MINIMAL = {"M_Terrain": "Granite", "M_CastleStone": "CastleStone", "M_Trim": "Trim", "M_Boulder": "Granite"}


def build_materials():
    REPORT.step("material parameter collection", build_mpc)
    mats = {}
    specs = [
        ("M_Terrain", build_terrain),
        ("M_Boulder", build_boulder),
        ("M_CastleStone", lambda: build_masonry("M_CastleStone", "CastleStone", 200.0, 1.0)),
        ("M_Trim", lambda: build_masonry("M_Trim", "Trim", 200.0, 0.6)),
        ("M_Slate", build_slate),
        ("M_Lead", lambda: build_simple("M_Lead", "Lead", metal=0.4)),
        ("M_Glass", build_glass),
        ("M_Wood", lambda: build_simple("M_Wood", "Wood", uv_tile=0.5)),
        ("M_Cloth", build_cloth),
        ("M_Marble", lambda: build_simple("M_Marble", None, (0.78, 0.77, 0.73), 0.35)),
        ("M_Lantern", lambda: build_emissive("M_Lantern", (1.0, 0.58, 0.22), 60.0)),
        ("M_Water", lambda: build_water("M_Water", False)),
        ("M_River", lambda: build_water("M_River", True)),
        ("M_Fall", build_fall),
        ("M_Mist", build_mist),
        ("M_Bark", lambda: build_bark("M_Bark", "Bark")),
        ("M_BirchBark", lambda: build_bark("M_BirchBark", "BirchBark")),
        ("M_DeadWood", lambda: build_bark("M_DeadWood", "Bark", tint=(0.95, 0.95, 1.0))),
        ("M_PineBark", lambda: build_bark("M_PineBark", "Bark", tint=(1.5, 0.85, 0.6))),
        ("M_Needles", lambda: build_foliage("M_Needles", "T_Needles_BC", "T_Needles_N", (0.78, 0.88, 0.74), (1.12, 1.05, 0.86), 0.55, 0.3)),
        ("M_Leaf", lambda: build_foliage("M_Leaf", "T_Leaf_BC", "T_Leaf_N", (0.82, 0.95, 0.70), (1.20, 1.10, 0.75), 0.45, 0.35)),
        ("M_Fern3D", lambda: build_foliage("M_Fern3D", "T_FernPinna_BC", None, rough=0.5)),
        ("M_Grass3D", lambda: build_foliage("M_Grass3D", "T_GrassBlade_BC", None, rough=0.55)),
        ("M_MossClump", build_moss_clump),
        ("M_Stars", lambda: build_sky_layer("M_Stars", "T_Stars", 2.0)),
        ("M_Moon", lambda: build_sky_layer("M_Moon", "T_Moon", 40.0)),
    ]
    for name, fn in specs:
        try:
            m, b = fn()
            if b.fails and name in MINIMAL:
                REPORT.note(f"{name}: {b.fails} wiring problems - rebuilding with a minimal graph")
                m, _ = build_simple(name, MINIMAL[name], uv_tile=0.25)
            mats[name] = m
        except Exception as e:                                     # noqa: BLE001
            REPORT.note(f"material {name} failed: {e}")
            if name in MINIMAL:
                try:
                    mats[name], _ = build_simple(name, MINIMAL[name], uv_tile=0.25)
                except Exception as e2:                            # noqa: BLE001
                    REPORT.note(f"material {name} minimal fallback failed too: {e2}")
    return mats


# --------------------------------------------------------------------------- 5. level
def subsystem(cls):
    return unreal.get_editor_subsystem(cls)


def make_level():
    les = subsystem(unreal.LevelEditorSubsystem)
    path = f"{PATHS['Maps']}/{CFG['level_name']}"
    if not asset_exists(path):
        if not les.new_level(path):
            raise RuntimeError(f"could not create level {path}")
    else:
        les.load_level(path)
        eas = subsystem(unreal.EditorActorSubsystem)
        tag = unreal.Name(CFG["actor_tag"])
        n = 0
        for a in eas.get_all_level_actors():
            if tag in a.tags:
                eas.destroy_actor(a)
                n += 1
        log(f"removed {n} actors from a previous build")
    return path


def spawn(cls, label, loc=None, rot=None, folder=None):
    eas = subsystem(unreal.EditorActorSubsystem)
    a = eas.spawn_actor_from_class(cls, loc or unreal.Vector(0, 0, 0), rot or unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0))
    if a is None:
        raise RuntimeError(f"could not spawn {label}")
    a.set_actor_label(label)
    sp(unreal.Actor, a, tags=[unreal.Name(CFG["actor_tag"])])
    if folder:
        a.set_folder_path(folder)
    return a


def place_meshes(manifest, meshes, mats, corr):
    yaw, scale, k, ok = corr
    default = mats.get("M_CastleStone")
    placed = {}
    for e in manifest["meshes"]:
        if e.get("library"):
            continue
        m = meshes.get(e["name"])
        if m is None:
            continue
        a = spawn(unreal.StaticMeshActor, e["name"], rot=unreal.Rotator(roll=0.0, pitch=0.0, yaw=yaw),
                  folder=f"WizardingWorld/{e.get('kind', 'other').capitalize()}")
        if not ok:
            a.set_actor_scale3d(scale)
        comp = a.static_mesh_component
        comp.set_static_mesh(m)
        try:
            comp.set_mobility(unreal.ComponentMobility.STATIC)
        except Exception:                                          # noqa: BLE001
            pass
        for i, slot in enumerate(m.static_materials):
            mat = mats.get(str(slot.material_slot_name)) or default
            if mat is not None:
                comp.set_material(i, mat)
        comp.set_cast_shadow(bool(e.get("shadow", True)))
        if e.get("kind") in ("water", "waterfall"):
            sp(unreal.StaticMeshComponent, comp, affect_distance_field_lighting=False)
        comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        placed[e["name"]] = a
    log(f"{len(placed)} mesh actors placed")
    return placed


# ---- instances ---------------------------------------------------------------------------------------------------
def read_instance_records(path):
    """float32 little-endian records: x y z (cm), qx qy qz qw, sx sy sz (already in Unreal's frame)."""
    import array

    a = array.array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def add_ism_component(actor):
    """Add an InstancedStaticMeshComponent to a placed actor the way the Details panel's '+ Add' does it."""
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    lib = unreal.SubobjectDataBlueprintFunctionLibrary
    handles = sds.k2_gather_subobject_data_for_instance(actor)
    if not handles:
        raise RuntimeError("no subobject data for the instance actor")
    params = unreal.AddNewSubobjectParams(parent_handle=handles[0], new_class=unreal.InstancedStaticMeshComponent)
    handle, fail = sds.add_new_subobject(params)
    if not lib.is_handle_valid(handle):
        raise RuntimeError(f"add_new_subobject failed: {fail}")
    comp = lib.get_object(lib.get_data(handle))
    if comp is None:
        raise RuntimeError("the new instanced static mesh component could not be resolved")
    return comp


def _place_instance_set(src, sname, entries, meshes, mats, corr_t, density):
    eas = subsystem(unreal.EditorActorSubsystem)
    made, total = [], 0
    batch_n = int(CFG.get("instance_batch", 20000))
    try:
        for e in entries:
            mesh = meshes.get(e["mesh"])
            if mesh is None:
                raise RuntimeError(f"library mesh {e['mesh']} was not imported")
            rec = read_instance_records(os.path.join(src, "fbx", e["file"]))
            n = len(rec) // 10
            if n != e["count"]:
                raise RuntimeError(f"{e['file']}: {n} records but the manifest says {e['count']}")
            a = spawn(unreal.Actor, f"Inst_{sname}_{e['variant']:02d}", folder=f"WizardingWorld/Instances/{sname}")
            made.append(a)
            comp = add_ism_component(a)
            comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
            comp.set_static_mesh(mesh)
            for i, slot in enumerate(mesh.static_materials):
                mat = mats.get(str(slot.material_slot_name)) or mats.get("M_Terrain")
                if mat is not None:
                    comp.set_material(i, mat)
            comp.set_cast_shadow(bool(e.get("shadow", True)))
            try:
                comp.set_mobility(unreal.ComponentMobility.STATIC)
            except Exception:                                      # noqa: BLE001
                pass
            if e.get("foliage") or sname in ("moss", "grass"):
                sp(unreal.StaticMeshComponent, comp, affect_distance_field_lighting=False)
            batch, added = [], 0
            for i in range(n):
                if density < 1.0 and (i * 0.6180339887498949) % 1.0 >= density:
                    continue
                o = 10 * i
                t = unreal.Transform()
                t.translation = unreal.Vector(rec[o], rec[o + 1], rec[o + 2])
                t.rotation = unreal.Quat(rec[o + 3], rec[o + 4], rec[o + 5], rec[o + 6])
                t.scale3d = unreal.Vector(rec[o + 7], rec[o + 8], rec[o + 9])
                if corr_t is not None:
                    t = corr_t.multiply(t)
                batch.append(t)
                if len(batch) >= batch_n:
                    comp.add_instances(batch, False, False)
                    added += len(batch)
                    batch = []
            if batch:
                comp.add_instances(batch, False, False)
                added += len(batch)
            got = comp.get_instance_count()
            if got != added:
                raise RuntimeError(f"{a.get_actor_label()}: {got} instances in the component, {added} added")
            total += added
    except Exception:
        for a in made:
            eas.destroy_actor(a)
        raise
    log(f"{sname}: {total:,} instances in {len(made)} instanced components")
    return total


def place_instances(src, manifest, meshes, mats, corr):
    if not CFG.get("instances", True):
        log("instancing disabled in CFG")
        return set()
    yaw, scale, k, ok = corr
    corr_t = None if ok else unreal.Transform(unreal.Vector(0, 0, 0), unreal.Rotator(roll=0.0, pitch=0.0, yaw=yaw), scale)
    density = max(0.0, min(1.0, float(CFG.get("instance_density", 1.0))))
    want = CFG.get("instance_sets")
    by_set = {}
    for e in manifest.get("instances", []):
        by_set.setdefault(e["set"], []).append(e)
    done, total = set(), 0
    for sname in sorted(by_set):
        if want and sname not in want:
            continue
        d = 1.0 if sname in ("boats", "willow") else density
        r = REPORT.step(f"instances {sname}", _place_instance_set, src, sname, by_set[sname], meshes, mats, corr_t, d)
        if r is not None:
            done.add(sname)
            total += r
    log(f"instances: {total:,} in {len(done)} sets")
    return done


# ---- sky, light, fog -----------------------------------------------------------------------------------------------
def sun_rotation(elev_deg, az_deg):
    """Directional light orientation (it shines along its forward axis, away from the sun).  Azimuth is clockwise from
    north; north is +Y in the generator = -Y in Unreal, east is +X in both: light from the north travels to +Y (yaw 90),
    from the east to -X (yaw 180)  ->  yaw = 90 + azimuth, pitch = -elevation."""
    return unreal.Rotator(roll=0.0, pitch=-elev_deg, yaw=(90.0 + az_deg) % 360.0)


def build_atmosphere(scene, mats, meshes):
    actors = {}
    sun = spawn(unreal.DirectionalLight, "Sun", folder="WizardingWorld/Lighting")
    sc = sun.get_component_by_class(unreal.DirectionalLightComponent)
    sp(unreal.DirectionalLightComponent, sc, atmosphere_sun_light=True, atmosphere_sun_light_index=0, light_source_angle=0.55,
       cast_shadows=True, cast_cloud_shadows=True, volumetric_scattering_intensity=1.0)
    actors["sun"] = sun
    moon = spawn(unreal.DirectionalLight, "MoonLight", folder="WizardingWorld/Lighting")
    mc = moon.get_component_by_class(unreal.DirectionalLightComponent)
    sp(unreal.DirectionalLightComponent, mc, atmosphere_sun_light=True, atmosphere_sun_light_index=1, light_source_angle=0.6,
       cast_shadows=True, intensity=0.0, light_color=unreal.Color(170, 195, 255, 255), volumetric_scattering_intensity=2.0)
    actors["moon"] = moon
    sky_atm = spawn(unreal.SkyAtmosphere, "SkyAtmosphere", folder="WizardingWorld/Lighting")
    actors["atmosphere"] = sky_atm
    sky = spawn(unreal.SkyLight, "SkyLight", loc=unreal.Vector(0, 0, 50000), folder="WizardingWorld/Lighting")
    sk = sky.get_component_by_class(unreal.SkyLightComponent)
    sp(unreal.SkyLightComponent, sk, real_time_capture=True, cast_shadows=True, lower_hemisphere_is_black=False, volumetric_scattering_intensity=1.0)
    actors["sky"] = sky
    clouds = spawn(unreal.VolumetricCloud, "VolumetricCloud", folder="WizardingWorld/Lighting")
    cc = clouds.get_component_by_class(unreal.VolumetricCloudComponent)
    sp(unreal.VolumetricCloudComponent, cc, layer_bottom_altitude=1.2, layer_height=4.5)
    actors["clouds"] = clouds
    fog = spawn(unreal.ExponentialHeightFog, "ExponentialHeightFog", loc=unreal.Vector(0, 0, 0), folder="WizardingWorld/Lighting")
    fc = fog.get_component_by_class(unreal.ExponentialHeightFogComponent)
    sp(unreal.ExponentialHeightFogComponent, fc, enable_volumetric_fog=True, volumetric_fog_distance=600000.0,
       volumetric_fog_scattering_distribution=0.72, volumetric_fog_albedo=unreal.Color(235, 240, 245, 255))
    actors["fog"] = fog
    ppv = spawn(unreal.PostProcessVolume, "PostProcess", folder="WizardingWorld/Lighting")
    sp(unreal.PostProcessVolume, ppv, unbound=True, priority=1.0)
    actors["pp"] = ppv
    # local fog volumes: mist banks in the gorge, over the lake below the castle, in the ravine (mist / sunset / night)
    banks = []
    for i, fv in enumerate(scene.get("fog_volumes", [])):
        try:
            v = spawn(unreal.LocalFogVolume, f"Mist_{fv['name']}", loc=to_ue(fv["center"]), folder="WizardingWorld/Lighting/Mist")
            v.set_actor_scale3d(unreal.Vector(fv["size"][0], fv["size"][1], fv["size"][2]))
            vc = v.get_component_by_class(unreal.LocalFogVolumeComponent)
            sp(unreal.LocalFogVolumeComponent, vc, radial_fog_extinction=0.0, height_fog_extinction=0.0, height_fog_falloff=fv.get("falloff", 1000.0),
               fog_albedo=unreal.LinearColor(0.85, 0.88, 0.9, 1.0))
            banks.append((v, fv))
        except Exception as e:                                     # noqa: BLE001
            REPORT.note(f"local fog volume {fv.get('name')} failed: {e}")
    actors["banks"] = banks
    # night sky: star dome + moon disc (additive, scaled by MPC Night)
    for key, mesh_name, mat_name in (("stars", "SM_SkyDome", "M_Stars"), ("moon_disc", "SM_MoonDisc", "M_Moon")):
        m = meshes.get(mesh_name)
        if m is None:
            continue
        a = spawn(unreal.StaticMeshActor, key, folder="WizardingWorld/Lighting")
        comp = a.static_mesh_component
        comp.set_static_mesh(m)
        if mats.get(mat_name):
            comp.set_material(0, mats[mat_name])
        comp.set_cast_shadow(False)
        comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        sp(unreal.StaticMeshComponent, comp, affect_distance_field_lighting=False)
        actors[key] = a
    # boat lanterns: a warm point light at every lantern (switched on by the night preset)
    lanterns = []
    for i, p in enumerate(scene.get("lanterns", [])):
        lt = spawn(unreal.PointLight, f"Lantern_{i:02d}", loc=unreal.Vector(p[0], p[1], p[2]), folder="WizardingWorld/Lighting/Lanterns")
        lc = lt.get_component_by_class(unreal.PointLightComponent)
        sp(unreal.PointLightComponent, lc, intensity=0.0, light_color=unreal.Color(255, 160, 80, 255), attenuation_radius=2500.0,
           source_radius=6.0, cast_shadows=True)
        lanterns.append(lt)
    actors["lanterns"] = lanterns
    ACTORS.update(actors)
    return actors


ACTORS = {}
SCENE = {}


def preset_values(name):
    P = dict(SCENE["presets"][name])
    P["fog_density_ue"] = P["fog_density"] * 1.0
    return P


def apply_preset(name):
    """Light the editor level for one preset (mist, day, sunset, dusk, night): sun / moon, sky, fog, mist banks, clouds,
    exposure, window glow, lanterns.  The sequences key the same values, so Movie Render Queue renders each shot with
    its own preset."""
    if not ACTORS:
        _rebind_actors()
    P = preset_values(name)
    night = bool(P.get("moon"))
    sun, moon = ACTORS.get("sun"), ACTORS.get("moon")
    if sun:
        sun.set_actor_rotation(sun_rotation(P["sun_elev"] if not night else -20.0, P["sun_az"]), False)
        sc = sun.get_component_by_class(unreal.DirectionalLightComponent)
        r, g, b = P["sun_color"]
        sp(unreal.DirectionalLightComponent, sc, intensity=0.0 if night else float(P["sun_lux"]), light_color=unreal.Color(r, g, b, 255))
    if moon:
        moon.set_actor_rotation(sun_rotation(P["sun_elev"], P["sun_az"]), False)
        mc = moon.get_component_by_class(unreal.DirectionalLightComponent)
        sp(unreal.DirectionalLightComponent, mc, intensity=float(P["sun_lux"]) if night else 0.0)
    if ACTORS.get("sky"):
        sp(unreal.SkyLightComponent, ACTORS["sky"].get_component_by_class(unreal.SkyLightComponent), intensity=float(P["sky"]))
    if ACTORS.get("fog"):
        fc = ACTORS["fog"].get_component_by_class(unreal.ExponentialHeightFogComponent)
        sp(unreal.ExponentialHeightFogComponent, fc, fog_density=float(P["fog_density"]), fog_height_falloff=float(P["fog_falloff"]),
           volumetric_fog_extinction_scale=1.0 if P.get("vol_fog") else 0.0)
        ACTORS["fog"].set_actor_location(unreal.Vector(0, 0, 100.0 * float(P.get("fog_height", 0.0))), False, False)
    for v, fv in ACTORS.get("banks", []):
        vc = v.get_component_by_class(unreal.LocalFogVolumeComponent)
        k = fv.get("strength", {}).get(name, 0.0)
        sp(unreal.LocalFogVolumeComponent, vc, radial_fog_extinction=float(fv.get("radial", 0.6)) * k, height_fog_extinction=float(fv.get("height", 1.0)) * k)
    if ACTORS.get("pp"):
        ppv = ACTORS["pp"]
        s = ppv.get_editor_property("settings")
        t = P.get("tint", (1.0, 1.0, 1.0))
        sat = P.get("saturation", 1.0)
        sp(unreal.PostProcessSettings, s, override_auto_exposure_bias=True, auto_exposure_bias=float(P["exposure"]),
           override_auto_exposure_method=True, auto_exposure_method=unreal.AutoExposureMethod.AEM_HISTOGRAM,
           override_bloom_intensity=True, bloom_intensity=0.6 if night else 0.45,
           override_vignette_intensity=True, vignette_intensity=0.35,
           override_film_grain_intensity=True, film_grain_intensity=0.05,
           override_color_saturation=True, color_saturation=unreal.Vector4(sat, sat, sat, 1.0),
           override_scene_color_tint=True, scene_color_tint=unreal.LinearColor(t[0], t[1], t[2], 1.0),
           override_motion_blur_amount=True, motion_blur_amount=0.0,
           override_lumen_scene_lighting_quality=True, lumen_scene_lighting_quality=2.0,
           override_lumen_final_gather_quality=True, lumen_final_gather_quality=2.0,
           override_lumen_reflection_quality=True, lumen_reflection_quality=2.0)
        ppv.set_editor_property("settings", s)
    for lt in ACTORS.get("lanterns", []):
        sp(unreal.PointLightComponent, lt.get_component_by_class(unreal.PointLightComponent), intensity=float(CFG["lantern_intensity"]) if night else 0.0)
    set_mpc("Night", 1.0 if night else 0.0)
    set_mpc("WindowGlow", float(CFG["window_glow"]) * float(P.get("windows", 0.0)))
    log(f"preset '{name}' applied")


def _rebind_actors():
    """Find the actors of a previous build (so apply_preset works in a fresh editor session)."""
    eas = subsystem(unreal.EditorActorSubsystem)
    tag = unreal.Name(CFG["actor_tag"])
    lanterns, banks = [], []
    for a in eas.get_all_level_actors():
        if tag not in a.tags:
            continue
        lab = a.get_actor_label()
        key = {"Sun": "sun", "MoonLight": "moon", "SkyLight": "sky", "ExponentialHeightFog": "fog", "PostProcess": "pp",
               "stars": "stars", "moon_disc": "moon_disc", "VolumetricCloud": "clouds"}.get(lab)
        if key:
            ACTORS[key] = a
        elif lab.startswith("Lantern_"):
            lanterns.append(a)
        elif lab.startswith("Mist_"):
            fv = next((f for f in SCENE.get("fog_volumes", []) if f"Mist_{f['name']}" == lab), None)
            if fv:
                banks.append((a, fv))
    ACTORS["lanterns"] = lanterns
    ACTORS["banks"] = banks
    if MPC["asset"] is None and asset_exists(f"{PATHS['Materials']}/MPC_World"):
        MPC["asset"] = load(f"{PATHS['Materials']}/MPC_World")


# --------------------------------------------------------------------------- 6. cameras + sequences
def unwrap(angles):
    out = [angles[0]]
    for a in angles[1:]:
        while a - out[-1] > 180:
            a -= 360
        while a - out[-1] < -180:
            a += 360
        out.append(a)
    return out


def look_rot(frm, to):
    return unreal.MathLibrary.find_look_at_rotation(to_ue(frm), to_ue(to))


def make_camera(name, cam):
    keys = cam["keys"]
    p0 = keys[0]
    a = spawn(unreal.CineCameraActor, name, loc=to_ue(p0["loc"]), rot=look_rot(p0["loc"], p0["look_at"]), folder="WizardingWorld/Cameras")
    cc = a.get_cine_camera_component()
    res = cam.get("res", [3840, 2160])
    fb = cc.get_editor_property("filmback")
    if res[0] >= res[1]:
        sp(unreal.CameraFilmbackSettings, fb, sensor_width=36.0, sensor_height=36.0 * res[1] / res[0])
    else:
        sp(unreal.CameraFilmbackSettings, fb, sensor_width=36.0 * res[0] / res[1], sensor_height=36.0)
    cc.set_editor_property("filmback", fb)
    ls = cc.get_editor_property("lens_settings")
    sp(unreal.CameraLensSettings, ls, min_focal_length=4.0, max_focal_length=1000.0, min_f_stop=1.2, max_f_stop=32.0)
    cc.set_editor_property("lens_settings", ls)
    sp(unreal.CineCameraComponent, cc, current_focal_length=float(cam["lens"]), current_aperture=float(cam.get("fstop", 8.0)))
    fs = cc.get_editor_property("focus_settings")
    dx = [p0["look_at"][i] - p0["loc"][i] for i in range(3)]
    dist_cm = 100.0 * math.sqrt(sum(v * v for v in dx))
    sp(unreal.CameraFocusSettings, fs, focus_method=unreal.CameraFocusMethod.MANUAL, manual_focus_distance=float(dist_cm))
    cc.set_editor_property("focus_settings", fs)
    return a


def _const_float_track(binding, prop_name, prop_path, value, last):
    track = binding.add_track(unreal.MovieSceneFloatTrack)
    track.set_property_name_and_path(prop_name, prop_path)
    sec = track.add_section()
    sec.set_range(0, last)
    ch = sec.get_all_channels()[0]
    ch.add_key(unreal.FrameNumber(0), float(value), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE)


def _component_binding(seq, actor, comp_cls):
    comp = actor.get_component_by_class(comp_cls)
    return seq.add_possessable(comp)


def key_preset(seq, preset, last):
    """Key the preset's lighting into a sequence (sun / moon intensity + direction, sky light, fog, mist banks, lanterns,
    MPC_World).  Every part is best effort: a failure is logged and the editor values (apply_preset) are used."""
    P = preset_values(preset)
    night = bool(P.get("moon"))
    steps = []

    def safe(label, fn):
        try:
            fn()
            steps.append(label)
        except Exception as e:                                     # noqa: BLE001
            REPORT.note(f"{seq.get_name()}: preset track '{label}' failed: {str(e)[:120]}")

    def sun_tracks():
        for key, on, elev in (("sun", not night, P["sun_elev"]), ("moon", night, P["sun_elev"])):
            actor = ACTORS.get(key)
            if actor is None:
                continue
            b = seq.add_possessable(actor)
            tr = b.add_track(unreal.MovieScene3DTransformTrack)
            sec = tr.add_section()
            sec.set_range(0, last)
            rot = sun_rotation(elev if on else -20.0, P["sun_az"])
            loc = actor.get_actor_location()
            vals = [loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw, 1.0, 1.0, 1.0]
            for c, v in zip(sec.get_all_channels(), vals):
                c.add_key(unreal.FrameNumber(0), float(v), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
            cb = _component_binding(seq, actor, unreal.DirectionalLightComponent)
            _const_float_track(cb, "Intensity", "Intensity", float(P["sun_lux"]) if on else 0.0, last)

    def sky_track():
        if ACTORS.get("sky"):
            _const_float_track(_component_binding(seq, ACTORS["sky"], unreal.SkyLightComponent), "Intensity", "Intensity", float(P["sky"]), last)

    def fog_tracks():
        if ACTORS.get("fog"):
            cb = _component_binding(seq, ACTORS["fog"], unreal.ExponentialHeightFogComponent)
            _const_float_track(cb, "FogDensity", "FogDensity", float(P["fog_density"]), last)
            _const_float_track(cb, "FogHeightFalloff", "FogHeightFalloff", float(P["fog_falloff"]), last)

    def bank_tracks():
        for v, fv in ACTORS.get("banks", []):
            k = fv.get("strength", {}).get(preset, 0.0)
            cb = _component_binding(seq, v, unreal.LocalFogVolumeComponent)
            _const_float_track(cb, "RadialFogExtinction", "RadialFogExtinction", float(fv.get("radial", 0.6)) * k, last)
            _const_float_track(cb, "HeightFogExtinction", "HeightFogExtinction", float(fv.get("height", 1.0)) * k, last)

    def lantern_tracks():
        for lt in ACTORS.get("lanterns", []):
            cb = _component_binding(seq, lt, unreal.PointLightComponent)
            _const_float_track(cb, "Intensity", "Intensity", float(CFG["lantern_intensity"]) if night else 0.0, last)

    def exposure_track():
        if ACTORS.get("pp"):
            b = seq.add_possessable(ACTORS["pp"])
            _const_float_track(b, "AutoExposureBias", "Settings.AutoExposureBias", float(P["exposure"]), last)

    def mpc_track():
        if MPC["asset"] is None:
            return
        tr = seq.add_track(unreal.MovieSceneMaterialParameterCollectionTrack)
        tr.set_editor_property("mpc", MPC["asset"])
        sec = tr.add_section()
        sec.set_range(0, last)
        sec.add_scalar_parameter_key("Night", unreal.FrameNumber(0), 1.0 if night else 0.0)
        sec.add_scalar_parameter_key("WindowGlow", unreal.FrameNumber(0), float(CFG["window_glow"]) * float(P.get("windows", 0.0)))

    safe("sun / moon", sun_tracks)
    safe("sky light", sky_track)
    safe("height fog", fog_tracks)
    safe("mist banks", bank_tracks)
    if night:
        safe("lanterns", lantern_tracks)
    safe("exposure", exposure_track)
    safe("MPC", mpc_track)
    return steps


def make_sequence(name, cam_actor, cam, fps=24):
    keys = cam["keys"]
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    path = f"{PATHS['Cinematics']}/{name}"
    if asset_exists(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    seq = tools.create_asset(name, PATHS["Cinematics"], unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(fps, 1))
    last = max(1, int(round(keys[-1]["t"] * fps)))
    seq.set_playback_start(0)
    seq.set_playback_end(last)
    binding = seq.add_possessable(cam_actor)
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    sec = track.add_section()
    sec.set_range(0, last)
    chans = sec.get_all_channels()
    locs = [to_ue(k["loc"]) for k in keys]
    rots = [look_rot(k["loc"], k["look_at"]) for k in keys]
    rolls = unwrap([r.roll for r in rots])
    pitches = unwrap([r.pitch for r in rots])
    yaws = unwrap([r.yaw for r in rots])
    for i, k in enumerate(keys):
        f = unreal.FrameNumber(int(round(k["t"] * fps)))
        vals = [locs[i].x, locs[i].y, locs[i].z, rolls[i], pitches[i], yaws[i], 1.0, 1.0, 1.0]
        for c, v in zip(chans, vals):
            c.add_key(f, float(v), 0.0, unreal.MovieSceneTimeUnit.DISPLAY_RATE)
    cut = seq.add_track(unreal.MovieSceneCameraCutTrack)
    cs = cut.add_section()
    cs.set_range(0, last)
    try:
        bid = unreal.MovieSceneSequenceExtensions.get_portable_binding_id(seq, seq, binding)
    except Exception:                                              # noqa: BLE001 - older engines
        bid = seq.get_binding_id(binding)
    cs.set_camera_binding_id(bid)
    steps = key_preset(seq, cam.get("preset", "day"), last)
    unreal.EditorAssetLibrary.save_loaded_asset(seq)
    log(f"{name}: {len(keys)} camera keys, {last} frames, preset '{cam.get('preset')}' keyed ({', '.join(steps)})")
    return seq


def queue_render_job(seq_name, level_path, resolution, label):
    """Best effort: the Movie Render Queue classes only exist when the plugin is enabled."""
    names = ["MoviePipelineQueueSubsystem", "MoviePipelineExecutorJob", "MoviePipelineOutputSetting", "MoviePipelineAntiAliasingSetting",
             "MoviePipelineDeferredPassBase", "MoviePipelineImageSequenceOutput_PNG"]
    cls = {n: getattr(unreal, n, None) for n in names}
    missing = [n for n, c in cls.items() if c is None]
    if missing:
        REPORT.note("Movie Render Queue plugin not available (" + ", ".join(missing) + "); add the job by hand - see README")
        return False
    sub = unreal.get_editor_subsystem(cls["MoviePipelineQueueSubsystem"])
    queue = sub.get_queue()
    for j in list(queue.get_jobs()):
        if str(j.get_editor_property("job_name")) == label:
            queue.delete_job(j)
    job = queue.allocate_new_job(cls["MoviePipelineExecutorJob"])
    sp(cls["MoviePipelineExecutorJob"], job, job_name=label)
    job.set_editor_property("map", unreal.SoftObjectPath(f"{level_path}.{CFG['level_name']}"))
    job.set_editor_property("sequence", unreal.SoftObjectPath(f"{PATHS['Cinematics']}/{seq_name}.{seq_name}"))
    cfg = job.get_configuration()
    out = cfg.find_or_add_setting_by_class(cls["MoviePipelineOutputSetting"])
    sp(cls["MoviePipelineOutputSetting"], out, output_resolution=unreal.IntPoint(resolution[0], resolution[1]),
       file_name_format="{sequence_name}.{frame_number}", override_existing_output=True, zero_pad_frame_numbers=4)
    try:
        d = unreal.DirectoryPath()
        d.set_editor_property("path", os.path.join(unreal.Paths.project_saved_dir(), "MovieRenders", label))
        out.set_editor_property("output_directory", d)
    except Exception as e:                                         # noqa: BLE001
        REPORT.note(f"render output directory not set ({str(e)[:80]}); set it in the Movie Render Queue")
    aa = cfg.find_or_add_setting_by_class(cls["MoviePipelineAntiAliasingSetting"])
    sp(cls["MoviePipelineAntiAliasingSetting"], aa, spatial_sample_count=CFG["spatial_samples"], temporal_sample_count=CFG["temporal_samples"],
       engine_warm_up_count=CFG["warmup_frames"], render_warm_up_count=CFG["warmup_frames"], use_camera_cut_for_warm_up=True,
       override_anti_aliasing=True)
    cfg.find_or_add_setting_by_class(cls["MoviePipelineDeferredPassBase"])
    cfg.find_or_add_setting_by_class(cls["MoviePipelineImageSequenceOutput_PNG"])
    if CFG.get("output_exr"):
        exr = getattr(unreal, "MoviePipelineImageSequenceOutput_EXR", None)
        if exr is not None:
            cfg.find_or_add_setting_by_class(exr)
    cv = getattr(unreal, "MoviePipelineConsoleVariableSetting", None)
    if cv is not None:
        try:
            c = cfg.find_or_add_setting_by_class(cv)
            for k, v in (("r.Shadow.Virtual.MaxPhysicalPages", 8192), ("r.Lumen.Reflections.DownsampleFactor", 1),
                         ("r.Lumen.ScreenProbeGather.DownsampleFactor", 8), ("r.VolumetricFog.GridPixelSize", 4),
                         ("r.VolumetricCloud.ViewRaySampleMaxCount", 512), ("r.ViewDistanceScale", 50)):
                c.add_or_update_console_variable(k, float(v))
        except Exception as e:                                     # noqa: BLE001
            REPORT.note(f"render console variables not set: {str(e)[:100]}")
    return True


def build_cinematics(scene, level_path):
    made = []
    for name, cam in scene["shots"].items():
        actor = make_camera(name, cam)
        made.append(name)
        seq_name = cam["sequence"]
        REPORT.step(f"sequence {seq_name}", make_sequence, seq_name, actor, cam)
        REPORT.step(f"render job {seq_name}", queue_render_job, seq_name, level_path, cam["res"], name)
    log(f"cameras created: {', '.join(made)}")


# --------------------------------------------------------------------------- 7. entry point
def save_all():
    try:
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    except Exception as e:                                         # noqa: BLE001
        REPORT.note(f"save_dirty_packages failed: {e}")
    try:
        subsystem(unreal.LevelEditorSubsystem).save_current_level()
    except Exception as e:                                         # noqa: BLE001
        REPORT.note(f"save_current_level failed: {e}")


def run():
    unreal.log("=" * 70)
    unreal.log("[WW] building the castle, the crag, the lake and the valley ...")
    src = REPORT.step("locate SourceAssets", find_source_dir)
    if not src:
        return
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    scene = json.load(open(os.path.join(src, "scene.json")))
    SCENE.update(scene)
    for p in PATHS.values():
        unreal.EditorAssetLibrary.make_directory(p)
    REPORT.step("import textures", import_textures, src)
    meshes = REPORT.step("import meshes", import_meshes, src, manifest) or {}
    corr = REPORT.step("calibrate scale/axes", calibrate, manifest, meshes) if meshes else None
    corr = corr or (0.0, unreal.Vector(1, 1, 1), 1.0, True)
    mats = REPORT.step("build materials", build_materials) or {}
    level_path = REPORT.step("create level", make_level)
    if level_path:
        REPORT.step("place meshes", place_meshes, manifest, meshes, mats, corr)
        REPORT.step("instanced vegetation, rocks and boats", place_instances, src, manifest, meshes, mats, corr)
        REPORT.step("sky, sun, moon, fog, clouds", build_atmosphere, scene, mats, meshes)
        REPORT.step("cameras and sequences", build_cinematics, scene, level_path)
        REPORT.step(f"apply preset '{CFG['start_preset']}'", apply_preset, CFG["start_preset"])
        REPORT.step("save", save_all)
    dt = time.time() - REPORT.t0
    unreal.log("=" * 70)
    unreal.log(f"[WW] finished in {dt / 60:.1f} min: {len(REPORT.ok)} steps ok, {len(REPORT.fail)} failed, {len(REPORT.warn)} warnings")
    for n, e in REPORT.fail:
        unreal.log_error(f"[WW]   FAILED  {n}: {e}")
    for w in REPORT.warn[:30]:
        unreal.log_warning(f"[WW]   note    {w}")
    unreal.log("[WW] open /Game/WizardingWorld/Maps/WizardingWorld; render with Window > Cinematics > Movie Render Queue (see README).")


if __name__ == "__main__":
    run()
