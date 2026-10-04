"""Export every generated mesh to FBX (centimetres, Z-up) + instance records + manifest.json for the Unreal builder.

FBX recipe (verified by parsing the output with fbx_inspect, as in the Flooded Rotunda pipeline):
    axis_forward='Y', axis_up='Z', bake_space_transform=True, apply_unit_scale=False, apply_scale_options='FBX_SCALE_NONE'
  -> vertices baked in centimetres, identity node transform, UnitScaleFactor 1; Blender (x, y, z) m -> Unreal (100x, -100y, 100z) cm.

Instances: GEO/instances/<set>.npz -> OUT/fbx/instances/<set>_<variant>.bin, little-endian float32 records
    x y z (cm), qx qy qz qw, sx sy sz   already in Unreal's frame (see instances.py).

usage: python export_world.py GEO_DIR OUT_DIR
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

import bpy

import blender_util as bu
import fbx_inspect as fi
import instances as ins
import world as W
from meshkit import Mesh

MAT_NAMES = {0: "M_Terrain", 1: "M_Boulder", 6: "M_Fern3D", 13: "M_MossClump", 14: "M_Fern3D", 15: "M_Grass3D",
             20: "M_CastleStone", 21: "M_Trim", 22: "M_Slate", 23: "M_Lead", 24: "M_Glass", 25: "M_Wood",
             26: "M_Cloth", 28: "M_Marble", 30: "M_Water", 31: "M_River", 32: "M_Fall", 33: "M_Mist", 34: "M_Lantern",
             35: "M_Stars", 36: "M_Moon",
             40: "M_Bark", 41: "M_Needles", 42: "M_Leaf", 43: "M_DeadWood", 44: "M_BirchBark", 45: "M_PineBark"}

# instance set -> library mesh per variant
INSTANCE_SETS = {
    "spruce": lambda v: f"lib_spruce_{v:02d}", "spruceyoung": lambda v: f"lib_spruceyoung_{v:02d}", "pine": lambda v: "lib_pine_00",
    "snag": lambda v: "lib_snag_00", "birch": lambda v: f"lib_birch_{v:02d}", "willow": lambda v: "lib_willow_00",
    "boulder": lambda v: f"lib_boulder_{v:02d}", "moss": lambda v: f"lib_moss_{v:02d}", "fern": lambda v: f"lib_fern_{v:02d}",
    "grass": lambda v: f"lib_grass_{v:02d}", "boats": lambda v: "lib_boat_00",
}
FOLIAGE_SETS = {"spruce", "spruceyoung", "pine", "snag", "birch", "willow", "fern", "grass"}

# how Unreal should treat each asset:  kind (folder / tag), nanite, shadow, and whether Lumen's distance fields matter
RULES = [
    ("terrain_", dict(kind="terrain", nanite=True, shadow=True)),
    ("castle_", dict(kind="architecture", nanite=True, shadow=True)),
    ("grounds_", dict(kind="architecture", nanite=True, shadow=True)),
    ("water_fall", dict(kind="waterfall", nanite=False, shadow=False)),
    ("water_mist", dict(kind="waterfall", nanite=False, shadow=False)),
    ("water_", dict(kind="water", nanite=False, shadow=False)),
    ("sky_", dict(kind="sky", nanite=False, shadow=False)),
]


def rules_for(name):
    for prefix, r in RULES:
        if name.startswith(prefix):
            return dict(r)
    return dict(kind="other", nanite=True, shadow=True)


def pascal(name):
    return "".join(p.capitalize() for p in name.split("_"))


def _export_mesh(m, base, out_dir, mats, extra, normals_angle):
    ids = sorted(set(int(i) for i in m.mat))
    remap = {old: new for new, old in enumerate(ids)}
    m.mat = np.array([remap[int(i)] for i in m.mat], np.int16)
    slots = [MAT_NAMES.get(i, "M_Terrain") for i in ids]
    for s in slots:
        if s not in mats:
            mats[s] = bpy.data.materials.new(s)
    name = "SM_" + pascal(base)
    for k in ("uv0", "uv1", "uv2", "uv3"):
        a = getattr(m, k)
        if a is not None:
            setattr(m, k, np.nan_to_num(a))
    if m.uv0 is None:
        m.uv_box(2.0, only_missing=False)
    ob = bu.add_object(name, m, [mats[s] for s in slots], normals_angle=normals_angle)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    fbx_path = f"{out_dir}/{name}.fbx"
    bpy.ops.export_scene.fbx(
        filepath=fbx_path, use_selection=True, object_types={"MESH"}, use_mesh_modifiers=False,
        add_leaf_bones=False, bake_anim=False, path_mode="STRIP", embed_textures=False, use_triangles=True,
        mesh_smooth_type="OFF", colors_type="NONE", use_tspace=False,
        axis_forward="Y", axis_up="Z", bake_space_transform=True,
        apply_unit_scale=False, apply_scale_options="FBX_SCALE_NONE", global_scale=1.0,
    )
    lo, hi = m.V.min(0), m.V.max(0)
    fn = m.face_normals()
    entry = dict(
        name=name, file=f"{name}.fbx", source=base, tris=int(m.nf), slots=slots,
        bbox_center_m=((lo + hi) / 2).round(4).tolist(), bbox_extent_m=((hi - lo) / 2).round(4).tolist(),
        uv_layers=[n for n, a in zip(("UV0", "UV1", "UV2", "UV3"), (m.uv0, m.uv1, m.uv2, m.uv3)) if a is not None],
        size_mb=round(os.path.getsize(fbx_path) / 1e6, 2), mean_up=round(float(fn[:, 2].mean()), 3), **extra,
    )
    bpy.data.objects.remove(ob)
    if name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[name])
    print(f"  {name:30s} {m.nf:9,d} tris  {entry['size_mb']:7.2f} MB  slots={slots}", flush=True)
    return entry


def export_instances(geo_dir, out_dir, lib_names):
    entries = []
    os.makedirs(f"{out_dir}/instances", exist_ok=True)
    for path in sorted(glob.glob(f"{geo_dir}/instances/*.npz")):
        sname = os.path.splitext(os.path.basename(path))[0]
        if sname not in INSTANCE_SETS:
            print(f"  (skipping unknown instance set {sname})")
            continue
        d = ins.load_set(path)
        rec = ins.ue_records(d["pos"], d["R"], d["scale"])
        for v in sorted(set(int(x) for x in d["variant"])):
            lib = INSTANCE_SETS[sname](v)
            if lib not in lib_names:
                print(f"  !! {sname} variant {v}: library mesh {lib} missing - skipped")
                continue
            sel = d["variant"] == v
            fn = f"instances/{sname}_{v:02d}.bin"
            rec[sel].tofile(f"{out_dir}/{fn}")
            entries.append(dict(set=sname, variant=v, mesh="SM_" + pascal(lib), file=fn, count=int(sel.sum()),
                                shadow=sname not in ("grass", "moss"), foliage=sname in FOLIAGE_SETS,
                                bbox_min_m=d["pos"][sel].min(0).round(2).tolist(), bbox_max_m=d["pos"][sel].max(0).round(2).tolist()))
        print(f"  instances {sname:12s} {len(rec):9,d}", flush=True)
    return entries


def export_all(geo_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bu.reset_scene()
    mats = {}
    manifest = {"unit": "cm", "mapping": "ue = (100*x, -100*y, 100*z) from Blender metres", "meshes": [], "instances": [],
                "instance_format": "float32 LE records: x y z (cm), qx qy qz qw, sx sy sz - Unreal frame"}
    for path in sorted(glob.glob(f"{geo_dir}/*.npz")):
        base = os.path.splitext(os.path.basename(path))[0]
        if base in ("core_columns", "core_top"):
            continue
        m = Mesh.load(path)
        if m.nf == 0:
            continue
        extra = rules_for(base)
        smooth = None if base.startswith(("terrain_", "water_", "sky_")) else 35.0
        manifest["meshes"].append(_export_mesh(m, base, out_dir, mats, extra, smooth))
    lib_paths = sorted(glob.glob(f"{geo_dir}/lib/lib_*.npz"))
    have_lib = {os.path.splitext(os.path.basename(p))[0] for p in lib_paths}
    for path in lib_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        m = Mesh.load(path)
        if base.startswith("lib_boulder"):
            m.mat[:] = 1                                       # boulders get their own rock material (per-instance variation)
        foliage = base.startswith(("lib_spruce", "lib_pine", "lib_snag", "lib_birch", "lib_willow", "lib_fern", "lib_grass", "lib_moss"))
        manifest["meshes"].append(_export_mesh(m, base, out_dir, mats, dict(kind="library", nanite=True, shadow=True, library=True,
                                                                           foliage=foliage), None if foliage else 30.0))
    manifest["instances"] = export_instances(geo_dir, out_dir, have_lib)
    json.dump(manifest, open(f"{out_dir}/manifest.json", "w"), indent=1)
    return manifest


def write_scene(geo_dir, out_path):
    """scene.json for the Unreal builder: lighting presets, the shots (cameras, keys, preset), mist banks, lanterns."""
    import shots as SH
    import scatter

    resolved = SH.resolve(lambda x, y: float(scatter.surface(np.array([x]), np.array([y]))[0][0]))
    lanterns = []
    bp = f"{geo_dir}/instances/boats.npz"
    if os.path.isfile(bp):
        d = ins.load_set(bp)
        for p, R in zip(d["pos"], d["R"]):
            q = p + R.astype(np.float64) @ np.array([2.12, 0.0, 0.98])
            lanterns.append([round(100.0 * q[0], 1), round(-100.0 * q[1], 1), round(100.0 * q[2], 1)])
    fog_volumes = [
        dict(name="Gorge", center=[-215.0, -40.0, 16.0], size=[90.0, 300.0, 26.0], radial=0.45, height=0.7,
             strength=dict(mist=1.0, sunset=0.35, dusk=0.6, night=0.5)),
        dict(name="LakeUnderCliff", center=[40.0, -330.0, 5.0], size=[560.0, 240.0, 16.0], radial=0.35, height=0.6,
             strength=dict(mist=0.9, sunset=0.6, dusk=0.7, night=0.8)),
        dict(name="Ravine", center=[226.0, -90.0, 30.0], size=[50.0, 120.0, 30.0], radial=0.5, height=0.6,
             strength=dict(mist=1.0, sunset=0.3, dusk=0.5, night=0.6)),
        dict(name="ValleyForest", center=[420.0, 650.0, 82.0], size=[900.0, 650.0, 30.0], radial=0.25, height=0.5,
             strength=dict(mist=1.0, dusk=0.5, night=0.3)),
        dict(name="WestHills", center=[-700.0, 100.0, 95.0], size=[700.0, 900.0, 40.0], radial=0.2, height=0.5,
             strength=dict(mist=0.8, dusk=0.4)),
    ]
    scene = dict(world=dict(lake_z=W.LAKE_Z, castle_z=W.CASTLE_Z, extent_m=W.EXTENT, moon_dir=list(SH.MOON_DIR)),
                 presets=SH.PRESETS, shots=resolved, fog_volumes=fog_volumes, lanterns=lanterns)
    json.dump(scene, open(out_path, "w"), indent=1)
    print(f"  scene.json: {len(resolved)} shots, {len(SH.PRESETS)} presets, {len(fog_volumes)} mist banks, {len(lanterns)} lanterns")


def verify_instances(out_dir, geo_dir, n_check=150):
    manifest = json.load(open(f"{out_dir}/manifest.json"))
    bad = 0
    rng = np.random.default_rng(5)
    E = W.EXTENT * 100.0 + 1000.0
    for e in manifest.get("instances", []):
        raw = np.fromfile(f"{out_dir}/{e['file']}", dtype="<f4")
        problems = []
        if raw.size != e["count"] * 10:
            problems.append(f"size {raw.size} != {e['count']}*10")
        rec = raw.reshape(-1, 10)
        if not np.isfinite(rec).all():
            problems.append("non-finite values")
        qn = np.linalg.norm(rec[:, 3:7], axis=1)
        if np.abs(qn - 1).max() > 1e-3:
            problems.append(f"quaternion norm off by {np.abs(qn - 1).max():.2e}")
        if np.abs(rec[:, :2]).max() > E or rec[:, 2].min() < -10000 or rec[:, 2].max() > 150000:
            problems.append("instances outside the world")
        src = ins.load_set(f"{geo_dir}/instances/{e['set']}.npz")
        sel = np.nonzero(src["variant"] == e["variant"])[0]
        Vl = Mesh.load(f"{geo_dir}/lib/{INSTANCE_SETS[e['set']](e['variant'])}.npz").V
        Vl = Vl[rng.choice(len(Vl), min(len(Vl), 48), replace=False)]
        Mx = np.array([1.0, -1.0, 1.0])
        worst = 0.0
        for k in rng.choice(len(sel), min(len(sel), n_check), replace=False):
            i = sel[k]
            pb = src["pos"][i] + (src["R"][i].astype(np.float64) @ (Vl * src["scale"][i]).T).T
            expect = pb * Mx * 100.0
            Rq = ins.matrix_from_quat(rec[k, 3:7].astype(np.float64)[None])[0]
            got = rec[k, :3] + (Rq @ ((Vl * Mx) * 100.0 * rec[k, 7:10]).T).T
            worst = max(worst, float(np.abs(got - expect).max()))
        if worst > 0.5:
            problems.append(f"frame mismatch {worst:.3f} cm")
        bad += bool(problems)
        print(f"  [{'OK ' if not problems else 'BAD'}] {e['file']:28s} " + ("; ".join(problems) if problems else
              f"{e['count']:8,d} inst -> {e['mesh']}, max placement error {worst:.3f} cm"))
    return bad


def verify(out_dir, tol_cm=1.0):
    manifest = json.load(open(f"{out_dir}/manifest.json"))
    bad = 0
    for e in manifest["meshes"]:
        s = fi.summarize(f"{out_dir}/{e['file']}")
        g = s["geometry"][0]
        gl = s["global"]
        problems = []
        if gl["UnitScaleFactor"] != 1.0:
            problems.append(f"UnitScaleFactor={gl['UnitScaleFactor']}")
        if (gl["UpAxis"], gl["UpAxisSign"], gl["FrontAxis"], gl["FrontAxisSign"], gl["CoordAxis"], gl["CoordAxisSign"]) != (2, 1, 1, -1, 0, 1):
            problems.append(f"axes={gl}")
        mod = s["models"][0]
        if any(abs(v) > 1e-6 for k in ("Lcl Rotation", "PreRotation") for v in mod.get(k, [0, 0, 0])) or any(abs(v - 1) > 1e-6 for v in mod.get("Lcl Scaling", [1, 1, 1])):
            problems.append(f"node transform not identity: {mod}")
        lo, hi = np.array(g["min"]), np.array(g["max"])
        ctr, ext = (lo + hi) / 2, (hi - lo) / 2
        exp_c = np.array(e["bbox_center_m"]) * 100.0 * np.array([1.0, 1.0, 1.0])
        exp_e = np.array(e["bbox_extent_m"]) * 100.0
        if np.abs(ctr - exp_c).max() > tol_cm or np.abs(ext - exp_e).max() > tol_cm:
            problems.append(f"extents differ: file ctr {ctr.round(1)} ext {ext.round(1)} vs manifest ctr {exp_c.round(1)} ext {exp_e.round(1)}")
        if g["polys"] != e["tris"]:
            problems.append(f"tri count {g['polys']} != {e['tris']}")
        if [u[0] for u in g["uv_layers"]] != e["uv_layers"]:
            problems.append(f"uv layers {g['uv_layers']} != {e['uv_layers']}")
        if s["materials"] != e["slots"]:
            problems.append(f"materials {s['materials']} != {e['slots']}")
        if e.get("kind") in ("terrain", "water") and e.get("mean_up", 1.0) < 0.3:
            problems.append(f"surface faces down (mean normal z {e.get('mean_up')})")
        bad += bool(problems)
        print(f"  [{'OK ' if not problems else 'BAD'}] {e['name']:30s} " + ("; ".join(problems) if problems else f"{g['polys']:,} tris"))
    return bad


if __name__ == "__main__":
    geo, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    export_all(geo, out)
    print(f"--- exported in {time.time() - t0:.0f}s; verifying FBX files by parsing them back ---")
    n_bad = verify(out)
    print("--- verifying instance files ---")
    n_bad += verify_instances(out, geo)
    print("ALL FBX OK" if n_bad == 0 else f"{n_bad} FILES HAVE PROBLEMS")
    import shutil

    shutil.copy(f"{out}/manifest.json", os.path.join(os.path.dirname(os.path.abspath(out)), "manifest.json"))   # the builder reads SourceAssets/manifest.json
    write_scene(geo, os.path.join(os.path.dirname(os.path.abspath(out)), "scene.json"))
    sys.exit(1 if n_bad else 0)
