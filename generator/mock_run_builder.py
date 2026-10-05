"""Execute build_world.run() against a fake `unreal` module.

It cannot prove that Unreal accepts every call (check_builder.py validates names against the real 5.6 API stub), but it
executes every line of the builder's own logic - material graph helpers, calibration maths, camera / sequence keys,
manifest handling - and fails on any Python-level mistake (bad variable, wrong helper arguments, KeyError, ...).

usage: python mock_run_builder.py SOURCE_ASSETS_DIR
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
from unittest.mock import MagicMock

PROJECT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WizardingWorld"))


class V3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return f"V3({self.x:.1f},{self.y:.1f},{self.z:.1f})"


class Rot:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll, self.pitch, self.yaw = float(roll), float(pitch), float(yaw)


class Plain:
    """Property bag that behaves like a UObject for set/get_editor_property."""

    def __init__(self, **kw):
        self.__dict__["_p"] = dict(kw)

    def set_editor_property(self, k, v):
        self._p[k] = v

    def get_editor_property(self, k):
        return self._p.get(k, MagicMock(name=k))

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return self._p.get(k) if k in self._p else MagicMock(name=k)


def make_unreal(src_dir, manifest):
    u = MagicMock(name="unreal")
    u.Vector, u.Rotator = V3, Rot
    u.LinearColor = lambda r=0, g=0, b=0, a=1: (r, g, b, a)
    u.Color = lambda r=0, g=0, b=0, a=255: (r, g, b, a)
    u.Vector4 = lambda *a: tuple(a)
    u.FrameRate = lambda a, b=1: (a, b)
    u.FrameNumber = lambda v: v
    u.IntPoint = lambda a, b: (a, b)
    u.Name = lambda s: s
    u.SoftObjectPath = lambda s: s
    u.log = lambda m: print("  [log]", m)
    u.log_warning = lambda m: print("  [warn]", m)
    u.log_error = lambda m: print("  [ERR ]", m)
    u.Paths.project_dir.return_value = PROJECT + "/"
    u.Paths.project_saved_dir.return_value = PROJECT + "/Saved/"
    # assets
    created = {}
    lib = u.EditorAssetLibrary
    lib.does_asset_exist.side_effect = lambda p: p in created
    meshes = {e["name"]: e for e in manifest["meshes"]}

    class Bounds:
        def __init__(self, c, e):
            self.origin, self.box_extent = V3(*c), V3(*e)

    def load_asset(path):
        name = path.rsplit("/", 1)[-1]
        if "/Meshes/" in path and name in meshes:
            e = meshes[name]
            c, x = e["bbox_center_m"], e["bbox_extent_m"]
            m = MagicMock(name=name)
            m.get_bounds.return_value = Bounds((100 * c[0], -100 * c[1], 100 * c[2]), (100 * x[0], 100 * x[1], 100 * x[2]))
            m.static_materials = [Plain(material_slot_name=s) for s in e["slots"]]
            m.get_name.return_value = name
            return m
        if path in created:
            return created[path]
        return MagicMock(name=name)

    lib.load_asset.side_effect = load_asset
    lib.save_loaded_asset.return_value = True
    tools = MagicMock(name="asset_tools")

    def create_asset(name, path, cls, factory):
        a = Plain(name=name)
        created[f"{path}/{name}"] = a
        return a

    tools.create_asset.side_effect = create_asset
    u.AssetToolsHelpers.get_asset_tools.return_value = tools
    u.AssetImportTask = lambda: Plain()
    u.FbxImportUI = lambda: Plain(static_mesh_import_data=Plain())

    # material editing: every expression is a Plain bag; connections always succeed
    mel = u.MaterialEditingLibrary
    mel.create_material_expression.side_effect = lambda mat, cls, x, y: Plain(_cls=getattr(cls, "_mock_name", str(cls)))
    mel.connect_material_expressions.return_value = True
    mel.connect_material_property.return_value = True
    # level / actors
    eas = MagicMock(name="EditorActorSubsystem")

    def spawn(cls, loc, rot):
        a = Plain(tags=[])
        a.set_actor_label = lambda s: None
        a.set_folder_path = lambda s: None
        a.set_actor_scale3d = lambda s: None
        a.get_component_by_class = lambda c: Plain()
        a.static_mesh_component = MagicMock()
        a.get_cine_camera_component = lambda: Plain(filmback=Plain(), lens_settings=Plain(), focus_settings=Plain(tracking_focus_settings=Plain()))
        return a

    eas.spawn_actor_from_class.side_effect = spawn
    eas.get_all_level_actors.return_value = []
    les = MagicMock(name="LevelEditorSubsystem")
    les.new_level.return_value = True
    u.get_editor_subsystem.side_effect = lambda cls: eas if "Actor" in str(getattr(cls, "_mock_name", cls)) else les
    u.MathLibrary.find_look_at_rotation.side_effect = lambda a, b: Rot(0.0, math.degrees(math.atan2(b.z - a.z, math.hypot(b.x - a.x, b.y - a.y))), math.degrees(math.atan2(b.y - a.y, b.x - a.x)))
    # transforms / quaternions as plain value holders so instance records can be checked after the run
    class Tr:
        def __init__(self, location=None, rotation=None, scale=None):
            self.translation, self.rotation, self.scale3d = location or V3(), rotation, scale or V3(1, 1, 1)

        def multiply(self, b):
            return b

    u.Transform = Tr
    u.Quat = lambda x=0.0, y=0.0, z=0.0, w=1.0: (float(x), float(y), float(z), float(w))

    # SubobjectDataSubsystem: every add_new_subobject creates a fake instanced static mesh component
    class FakeISM:
        def __init__(self):
            self.n, self.mesh, self.mats, self.samples, self.calls = 0, None, {}, [], []

        def add_instances(self, ts, should_return_indices, world_space=False):
            self.n += len(ts)
            self.samples.extend(ts[:: max(1, len(ts) // 50)])
            self.calls.append((len(ts), should_return_indices, world_space))
            return []

        def get_instance_count(self):
            return self.n

        def set_static_mesh(self, m):
            self.mesh = m

        def set_material(self, i, m):
            self.mats[i] = m

        def __getattr__(self, k):
            if k.startswith("__"):
                raise AttributeError(k)
            return MagicMock(name=k)

    isms = []
    sds = MagicMock(name="SubobjectDataSubsystem")

    def add_new_subobject(params):
        c = FakeISM()
        isms.append(c)
        return c, ""

    sds.add_new_subobject.side_effect = add_new_subobject
    sds.k2_gather_subobject_data_for_instance.return_value = ["root-handle"]
    u.get_engine_subsystem.side_effect = lambda cls: sds
    sdl = u.SubobjectDataBlueprintFunctionLibrary
    sdl.is_handle_valid.side_effect = lambda h: h is not None
    sdl.get_data.side_effect = lambda h: h
    sdl.get_object.side_effect = lambda d: d
    u._fake_isms = isms
    seq = MagicMock(name="sequence")
    seq.get_binding_id.return_value = 1
    tools.create_asset.side_effect = lambda name, path, cls, factory: (seq if "Sequence" in str(getattr(cls, "_mock_name", cls)) else create_asset(name, path, cls, factory))
    return u


def main(src):
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    os.environ["WW_SOURCE"] = src
    u = make_unreal(src, manifest)
    sys.modules["unreal"] = u
    path = os.path.join(PROJECT, "Content", "Python", "build_world.py")
    spec = importlib.util.spec_from_file_location("build_world", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for kv in OVERRIDES:                                   # --cfg key=<json> (e.g. instances=false, instance_sets=["moss"])
        k, v = kv.split("=", 1)
        mod.CFG[k] = json.loads(v)
        print(f"CFG override: {k} = {mod.CFG[k]!r}")
    mod.run()
    rep = mod.REPORT
    # instanced detail: every manifest record must have reached an ISM, with unit quaternions and positive scales
    isms = u._fake_isms
    want = sum(e["count"] for e in manifest.get("instances", []))
    got = sum(c.n for c in isms)
    bad_q = bad_s = 0
    for c in isms:
        for t in c.samples:
            q = t.rotation
            if abs(math.sqrt(sum(v * v for v in q)) - 1.0) > 1e-3:
                bad_q += 1
            if min(t.scale3d.x, t.scale3d.y, t.scale3d.z) <= 0:
                bad_s += 1
    no_mat = sum(1 for c in isms if not c.mats)
    print(f"\ninstanced components: {len(isms)}, instances {got:,} of {want:,} in the manifest; "
          f"sampled transforms with bad quaternion {bad_q}, bad scale {bad_s}; components without materials {no_mat}")
    dens = max(0.0, min(1.0, float(mod.CFG.get("instance_density", 1.0))))

    def kept(n):                                           # the builder's deterministic thinning
        return n if dens >= 1.0 else sum(1 for i in range(n) if (i * 0.6180339887498949) % 1.0 < dens)

    sets = mod.CFG.get("instance_sets")
    want = 0 if not mod.CFG.get("instances", True) else sum((e["count"] if e["set"] in ("boats", "willow") else kept(e["count"]))
                                                           for e in manifest.get("instances", []) if not sets or e["set"] in sets)
    if manifest.get("instances") and (got != want or bad_q or bad_s or no_mat):
        rep.fail.append(("instanced detail check", f"got {got} want {want}, bad_q {bad_q}, bad_s {bad_s}, no_mat {no_mat}"))
    print(f"\nMOCK RUN: {len(rep.ok)} steps ok, {len(rep.fail)} failed, {len(rep.warn)} warnings")
    for n, e in rep.fail:
        print("  FAILED", n, e)
    return 1 if rep.fail else 0


OVERRIDES = []

if __name__ == "__main__":
    args = sys.argv[1:]
    while "--cfg" in args:
        i = args.index("--cfg")
        OVERRIDES.append(args[i + 1])
        del args[i:i + 2]
    sys.exit(main(args[0]))
