"""quickview - fast Cycles look-dev renders of generated .npz meshes (clay + mask colours), for iterating on shapes.

usage: python quickview.py OUT.png --cam x,y,z --at x,y,z [--lens 35] [--res 960x540] [--samples 16] [--sun elev,az]
                           mesh.npz [mesh2.npz ...]
Mask colouring: rock grey, moss green (UV1.y), forest floor brown (UV2.x), shore dark (UV2.y), AO (UV3.y).
Library instance sets can be previewed with --inst set.npz:lib.npz (geometry nodes instancing).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import bpy
import mathutils

import blender_util as bu
from meshkit import Mesh


def mask_material(name="mask"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N = nt.nodes
    L = nt.links
    bsdf = N["Principled BSDF"]

    def uvsep(uv):
        u = N.new("ShaderNodeUVMap")
        u.uv_map = uv
        s = N.new("ShaderNodeSeparateXYZ")
        L.new(u.outputs[0], s.inputs[0])
        return s

    def mix(fac, a, b):
        mx = N.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        if isinstance(fac, float):
            mx.inputs[0].default_value = fac
        else:
            L.new(fac, mx.inputs[0])
        for idx, v in ((6, a), (7, b)):
            if isinstance(v, tuple):
                mx.inputs[idx].default_value = (*v, 1)
            else:
                L.new(v, mx.inputs[idx])
        return mx.outputs[2]

    s1, s2, s3 = uvsep("UV1"), uvsep("UV2"), uvsep("UV3")
    # noise for a little colour breakup
    tc = N.new("ShaderNodeTexCoord")
    nz = N.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.35
    nz.inputs["Detail"].default_value = 8
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    grass = mix(nz.outputs["Fac"], (0.16, 0.24, 0.07), (0.25, 0.30, 0.10))
    c = mix(s2.outputs[0], grass, (0.10, 0.075, 0.045))                       # forest floor
    rock = mix(nz.outputs["Fac"], (0.30, 0.29, 0.27), (0.45, 0.43, 0.40))
    c = mix(s1.outputs[0], c, rock)                                           # rock
    c = mix(s1.outputs[1], c, (0.10, 0.20, 0.035))                            # moss
    c = mix(s2.outputs[1], c, (0.06, 0.055, 0.045))                           # shore / wet
    c = mix(s3.outputs[0], c, (0.30, 0.25, 0.18))                             # road
    ao = N.new("ShaderNodeMath")
    ao.operation = "POWER"
    L.new(s3.outputs[1], ao.inputs[0])
    ao.inputs[1].default_value = 1.5
    mul = N.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs[0].default_value = 1.0
    L.new(c, mul.inputs[6])
    comb = N.new("ShaderNodeCombineXYZ")
    for i in range(3):
        L.new(ao.outputs[0], comb.inputs[i])
    L.new(comb.outputs[0], mul.inputs[7])
    L.new(mul.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    return m


def setup_world(sun_elev=25.0, sun_az=120.0, strength=1.0):
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    nt = w.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(sun_elev)
        sky.sun_rotation = math.radians(90.0 - sun_az)
        sky.altitude = 200
        sky.air_density = 1.0
        sky.dust_density = 2.0
    except Exception:
        pass
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.25 * strength
    nt.links.new(sky.outputs[0], bg.inputs[0])
    bpy.context.scene.world = w
    el, az = math.radians(sun_elev), math.radians(sun_az)
    d = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    bu.add_sun(d, strength=3.2 * strength, angle_deg=0.6, color=(1.0, 0.95, 0.86))


def instance_object(set_path, lib_path, mat):
    """Point cloud with per-point rotation / scale -> geometry nodes 'Instance on Points' of the library mesh."""
    import instances as ins

    d = ins.load_set(set_path)
    lib = Mesh.load(lib_path)
    lob = bu.add_object(os.path.basename(lib_path), lib, [mat], normals_angle=None)
    lob.hide_render = True
    lob.hide_viewport = True
    pos = d["pos"].astype(np.float64)
    me = bpy.data.meshes.new("pts")
    me.vertices.add(len(pos))
    me.vertices.foreach_set("co", pos.astype(np.float32).ravel())
    eul = ins.euler_xyz(d["R"])
    a = me.attributes.new("rot", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", eul.astype(np.float32).ravel())
    s = me.attributes.new("scl", "FLOAT_VECTOR", "POINT")
    s.data.foreach_set("vector", d["scale"].astype(np.float32).ravel())
    ob = bpy.data.objects.new("inst", me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("gn", "NODES")
    ng = bpy.data.node_groups.new("inst", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    iop = ng.nodes.new("GeometryNodeInstanceOnPoints")
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs[0].default_value = lob
    oi.transform_space = "ORIGINAL"
    ra = ng.nodes.new("GeometryNodeInputNamedAttribute")
    ra.data_type = "FLOAT_VECTOR"
    ra.inputs[0].default_value = "rot"
    sa = ng.nodes.new("GeometryNodeInputNamedAttribute")
    sa.data_type = "FLOAT_VECTOR"
    sa.inputs[0].default_value = "scl"
    ng.links.new(gi.outputs[0], iop.inputs["Points"])
    ng.links.new(oi.outputs["Geometry"], iop.inputs["Instance"])
    ng.links.new(ra.outputs[0], iop.inputs["Rotation"])
    ng.links.new(sa.outputs[0], iop.inputs["Scale"])
    ng.links.new(iop.outputs[0], go.inputs[0])
    mod.node_group = ng
    return ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("meshes", nargs="*")
    ap.add_argument("--cam", required=True)
    ap.add_argument("--at", required=True)
    ap.add_argument("--lens", type=float, default=35.0)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--sun", default="25,120")
    ap.add_argument("--clay", action="store_true")
    ap.add_argument("--inst", action="append", default=[])
    ap.add_argument("--water", type=float, default=None)
    ap.add_argument("--fog", type=float, default=0.0)
    a = ap.parse_args()
    bu.reset_scene()
    res = tuple(int(v) for v in a.res.split("x"))
    bu.setup_cycles(samples=a.samples, res=res, max_bounces=4)
    el, az = (float(v) for v in a.sun.split(","))
    setup_world(el, az)
    mat = bu.clay_material() if a.clay else mask_material()
    pal = {20: (0.42, 0.40, 0.36), 21: (0.55, 0.52, 0.47), 22: (0.10, 0.11, 0.13), 23: (0.05, 0.05, 0.06), 24: (0.02, 0.03, 0.05), 25: (0.20, 0.12, 0.06)}
    mats_by_id = {}
    for p in a.meshes:
        m = Mesh.load(p)
        ids = sorted(set(int(i) for i in m.mat))
        if len(ids) == 1 and ids[0] == 0:
            bu.add_object(os.path.basename(p), m, [mat], normals_angle=None)
            continue
        for i in ids:
            if i not in mats_by_id:
                mats_by_id[i] = bu.clay_material(pal.get(i, (0.5, 0.5, 0.5)), 0.35 if i in (22, 23, 24) else 0.8)
        remap = {old: new for new, old in enumerate(ids)}
        m.mat = np.array([remap[int(i)] for i in m.mat], np.int16)
        bu.add_object(os.path.basename(p), m, [mats_by_id[i] for i in ids], normals_angle=30.0)
    for spec in a.inst:
        sp, lp = spec.split(":")
        instance_object(sp, lp, bu.clay_material((0.12, 0.2, 0.06), 0.8))
    if a.water is not None:
        bpy.ops.mesh.primitive_plane_add(size=6000, location=(0, 0, a.water))
        wm = bpy.data.materials.new("water")
        wm.use_nodes = True
        b = wm.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.01, 0.025, 0.03, 1)
        b.inputs["Roughness"].default_value = 0.05
        bpy.context.object.data.materials.append(wm)
    cam = bu.add_camera([float(v) for v in a.cam.split(",")], [float(v) for v in a.at.split(",")], lens=a.lens)
    cam.data.clip_end = 20000
    if a.fog > 0:
        sc = bpy.context.scene
        sc.view_layers[0].use_pass_mist = True
    t = bu.render_to(a.out)
    print(f"rendered {a.out} in {t:.0f}s")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.argv = [sys.argv[0]] + argv
    main()
