"""massing_view - render the castle block-out (castle_massing.py) from the reference viewpoints, in clay, with Cycles.

    python massing_view.py OUT/massing [view,view...]
Views: poster (the day / night posters, from the lake), boats (shot 02 / ref 2), gorge (shot 01 / ref 1), top (plan),
grounds (from the north-east), east (from the east shore)."""
from __future__ import annotations

import math
import os
import sys

import bpy

import blender_util as bu
from meshkit import Mesh

VIEWS = {
    # the boats' approach up the east arm (the night poster / ref 2): the Great Hall's long side, the Grand Staircase
    # Tower, the Viaduct court, the grand viaduct over the inlet, the north-west rock, the spur with the entry stairs
    "night_poster": dict(loc=(560.0, 40.0, 60.0), at=(60.0, 30.0, 118.0), lens=32.0, res=(1500, 1000)),
    "day_poster": dict(loc=(520.0, -170.0, 46.0), at=(90.0, -10.0, 120.0), lens=32.0, res=(1500, 1000)),
    "bay": dict(loc=(420.0, 150.0, 18.0), at=(40.0, 100.0, 90.0), lens=28.0, res=(1500, 1000)),
    "top": dict(loc=(10.0, 0.0, 1800.0), at=(10.0, 1.0, 0.0), lens=50.0, res=(1400, 1400), ortho=460.0),
    "gorge": dict(loc=(-300.0, -176.0, 112.0), at=(-118.0, -40.0, 100.0), lens=24.0, res=(900, 1600)),
    "south": dict(loc=(-40.0, -700.0, 30.0), at=(0.0, -20.0, 115.0), lens=35.0, res=(1600, 900)),
}


def mat(name, color, rough=0.8, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def terrain_mat():
    m = bpy.data.materials.new("Terrain")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.62
    ramp.color_ramp.elements[0].color = (0.20, 0.19, 0.17, 1)      # rock
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = (0.11, 0.16, 0.07, 1)      # grass / moss
    nt.links.new(sep.outputs[2], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.9
    return m


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "OUT/massing"
    views = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VIEWS)
    bu.reset_scene()
    sc = bpy.context.scene
    bu.setup_cycles(samples=24, res=(1600, 900), max_bounces=4, clamp=6.0)
    sc.view_settings.look = "Medium High Contrast"
    M = {20: mat("Wall", (0.50, 0.46, 0.40)), 21: mat("Trim", (0.58, 0.54, 0.47)), 22: mat("Roof", (0.06, 0.075, 0.09), 0.5),
         24: mat("Glass", (0.05, 0.05, 0.05), 0.1)}
    cm = Mesh.load(f"{d}/massing.npz")
    ids = sorted(set(int(i) for i in cm.mat))
    remap = {o: n for n, o in enumerate(ids)}
    import numpy as np
    cm.mat = np.array([remap[int(i)] for i in cm.mat], np.int16)
    bu.add_object("Castle", cm, [M[i] for i in ids], normals_angle=30.0)
    tm = terrain_mat()
    for n in ("near", "far"):
        bu.add_object(f"Terrain_{n}", Mesh.load(f"{d}/terrain_{n}.npz"), [tm], normals_angle=None)
    water = mat("Water", (0.01, 0.02, 0.025), 0.05)
    lake = Mesh(np.array([[-5000, -5000, 0.0], [5000, -5000, 0.0], [5000, 5000, 0.0], [-5000, 5000, 0.0]]),
                np.array([[0, 1, 2], [0, 2, 3]]), np.zeros(2, np.int16))
    bu.add_object("Lake", lake, [water], normals_angle=None)
    # sky + sun from the south-west (the day poster's light)
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.55, 0.65, 0.80, 1.0)
    bg.inputs[1].default_value = 0.9
    sc.world = w
    el, az = math.radians(32.0), math.radians(215.0)
    bu.add_sun((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)), strength=3.2, angle_deg=1.0,
               color=(1.0, 0.95, 0.88), name="Sun")
    for v in views:
        V = VIEWS[v]
        for ob in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(ob)
        sc.render.resolution_x, sc.render.resolution_y = V["res"]
        cam = bu.add_camera(V["loc"], V["at"], lens=V["lens"], sensor=36.0, name=v)
        cam.data.sensor_fit = "HORIZONTAL" if V["res"][0] >= V["res"][1] else "VERTICAL"
        if V["res"][0] < V["res"][1]:
            cam.data.sensor_height = 36.0
        cam.data.clip_end = 20000.0
        if V.get("ortho"):
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = V["ortho"]
        bu.render_to(f"{d}/view_{v}.png")
        print("rendered", v, flush=True)


if __name__ == "__main__":
    main()
