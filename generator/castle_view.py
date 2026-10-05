"""castle_view - clay renders of the detailed castle (castle.py: unique shells + instanced detail modules) on the core
terrain, from the reference viewpoints and from close up - the quick check between castle builds.

    python castle_view.py OUT/geo OUT/castle_view [view,view...] [--samples N] [--long PX]
Views: those of massing_view.py plus close-ups (hall, staircase, viaduct, stairs, astronomy, clock)."""
from __future__ import annotations

import argparse
import glob
import math
import os
import time

import bpy
import numpy as np

import blender_util as bu
import preview as PV
from massing_view import VIEWS as MASSING_VIEWS
from massing_view import mat, terrain_mat
from meshkit import Mesh

VIEWS = dict(MASSING_VIEWS)
VIEWS.update({
    # the Great Hall's long side over the east arm, from a boat
    "hall": dict(loc=(300.0, -40.0, 12.0), at=(150.0, -60.0, 112.0), lens=30.0, res=(1500, 1000)),
    # the Grand Staircase Tower and the Headmaster's Tower from the Viaduct court's roofs
    "staircase": dict(loc=(150.0, 60.0, 150.0), at=(56.0, -28.0, 140.0), lens=35.0, res=(1000, 1400)),
    # the grand viaduct from the water of the inlet's mouth
    "viaduct": dict(loc=(110.0, 150.0, 6.0), at=(40.0, 100.0, 50.0), lens=24.0, res=(1500, 1000)),
    # the entry stairs, the boathouse and the waterfall from the bay
    "stairs": dict(loc=(230.0, 110.0, 8.0), at=(160.0, 200.0, 40.0), lens=28.0, res=(1500, 1000)),
    "stairs_wide": dict(loc=(262.0, 40.0, 22.0), at=(140.0, 195.0, 48.0), lens=26.0, res=(1500, 1000)),
    # the covered bridge over the stream's gorge, from the grounds (north-east) and from the gorge
    "bridge": dict(loc=(236.0, 336.0, 92.0), at=(122.0, 266.0, 74.0), lens=30.0, res=(1500, 1000)),
    "bridge_gorge": dict(loc=(116.0, 318.0, 66.0), at=(127.0, 274.0, 78.0), lens=26.0, res=(1500, 1000)),
    # the north-west rock's west face over the gorge (the night reference's left side)
    "west_cliff": dict(loc=(-330.0, 70.0, 92.0), at=(-150.0, 30.0, 80.0), lens=30.0, res=(1500, 1000)),
    # the south-east rock's north-east corner over the bay (the deep walls and a cliff tower)
    "se_corner": dict(loc=(250.0, 140.0, 20.0), at=(170.0, 70.0, 60.0), lens=30.0, res=(1500, 1000)),
    # the Astronomy Tower's stages and crown
    "astronomy": dict(loc=(-200.0, -60.0, 140.0), at=(-98.0, 20.0, 170.0), lens=40.0, res=(1000, 1400)),
    # the Clock Tower over the clock court
    "clock": dict(loc=(60.0, -200.0, 70.0), at=(-16.0, -114.0, 130.0), lens=35.0, res=(1000, 1400)),
    # window / buttress detail on the hall
    "detail": dict(loc=(184.0, -50.0, 96.0), at=(165.0, -62.0, 104.0), lens=35.0, res=(1500, 1000)),
})

CLAY = {"M_CastleStone": ((0.50, 0.46, 0.40), 0.8), "M_Trim": ((0.60, 0.56, 0.49), 0.75), "M_Slate": ((0.07, 0.085, 0.10), 0.45),
        "M_Lead": ((0.20, 0.21, 0.22), 0.4), "M_Glass": ((0.02, 0.025, 0.03), 0.08), "M_GlassInst": ((0.02, 0.025, 0.03), 0.08),
        "M_Wood": ((0.25, 0.17, 0.10), 0.8), "M_ClockFace": ((0.75, 0.70, 0.55), 0.5), "M_Lantern": ((0.9, 0.7, 0.4), 0.3),
        "M_River": ((0.03, 0.06, 0.07), 0.05), "M_Fall": ((0.85, 0.9, 0.92), 0.4), "M_Mist": ((0.9, 0.92, 0.95), 0.9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geo")
    ap.add_argument("out")
    ap.add_argument("views", nargs="?", default="")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--long", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    bu.reset_scene()
    sc = bpy.context.scene
    bu.setup_cycles(samples=a.samples, res=(1600, 900), max_bounces=4, clamp=6.0)
    sc.view_settings.look = "Medium High Contrast"
    M = {k: mat(k, c, r) for k, (c, r) in CLAY.items()}
    M["M_Terrain"] = terrain_mat()
    for p in sorted(glob.glob(f"{a.geo}/castle_*.npz")) + sorted(glob.glob(f"{a.geo}/terrain_core_*.npz")):
        PV.add_mesh(p, M)
    outer = sorted(glob.glob(f"{a.geo}/terrain_outer_*.npz"))
    if outer:
        for p in outer:
            PV.add_mesh(p, M)
    else:
        far = Mesh.load("OUT/massing/terrain_far.npz")     # the analytic backdrop, cut where the core terrain is
        P = far.V[far.F]
        inside = ((P[:, :, 0] > -598) & (P[:, :, 0] < 598) & (P[:, :, 1] > -518) & (P[:, :, 1] < 678)).all(1)
        bu.add_object("Terrain_far", Mesh(far.V, far.F[~inside], far.mat[~inside]), [M["M_Terrain"]], normals_angle=None)
    for p in ("water_stream", "water_fall", "water_mist"):
        if os.path.isfile(f"{a.geo}/{p}.npz"):
            PV.add_mesh(f"{a.geo}/{p}.npz", M)
    lib_obs = {}
    for p in sorted(glob.glob(f"{a.geo}/lib/lib_c_*.npz")):
        ob = PV.add_mesh(p, M)
        ob.hide_render = True
        ob.hide_viewport = True
        lib_obs[os.path.splitext(os.path.basename(p))[0]] = ob
    n = 0
    for p in sorted(glob.glob(f"{a.geo}/instances/castle_*.npz")):
        n += len(PV.add_instances(p, lib_obs))
    water = mat("Water", (0.01, 0.02, 0.025), 0.05)
    lake = Mesh(np.array([[-5000, -5000, 0.0], [5000, -5000, 0.0], [5000, 5000, 0.0], [-5000, 5000, 0.0]]),
                np.array([[0, 1, 2], [0, 2, 3]]), np.zeros(2, np.int16))
    bu.add_object("Lake", lake, [water], normals_angle=None)
    w = bpy.data.worlds.new("W")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.55, 0.65, 0.80, 1.0)
    bg.inputs[1].default_value = 0.9
    sc.world = w
    el, az = math.radians(32.0), math.radians(215.0)
    bu.add_sun((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)), strength=3.2, angle_deg=1.0,
               color=(1.0, 0.95, 0.88), name="Sun")
    print(f"scene built in {time.time() - t0:.0f}s ({n} instancers)", flush=True)
    for v in (a.views.split(",") if a.views else list(VIEWS)):
        V = VIEWS[v]
        for ob in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(ob)
        rx, ry = V["res"]
        k = a.long / max(rx, ry)
        sc.render.resolution_x, sc.render.resolution_y = int(rx * k), int(ry * k)
        cam = bu.add_camera(V["loc"], V["at"], lens=V["lens"], sensor=36.0, name=v)
        cam.data.sensor_fit = "HORIZONTAL" if rx >= ry else "VERTICAL"
        if rx < ry:
            cam.data.sensor_height = 36.0
        cam.data.clip_end = 20000.0
        cam.data.clip_start = 0.3
        if V.get("ortho"):
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = V["ortho"]
        t = time.time()
        bu.render_to(f"{a.out}/{v}.png")
        print(f"rendered {v} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
