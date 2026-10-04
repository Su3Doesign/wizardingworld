"""build_all - regenerate the whole world from scratch and package it for Unreal (and render the Cycles previews).

    python build_all.py                       # standard quality (what the repository ships): ~12 M unique tris, 620 k instances
    python build_all.py --quality ultra       # local build: denser rock (0.5 m voxels, 9 M tris core), denser forests
    python build_all.py --skip-previews       # stop after the Unreal SourceAssets
    python build_all.py --only scatter,export # re-run some stages (they read the previous stages' OUT/ files)
    python build_all.py --only scatter,instances,previews   # after moving a camera in shots.py

Stages (each a module of this folder; all deterministic, fixed seeds):
    erode     world.py       hydraulic + thermal erosion of the mountain ring              (cache/erosion.npz)
    core      core.py        the 1.2 km hero terrain: height field + granite SDF (crag, gorge, ravine, lake cliffs)
    castle    castle.py      the castle on its crag (castle_kit.py)
    terrain   terrain.py     the outer valley and the mountains (adaptive height-field mesh)
    grounds   grounds.py     stadium, hut, greenhouses, stone circle, gates, station, Hogsmeade, boats, water surfaces
    sky       sky.py         star dome + moon disc
    trees     trees.py       spruce / pine / birch / willow / snag / boulder library
    plants    smallplants.py moss cushions, ferns, grass clumps
    scatter   scatter.py     forests, crag ledges, moss, boulders, understory (+ sight-line clearings for the shots)
    textures  tex_world.py   the procedural texture library
    export    export_world.py  FBX + instance records + manifest.json + scene.json -> ../WizardingWorld/SourceAssets
    instances export_world.py --instances-only: only the instance records + scene.json (FBX untouched); run on request
    previews  preview.py     Cycles previews from every shot -> ../Previews (+ contact_sheet.py: all shots on one page)
Requirements: Python 3.11 + requirements.txt (bpy 4.5 = Blender as a module).  ~25 min on 4 cores + previews.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "OUT")
GEO = os.path.join(OUT, "geo")
TEX = os.path.join(OUT, "tex")
SA = os.path.normpath(os.path.join(HERE, "..", "WizardingWorld", "SourceAssets"))

QUALITY = {
    "standard": dict(vox=0.75, core_tris=3_600_000, outer_scale=1.0, tree_density=1.0),
    "ultra": dict(vox=0.5, core_tris=9_000_000, outer_scale=0.7, tree_density=1.25),
}


def run(stage, args):
    t = time.time()
    print(f"=== {stage}: python {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable] + args, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"stage {stage} failed (exit {r.returncode})")
    print(f"=== {stage} done in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="standard", choices=sorted(QUALITY))
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-previews", action="store_true")
    a = ap.parse_args()
    q = QUALITY[a.quality]
    only = set(s for s in a.only.split(",") if s)

    def want(s):
        return not only or s in only

    os.makedirs(GEO, exist_ok=True)
    t0 = time.time()
    if want("erode"):
        run("erode", ["world.py", "erode"])
    if want("core"):
        run("core", ["core.py", GEO, str(q["vox"]), str(q["core_tris"])])
    if want("castle"):
        run("castle", ["castle.py", GEO])
    if want("terrain"):
        run("terrain", ["terrain.py", GEO, str(q["outer_scale"])])
    if want("grounds"):
        run("grounds", ["grounds.py", GEO])
    if want("sky"):
        run("sky", ["sky.py", GEO])
    if want("trees"):
        run("trees", ["trees.py", os.path.join(GEO, "lib")])
    if want("plants"):
        run("plants", ["smallplants.py", os.path.join(GEO, "lib"), TEX])
    if want("scatter"):
        run("scatter", ["scatter.py", GEO, str(q["tree_density"])])
    if want("textures"):
        run("textures", ["tex_world.py", TEX])
        run("plants-textures", ["smallplants.py", os.path.join(GEO, "lib"), TEX])
    if want("export"):
        fbx = os.path.join(SA, "fbx")
        if os.path.isdir(fbx):
            shutil.rmtree(fbx)
        os.makedirs(os.path.join(SA, "textures"), exist_ok=True)
        for f in os.listdir(TEX):
            shutil.copy(os.path.join(TEX, f), os.path.join(SA, "textures", f))
        run("export", ["export_world.py", GEO, fbx])
    if "instances" in only:                                   # a lighter export after a re-scatter: FBX files untouched
        run("instances", ["export_world.py", GEO, os.path.join(SA, "fbx"), "--instances-only"])
    if want("previews") and not a.skip_previews:
        prev = os.path.join(HERE, "..", "Previews")
        run("previews", ["preview.py", GEO, TEX, prev, "--samples", "40", "--long", "1280"])
        run("contact-sheet", ["contact_sheet.py", prev, os.path.join(prev, "contact_sheet.jpg")])
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
