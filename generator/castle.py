"""castle - composes the castle on its crag from castle_kit pieces (layout after the references).

Silhouette (seen from the gorge, ref 1; from the lake, ref 2; at sunset from the south-west, ref 4):
    the Great Hall along the west cliff edge, its buttressed long side towards the gorge; the Great Tower - the massive
    round tower with the tall cone - rising behind its south end; the lake-front range with its towers along the south
    cliff; the Grand Staircase and Clock towers in the middle; the Astronomy Tower (tallest spire) at the north-east;
    ranges round three courtyards; the twin-towered gatehouse on the north (grounds) side; the viaduct leaving the
    east side across the ravine; the boathouse and its stair tower at the foot of the lake cliff; a small tower and
    house on a ledge of the gorge wall (ref 1).

Output: OUT/geo/castle_<group>.npz (groups keep every FBX well under 100 MB and let Unreal cull them separately).
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

import castle_kit as ck
import meshkit as mk
import world as W

Z = W.CASTLE_Z


_TOP = {}


def surface_top():
    """Height map (1 m) of the top of the actual core terrain mesh (with the granite displacement), cached."""
    if "grid" in _TOP:
        return _TOP["grid"]
    import glob

    from scipy import ndimage

    cache = "OUT/geo/core_top.npz"
    files = sorted(glob.glob("OUT/geo/terrain_core_*.npz"))
    if os.path.isfile(cache) and all(os.path.getmtime(cache) > os.path.getmtime(f) for f in files):
        z = np.load(cache)
        _TOP["grid"] = (z["top"], float(z["x0"]), float(z["y0"]))
        return _TOP["grid"]
    x0, y0, cell = -610.0, -530.0, 1.0
    n = 1221
    top = np.full((n, n), -np.inf)
    for f in files:
        V = mk.Mesh.load(f).V
        i = np.clip(((V[:, 0] - x0) / cell).astype(int), 0, n - 1)
        j = np.clip(((V[:, 1] - y0) / cell).astype(int), 0, n - 1)
        np.maximum.at(top, (i, j), V[:, 2])
    bad = ~np.isfinite(top)
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        top = top[tuple(idx)]
    np.savez_compressed(cache, top=top.astype(np.float32), x0=x0, y0=y0)
    _TOP["grid"] = (top, x0, y0)
    return _TOP["grid"]


def top_at(x, y):
    top, x0, y0 = surface_top()
    i = np.clip(((np.asarray(x) - x0)).astype(int), 0, top.shape[0] - 1)
    j = np.clip(((np.asarray(y) - y0)).astype(int), 0, top.shape[1] - 1)
    return top[i, j]


def foot(x, y, r=8.0, deep=2.5):
    """Foundation depth: down to the actual rock under the footprint (a disc of radius r), plus a little."""
    a = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    xs = np.concatenate([[x], x + r * np.cos(a), x + 0.6 * r * np.cos(a)])
    ys = np.concatenate([[y], y + r * np.sin(a), y + 0.6 * r * np.sin(a)])
    return float(min(Z - 3.0, top_at(xs, ys).min() - deep))


def foot_line(p0, p1, half=8.0, deep=2.5):
    t = np.linspace(0, 1, 25)
    pts = np.asarray(p0)[None] * (1 - t[:, None]) + np.asarray(p1)[None] * t[:, None]
    d = np.asarray(p1, np.float64) - np.asarray(p0, np.float64)
    nb = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    xs = np.concatenate([pts[:, 0] + nb[0] * k for k in (-half, 0.0, half)])
    ys = np.concatenate([pts[:, 1] + nb[1] * k for k in (-half, 0.0, half)])
    return float(min(Z - 3.0, top_at(xs, ys).min() - deep))


def cliff_point(x, y0, y1, z_min=30.0):
    """Walk from (x, y0) towards (x, y1) and return the first point where the rock surface rises above z_min."""
    for y in np.linspace(y0, y1, 200):
        if float(top_at(x, y)) > z_min:
            return float(x), float(y)
    return float(x), float(y1)


def find_ledge(y_range=(-120.0, 40.0), z_range=(28.0, 58.0)):
    """A flattish spot on the gorge-side ledge of the crag (for the small tower + house of ref 1)."""
    best = None
    for y in np.arange(y_range[0], y_range[1], 4.0):
        for x in np.arange(-215.0, -165.0, 1.5):
            h = float(top_at(x, y))
            if not (z_range[0] < h < z_range[1]):
                continue
            hs = top_at(np.array([x - 4, x + 4, x, x]), np.array([y, y, y - 4, y + 4]))
            slope = float(np.ptp(hs)) / 6.0
            score = slope + abs(h - 44.0) * 0.01 + abs(y + 70.0) * 0.004
            if best is None or score < best[0]:
                best = (score, x, y, h)
    return best


def build(out_dir, seed=21):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    groups = {}

    def add(group, mesh):
        if mesh is not None and mesh.nf:
            groups.setdefault(group, []).append(mesh)

    def turrets(group, pts, z0, r=2.3, h=7.0):
        for p in pts:
            add(group, ck.bartizan(p, z0, r=r, h=h, roof_h=r * rng.uniform(3.8, 4.8), rng=rng))

    RT = ck.round_tower
    # ------------------------------------------------------------------ west: Great Hall, Great Tower, entrance
    hp0, hp1 = (-138.0, -118.0), (-138.0, -28.0)
    add("west", ck.hall(hp0, hp1, 26.0, Z, 27.0, rng=rng, z_foot=foot_line((-138, -118), (-138, -28), 14.0), bays=8))
    add("west", RT((-100.0, -128.0), 12.0, Z, 58.0, roof_h=64.0, z_foot=foot(-100, -128, 13), rng=rng, seg=80, per_floor=7,
                   win_w=1.5, win_h=3.4))                                                           # the Great Tower
    add("west", RT((-88.5, -141.5), 3.8, Z, 66.0, roof_h=15.0, z_foot=foot(-88.5, -141.5, 5), rng=rng, seg=24, per_floor=2,
                   win_w=0.8, win_h=1.9))
    add("west", RT((-113.0, -118.0), 3.2, Z + 30.0, 34.0, roof_h=13.0, z_foot=Z + 28.0, rng=rng, seg=24, per_floor=2,
                   win_w=0.7, win_h=1.6, courses=False))
    add("west", ck.wing((-112.0, -95.0), (-112.0, -42.0), 30.0, Z, 34.0, rng=rng, crenel=True, pitch=48.0, z_foot=foot_line((-112, -95), (-112, -42), 15.0)))
    add("west", ck.square_tower((-96.0, -30.0), 16.0, 16.0, Z, 70.0, rng=rng, roof="pyramid", roof_h=19.0, corner_turrets=True,
                                z_foot=foot(-96, -30, 11)))
    add("west", ck.wing((-150.0, -10.0), (-150.0, 150.0), 16.0, Z, 28.0, rng=rng, z_foot=foot_line((-150, -10), (-150, 150), 8.0)))
    add("west", RT((-150.0, -20.0), 6.0, Z, 50.0, roof_h=24.0, z_foot=foot(-150, -20, 7), rng=rng, seg=40))
    add("west", RT((-152.0, 64.0), 5.0, Z, 44.0, roof_h=21.0, z_foot=foot(-152, 64, 6), rng=rng, seg=40))
    add("west", RT((-150.0, 160.0), 6.5, Z, 52.0, roof_h=26.0, z_foot=foot(-150, 160, 7), rng=rng, seg=40))
    turrets("west", [(-158.5, 30.0), (-158.5, 110.0)], Z + 22.0)

    # ------------------------------------------------------------------ south: the lake front
    add("south", ck.wing((-82.0, -146.0), (56.0, -146.0), 16.0, Z, 26.0, rng=rng, z_foot=foot_line((-82, -146), (56, -146), 8.0)))
    add("south", RT((-36.0, -156.0), 6.5, Z, 46.0, roof_h=24.0, z_foot=foot(-36, -156, 7), rng=rng, seg=48))
    add("south", RT((16.0, -152.0), 5.5, Z, 56.0, roof_h=26.0, z_foot=foot(16, -152, 6), rng=rng, seg=48))
    add("south", RT((66.0, -130.0), 7.5, Z, 76.0, roof_h=32.0, z_foot=foot(66, -130, 8), rng=rng, seg=56))     # Headmaster's
    add("south", ck.wing((74.0, -118.0), (140.0, -118.0), 15.0, Z, 24.0, rng=rng, z_foot=foot_line((74, -118), (140, -118), 7.5)))
    add("south", ck.square_tower((150.0, -110.0), 15.0, 15.0, Z, 46.0, rng=rng, roof="pyramid", roof_h=16.0, corner_turrets=True,
                                 z_foot=foot(150, -110, 10)))
    turrets("south", [(-10.0, -154.5), (40.0, -154.5)], Z + 18.0)

    # ------------------------------------------------------------------ centre: clock tower, cross ranges, courtyards
    add("centre", ck.wing((-80.0, 20.0), (166.0, 20.0), 16.0, Z, 26.0, rng=rng, z_foot=Z - 4))
    add("centre", ck.square_tower((95.0, 22.0), 13.0, 13.0, Z, 72.0, rng=rng, roof="spire", roof_h=32.0, clock=True,
                                  corner_turrets=True, z_foot=Z - 4))
    add("centre", ck.wing((20.0, 28.0), (20.0, 188.0), 16.0, Z, 25.0, rng=rng, z_foot=Z - 4))
    add("centre", ck.square_tower((20.0, 20.0), 12.0, 12.0, Z, 54.0, rng=rng, roof="pyramid", roof_h=14.0, z_foot=Z - 4))
    add("centre", RT((-60.0, 22.0), 5.0, Z, 50.0, roof_h=22.0, z_foot=Z - 4, rng=rng, seg=40))
    add("centre", RT((58.0, 104.0), 6.0, Z, 60.0, roof_h=27.0, z_foot=Z - 4, rng=rng, seg=40))
    add("centre", ck.wing((-80.0, -10.0), (-80.0, 14.0), 14.0, Z, 22.0, rng=rng, z_foot=Z - 4, dormers=False))
    add("centre", ck.wing((100.0, -112.0), (100.0, 14.0), 15.0, Z, 24.0, rng=rng, z_foot=Z - 4))
    turrets("centre", [(-40.0, 28.5), (60.0, 28.5), (140.0, 28.5), (28.5, 80.0), (28.5, 150.0)], Z + 19.0, r=2.0, h=6.0)

    # ------------------------------------------------------------------ north: entrance front, towers
    add("north", ck.wing((-110.0, 198.0), (150.0, 200.0), 18.0, Z, 28.0, rng=rng, z_foot=Z - 6))
    add("north", RT((-116.0, 150.0), 9.0, Z, 74.0, roof_h=38.0, z_foot=Z - 6, rng=rng, seg=64))
    add("north", RT((95.0, 188.0), 8.5, Z, 82.0, roof_h=36.0, z_foot=Z - 6, rng=rng, seg=64))
    add("north", RT((-60.0, 190.0), 5.0, Z, 46.0, roof_h=22.0, z_foot=Z - 6, rng=rng, seg=40))
    for gx in (-4.0, 28.0):
        add("north", RT((gx, 214.0), 5.5, Z, 36.0, roof_h=18.0, z_foot=Z - 6, rng=rng, seg=40, door=(math.pi / 2 if gx < 0 else None)))
    add("north", ck.wing((-4.0, 214.0), (28.0, 214.0), 9.0, Z, 22.0, rng=rng, z_foot=Z - 6, dormers=False, crenel=True, roof=None,
                         end_gables=False))
    add("north", ck.wing((-110.0, 40.0), (-110.0, 190.0), 15.0, Z, 24.0, rng=rng, z_foot=Z - 6))
    turrets("north", [(-80.0, 207.5), (60.0, 207.5), (120.0, 207.5)], Z + 21.0)

    # ------------------------------------------------------------------ east: ranges over the ravine, Astronomy Tower
    add("east", ck.wing((178.0, -86.0), (180.0, 170.0), 16.0, Z, 30.0, rng=rng, crenel=True, z_foot=foot_line((178, -86), (180, 170), 8.0)))
    add("east", RT((180.0, -92.0), 7.0, Z, 52.0, roof_h=26.0, z_foot=foot(180, -92, 8), rng=rng, seg=48))
    add("east", RT((176.0, 40.0), 6.0, Z, 58.0, roof_h=26.0, z_foot=foot(176, 40, 7), rng=rng, seg=48))
    add("east", RT((158.0, 142.0), 6.8, Z, 110.0, roof_h=28.0, z_foot=Z - 6, rng=rng, seg=56, parapet=True))   # Astronomy Tower
    add("east", RT((176.0, 178.0), 6.0, Z, 60.0, roof_h=26.0, z_foot=foot(176, 178, 7), rng=rng, seg=48))
    add("east", ck.wing((130.0, 60.0), (130.0, 150.0), 14.0, Z, 26.0, rng=rng, z_foot=Z - 4))
    add("east", RT((140.0, 100.0), 4.5, Z, 70.0, roof_h=22.0, z_foot=Z - 4, rng=rng, seg=40))
    turrets("east", [(188.5, -30.0), (188.5, 100.0)], Z + 24.0)

    # ------------------------------------------------------------------ curtain walls on the rim
    rim = W.CRAG_OUTLINE
    cen = rim.mean(0)
    inward = rim + (cen - rim) / np.linalg.norm(cen - rim, axis=1, keepdims=True) * 3.0
    ang = np.degrees(np.arctan2(rim[:, 1] - cen[1], rim[:, 0] - cen[0]))
    for a0, a1, grp in ((-128.0, -100.0, "south"), (-80.0, -60.0, "south"), (-35.0, -20.0, "east"), (95.0, 115.0, "north"),
                        (150.0, 175.0, "west"), (-175.0, -150.0, "west")):
        sel = (ang >= a0) & (ang <= a1)
        pts = inward[sel]
        if len(pts) < 2:
            continue
        order = np.argsort(np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0]))
        pts = pts[order][::4]
        for i in range(len(pts) - 1):                       # per segment: each piece reaches down to its own rock
            zf = foot_line(pts[i], pts[i + 1], 1.5)
            add(grp, ck.curtain_wall(pts[i:i + 2], Z, 8.0, z_foot=zf, rng=rng))

    # ------------------------------------------------------------------ viaduct
    v0, v1 = W.VIADUCT
    add("viaduct", ck.viaduct(v0, v1, Z - 1.0, lambda x, y: float(top_at(x, y)), width=6.5, rng=rng))
    add("viaduct", RT(v1, 4.5, Z - 1.0, 12.0, roof_h=11.0, z_foot=foot(*v1, 5), rng=rng, seg=32, door=math.pi))

    # ------------------------------------------------------------------ boathouse + stair tower against the lake cliff
    bh = W.BOATHOUSE
    add("lake", ck.wing((bh[0] - 12.0, bh[1] - 2.0), (bh[0] + 12.0, bh[1] - 2.0), 12.0, 0.6, 9.0, rng=rng, z_foot=-4.0, dormers=False,
                        chimneys=False))
    sx, sy = cliff_point(bh[0] + 20.0, bh[1] - 6.0, bh[1] + 30.0, z_min=25.0)
    add("lake", RT((sx, sy + 3.0), 5.5, 0.6, 30.0, roof_h=13.0, z_foot=-4.0, rng=rng, seg=40, per_floor=3, win_w=0.9, win_h=2.0))
    # ------------------------------------------------------------------ the small tower + house on the gorge ledge (ref 1)
    led = find_ledge()
    if led:
        _, lx, ly, lz = led
        add("ledge", RT((lx, ly), 4.0, lz, 16.0, roof_h=13.0, z_foot=lz - 6.0, rng=rng, seg=32, per_floor=3, win_w=0.9, win_h=2.0))
        add("ledge", ck.wing((lx + 3.0, ly - 10.0), (lx + 3.0, ly - 24.0), 8.0, lz, 7.0, rng=rng, z_foot=lz - 6.0, dormers=False,
                             chimneys=True))
        print(f"  ledge buildings at ({lx:.0f}, {ly:.0f}, {lz:.1f})", flush=True)

    out = []
    for g, parts in groups.items():
        m = ck.finish(mk.merge(parts), Z if g not in ("lake", "ledge") else None)
        name = f"castle_{g}"
        m.save(f"{out_dir}/{name}.npz")
        out.append((name, m.nf))
        print(f"  {name:16s} {m.nf:9,d} tris", flush=True)
    print(f"  castle: {sum(n for _, n in out):,} tris in {len(out)} groups ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo")
