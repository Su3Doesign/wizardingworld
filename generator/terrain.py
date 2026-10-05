"""terrain - the outer valley and the mountains: an adaptive Delaunay height-field mesh round the core (core.py).

Point spacing grows with the distance from the core region (3 m next to it, 6 m on the near hills, 8 m out to 1.5 km,
12 m on the far mountains), so ridge lines stay smooth against the sky while the file stays small.  The mesh has a
hole a few metres inside the core region; in that overlap band the core surface dips below this one (no seam, no
z-fighting).  Per-vertex masks are the same as the core's (UV1 rock/moss, UV2 forest/shore, UV3 road/ao).
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.spatial import Delaunay

import core
import world as W
from meshkit import Mesh

BANDS = [(0.0, 150.0, 3.0), (150.0, 600.0, 6.0), (600.0, 1500.0, 8.0), (1500.0, 1e9, 12.0)]


def rect_distance(x, y, r):
    dx = np.maximum.reduce([r["x"][0] - x, x - r["x"][1], np.zeros_like(x)])
    dy = np.maximum.reduce([r["y"][0] - y, y - r["y"][1], np.zeros_like(y)])
    return np.hypot(dx, dy)


def sample_points(seed=3, scale=1.0):
    rng = np.random.default_rng(seed)
    E = W.EXTENT
    R = core.REGION
    hole = dict(x=(R["x"][0] + core.EDGE_TAPER, R["x"][1] - core.EDGE_TAPER), y=(R["y"][0] + core.EDGE_TAPER, R["y"][1] - core.EDGE_TAPER))
    pts = []
    for d0, d1, s in BANDS:
        s = s * scale
        xs = np.arange(-E, E + s, s)
        X, Y = np.meshgrid(xs, xs)
        X = X + rng.uniform(-0.35, 0.35, X.shape) * s
        Y = Y + rng.uniform(-0.35, 0.35, Y.shape) * s
        X = np.clip(X, -E, E)
        Y = np.clip(Y, -E, E)
        d = rect_distance(X, Y, hole)
        inside_hole = (X > hole["x"][0]) & (X < hole["x"][1]) & (Y > hole["y"][0]) & (Y < hole["y"][1])
        sel = (d >= d0) & (d < d1) & ~inside_hole
        pts.append(np.stack([X[sel], Y[sel]], 1))
    # exact borders: the hole rectangle (1.5 m) and the map edge
    hb = []
    for (xa, ya), (xb, yb) in (((hole["x"][0], hole["y"][0]), (hole["x"][1], hole["y"][0])), ((hole["x"][1], hole["y"][0]), (hole["x"][1], hole["y"][1])),
                               ((hole["x"][1], hole["y"][1]), (hole["x"][0], hole["y"][1])), ((hole["x"][0], hole["y"][1]), (hole["x"][0], hole["y"][0]))):
        n = int(math.hypot(xb - xa, yb - ya) / 1.5)
        t = np.linspace(0, 1, n, endpoint=False)
        hb.append(np.stack([xa + (xb - xa) * t, ya + (yb - ya) * t], 1))
    eb = []
    for k in np.linspace(-E, E, int(2 * E / (12.0 * scale)) + 1):
        eb += [(k, -E), (k, E), (-E, k), (E, k)]
    P = np.vstack(pts + hb + [np.array(eb)])
    P = np.unique(np.round(P, 3), axis=0)
    return P, hole


def build(out_dir, scale=1.0, tiles=(4, 4)):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    P, hole = sample_points(scale=scale)
    tri = Delaunay(P).simplices
    C = P[tri].mean(1)
    inside = (C[:, 0] > hole["x"][0]) & (C[:, 0] < hole["x"][1]) & (C[:, 1] > hole["y"][0]) & (C[:, 1] < hole["y"][1])
    tri = tri[~inside]
    print(f"  {len(P):,} points, {len(tri):,} triangles ({time.time() - t0:.0f}s)", flush=True)
    F = W.Fields(P[:, 0], P[:, 1])
    Hh = F.height()
    V = np.stack([P[:, 0], P[:, 1], Hh], 1)
    # counter-clockwise (upward) winding
    fn = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])
    flip = fn[:, 2] < 0
    tri[flip] = tri[flip][:, ::-1]
    print(f"  heights {Hh.min():.0f} .. {Hh.max():.0f} m ({time.time() - t0:.0f}s)", flush=True)
    N = core.vertex_normals(V, tri)
    m = core.masks(V, N, None, rock_w=np.zeros(len(V)))
    # far terrain: the AO channel carries a soft concavity term instead of the SDF occlusion
    m["ao"] = np.clip(0.75 + 0.25 * N[:, 2], 0.5, 1.0)
    x0, x1 = -W.EXTENT, W.EXTENT
    Cx = V[tri].mean(1)
    tx = np.clip(((Cx[:, 0] - x0) / (x1 - x0) * tiles[0]).astype(int), 0, tiles[0] - 1)
    ty = np.clip(((Cx[:, 1] - x0) / (x1 - x0) * tiles[1]).astype(int), 0, tiles[1] - 1)
    out = []
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            sel = (tx == i) & (ty == j)
            if not sel.any():
                continue
            sub = Mesh(V, tri[sel], 0).compact()
            idx = np.unique(tri[sel])
            mm = {k: v[idx] for k, v in m.items()}
            sub.uv0 = (sub.V[:, :2] / 4.0)[sub.F]
            core.pack(sub, mm)
            name = f"terrain_outer_{i}{j}"
            sub.save(f"{out_dir}/{name}.npz")
            out.append((name, sub.nf))
    print(f"  outer terrain: {sum(n for _, n in out):,} tris in {len(out)} tiles ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo", float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
