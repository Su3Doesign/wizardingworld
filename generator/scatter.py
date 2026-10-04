"""scatter - instance sets for the vegetation and rocks (positions, rotations, scales, variants).

Sets (each becomes per-variant instance records for Unreal and geometry-node instances for the Cycles previews):
    spruce        the forests: hills, gorge sides, the eastern (forbidden) forest, mountain slopes up to the tree line
    spruceyoung   forest edges, clearings and - with the small spruces - the ledges and talus of the castle crag (ref 1)
    pine, snag    rocky ridges and lake shores; a few dead trees in the forest
    birch         the lower gorge, the river banks, the lake shore and the edges of the grounds
    willow        the old willow of the grounds (one)
    boulder       talus below the cliffs, the river bed, the lake shore, forest floor, mountain slopes
    moss          moss cushions on the up-facing rock of the crag, the gorge and the ravine (ref 1 / 2 moss)
    fern, grass   forest floor and lawns near the castle and the gorge rim (the hero areas)

Ground in the core region comes from the top-surface map of the actual core mesh (granite included); elsewhere from H.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

import castle as C
import core
import instances as ins
import world as W
from meshkit import Mesh

R = core.REGION


def in_core(x, y, m=4.0):
    return (x > R["x"][0] + m) & (x < R["x"][1] - m) & (y > R["y"][0] + m) & (y < R["y"][1] - m)


def surface(x, y):
    """Ground height and (approximate) normal z at points: core top map inside the core, H outside."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    h = np.empty(len(x))
    nz = np.empty(len(x))
    ic = in_core(x, y)
    if ic.any():
        top, x0, y0 = C.surface_top()
        from scipy import ndimage

        ci = x[ic] - x0
        cj = y[ic] - y0
        h[ic] = ndimage.map_coordinates(top, [ci, cj], order=1, mode="nearest")
        gx = (ndimage.map_coordinates(top, [ci + 1.5, cj], order=1, mode="nearest") - ndimage.map_coordinates(top, [ci - 1.5, cj], order=1, mode="nearest")) / 3.0
        gy = (ndimage.map_coordinates(top, [ci, cj + 1.5], order=1, mode="nearest") - ndimage.map_coordinates(top, [ci, cj - 1.5], order=1, mode="nearest")) / 3.0
        nz[ic] = 1.0 / np.sqrt(1 + gx * gx + gy * gy)
    oc = ~ic
    if oc.any():
        e = 2.0
        xs, ys = x[oc], y[oc]
        hh = W.ground(np.concatenate([xs, xs + e, xs - e, xs, xs]), np.concatenate([ys, ys, ys, ys + e, ys - e]))
        n = len(xs)
        h[oc] = hh[:n]
        gx = (hh[n:2 * n] - hh[2 * n:3 * n]) / (2 * e)
        gy = (hh[3 * n:4 * n] - hh[4 * n:]) / (2 * e)
        nz[oc] = 1.0 / np.sqrt(1 + gx * gx + gy * gy)
    return h, nz


def jitter_grid(spacing, x_range, y_range, rng):
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X = X.ravel() + rng.uniform(0, spacing, X.size)
    Y = Y.ravel() + rng.uniform(0, spacing, Y.size)
    return X, Y


def upright(n, rng, lean=0.03):
    """Rotation matrices: vertical with a small random lean and a random yaw (vectorised)."""
    yaw = rng.uniform(0, 2 * math.pi, n)
    ax, ay = rng.normal(0, lean, n), rng.normal(0, lean, n)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.zeros((n, 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.0
    Rx = np.zeros((n, 3, 3))
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = np.cos(ax), -np.sin(ax), np.sin(ax), np.cos(ax)
    Ry = np.zeros((n, 3, 3))
    Ry[:, 1, 1] = 1.0
    Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 2, 0], Ry[:, 2, 2] = np.cos(ay), np.sin(ay), -np.sin(ay), np.cos(ay)
    return Ry @ Rx @ Rz


def random_rot(n, rng):
    """Uniformly random rotations (boulders)."""
    q = rng.normal(0, 1, (n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return ins.matrix_from_quat(q)


class Sets:
    def __init__(self):
        self.d = {}

    def add(self, name, pos, R_, scale, variant):
        if len(pos) == 0:
            return
        e = self.d.setdefault(name, dict(pos=[], R=[], scale=[], variant=[]))
        e["pos"].append(np.asarray(pos, np.float64))
        e["R"].append(np.asarray(R_, np.float64))
        sc = np.asarray(scale, np.float64)
        if sc.ndim == 1:
            sc = np.repeat(sc[:, None], 3, 1)
        e["scale"].append(sc)
        e["variant"].append(np.asarray(variant, np.int16))

    def clear_sightlines(self):
        """Remove trees (and big boulders) in front of the cameras of shots.py, so every shot has a clear view."""
        import shots

        cl = shots.clearings(shots.resolve(lambda x, y: float(surface(np.array([x]), np.array([y]))[0][0])))
        for name, e in self.d.items():
            if name in ("moss", "grass", "fern", "willow"):
                continue
            height = {"spruce": 29.0, "spruceyoung": 8.0, "pine": 18.0, "snag": 22.0, "birch": 14.0, "boulder": 1.2}.get(name, 20.0)
            pos = np.concatenate(e["pos"])
            sc = np.concatenate(e["scale"])[:, 2]
            keep = np.ones(len(pos), bool)
            for p, d, low, ha in cl:
                v = pos - p
                dist = np.linalg.norm(v[:, :2], axis=1)
                dh = np.array([d[0], d[1]]) / (np.linalg.norm(d[:2]) + 1e-9)
                along = v[:, :2] @ dh
                lat = np.abs(v[:, 0] * dh[1] - v[:, 1] * dh[0])
                tgt_d = float(np.linalg.norm((low - p)[:2]))
                in_cone = (along > -3.0) & (along < tgt_d - 15.0) & (lat < 6.0 + along * math.tan(math.radians(ha)))
                # height of the sight line (camera -> low subject point) above each candidate
                line_z = p[2] + (low[2] - p[2]) * np.clip(along / max(tgt_d, 1.0), 0, 1)
                top = pos[:, 2] + height * sc
                blocks = in_cone & ((top > line_z - 2.0) | (dist < 8.0))
                keep &= ~blocks
            n0 = len(pos)
            e["pos"] = [pos[keep]]
            e["R"] = [np.concatenate(e["R"])[keep]]
            e["scale"] = [np.concatenate(e["scale"])[keep]]
            e["variant"] = [np.concatenate(e["variant"])[keep]]
            if n0 - keep.sum():
                print(f"  sight lines: removed {n0 - int(keep.sum()):,} {name}", flush=True)

    def save(self, out_dir):
        os.makedirs(f"{out_dir}/instances", exist_ok=True)
        tot = 0
        for name, e in self.d.items():
            pos = np.concatenate(e["pos"])
            ins.save_set(f"{out_dir}/instances/{name}.npz", pos, np.concatenate(e["R"]), np.concatenate(e["scale"]), np.concatenate(e["variant"]))
            print(f"  {name:12s} {len(pos):9,d} instances", flush=True)
            tot += len(pos)
        print(f"  total {tot:,} instances", flush=True)


# ----------------------------------------------------------------------------------------------- forests
def forests(S, rng, density=1.0):
    t = time.time()
    E = W.EXTENT - 30.0
    X, Y = jitter_grid(4.6 / math.sqrt(density), (-E, E), (-E, E), rng)
    F = W.Fields(X, Y)
    f = F.forest()
    # nearer the castle the forest is denser; far away a little thinner
    d0 = np.hypot(X - 0.0, Y - 150.0)
    p = f * (0.62 - 0.22 * W.sstep(1500.0, 3500.0, d0))
    keep = rng.random(len(X)) < p
    X, Y = X[keep], Y[keep]
    h, nz = surface(X, Y)
    ok = (nz > 0.66) & (h > W.LAKE_Z + 1.2)
    X, Y, h, nz = X[ok], Y[ok], h[ok], nz[ok]
    n = len(X)
    kind = rng.random(n)
    elev = h
    # species mix: spruce (most), young spruce at edges (low forest mask), pine on dry ridges / high, snags, birch low by water
    fm = W.Fields(X, Y).forest()
    birch_zone = (W.sstep(140.0, 30.0, W.Fields(X, Y).gorge()[0]) * W.sstep(40.0, 5.0, h)) + W.sstep(25.0, 5.0, h - W.LAKE_Z) * 0.6
    is_birch = kind < 0.04 + 0.45 * np.clip(birch_zone, 0, 1)
    is_snag = (~is_birch) & (kind > 0.988)
    is_pine = (~is_birch) & (~is_snag) & (kind > 0.93 - 0.25 * W.sstep(300.0, 500.0, elev))
    is_young = (~is_birch) & (~is_snag) & (~is_pine) & (rng.random(n) < 0.12 + 0.5 * (1 - fm))
    is_spruce = ~(is_birch | is_snag | is_pine | is_young)
    pos = np.stack([X, Y, h - 0.35], 1)
    for name, sel, nvar, smin, smax in (("spruce", is_spruce, 4, 0.55, 1.2), ("spruceyoung", is_young, 2, 0.6, 1.35),
                                        ("pine", is_pine, 1, 0.75, 1.25), ("snag", is_snag, 1, 0.6, 1.1), ("birch", is_birch, 2, 0.7, 1.2)):
        k = int(sel.sum())
        if not k:
            continue
        sc = rng.uniform(smin, smax, k)
        # slightly smaller trees high up and on the poorest ground
        sc = sc * (1.0 - 0.3 * W.sstep(300.0, 520.0, elev[sel]))
        S.add(name, pos[sel], upright(k, rng, 0.025), sc, rng.integers(0, nvar, k))
    print(f"  forests: {n:,} trees ({time.time() - t:.0f}s)", flush=True)


def crag_ledges(S, rng):
    """Small spruces and young trees on the ledges and talus of the crag, the gorge and the ravine (ref 1)."""
    t = time.time()
    top, x0, y0 = C.surface_top()
    X, Y = jitter_grid(3.2, R["x"], R["y"], rng)
    ok = in_core(X, Y, 8.0)
    X, Y = X[ok], Y[ok]
    h, nz = surface(X, Y)
    F = W.Fields(X, Y)
    cd = F.crag()
    # 'cliff zone': steep ground within ~14 m (from the top map)
    from scipy import ndimage

    gy, gx = np.gradient(top)
    steep = np.hypot(gx, gy) > 1.2
    near = ndimage.binary_dilation(steep, iterations=14)
    ci = np.clip((X - x0).astype(int), 0, top.shape[0] - 1)
    cj = np.clip((Y - y0).astype(int), 0, top.shape[1] - 1)
    ledge = near[ci, cj] & (nz > 0.72) & (h > W.LAKE_Z + 1.5) & (h < W.CASTLE_Z - 3.0) & (cd > 3.0)
    p = np.where(ledge, 0.55, 0.0)
    keep = rng.random(len(X)) < p
    X, Y, h = X[keep], Y[keep], h[keep]
    n = len(X)
    young = rng.random(n) < 0.55
    pos = np.stack([X, Y, h - 0.3], 1)
    S.add("spruceyoung", pos[young], upright(int(young.sum()), rng, 0.06), rng.uniform(0.6, 1.5, int(young.sum())), rng.integers(0, 2, int(young.sum())))
    k = int((~young).sum())
    S.add("spruce", pos[~young], upright(k, rng, 0.05), rng.uniform(0.3, 0.6, k), rng.integers(0, 4, k))
    print(f"  crag ledges: {n:,} trees ({time.time() - t:.0f}s)", flush=True)


# ----------------------------------------------------------------------------------------------- rock + moss from the core mesh faces
def core_faces():
    import glob

    Vs, Fs, M1, M2, M3 = [], [], [], [], []
    off = 0
    for f in sorted(glob.glob("OUT/geo/terrain_core_*.npz")):
        m = Mesh.load(f)
        Vs.append(m.V)
        Fs.append(m.F + off)
        off += m.nv
        M1.append(m.uv1.mean(1))
        M2.append(m.uv2.mean(1))
        M3.append(m.uv3.mean(1))
    V = np.concatenate(Vs)
    F = np.concatenate(Fs)
    return V, F, np.concatenate(M1), np.concatenate(M2), np.concatenate(M3)


def sample_faces(V, F, weight, rng):
    """Points on triangles: Poisson count per face = weight (expected number), uniform barycentrics."""
    cnt = rng.poisson(np.clip(weight, 0, None))
    idx = np.repeat(np.arange(len(F)), cnt)
    if len(idx) == 0:
        return np.zeros((0, 3)), idx
    r1 = np.sqrt(rng.random(len(idx)))
    r2 = rng.random(len(idx))
    A, B, Cc = V[F[idx, 0]], V[F[idx, 1]], V[F[idx, 2]]
    P = (1 - r1)[:, None] * A + (r1 * (1 - r2))[:, None] * B + (r1 * r2)[:, None] * Cc
    return P, idx


def rocks_and_moss(S, rng, moss_density=4.0):
    t = time.time()
    V, F, m1, m2, m3 = core_faces()
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = 0.5 * np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(2 * area, 1e-12)[:, None]
    C_ = V[F].mean(1)
    rock, moss_m = m1[:, 0], m1[:, 1]
    shore = m2[:, 1]
    Fd = W.Fields(C_[:, 0], C_[:, 1])
    cd = Fd.crag()
    plateau = cd < -2.0
    # --- moss cushions on up-facing rock (ledges, block tops, the rim below the castle)
    w = area * moss_density * np.clip(rock, 0, 1) * W.sstep(0.35, 0.8, fn[:, 2]) * (0.35 + 0.65 * moss_m) * (~plateau) * (C_[:, 2] > W.LAKE_Z + 0.6)
    P, idx = sample_faces(V, F, w, rng)
    k = len(P)
    Rm = ins.frames_from_normals(fn[idx], rng, up_blend=0.35)
    S.add("moss", P - fn[idx] * 0.05, Rm, rng.uniform(0.18, 0.75, k) * np.array([1.0])[0], rng.integers(0, 4, k))
    # --- talus boulders: moderate slopes below steep ground, lake shore, river bed
    gd, gs, _ = Fd.gorge()
    talus = W.sstep(0.95, 0.75, fn[:, 2]) * W.sstep(0.3, 0.5, fn[:, 2]) * (C_[:, 2] < W.CASTLE_Z - 4) * (~plateau)
    river_bed = W.sstep(12.0, 4.0, gd) * (C_[:, 2] < W.gorge_bed(gs) + 2.0)
    wb = area * (0.020 * talus * (0.4 + rock) + 0.045 * river_bed + 0.02 * shore * (C_[:, 2] < 2.0))
    P, idx = sample_faces(V, F, wb, rng)
    k = len(P)
    big = rng.random(k) < 0.25
    sc = np.where(big, rng.uniform(1.8, 4.5, k), rng.uniform(0.4, 1.6, k))
    S.add("boulder", P - np.array([0, 0, 0.1]) * sc[:, None], random_rot(k, rng), sc[:, None] * rng.uniform(0.8, 1.2, (k, 3)), rng.integers(0, 8, k))
    print(f"  core rock: {int((w > 0).sum()):,} mossy faces -> moss/boulders ({time.time() - t:.0f}s)", flush=True)


def outer_boulders(S, rng):
    """Big boulders on the mountain slopes and in the forests (break up the far terrain), shore stones round the lake."""
    t = time.time()
    E = W.EXTENT - 30.0
    X, Y = jitter_grid(14.0, (-E, E), (-E, E), rng)
    ok = ~in_core(X, Y, -10.0)
    X, Y = X[ok], Y[ok]
    h, nz = surface(X, Y)
    F = W.Fields(X, Y)
    f = F.forest()
    p = 0.10 * W.sstep(0.95, 0.7, nz) + 0.03 * f + 0.15 * W.sstep(3.0, 0.5, h - W.LAKE_Z) * W.sstep(-1.5, 0.0, h - W.LAKE_Z)
    keep = rng.random(len(X)) < p
    X, Y, h = X[keep], Y[keep], h[keep]
    k = len(X)
    sc = rng.uniform(1.0, 5.5, k) * (1 + 0.6 * (rng.random(k) < 0.1))
    S.add("boulder", np.stack([X, Y, h - 0.2 * sc], 1), random_rot(k, rng), sc[:, None] * rng.uniform(0.8, 1.2, (k, 3)), rng.integers(0, 8, k))
    print(f"  outer boulders: {k:,} ({time.time() - t:.0f}s)", flush=True)


def understory(S, rng, grass_density=0.6, fern_density=0.10):
    """Ferns in the forest floor and grass tufts on lawns / rims near the castle (hero areas only)."""
    t = time.time()
    X, Y = jitter_grid(1.0 / math.sqrt(max(grass_density, fern_density)), (-560.0, 560.0), (-480.0, 640.0), rng)
    h, nz = surface(X, Y)
    F = W.Fields(X, Y)
    cd = F.crag()
    f = F.forest()
    lawn = (1 - f) * W.sstep(0.8, 0.95, nz) * (h > W.LAKE_Z + 1.5) * (cd > 1.0) * W.sstep(4.0, 9.0, F.road())
    near = W.sstep(650.0, 250.0, np.hypot(X - 0.0, Y - 60.0))
    pg = grass_density / max(grass_density, fern_density) * lawn * near * 0.8
    pf = fern_density / max(grass_density, fern_density) * f * W.sstep(0.7, 0.85, nz) * near
    u = rng.random(len(X))
    g = u < pg
    fe = (~g) & (u < pg + pf)
    for name, sel, lo, hi in (("grass", g, 0.35, 0.8), ("fern", fe, 0.6, 1.4)):
        k = int(sel.sum())
        S.add(name, np.stack([X[sel], Y[sel], h[sel] - 0.05], 1), upright(k, rng, 0.08), rng.uniform(lo, hi, k), rng.integers(0, 4, k))
    print(f"  understory: {int(g.sum()):,} grass tufts, {int(fe.sum()):,} ferns ({time.time() - t:.0f}s)", flush=True)


def build(out_dir="OUT/geo", seed=13, density=1.0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    S = Sets()
    forests(S, rng, density)
    crag_ledges(S, rng)
    rocks_and_moss(S, rng)
    outer_boulders(S, rng)
    understory(S, rng)
    wx, wy = W.WILLOW
    S.add("willow", np.array([[wx, wy, float(W.ground(wx, wy)[0]) - 0.6]]), upright(1, rng, 0.0), np.array([1.25]), np.array([0]))
    S.clear_sightlines()
    S.save(out_dir)
    print(f"  scatter done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo", density=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
