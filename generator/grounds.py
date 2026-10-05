"""grounds - everything round the castle: the Quidditch stadium, the gamekeeper's hut, the greenhouses, the stone circle,
the gates, the station, Hogsmeade, the boats with their lanterns, the tomb on the island; and the water surfaces
(the Black Lake, the river in the gorge, the ravine stream and the waterfall).

Materials (besides the castle_kit ones): CLOTH (stand canopies / banners, colour per stand from UV2.x), MARBLE (tomb),
LANTERN (emissive lantern glass), WATER (lake), RIVER (flowing water, UV0.v along the flow), FALL (waterfall sheet),
MIST (waterfall spray, translucent).
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
from meshkit import Mesh

CLOTH, MARBLE, LANTERN, WATER, RIVER, FALL, MIST = 26, 28, 34, 30, 31, 32, 33
STONE, TRIM, SLATE, LEAD, GLASS, WOOD = ck.STONE, ck.TRIM, ck.SLATE, ck.LEAD, ck.GLASS, ck.WOOD


def g(x, y):
    return float(W.ground(x, y)[0])


def gmin(x, y, r):
    a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    return float(W.ground(np.concatenate([[x], x + r * np.cos(a)]), np.concatenate([[y], y + r * np.sin(a)])).min())


def _bx(c, size, yaw, mat):
    a = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    b = np.array([-a[1], a[0], 0.0])
    return ck._mesh_box(c, size, a, b, mat)


# ----------------------------------------------------------------------------------------------- Quidditch stadium
def stadium(rng):
    cx, cy = W.PITCH
    W2, L2 = W.PITCH_SIZE[0] / 2, W.PITCH_SIZE[1] / 2
    z0 = g(cx, cy)
    parts = []
    n = 22
    houses = [0.02, 0.27, 0.52, 0.77]                     # four house colours (UV2.x picks the palette entry)
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        x = cx + (W2 + 6.0) * math.cos(a)
        y = cy + (L2 + 6.0) * math.sin(a)
        zt = z0 + rng.uniform(17.0, 27.0)
        col = houses[(k * 4) // n]
        # timber tower: four posts, cross braces, a cloth-wrapped stand box on top, pointed canvas roof
        yaw = a
        for sx in (-1, 1):
            for sy in (-1, 1):
                p = np.array([x + sx * 2.2 * math.cos(yaw) - sy * 2.2 * math.sin(yaw), y + sx * 2.2 * math.sin(yaw) + sy * 2.2 * math.cos(yaw), 0.0])
                p[2] = (z0 + zt) / 2
                parts.append(_bx(p, (0.45, 0.45, zt - z0), yaw, WOOD))
        for zz in np.arange(z0 + 3.0, zt - 4.0, 4.0):
            parts.append(_bx((x, y, zz), (5.0, 5.0, 0.3), yaw, WOOD))
        stand = _bx((x, y, zt - 2.0), (5.6, 5.6, 4.2), yaw, CLOTH)
        stand.uv2 = np.broadcast_to(np.array([col, 1.0]), (stand.nf, 3, 2)).copy()
        parts.append(stand)
        roof = ck._pyramid_roof((x, y), 6.2, 6.2, zt + 0.1, rng.uniform(5.5, 8.0), yaw)
        roof.mat[:] = CLOTH
        roof.uv2 = np.broadcast_to(np.array([col, 1.0]), (roof.nf, 3, 2)).copy()
        parts.append(roof)
        # pennant
        pole = _bx((x, y, zt + 8.5), (0.12, 0.12, 4.0), yaw, LEAD)
        parts.append(pole)
    # low stands joining the towers
    for k in range(n):
        a0 = 2 * math.pi * (k + 0.5) / n
        a1 = 2 * math.pi * (k + 1.5) / n
        p0 = np.array([cx + (W2 + 6.0) * math.cos(a0), cy + (L2 + 6.0) * math.sin(a0)])
        p1 = np.array([cx + (W2 + 6.0) * math.cos(a1), cy + (L2 + 6.0) * math.sin(a1)])
        mid = (p0 + p1) / 2
        L = float(np.linalg.norm(p1 - p0))
        yaw = math.atan2(*(p1 - p0)[::-1])
        st = _bx((mid[0], mid[1], z0 + 3.5), (L - 4.0, 3.2, 7.0), yaw, WOOD)
        parts.append(st)
        cl = _bx((mid[0], mid[1], z0 + 7.6), (L - 4.0, 3.4, 1.2), yaw, CLOTH)
        cl.uv2 = np.broadcast_to(np.array([houses[(k * 4) // n], 1.0]), (cl.nf, 3, 2)).copy()
        parts.append(cl)
    # goal hoops: three at each end (poles with rings)
    for end in (-1, 1):
        for i, (dx, hgt) in enumerate(((-7.0, 15.0), (0.0, 18.0), (7.0, 15.0))):
            x, y = cx + dx, cy + end * (L2 - 12.0)
            parts.append(_bx((x, y, z0 + hgt / 2), (0.35, 0.35, hgt), 0.0, LEAD))
            ring = mk.revolve([(2.0, -0.18), (2.35, -0.18), (2.35, 0.18), (2.0, 0.18)], seg=40, mat=LEAD)
            ring.rotate_x(math.pi / 2)
            ring.translate((x, y, z0 + hgt + 2.3))
            ring.uv_box(1.0, only_missing=False)
            parts.append(ring)
    return mk.merge(parts)


# ----------------------------------------------------------------------------------------------- hut, greenhouses, stones
def hut(rng):
    x, y = W.HUT
    z = g(x, y)
    parts = [ck.round_tower((x, y), 4.2, z, 4.0, roof_h=6.5, z_foot=z - 1.5, rng=rng, seg=24, per_floor=3, win_w=0.9, win_h=1.4,
                           courses=False, door=-math.pi / 2, cone_eave=0.9)]
    parts.append(_bx((x + 2.6, y + 2.6, z + 6.0), (1.2, 1.2, 8.0), 0.6, STONE))     # chimney
    # fence + pumpkin patch posts
    for k in range(16):
        a = -1.2 + 0.12 * k
        parts.append(_bx((x + 12 * math.cos(a), y + 12 * math.sin(a), g(x + 12 * math.cos(a), y + 12 * math.sin(a)) + 0.6), (0.16, 0.16, 1.2), a, WOOD))
    return mk.merge(parts)


def greenhouses(rng):
    cx, cy = W.GREENHOUSES
    z = g(cx, cy)
    parts = []
    for i, dy in enumerate((-18.0, 0.0, 18.0)):
        L, Wd = 34.0, 9.0
        c = (cx, cy + dy)
        parts.append(_bx((c[0], c[1], z + 0.5), (L, Wd, 1.6), 0.0, STONE))          # dwarf brick wall
        # glass gable roof on an iron frame
        roof, _ = ck.gable_roof((c[0], c[1], 0.0), (1.0, 0.0), L, Wd, z + 3.4, 40.0, overhang=0.1, course=1.2, step=0.0, ridge=True)
        roof.mat[:] = GLASS
        roof.uv2 = np.broadcast_to(np.array([rng.random(), 0.0]), (roof.nf, 3, 2)).copy()
        parts.append(roof)
        for side in (-1, 1):
            wall = _bx((c[0], c[1] + side * Wd / 2, z + 2.3), (L, 0.08, 2.2), 0.0, GLASS)
            wall.uv2 = np.zeros((wall.nf, 3, 2))
            parts.append(wall)
        for k in range(int(L // 2.0) + 1):
            x = c[0] - L / 2 + 2.0 * k
            for side in (-1, 1):
                parts.append(_bx((x, c[1] + side * Wd / 2, z + 2.3), (0.1, 0.12, 2.3), 0.0, LEAD))
            parts.append(_bx((x, c[1], z + 3.4 + (Wd / 2) * math.tan(math.radians(40)) / 2), (0.1, Wd, 0.1), 0.0, LEAD))
    return mk.merge(parts)


def stone_circle(rng):
    cx, cy = W.STONE_CIRCLE
    parts = []
    for k in range(13):
        a = 2 * math.pi * k / 13 + rng.normal(0, 0.05)
        x, y = cx + 15.0 * math.cos(a), cy + 15.0 * math.sin(a)
        z = g(x, y)
        h = rng.uniform(3.2, 5.4)
        st = mk.rounded_box((0, 0, 0), (1.3, 0.8, h), r=0.18, max_edge=0.4)
        st.V = st.V + 0.08 * mk.fbm(st.V * 1.3 + k, 3, seed=k)[:, None] * st.vertex_normals()
        st.rotate_z(a + rng.normal(0, 0.15))
        st.rotate_y(rng.normal(0, 0.06))
        st.translate((x, y, z + h / 2 - 0.4))
        st.mat[:] = STONE
        st.uv_box(1.0, only_missing=False)
        parts.append(st)
    return mk.merge(parts)


def gates(rng):
    x, y = W.GATES
    z = g(x, y)
    parts = []
    for side in (-1, 1):
        px = x + side * 7.0
        parts.append(_bx((px, y, z + 4.0), (2.4, 2.4, 8.0), 0.0, STONE))
        parts.append(_bx((px, y, z + 8.2), (2.9, 2.9, 0.5), 0.0, TRIM))
        # the winged boar: a heavy body block with two swept wings (stylised)
        body = mk.rounded_box((px, y, z + 9.6), (1.0, 2.1, 1.3), r=0.35, max_edge=0.3)
        body.mat[:] = TRIM
        parts.append(body)
        for wsgn in (-1, 1):
            wing = mk.rounded_box((0, 0, 0), (0.18, 1.8, 1.3), r=0.06, max_edge=0.3)
            wing.rotate_y(wsgn * 0.6)
            wing.translate((px + wsgn * 0.85, y - 0.2, z + 10.7))
            wing.mat[:] = TRIM
            parts.append(wing)
        # flanking wall
        parts.append(_bx((px + side * 16.0, y, z + 2.2), (28.0, 1.2, 4.4), 0.0, STONE))
    # wrought iron gate leaves
    for k in range(17):
        parts.append(_bx((x - 5.6 + 0.7 * k, y, z + 3.2), (0.08, 0.08, 6.0 + 0.6 * math.sin(k / 16 * math.pi)), 0.0, LEAD))
    for zz in (z + 1.0, z + 5.0):
        parts.append(_bx((x, y, zz), (11.6, 0.12, 0.12), 0.0, LEAD))
    out = mk.merge(parts)
    out.uv_box(1.0, only_missing=False)
    return out


def tomb(rng):
    x, y = W.ISLANDS[0][:2]
    z = g(x, y)
    parts = [_bx((x, y, z + 0.4), (4.4, 2.6, 0.8), 0.3, MARBLE), _bx((x, y, z + 1.4), (3.8, 2.0, 1.4), 0.3, MARBLE),
             _bx((x, y, z + 2.25), (4.0, 2.2, 0.3), 0.3, MARBLE)]
    return mk.merge(parts)


# ----------------------------------------------------------------------------------------------- station + village
def station(rng):
    x, y = W.STATION
    z = W.LAKE_Z + 3.2
    parts = [_bx((x, y, z - 0.4), (120.0, 9.0, 1.6), 0.55, STONE),
             ck.wing((x - 22.0, y + 12.0), (x + 22.0, y + 12.0 + 44.0 * 0.0), 10.0, z, 7.5, rng=rng, z_foot=z - 2.0, dormers=False,
                     chimneys=True)]
    # lamp posts with lanterns along the platform
    a = 0.55
    for k in range(9):
        u = -48.0 + 12.0 * k
        px, py = x + u * math.cos(a), y + u * math.sin(a)
        parts.append(_bx((px, py, z + 2.0), (0.14, 0.14, 4.0), a, LEAD))
        lan = _bx((px, py, z + 4.25), (0.45, 0.45, 0.6), a, LANTERN)
        parts.append(lan)
    # rails heading north-east
    for side in (-0.75, 0.75):
        for k in range(60):
            u = -60.0 + 20.0 * k
            px, py = x + u * math.cos(a) + side * -math.sin(a) - 0.0, y + u * math.sin(a) + side * math.cos(a) - 7.0
            gz = g(px, py)
            if gz < W.LAKE_Z + 0.5:
                continue
            parts.append(_bx((px, py, gz + 0.25), (20.0, 0.12, 0.18), a, LEAD))
    return mk.merge(parts)


def village(rng):
    cx, cy = W.HOGSMEADE
    parts = []
    street = W.catmull([(cx - 150, cy - 160), (cx - 40, cy - 30), (cx + 10, cy + 60), (cx - 20, cy + 190)], step=2.0)
    cross = W.catmull([(cx - 170, cy + 40), (cx, cy + 30), (cx + 170, cy + 10)], step=2.0)
    for path in (street, cross):
        seg = np.diff(path, axis=0)
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
        u = 6.0
        while u < L[-1] - 6.0:
            i = np.searchsorted(L, u) - 1
            p = path[i]
            d = seg[min(i, len(seg) - 1)]
            d = d / (np.linalg.norm(d) + 1e-9)
            nrm = np.array([-d[1], d[0]])
            for side in (-1, 1):
                if rng.random() < 0.15:
                    continue
                w = rng.uniform(7.0, 11.0)
                dp = rng.uniform(6.5, 8.5)
                c = p + nrm * side * (6.0 + dp / 2)
                z = gmin(c[0], c[1], 5.0)
                yaw = math.atan2(d[1], d[0])
                a = np.array([math.cos(yaw), math.sin(yaw)])
                h = rng.uniform(5.5, 8.5)
                parts.append(ck.wing(c - a * w / 2, c + a * w / 2, dp, z, h, rng=rng, z_foot=z - 2.0, pitch=rng.uniform(48, 58),
                                     dormers=rng.random() < 0.5, chimneys=True, win_kind="flat", floor_h=2.9))
            u += rng.uniform(10.0, 14.0)
    return mk.merge(parts)


# ----------------------------------------------------------------------------------------------- boats (library + route)
def boat_mesh(seed=0):
    """A small rowing boat (~4.2 m) with a lantern on a pole at the bow; local +X = forward."""
    rng = np.random.default_rng(seed)
    L, B, D = 4.2, 1.35, 0.55
    u = np.linspace(-1, 1, 25)
    v = np.linspace(0, 1, 9)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    half = B / 2 * (1 - np.abs(U) ** 2.2) ** 0.55
    x = U * L / 2
    yl = -half * np.sin(Vv * math.pi / 2)
    zl = -D * np.cos(Vv * math.pi / 2) * (1 - 0.35 * np.abs(U) ** 3) + 0.25 * U ** 4
    P1 = np.stack([x, yl, zl], -1)
    P2 = np.stack([x, -yl, zl], -1)
    hull_out = mk.merge([mk.grid(P1, mat=WOOD), mk.grid(P2[:, ::-1], mat=WOOD)])
    if np.mean(hull_out.face_normals()[:, 2]) > 0:
        pass
    inner = hull_out.copy()
    inner.V = inner.V * np.array([0.96, 0.93, 0.93]) + np.array([0, 0, 0.03])
    inner.flip()
    parts = [hull_out, inner]
    for xs in (-0.8, 0.6):
        parts.append(_bx((xs, 0, -0.18), (0.3, B * 0.85, 0.05), 0.0, WOOD))
    parts.append(_bx((1.75, 0, 0.55), (0.06, 0.06, 1.3), 0.0, WOOD))
    parts.append(_bx((1.95, 0, 1.15), (0.45, 0.04, 0.04), 0.0, WOOD))
    lan = _bx((2.12, 0, 0.98), (0.22, 0.22, 0.32), 0.0, LANTERN)
    parts.append(lan)
    parts.append(_bx((2.12, 0, 1.17), (0.26, 0.26, 0.05), 0.0, LEAD))
    m = mk.merge(parts)
    m.uv_box(0.6, only_missing=False)
    return m


def boat_route(n=14, seed=4):
    """Boats spread along the crossing from the station dock to the boathouse (for the night arrival shot)."""
    rng = np.random.default_rng(seed)
    a = np.array(W.BOAT_DOCK_STATION)
    b = np.array([W.BOATHOUSE[0] + 4.0, W.BOATHOUSE[1] - 28.0])
    mid = (a + b) / 2 + np.array([-120.0, -170.0])
    route = W.catmull([a, mid, b], step=4.0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    pos, R, scl = [], [], []
    for k in range(n):
        t = 0.45 + 0.5 * k / n + rng.uniform(-0.01, 0.01)
        u = t * s[-1]
        i = np.searchsorted(s, u) - 1
        p = route[i]
        d = route[min(i + 1, len(route) - 1)] - route[i]
        yaw = math.atan2(d[1], d[0]) + rng.normal(0, 0.06)
        off = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * rng.normal(0, 9.0)
        c, s_ = math.cos(yaw), math.sin(yaw)
        pos.append([p[0] + off[0], p[1] + off[1], W.LAKE_Z + 0.12])
        R.append([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        scl.append([1.0, 1.0, 1.0])
    return np.array(pos), np.array(R), np.array(scl), route


# ----------------------------------------------------------------------------------------------- water surfaces
def lake_surface():
    """The Black Lake: a disc-like grid slightly larger than the shoreline (it runs under the terrain), z = 0."""
    P = W.LAKE_OUTLINE
    cen = P.mean(0)
    ext = (P - cen) * 1.06 + cen
    xs = np.arange(ext[:, 0].min() - 40, ext[:, 0].max() + 40, 25.0)
    ys = np.arange(ext[:, 1].min() - 40, ext[:, 1].max() + 40, 25.0)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Pz = np.stack([X, Y, np.full_like(X, W.LAKE_Z)], -1)
    m = mk.grid(Pz, uv_tile=1.0, mat=WATER)
    C = m.V[m.F].mean(1)
    F = W.Fields(C[:, 0], C[:, 1])
    keep = F.lake() < 60.0
    m = m.select_faces(keep)
    m.uv0 = (m.V[:, :2] / 50.0)[m.F]
    if np.mean(m.face_normals()[:, 2]) < 0:
        m.flip()
    return m


def ribbon(path, z_of_s, width_of_s, mat, step=2.0, s_range=None, uv_scale=4.0):
    """Water ribbon along a centre line: UV0 u across (0..1), v = distance along the flow / uv_scale."""
    P = np.asarray(path, np.float64)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    lo, hi = s_range or (0.0, s[-1])
    ss = np.arange(lo, hi, step)
    xs = np.interp(ss, s, P[:, 0])
    ys = np.interp(ss, s, P[:, 1])
    t = np.gradient(np.stack([xs, ys], 1), axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    wv = width_of_s(ss)
    zv = z_of_s(ss)
    across = np.linspace(-1, 1, 7)
    G = np.empty((len(ss), len(across), 3))
    G[:, :, 0] = xs[:, None] + nrm[:, None, 0] * across[None] * wv[:, None] / 2
    G[:, :, 1] = ys[:, None] + nrm[:, None, 1] * across[None] * wv[:, None] / 2
    G[:, :, 2] = zv[:, None] + 0.0 * across[None]
    m = mk.grid(G, mat=mat)
    UV = np.stack(np.meshgrid(ss / uv_scale, (across + 1) / 2, indexing="ij"), -1)[..., ::-1].reshape(-1, 2)
    m.uv0 = UV[m.F]
    if np.mean(m.face_normals()[:, 2]) < 0:
        m.flip()
    return m


def river():
    path = W.GORGE_PATH
    L = W.pathfield("gorge").length
    s_end = L - 30.0                                       # it meets the lake

    def z(s):
        return np.maximum(W.gorge_bed(s) + 0.7, W.LAKE_Z + 0.02)

    def w(s):
        return 15.0 + 6.0 * np.clip((s - (L - 900.0)) / 900.0, 0, 1)

    return ribbon(path, z, w, RIVER, step=2.0, s_range=(0.0, s_end), uv_scale=6.0)


def stream_and_fall():
    """The stream across the grounds, then down the cleft through the entry stairs (world.FALL_BED): it runs under each
    flight's arch and falls on the rock banks between them, the last fall into the bay.  Returns the water ribbons (runs),
    the falling sheets and the spray."""
    path = W.RAVINE_PATH
    L = W.pathfield("ravine").length

    def z(s):
        return W.ravine_bed(s) + 0.35

    def w(s):
        return 7.0 - 1.0 * np.clip((s - (L - 120.0)) / 120.0, 0, 1)

    runs = [ribbon(path, z, w, RIVER, step=1.5, s_range=(0.0, L - 0.3), uv_scale=4.0)]
    falls, mists = [], []
    B = W.FALL_BED
    x = W.FALL_X
    y_end = float(path[-1][1])
    for i in range(len(B) - 1):
        (ya, za), (yb, zb) = B[i], B[i + 1]
        if ya > y_end or zb < W.LAKE_Z - 1.0 and za < W.LAKE_Z:
            continue
        drop = za - zb
        if drop < 3.0:                                     # a run: the water's surface follows the bed
            yy = np.arange(min(ya, y_end), yb - 0.01, -1.0)
            if len(yy) < 2:
                continue
            P = np.stack([np.full(len(yy), x), yy], 1)
            runs.append(ribbon(P, lambda s_, ya_=min(ya, y_end), yb_=yb, za_=za, zb_=zb: np.interp(s_, [0.0, ya_ - yb_], [za_, zb_]) + 0.35,
                               lambda s_: np.full_like(s_, 5.6), RIVER, step=1.0, uv_scale=4.0))
            continue
        # a fall: the sheet leaves the lip level and curves down to the pool (a free-fall parabola), spreading a little
        zb_eff = max(zb, W.LAKE_Z - 0.3)
        rows = np.linspace(0, 1, max(12, int(drop / 0.8)))
        cols = np.linspace(-1, 1, 11)
        G = np.empty((len(rows), len(cols), 3))
        for k, t in enumerate(rows):
            wid = 5.2 + 1.6 * t
            G[k, :, 0] = x + cols * wid / 2
            G[k, :, 1] = ya + 0.2 - (ya - yb + 0.6) * t - 0.25 * np.cos(cols * 2.0) * t
            G[k, :, 2] = za + 0.35 - (za + 0.35 - zb_eff) * t * t
        f = mk.grid(G, mat=FALL)
        UV = np.stack(np.meshgrid(rows * drop / 4.0, (cols + 1) / 2, indexing="ij"), -1)[..., ::-1].reshape(-1, 2)
        f.uv0 = UV[f.F]
        if np.mean(f.face_normals()[:, 1]) > 0:          # the sheet faces the lake (south)
            f.flip()
        falls.append(f)
        r = 3.0 + 0.35 * drop
        mist = mk.revolve([(0.0, 0.0), (r, 0.3), (r * 1.15, r * 0.35), (r * 0.65, r * 0.85), (0.0, r * 1.05)], seg=24, mat=MIST)
        mist.translate((x, yb - 1.5, zb_eff - 0.3))
        if mist.volume() < 0:
            mist.flip()
        mist.uv_box(10.0, only_missing=False)
        mists.append(mist)
    return mk.merge(runs), mk.merge(falls), mk.merge(mists)


# ----------------------------------------------------------------------------------------------- driver
def build(out_dir, seed=8):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def save(name, m, z_ground=None):
        if m is None or m.nf == 0:
            return
        m = ck.finish(m, z_ground)
        m.save(f"{out_dir}/{name}.npz")
        out.append((name, m.nf))
        print(f"  {name:22s} {m.nf:9,d} tris", flush=True)

    save("grounds_stadium", stadium(rng))
    save("grounds_hut", hut(rng))
    save("grounds_greenhouses", greenhouses(rng))
    save("grounds_stonecircle", stone_circle(rng))
    save("grounds_gates", gates(rng))
    save("grounds_tomb", tomb(rng))
    save("grounds_station", station(rng))
    save("grounds_hogsmeade", village(rng))
    # water (no stone masks)
    lk = lake_surface()
    lk.save(f"{out_dir}/water_lake.npz")
    out.append(("water_lake", lk.nf))
    rv = river()
    rv.save(f"{out_dir}/water_river.npz")
    out.append(("water_river", rv.nf))
    st, fall, mist = stream_and_fall()
    st.save(f"{out_dir}/water_stream.npz")
    fall.save(f"{out_dir}/water_fall.npz")
    mist.save(f"{out_dir}/water_mist.npz")
    out += [("water_stream", st.nf), ("water_fall", fall.nf), ("water_mist", mist.nf)]
    # boats: library mesh + instance set
    import instances as ins

    os.makedirs(f"{out_dir}/lib", exist_ok=True)
    os.makedirs(f"{out_dir}/instances", exist_ok=True)
    bm = boat_mesh()
    bm.uv1 = np.zeros((bm.nf, 3, 2))
    bm.uv2 = np.zeros((bm.nf, 3, 2))
    bm.save(f"{out_dir}/lib/lib_boat_00.npz")
    pos, R, scl, route = boat_route()
    ins.save_set(f"{out_dir}/instances/boats.npz", pos, R, scl, np.zeros(len(pos), np.int16))
    np.save(f"{out_dir}/boat_route.npy", route)
    print(f"  boats: {len(pos)} on the crossing; grounds done in {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo")
