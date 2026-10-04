"""trees - high-poly vegetation library (real geometry, no alpha cards: opaque, Nanite-friendly).

    lib_spruce_00..03   Norway spruce, 26-30 m: tapered trunk with root flare, whorls of drooping branches that curve up at
                        the tips, curtains of pendulous needle sprays (serrated strips carrying the needle texture)
    lib_spruce_young_00..01   dense young spruces (~7 m) for edges, ledges and clearings
    lib_pine_00         Scots pine: orange upper bark, high umbrella crown of needle tufts
    lib_snag_00         dead spruce: bare grey trunk and broken branches
    lib_birch_00..01    birch / rowan: white-grey trunk, fine twigs, small leaves (lower gorge, the lake shore)
    lib_willow_00       the gnarled old willow of the grounds
    lib_boulder_00..07  granite boulders: faceted blocks with rounded weathered edges (talus, shores, river bed, forest)

Units: metres, the trunk base at the origin, +Z up.  Material ids: 40 bark, 41 needles, 42 leaves, 43 dead wood,
44 birch bark, 0 stone (boulders use the terrain's rock material).  UV1.x = per-spray / per-leaf tint (0..1),
UV1.y = 'sun' (outer crown, 1) vs inner shade (0) - the materials use both for natural colour variation.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

import meshkit as mk
from meshkit import Mesh

BARK, NEEDLES, LEAVES, DEADWOOD, BIRCHBARK, ROCK = 40, 41, 42, 43, 44, 0


# ----------------------------------------------------------------------------------------------- primitives
def tube(points, radius, sides=6, mat=BARK, v_scale=1.0, closed_tip=True):
    """Tapered tube along a polyline; UV u round, v along (metres / v_scale)."""
    P = np.asarray(points, np.float64)
    k = len(P)
    r = np.broadcast_to(np.asarray(radius, np.float64), (k,))
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    ref = np.where(np.abs(T[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    S = np.cross(T, ref)
    S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    B = np.cross(T, S)
    ang = np.linspace(0, 2 * math.pi, sides + 1)
    ring = (np.cos(ang)[None, :, None] * S[:, None, :] + np.sin(ang)[None, :, None] * B[:, None, :]) * r[:, None, None]
    G = P[:, None, :] + ring
    m = mk.grid(G.transpose(1, 0, 2), mat=mat)                 # (sides+1, k, 3): u round, v along
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    circ = 2 * math.pi * float(r.mean())
    UV = np.stack(np.meshgrid(ang / (2 * math.pi) * max(circ, 0.05) / v_scale, s / v_scale, indexing="ij"), -1).reshape(-1, 2)
    m.uv0 = UV[m.F]
    return m                     # callers fix the winding with _orient_out


def _orient_out(m, axis_pts):
    """Flip a tube if its normals point inwards (test against the nearest axis point)."""
    from scipy.spatial import cKDTree

    C = m.V[m.F].mean(1)
    _, i = cKDTree(axis_pts).query(C)
    out = C - axis_pts[i]
    if np.mean(np.einsum("ij,ij->i", m.face_normals(), out)) < 0:
        m.flip()
    return m


def strip(center, direction, normal, length, width, segs=8, serr=0.35, mat=NEEDLES, tint=0.5, sun=1.0, droop=0.0, curl=0.0):
    """A needle spray / leaflet strip: V-section ribbon with serrated edges (the texture carries the needles).
    UV0: u across (0..1), v along (0..1)."""
    d = np.asarray(direction, np.float64)
    d /= np.linalg.norm(d)
    n = np.asarray(normal, np.float64)
    n = n - d * np.dot(n, d)
    n /= np.linalg.norm(n) + 1e-12
    w = np.cross(n, d)
    s = np.linspace(0, 1, segs + 1)
    wid = width * np.sin(math.pi * np.clip(0.08 + 0.92 * s, 0, 1)) ** 0.55
    wid = wid * (1.0 + serr * np.where(np.arange(segs + 1) % 2 == 0, 0.5, -0.5))
    wid[-1] = 0.0
    cen = np.asarray(center)[None] + d[None] * (length * s)[:, None] - n[None] * (droop * length * s * s)[:, None]
    ridge = n[None] * (wid * 0.22)[:, None]
    L = cen + w[None] * (wid / 2 * np.cos(curl))[:, None] - n[None] * (wid / 2 * np.sin(curl))[:, None]
    R = cen - w[None] * (wid / 2 * np.cos(curl))[:, None] - n[None] * (wid / 2 * np.sin(curl))[:, None]
    M = cen + ridge
    V = np.concatenate([L, M, R])
    k = segs + 1
    F = []
    for c in range(2):
        a = np.arange(k) + c * k
        b = a + k
        for q in range(segs):
            F += [(a[q], b[q], b[q + 1]), (a[q], b[q + 1], a[q + 1])]
    F = np.array(F)
    m = Mesh(V, F, mat)
    UVv = np.concatenate([np.stack([np.zeros(k), s], 1), np.stack([np.full(k, 0.5), s], 1), np.stack([np.ones(k), s], 1)])
    m.uv0 = UVv[F]
    m.uv1 = np.broadcast_to(np.array([tint, sun]), (m.nf, 3, 2)).copy()
    # face 'up' side towards the normal
    if np.mean(m.face_normals() @ n) < 0:
        m.flip()
    return m


def _curve(p0, dirs, lengths):
    pts = [np.asarray(p0, np.float64)]
    for d, L in zip(dirs, lengths):
        pts.append(pts[-1] + np.asarray(d) * L)
    return np.array(pts)


# ----------------------------------------------------------------------------------------------- spruce
def spruce(seed, H=28.0, crown=0.20, young=False):
    rng = np.random.default_rng(seed)
    parts = []
    # trunk: slight sway, root flare
    nz = 40
    z = np.linspace(0, H, nz)
    sway = np.stack([0.25 * np.sin(z / H * 3.1 + rng.uniform(0, 6)) * (z / H), 0.25 * np.cos(z / H * 2.3 + rng.uniform(0, 6)) * (z / H), z], 1)
    r_base = H * (0.013 if not young else 0.018)
    rad = r_base * (1 - z / H) ** 0.9 + 0.012
    rad[0] *= 1.65
    rad[1] *= 1.25
    trunk = tube(sway, rad, sides=10 if not young else 7, mat=BARK, v_scale=1.2)
    parts.append(_orient_out(trunk, sway))
    # roots: a few flared buttresses
    for k in range(5):
        a = 2 * math.pi * k / 5 + rng.uniform(-0.3, 0.3)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        pts = np.array([[0, 0, 0.6], d * 0.5 + [0, 0, 0.15], d * 1.1 + [0, 0, -0.05], d * 1.6 + [0, 0, -0.25]])
        rt = tube(pts, np.array([rad[0] * 0.55, rad[0] * 0.4, rad[0] * 0.22, 0.03]), sides=6, mat=BARK)
        parts.append(_orient_out(rt, pts))
    h0 = H * (0.10 if not young else 0.02)
    whorl = 0.62 if not young else 0.42
    zs = np.arange(h0, H - 0.3, whorl)
    Rmax = H * crown
    n_spray = 0
    for zi in zs:
        t = (zi - h0) / (H - h0)
        ztop = 1 - t
        Rz = Rmax * (ztop ** 0.92) * (1.0 + 0.12 * rng.normal()) + 0.25
        if not young and t < 0.12:
            Rz *= 0.55                                       # lowest branches short / partly dead
        nb = int(rng.integers(4, 7))
        base_a = rng.uniform(0, 2 * math.pi)
        c0 = np.interp(zi, z, sway[:, 0]), np.interp(zi, z, sway[:, 1])
        for k in range(nb):
            a = base_a + 2 * math.pi * k / nb + rng.normal(0, 0.25)
            out = np.array([math.cos(a), math.sin(a), 0.0])
            L = Rz * rng.uniform(0.8, 1.1)
            # branch: leaves the trunk slightly downwards, droops, the tip turns up
            dip = -0.35 - 0.25 * (1 - t)
            dirs = [out + [0, 0, dip * 0.6], out + [0, 0, dip], out + [0, 0, dip * 0.7], out + [0, 0, 0.05], out + [0, 0, 0.35]]
            dirs = [np.asarray(dd) / np.linalg.norm(dd) for dd in dirs]
            segL = [L * f for f in (0.22, 0.24, 0.22, 0.18, 0.14)]
            bp = _curve((c0[0], c0[1], zi), dirs, segL)
            br = 0.035 * (L / 4.0) ** 0.8 + 0.006
            parts.append(_orient_out(tube(bp, np.linspace(br, br * 0.25, len(bp)), sides=4, mat=BARK, v_scale=0.6), bp))
            n_spray += _spruce_foliage(parts, bp, L, ztop, young, rng)
        # intermediate (smaller) branches between the whorls
        for k in range(2 if not young else 1):
            a = base_a + rng.uniform(0, 2 * math.pi)
            zm = zi + whorl * rng.uniform(0.3, 0.7)
            if zm > H - 0.6:
                continue
            out = np.array([math.cos(a), math.sin(a), 0.0])
            L = Rz * rng.uniform(0.45, 0.7)
            dirs = [out + [0, 0, -0.3], out + [0, 0, -0.35], out + [0, 0, 0.1]]
            dirs = [np.asarray(dd) / np.linalg.norm(dd) for dd in dirs]
            c1 = np.interp(zm, z, sway[:, 0]), np.interp(zm, z, sway[:, 1])
            bp = _curve((c1[0], c1[1], zm), dirs, [L * 0.4, L * 0.35, L * 0.25])
            parts.append(_orient_out(tube(bp, np.linspace(0.02, 0.006, len(bp)), sides=3, mat=BARK, v_scale=0.6), bp))
            n_spray += _spruce_foliage(parts, bp, L, ztop, young, rng, sparse=True)
    # leader (top shoot) with a few upright sprays
    top = np.array([sway[-1, 0], sway[-1, 1], H])
    for k in range(6):
        a = 2 * math.pi * k / 6
        d = np.array([math.cos(a) * 0.4, math.sin(a) * 0.4, 1.0])
        parts.append(strip(top - [0, 0, 0.8], d, [math.cos(a), math.sin(a), 0.2], 0.9, 0.16, segs=6, tint=rng.random(), sun=1.0, curl=0.3))
    m = mk.merge(parts)
    _defaults(m)
    return m


def _spruce_foliage(parts, bp, L, ztop, young, rng, sparse=False):
    """Needle sprays along a branch polyline: pendulous curtains either side, a flat spray on top, side branchlets."""
    n = 0
    step = 0.2 if not sparse else 0.3
    n_s = max(3, int(L / step))
    segs = len(bp) - 1
    for j in range(n_s):
        u = 0.1 + 0.9 * j / n_s
        seg_i = min(int(u * segs), segs - 1)
        f = u * segs - seg_i
        p = bp[seg_i] * (1 - f) + bp[seg_i + 1] * f
        dseg = bp[seg_i + 1] - bp[seg_i]
        dseg /= np.linalg.norm(dseg)
        side = np.cross(dseg, [0, 0, 1.0])
        side /= np.linalg.norm(side) + 1e-12
        sun = float(np.clip(0.3 + 0.7 * u * (0.55 + 0.45 * ztop), 0, 1))
        for sgn in (-1.0, 1.0):
            hang = side * sgn * rng.uniform(0.35, 0.8) + np.array([0, 0, -1.0]) * (0.8 if not young else 0.5) + dseg * 0.3
            hang /= np.linalg.norm(hang)
            ls = rng.uniform(0.45, 0.85) * (0.7 + 0.4 * (1 - u)) * (1.0 if not young else 0.75)
            parts.append(strip(p, hang, side * sgn + [0, 0, 0.35], ls, rng.uniform(0.16, 0.24), segs=4, tint=rng.random(), sun=sun,
                               droop=0.1, curl=0.55))
            n += 1
        fwd = dseg + side * rng.normal(0, 0.45) + [0, 0, 0.12]
        parts.append(strip(p, fwd, [0, 0, 1.0], rng.uniform(0.4, 0.65), rng.uniform(0.2, 0.28), segs=4, tint=rng.random(), sun=sun,
                           droop=-0.04, curl=0.35))
        n += 1
        if not sparse and j % 3 == 1 and u < 0.85:
            # side branchlet with its own sprays
            sd = side * rng.choice([-1.0, 1.0])
            bdir = sd * 0.8 + dseg * 0.5 + [0, 0, -0.25]
            bdir /= np.linalg.norm(bdir)
            bl = L * (1 - u) * rng.uniform(0.35, 0.55) + 0.2
            for q in range(3):
                pq = p + bdir * bl * (q + 0.5) / 3
                hang = bdir * 0.3 + [0, 0, -0.9]
                parts.append(strip(pq, hang, bdir + [0, 0, 0.4], rng.uniform(0.35, 0.6), rng.uniform(0.15, 0.22), segs=4, tint=rng.random(),
                                   sun=sun * 0.9, droop=0.08, curl=0.5))
                n += 1
    return n


def scots_pine(seed, H=21.0):
    rng = np.random.default_rng(seed)
    parts = []
    nz = 30
    z = np.linspace(0, H * 0.82, nz)
    bend = np.stack([0.9 * np.sin(z / H * 2.0 + 1.0) * z / H, 0.5 * np.sin(z / H * 1.3) * z / H, z], 1)
    rad = 0.30 * (1 - z / (H * 0.95)) ** 0.8 + 0.03
    tr = tube(bend, rad, sides=9, mat=BARK, v_scale=1.0)
    # upper trunk = orange 'fox' bark
    C = tr.V[tr.F].mean(1)
    tr.mat[C[:, 2] > H * 0.45] = DEADWOOD + 2                   # 45 = pine upper bark
    parts.append(_orient_out(tr, bend))
    # umbrella crown: several big limbs from 60 % of the height, needle tufts at their ends
    for k in range(9):
        a = 2 * math.pi * k / 9 + rng.normal(0, 0.3)
        zb = H * rng.uniform(0.55, 0.8)
        cb = np.array([np.interp(zb, z, bend[:, 0]), np.interp(zb, z, bend[:, 1]), zb])
        out = np.array([math.cos(a), math.sin(a), 0.0])
        L = rng.uniform(2.5, 4.8)
        pts = _curve(cb, [out + [0, 0, 0.45], out + [0, 0, 0.25], out + [0, 0, 0.1]], [L * 0.4, L * 0.35, L * 0.25])
        parts.append(_orient_out(tube(pts, np.linspace(0.11, 0.03, len(pts)), sides=5, mat=DEADWOOD + 2), pts))
        for j in range(12):
            u = rng.uniform(0.35, 1.0)
            i = min(int(u * (len(pts) - 1)), len(pts) - 2)
            p = pts[i] + (pts[i + 1] - pts[i]) * (u * (len(pts) - 1) - i) + rng.normal(0, 0.35, 3)
            for q in range(7):
                d = rng.normal(0, 1, 3) + [0, 0, 0.9]
                parts.append(strip(p, d, rng.normal(0, 1, 3) + [0, 0, 1.0], rng.uniform(0.35, 0.55), 0.13, segs=5, tint=rng.random(),
                                   sun=0.8, curl=0.4))
    m = mk.merge(parts)
    _defaults(m)
    return m


def snag(seed, H=22.0):
    rng = np.random.default_rng(seed)
    z = np.linspace(0, H, 24)
    sway = np.stack([0.3 * np.sin(z / H * 2.7) * z / H, 0.2 * np.cos(z / H * 1.9) * z / H, z], 1)
    rad = 0.28 * (1 - z / H) ** 0.8 + 0.02
    parts = [_orient_out(tube(sway, rad, sides=8, mat=DEADWOOD, v_scale=1.0), sway)]
    for zi in np.arange(H * 0.15, H * 0.9, 0.7):
        for k in range(int(rng.integers(1, 4))):
            if rng.random() < 0.4:
                continue
            a = rng.uniform(0, 2 * math.pi)
            out = np.array([math.cos(a), math.sin(a), -0.25])
            L = rng.uniform(0.4, 2.6) * (1 - zi / H)
            pts = _curve((np.interp(zi, z, sway[:, 0]), np.interp(zi, z, sway[:, 1]), zi), [out, out + [0, 0, -0.2]], [L * 0.6, L * 0.4])
            parts.append(_orient_out(tube(pts, [0.04, 0.025, 0.008], sides=4, mat=DEADWOOD), pts))
    m = mk.merge(parts)
    _defaults(m)
    return m


def birch(seed, H=14.0):
    """Birch / rowan: slender pale trunk, ascending branches, twigs carrying small leaves (4-tri quads with a midrib)."""
    rng = np.random.default_rng(seed)
    parts = []
    z = np.linspace(0, H * 0.9, 22)
    lean = np.stack([0.5 * np.sin(z / H * 1.7) * z / H + 0.4 * z / H, 0.3 * np.sin(z / H * 2.3) * z / H, z], 1)
    rad = 0.16 * (1 - z / H) ** 0.85 + 0.02
    parts.append(_orient_out(tube(lean, rad, sides=8, mat=BIRCHBARK, v_scale=1.0), lean))
    n_leaf = 0
    for zi in np.arange(H * 0.3, H * 0.92, 0.55):
        for k in range(2):
            a = rng.uniform(0, 2 * math.pi)
            out = np.array([math.cos(a), math.sin(a), 0.0])
            cb = np.array([np.interp(zi, z, lean[:, 0]), np.interp(zi, z, lean[:, 1]), zi])
            L = rng.uniform(1.5, 3.4) * (1.1 - (zi / H) ** 2)
            pts = _curve(cb, [out + [0, 0, 0.9], out + [0, 0, 0.5], out + [0, 0, 0.1]], [L * 0.35, L * 0.35, L * 0.3])
            parts.append(_orient_out(tube(pts, np.linspace(0.05, 0.012, len(pts)), sides=4, mat=BIRCHBARK), pts))
            # twigs with leaves
            for j in range(16):
                u = rng.uniform(0.3, 1.0)
                i = min(int(u * (len(pts) - 1)), len(pts) - 2)
                p = pts[i] + (pts[i + 1] - pts[i]) * (u * (len(pts) - 1) - i)
                twig_d = out + rng.normal(0, 0.6, 3) + [0, 0, -0.3]
                twig_d /= np.linalg.norm(twig_d)
                tp = _curve(p, [twig_d, twig_d + [0, 0, -0.4]], [0.35, 0.3])
                parts.append(_orient_out(tube(tp, [0.01, 0.006, 0.003], sides=3, mat=BIRCHBARK), tp))
                for q in range(16):
                    lp = tp[0] + (tp[-1] - tp[0]) * rng.uniform(0.1, 1.0) + rng.normal(0, 0.12, 3)
                    ld = rng.normal(0, 1, 3) + [0, 0, -0.6]
                    parts.append(strip(lp, ld, rng.normal(0, 1, 3) + [0, 0, 1.0], rng.uniform(0.08, 0.12), rng.uniform(0.045, 0.065), segs=2,
                                       serr=0.15, mat=LEAVES, tint=rng.random(), sun=0.6 + 0.4 * u, curl=0.6))
                    n_leaf += 1
    m = mk.merge(parts)
    _defaults(m)
    return m


def willow(seed=5):
    """The old willow of the grounds: a short gnarled bole, writhing limbs, long hanging leafy whips."""
    rng = np.random.default_rng(seed)
    parts = []
    z = np.linspace(0, 6.0, 14)
    bole = np.stack([0.6 * np.sin(z * 0.7), 0.4 * np.cos(z * 0.9) - 0.4, z], 1)
    parts.append(_orient_out(tube(bole, np.linspace(1.3, 0.9, len(z)), sides=14, mat=BARK, v_scale=1.5), bole))
    for k in range(9):
        a = 2 * math.pi * k / 9 + rng.normal(0, 0.2)
        out = np.array([math.cos(a), math.sin(a), 0.0])
        pts = [bole[-1] + rng.normal(0, 0.3, 3)]
        d = out + [0, 0, 0.9]
        for s in range(7):
            d = d / np.linalg.norm(d) + rng.normal(0, 0.35, 3) * 0.6 + out * 0.15
            pts.append(pts[-1] + d / np.linalg.norm(d) * rng.uniform(1.0, 1.6))
        pts = np.array(pts)
        parts.append(_orient_out(tube(pts, np.linspace(0.5, 0.06, len(pts)), sides=7, mat=BARK, v_scale=1.0), pts))
        for j in range(16):
            u = rng.uniform(0.4, 1.0)
            i = min(int(u * (len(pts) - 1)), len(pts) - 2)
            p = pts[i] + (pts[i + 1] - pts[i]) * (u * (len(pts) - 1) - i)
            whip_len = rng.uniform(3.0, 7.5)
            wp = _curve(p, [out * 0.4 + [0, 0, 0.3], [0, 0, -1.0], [0, 0, -1.0]], [0.5, whip_len * 0.5, whip_len * 0.5])
            parts.append(_orient_out(tube(wp, [0.03, 0.02, 0.01, 0.004], sides=3, mat=BARK), wp))
            for q in range(22):
                lp = wp[1] + (wp[-1] - wp[1]) * rng.uniform(0, 1)
                parts.append(strip(lp + rng.normal(0, 0.05, 3), rng.normal(0, 0.4, 3) + [0, 0, -1.0], rng.normal(0, 1, 3), rng.uniform(0.1, 0.16), 0.03,
                                   segs=2, serr=0.1, mat=LEAVES, tint=rng.random() * 0.6, sun=0.7, curl=0.4))
    m = mk.merge(parts)
    _defaults(m)
    return m


# ----------------------------------------------------------------------------------------------- boulders
def boulder(seed, size=1.0):
    """Granite boulder: a block with planar facets (random plane cuts), rounded by weathering, then fine noise."""
    import manifold3d as m3d

    rng = np.random.default_rng(seed)
    b = m3d.Manifold.sphere(1.0, 48).scale([1.0, rng.uniform(0.7, 0.95), rng.uniform(0.45, 0.75)])
    for k in range(int(rng.integers(5, 9))):
        n = rng.normal(0, 1, 3)
        n[2] = abs(n[2]) * 0.6 + (0.3 if k == 0 else 0.0)
        n /= np.linalg.norm(n)
        b = b.trim_by_plane(list(-n), -float(rng.uniform(0.55, 0.85)))
    b = b.refine_to_length(0.06)
    mm = b.to_mesh()
    V = np.asarray(mm.vert_properties, np.float64)[:, :3]
    F = np.asarray(mm.tri_verts, np.int64)
    m = Mesh(V, F, ROCK).weld(1e-6)
    # round the sharp edges a little (Laplacian) and add grain
    import core

    m.V = core.taubin(m.V, m.F, iters=4, lam=0.45, mu=-0.48)
    nrm = m.vertex_normals()
    m.V = m.V + nrm * (0.035 * mk.fbm(m.V * 3.0 + seed, 4, seed=seed) + 0.012 * mk.fbm(m.V * 11.0, 2, seed=seed + 1))[:, None]
    m.V[:, 2] -= m.V[:, 2].min() + 0.15 * (m.V[:, 2].max() - m.V[:, 2].min())       # sink the base
    m.V *= size
    if m.volume() < 0:
        m.flip()
    m.uv_box(2.0, only_missing=False)
    up = m.face_normals()[:, 2]
    m.uv1 = np.stack([np.ones_like(up), np.clip(up * 1.2 - 0.2, 0, 1)], 1)[:, None, :].repeat(3, 1)       # rock 1, moss on top
    m.uv2 = np.zeros((m.nf, 3, 2))
    m.uv3 = np.stack([np.zeros_like(up), np.full_like(up, 0.9)], 1)[:, None, :].repeat(3, 1)
    return m


def _defaults(m):
    if m.uv0 is None:
        m.uv0 = np.zeros((m.nf, 3, 2))
    m.uv0 = np.nan_to_num(m.uv0)
    if m.uv1 is None:
        m.uv1 = np.full((m.nf, 3, 2), 0.5)
    m.uv1 = np.where(np.isnan(m.uv1), 0.5, m.uv1)
    m.uv2 = np.zeros((m.nf, 3, 2))
    return m


def build_library(lib_dir, quick=False):
    os.makedirs(lib_dir, exist_ok=True)
    t0 = time.time()
    out = []

    def save(name, m):
        m.save(f"{lib_dir}/{name}.npz")
        out.append((name, m.nf))
        print(f"  [{time.time() - t0:5.0f}s] {name:22s} {m.nf:8,d} tris  h={m.V[:, 2].max():.1f} m", flush=True)

    for i, (H, cr) in enumerate(((28.0, 0.19), (26.0, 0.22), (30.0, 0.17), (24.0, 0.21))):
        save(f"lib_spruce_{i:02d}", spruce(100 + i, H, cr))
    for i in range(2):
        save(f"lib_spruceyoung_{i:02d}", spruce(200 + i, 7.0 + i, 0.30, young=True))
    save("lib_pine_00", scots_pine(300))
    save("lib_snag_00", snag(400))
    for i in range(2):
        save(f"lib_birch_{i:02d}", birch(500 + i, 13.0 + 2 * i))
    save("lib_willow_00", willow())
    for i in range(8):
        save(f"lib_boulder_{i:02d}", boulder(600 + i, 1.0))
    return out


if __name__ == "__main__":
    build_library(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo/lib")
