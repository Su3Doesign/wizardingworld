"""grounds - everything round the castle: the Quidditch stadium, the gamekeeper's hut, the greenhouses, the stone circle,
the gates, the station, Hogsmeade, the boats with their lanterns, the tomb on the island, the props of props.py (the
flying lawn, lamps, braziers, the courts' fountain and statues); and the water surfaces (the Black Lake, the river in
the gorge, the ravine stream and the waterfall).

Materials (besides the castle_kit ones): CLOTH (the stadium's checkered cloth, banners, pennants; palette entry from
UV2.x = (k + 0.5) / 8: red, gold, green, silver, blue, bronze, yellow, black), MARBLE (the pitch markings, the tomb),
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


# ----------------------------------------------------------------------------------------------- small mesh helpers
UPV = np.array([0.0, 0.0, 1.0])
_BOX_F = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6],
                   [1, 3, 5], [3, 7, 5]])
# M_Cloth palette (UV2.x = (k + 0.5) / 8): the four houses' colour pairs
RED, GOLD, GREEN, SILVER, BLUE, BRONZE, YELLOW, BLACK = range(8)
HOUSES = ((RED, GOLD), (GREEN, SILVER), (BLUE, BRONZE), (YELLOW, BLACK))


def colour(m, k):
    """Set a cloth mesh's palette entry (M_Cloth reads UV2.x)."""
    m.uv2 = np.zeros((m.nf, 3, 2))
    m.uv2[:, :, 0] = (k + 0.5) / 8.0
    m.uv2[:, :, 1] = 1.0
    return m


def _box8(corners, mat):
    """Box from its 8 corners, ordered (z, y, x) bit-wise: index = 4 z + 2 y + x."""
    m = Mesh(np.asarray(corners, np.float64), _BOX_F, mat)
    if m.volume() < 0:
        m.flip()
    return m


def beam(pa, pb, w, d, mat=WOOD):
    """A timber of section w x d between two 3D points (w horizontal, d in the vertical plane through the timber)."""
    pa, pb = np.asarray(pa, np.float64), np.asarray(pb, np.float64)
    e = pb - pa
    L = float(np.linalg.norm(e))
    e = e / L
    h = np.cross(UPV, e) if abs(e[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    h = h / np.linalg.norm(h)
    v = np.cross(e, h)
    return _box8([pa + e * x + h * y + v * z for z in (-d / 2, d / 2) for y in (-w / 2, w / 2) for x in (0.0, L)], mat)


def prism(q, za, zb, mat):
    """Upright prism on a convex plan quad q = (q00, q10, q01, q11) (xy), from za to zb."""
    q = [np.array([p[0], p[1]], np.float64) for p in q]
    return _box8([[p[0], p[1], z] for z in (za, zb) for p in q], mat)


def quad(p0, p1, p2, p3, mat, k=None):
    """A single (two-sided material) quad p0 p1 p2 p3."""
    m = Mesh(np.array([p0, p1, p2, p3], np.float64), np.array([[0, 1, 2], [0, 2, 3]]), mat)
    m.uv_box(1.0, only_missing=False)
    return colour(m, k) if k is not None else m


def checker(o, a, b, na, nb, ka, kb, mat=CLOTH):
    """A checkered cloth panel: origin o, edge vectors a (na cells) and b (nb cells), colours ka / kb."""
    o, a, b = (np.asarray(x, np.float64) for x in (o, a, b))
    out = []
    for i in range(na):
        for j in range(nb):
            p = o + a * (i / na) + b * (j / nb)
            out.append(quad(p, p + a / na, p + a / na + b / nb, p + b / nb, mat, ka if (i + j) % 2 == 0 else kb))
    return mk.merge(out)


def torus(c, axis, R, r, seg=40, tube=10, mat=LEAD):
    """A ring of radius R (tube radius r) round `axis` through c."""
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    e1 = np.cross(axis, UPV) if abs(axis[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    th = np.linspace(0, 2 * math.pi, seg + 1)
    ph = np.linspace(0, 2 * math.pi, tube + 1)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    radial = np.cos(T)[..., None] * e1 + np.sin(T)[..., None] * e2
    G = np.asarray(c)[None, None] + radial * (R + r * np.cos(Ph))[..., None] + axis[None, None] * (r * np.sin(Ph))[..., None]
    m = mk.grid(G, mat=mat, uv_tile=1.0)
    if m.volume() < 0:
        m.flip()
    return m


def flat_ribbon(P, width, z, closed, mat=MARBLE):
    """A painted line: a flat strip of `width` along the polyline P (xy), facing up, at height z."""
    P = np.asarray(P, np.float64)
    if closed:
        P = np.vstack([P, P[:1]])
    t = np.gradient(P, axis=0)
    if closed:
        t[0] = t[-1] = P[1] - P[-2]
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    G = np.empty((len(P), 2, 3))
    for k, sgn in enumerate((-0.5, 0.5)):
        G[:, k, :2] = P + nrm * width * sgn
        G[:, k, 2] = z
    m = mk.grid(G, mat=mat, uv_tile=1.0)
    if np.mean(m.face_normals()[:, 2]) < 0:
        m.flip()
    return m


def oval(a, b, step):
    """Points along the ellipse (a across, b along) equally spaced in arc length (~step): u, v, unit tangent (tu, tv),
    outward normal (nu, nv), arc position s, perimeter."""
    t = np.linspace(0, 2 * math.pi, 6001)
    U, V = a * np.cos(t), b * np.sin(t)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(U), np.diff(V)))])
    n = max(8, int(round(s[-1] / step)))
    ss = np.arange(n) * s[-1] / n
    tt = np.interp(ss, s, t)
    tu, tv = -a * np.sin(tt), b * np.cos(tt)
    L = np.hypot(tu, tv)
    tu, tv = tu / L, tv / L
    return a * np.cos(tt), b * np.sin(tt), tu, tv, tv, -tu, ss, float(s[-1])


# ----------------------------------------------------------------------------------------------- Quidditch stadium
def quidditch_tower(c, tan, out, z0, H, house, rng, wind):
    """A spectators' tower: four battered timber posts with ring beams and X-braces in every bay, a box at the top whose
    parapet and skirt are wrapped in the house's checkered cloth, an open viewing band, a canvas pyramid roof with a
    checkered valance, and a long two-coloured streamer on a pole at the apex."""
    ka, kb = HOUSES[house]
    c = np.asarray(c, np.float64)
    tan = np.asarray(tan, np.float64)
    out = np.asarray(out, np.float64)
    hs0, hs1 = 2.6, 2.2
    zt = z0 + H                                              # the box floor
    ze = zt + 3.3                                            # the roof's eave

    def hs(z):
        return hs0 + (hs1 - hs0) * np.clip((z - z0) / H, 0.0, 1.0)

    def corner(a, b, z, extra=0.0):
        h = hs(z) + extra
        return np.array([c[0] + tan[0] * a * h + out[0] * b * h, c[1] + tan[1] * a * h + out[1] * b * h, z])

    parts = []
    cs = ((-1, -1), (1, -1), (1, 1), (-1, 1))                 # round the box: inner face (b = -1) first
    for a, b in cs:
        parts.append(beam(corner(a, b, z0 - 0.3), corner(a, b, zt), 0.48, 0.48))
        parts.append(beam(corner(a, b, zt), corner(a, b, ze), 0.36, 0.36))
        p = corner(a, b, z0)
        parts.append(_bx((p[0], p[1], z0 + 0.1), (1.1, 1.1, 0.8), math.atan2(tan[1], tan[0]), TRIM))     # stone pad
    nb = max(4, int(round(H / 4.6)))
    skirt0 = zt - 5.6                                         # the cloth hangs down to here
    levels = [z0 + H * k / nb for k in range(nb + 1)]
    for k, z in enumerate(levels[1:], 1):
        for i in range(4):
            (a0, b0), (a1, b1) = cs[i], cs[(i + 1) % 4]
            parts.append(beam(corner(a0, b0, z), corner(a1, b1, z), 0.3, 0.3))
    for k in range(nb):
        za, zb = levels[k], levels[k + 1]
        if za >= skirt0 - 0.5:
            break
        for i in range(4):
            (a0, b0), (a1, b1) = cs[i], cs[(i + 1) % 4]
            parts.append(beam(corner(a0, b0, za), corner(a1, b1, zb), 0.22, 0.22))
            parts.append(beam(corner(a1, b1, za), corner(a0, b0, zb), 0.22, 0.22))
    # the box: floor, checkered skirt + parapet, a rail, the valance under the eave
    zp = zt + 1.25
    parts.append(prism([corner(-1, -1, zt, 0.35)[:2], corner(1, -1, zt, 0.35)[:2], corner(-1, 1, zt, 0.35)[:2], corner(1, 1, zt, 0.35)[:2]],
                       zt - 0.2, zt + 0.1, WOOD))
    for i in range(4):
        (a0, b0), (a1, b1) = cs[i], cs[(i + 1) % 4]
        p0, p1 = corner(a0, b0, skirt0, 0.32), corner(a1, b1, skirt0, 0.32)
        q0, q1 = corner(a0, b0, zp, 0.32), corner(a1, b1, zp, 0.32)
        # the skirt: a ruled panel between the (battered) bottom edge and the top edge, in checks
        rows, cols = 5, 4
        for r_ in range(rows):
            for c_ in range(cols):
                f0, f1 = c_ / cols, (c_ + 1) / cols
                g0, g1 = r_ / rows, (r_ + 1) / rows
                lo0, lo1 = p0 + (p1 - p0) * f0, p0 + (p1 - p0) * f1
                hi0, hi1 = q0 + (q1 - q0) * f0, q0 + (q1 - q0) * f1
                parts.append(quad(lo0 + (hi0 - lo0) * g0, lo1 + (hi1 - lo1) * g0, lo1 + (hi1 - lo1) * g1, lo0 + (hi0 - lo0) * g1, CLOTH,
                                  ka if (r_ + c_) % 2 == 0 else kb))
        parts.append(beam(corner(a0, b0, zp + 0.08, 0.32), corner(a1, b1, zp + 0.08, 0.32), 0.16, 0.16))
        v0, v1 = corner(a0, b0, ze - 0.7, 0.42), corner(a1, b1, ze - 0.7, 0.42)
        parts.append(checker(v0, v1 - v0, np.array([0.0, 0.0, 0.7]), 8, 1, kb, ka))
    # canvas pyramid roof
    rh = float(rng.uniform(5.0, 7.0))
    apex = np.array([c[0], c[1], ze + rh])
    eaves = [corner(a, b, ze, 0.75) for a, b in cs]
    for i in range(4):
        e0, e1 = eaves[i], eaves[(i + 1) % 4]
        tri = Mesh(np.array([e0, e1, apex]), np.array([[0, 1, 2]]), CLOTH)
        if np.cross(e1 - e0, apex - e0) @ (((e0 + e1) / 2 - np.array([c[0], c[1], ze]))) < 0:
            tri.flip()
        tri.uv_box(1.0, only_missing=False)
        parts.append(colour(tri, ka))
    # the streamer: two bands of the house's colours, tapering and waving downwind from a pole on the apex
    parts.append(beam(apex - np.array([0, 0, 0.6]), apex + np.array([0, 0, 3.8]), 0.12, 0.12, LEAD))
    w = np.array([wind[0], wind[1], 0.0])
    side = np.cross(UPV, w)
    top = apex + np.array([0, 0, 3.7])
    ph = rng.uniform(0, 6)
    us = np.linspace(0, 1, 15)
    for band, (v0, v1, kk) in enumerate(((-1.0, 0.0, kb), (0.0, 1.0, ka))):
        U, Vb = np.meshgrid(us, np.linspace(v0, v1, 2), indexing="ij")
        half = 0.5 * (1 - 0.72 * U)
        G = top[None, None] + w[None, None] * (U * 7.5)[..., None] + UPV[None, None] * ((Vb - 1.0) * half - 0.35 * U)[..., None] \
            + side[None, None] * (0.45 * np.sin(U * 7.0 + ph) * U)[..., None]
        parts.append(colour(mk.grid(G, mat=CLOTH, uv_tile=1.0), kk))
    return mk.merge(parts)


def scoring_arc(ap, bp):
    """The scoring area's line across the +v end of the pitch (Quidditch Through the Ages): an arc bulging towards the
    centre, v = v0 + k u^2, from boundary to boundary.  Returns (u, v) points."""
    v0, k = bp - 28.0, 0.0554
    uu = np.linspace(0.0, ap, 4001)
    vb = bp * np.sqrt(np.clip(1.0 - (uu / ap) ** 2, 0.0, 1.0))
    u1 = float(uu[np.argmax(v0 + k * uu ** 2 >= vb)])
    u = np.linspace(-u1, u1, 41)
    return u, v0 + k * u ** 2


def stadium(rng):
    """The Quidditch stadium beside the castle, after the film's timber towers, the studio model and the diagram in
    Quidditch Through the Ages: an oval pitch with white markings (the boundary, the halfway line, the centre circle, the
    scoring areas' arcs), sandy patches worn round the goal hoops (pitch_sand), three hoops of different heights at each
    end; a ring of timber stands whose front facing the pitch is a band of checkered house cloth under a rail; a low
    crenellated stone wall round it all, hung outside with long house banners; and sixteen tall X-braced timber towers
    whose spectator boxes are wrapped in checkered house colours, under canvas pyramid roofs with streamers."""
    cx, cy = W.PITCH
    yaw = math.radians(W.PITCH_YAW)
    ex = np.array([math.cos(yaw), -math.sin(yaw), 0.0])      # across the pitch
    ey = np.array([math.sin(yaw), math.cos(yaw), 0.0])       # along it
    z0 = g(cx, cy)
    C = np.array([cx, cy, z0])

    def P(u, v, z=0.0):
        return C + ex * u + ey * v + UPV * z

    def xy(u, v):
        return (C + ex * u + ey * v)[:2]

    aw, bw = W.PITCH_SIZE[0] / 2, W.PITCH_SIZE[1] / 2        # the stands' front (their inner oval)
    ap, bp = aw - 6.0, bw - 8.0                              # the pitch's boundary line
    parts = []
    # --- markings
    zl = z0 + 0.05
    u, v, *_ = oval(ap, bp, 1.0)
    parts.append(flat_ribbon([xy(a, b) for a, b in zip(u, v)], 0.35, zl, True))
    parts.append(flat_ribbon([xy(a, 0.0) for a in np.linspace(-ap, ap, 30)], 0.35, zl, False))
    u, v, *_ = oval(7.0, 7.0, 0.8)
    parts.append(flat_ribbon([xy(a, b) for a, b in zip(u, v)], 0.35, zl, True))
    parts.append(flat_ribbon([xy(a, b) for a, b in zip(*oval(0.6, 0.6, 0.3)[:2])], 0.5, zl, True))     # where the balls are released
    au, av = scoring_arc(ap, bp)
    for end in (-1, 1):
        parts.append(flat_ribbon([xy(a, end * b) for a, b in zip(au, av)], 0.35, zl, False))
    # --- goal hoops: three at each end, the middle one the highest
    for end in (-1, 1):
        for du, hh in ((-8.0, 14.5), (0.0, 17.5), (8.0, 14.5)):
            b = P(du, end * (bp - 13.0))
            R = 1.75
            pole = mk.revolve([(0.001, -0.3), (0.95, -0.3), (0.95, 0.35), (0.55, 0.55), (0.26, 0.95), (0.22, 1.4), (0.13, hh - R - 0.1),
                               (0.17, hh - R + 0.05), (0.001, hh - R + 0.05)], seg=20, mat=LEAD)
            pole.translate(b)
            pole.uv_box(1.0, only_missing=False)
            parts.append(pole)
            parts.append(torus(b + UPV * hh, ey, R, 0.13))
    # --- the towers' places round the stands, the houses in quarters
    u, v, tu, tv, nu, nv, s, perim = oval(aw, bw, 2.5)
    n = len(u)
    n_t = 16
    s_t = (np.arange(n_t) + 0.5) * perim / n_t
    house_of = [(k // 4) % 4 for k in range(n_t)]

    def off(k_, o):
        return xy(u[k_] + nu[k_] * o, v[k_] + nv[k_] * o)

    def near_tower(sm):
        d = np.abs((sm - s_t + perim / 2) % perim - perim / 2)
        return float(d.min()), int(np.argmin(d))
    # --- the stands: a boarded front with a band of checkered house cloth facing the pitch under a capping rail, three
    # timber tiers behind it, and the low crenellated stone wall round the outside
    for i in range(n):
        j = (i + 1) % n
        smid = (s[i] + (s[j] if j else perim)) / 2
        dt, k_near = near_tower(smid)
        ka, kb = HOUSES[house_of[k_near]]
        parts.append(prism([off(i, 0.0), off(j, 0.0), off(i, 0.25), off(j, 0.25)], z0 - 0.3, z0 + 2.0, WOOD))
        a0, a1 = off(i, -0.03), off(j, -0.03)
        parts.append(checker([a0[0], a0[1], z0 + 0.55], [a1[0] - a0[0], a1[1] - a0[1], 0.0], [0.0, 0.0, 1.35], 4, 2, ka, kb))
        r0, r1 = off(i, 0.12), off(j, 0.12)
        parts.append(beam([r0[0], r0[1], z0 + 2.08], [r1[0], r1[1], z0 + 2.08], 0.36, 0.16))
        if dt >= 3.5:                                     # the tiers stop round the towers
            for t_ in range(3):
                oa, ob = 0.25 + 2.1 * t_, 0.25 + 2.1 * (t_ + 1)
                parts.append(prism([off(i, oa), off(j, oa), off(i, ob), off(j, ob)], z0 - 0.3, z0 + 1.15 + 0.9 * t_, WOOD))
        q = [off(i, 6.55), off(j, 6.55), off(i, 7.55), off(j, 7.55)]
        parts.append(prism(q, z0 - 0.5, z0 + 5.4, STONE))
        parts.append(prism([off(i, 6.4), off(j, 6.4), off(i, 7.7), off(j, 7.7)], z0 + 5.4, z0 + 5.65, TRIM))
        if i % 2 == 0:
            m0, m1 = (off(i, 6.55) + off(j, 6.55)) / 2, (off(i, 7.55) + off(j, 7.55)) / 2
            tt = (off(j, 7.05) - off(i, 7.05))
            tt = tt / (np.linalg.norm(tt) + 1e-9) * 0.6
            parts.append(prism([m0 - tt, m0 + tt, m1 - tt, m1 + tt], z0 + 5.65, z0 + 6.55, STONE))
    wind = np.array([math.cos(math.radians(25.0)), math.sin(math.radians(25.0))])
    for k in range(n_t):
        i = int(np.argmin(np.abs(s - s_t[k])))
        tw = np.interp(s_t[k], np.append(s, perim), np.append(u, u[0]))
        th = np.interp(s_t[k], np.append(s, perim), np.append(v, v[0]))
        tn = np.array([nu[i], nv[i]])
        c = xy(tw + tn[0] * 3.9, th + tn[1] * 3.9)
        out = (ex * tn[0] + ey * tn[1])[:2]
        tan = np.array([-out[1], out[0]])
        H = float(np.clip(25.0 + 6.0 * (0.5 + 0.5 * math.sin(k * 2.4 + 0.7)) + rng.normal(0.0, 1.2), 23.0, 33.0))
        parts.append(quidditch_tower(c, tan, out, z0, H, house_of[k], rng, wind))
        # a long house banner on the outer wall, between this tower and the next
        ka, kb = HOUSES[house_of[k]]
        s_b = (s_t[k] + perim / n_t / 2) % perim
        ib = int(np.argmin(np.abs(s - s_b)))
        nb_ = np.array([nu[ib], nv[ib]])
        bc = xy(u[ib] + nb_[0] * 7.6, v[ib] + nb_[1] * 7.6)
        bo = (ex * nb_[0] + ey * nb_[1])[:2]
        bt = np.array([-bo[1], bo[0], 0.0])
        o_ = np.array([bc[0], bc[1], z0]) - bt * 1.1
        for za, zb, kk in ((0.9, 1.7, kb), (1.7, 4.5, ka), (4.5, 5.3, kb)):
            parts.append(quad(o_ + UPV * za, o_ + bt * 2.2 + UPV * za, o_ + bt * 2.2 + UPV * zb, o_ + UPV * zb, CLOTH, kk))
    return mk.merge(parts)


def pitch_sand():
    """The sandy patches worn round the goal hoops, as a skin over the turf in the terrain's own material (M_Terrain with
    its dirt mask in UV3.x fading to grass at the edges, so it blends into the terrain)."""
    cx, cy = W.PITCH
    yaw = math.radians(W.PITCH_YAW)
    ex = np.array([math.cos(yaw), -math.sin(yaw)])
    ey = np.array([math.sin(yaw), math.cos(yaw)])
    bp = W.PITCH_SIZE[1] / 2 - 8.0
    z = g(cx, cy) + 0.03
    out = []
    for end in (-1, 1):
        vc = end * (bp - 12.5)
        us = np.arange(-16.0, 16.01, 0.5)
        vs = np.arange(-10.0, 10.01, 0.5)
        U, V = np.meshgrid(us, vs, indexing="ij")
        X = cx + U * ex[0] + (vc + V) * ey[0]
        Y = cy + U * ex[1] + (vc + V) * ey[1]
        r = np.hypot(U / 13.5, V / 8.0) + 0.09 * fn_noise(X, Y)
        dirt = np.clip((1.0 - r) / 0.28, 0.0, 1.0)
        dirt = dirt * dirt * (3 - 2 * dirt)
        G_ = np.stack([X, Y, np.full_like(X, z)], -1)
        m = mk.grid(G_, mat=0, uv_tile=1.0)
        d = dirt.reshape(-1)[m.F]
        keep = d.max(1) > 0.0
        m = m.select_faces(keep)
        d = d[keep]
        if np.mean(m.face_normals()[:, 2]) < 0:
            m.flip()
        m.uv1 = np.zeros((m.nf, 3, 2))
        m.uv2 = np.zeros((m.nf, 3, 2))
        m.uv3 = np.stack([d, np.ones_like(d)], -1)
        out.append(m)
    return mk.merge(out)


def fn_noise(x, y):
    import fastnoise as fn

    return fn.fbm2(np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel(), 4.0, 3, 311).reshape(np.shape(x))


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

    save("grounds_stadium", stadium(np.random.default_rng(seed + 2)))
    rng.random(44)              # advance the shared stream as the first stadium did: the hut, the greenhouses, the stones,
                                # the station and Hogsmeade keep their shapes
    sand = pitch_sand()                                   # terrain material, its own masks: no stone masks
    sand.save(f"{out_dir}/grounds_pitchsand.npz")
    out.append(("grounds_pitchsand", sand.nf))
    import props                                          # the flying lawn, lamps, braziers, the fountain, statues, odds and ends

    pm, lights = props.build(np.random.default_rng(seed + 1))
    save("grounds_props", pm)
    np.save(f"{out_dir}/grounds_lights.npy", lights)
    print(f"  grounds lights: {len(lights)} (braziers, lamp posts)", flush=True)
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
