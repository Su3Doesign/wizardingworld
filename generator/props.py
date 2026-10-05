"""props - the small things that make the grounds and the courts look lived in: the flying-lesson lawn beside the
Quidditch stadium (two rows of brooms laid on the grass, broom racks, the open ball trunk with the Quaffle, the Bludgers
and the Snitch, practice hoops, benches), lamp posts along the road from the castle to the gates, braziers at the
entrances, a tiered fountain with falling water in the Transfiguration court, statues of robed wizards, benches, a
sundial in the clock court, and barrels, crates and a woodpile by the greenhouses and the hut.

    props.build(rng) -> (mesh, lights)      called by grounds.py: grounds_props.npz + grounds_lights.npy (warm point
                                            lights at the braziers and the lamps, lit by the night presets)
"""
from __future__ import annotations

import math

import numpy as np

import grounds as G
import meshkit as mk
import world as W
from meshkit import Mesh

STONE, TRIM, WOOD, LEAD, CLOTH, MARBLE, LANTERN, WATER, FALL = (G.STONE, G.TRIM, G.WOOD, G.LEAD, G.CLOTH, G.MARBLE, G.LANTERN,
                                                                G.WATER, G.FALL)
UP = G.UPV


def ground(x, y):
    return float(W.ground(x, y)[0])


def at(m, pos, yaw=0.0):
    """Turn a prop built at the origin (x forward, z up) by yaw (rad) and set it down at pos."""
    m.rotate_z(yaw)
    m.translate(pos)
    return m


def solid(profile, seg, mat):
    """Closed surface of revolution about z (profile bottom to top, from the axis back to the axis)."""
    m = mk.revolve(profile, seg, mat=mat, uv_tile=1.0)
    if m.volume() < 0:
        m.flip()
    m.uv_box(1.0, only_missing=False)
    return m


def sphere(c, r, mat, seg=16, rings=8):
    t = np.linspace(-math.pi / 2, math.pi / 2, rings + 1)
    prof = [(max(r * math.cos(a), 0.001), r * math.sin(a)) for a in t]
    return solid(prof, seg, mat).translate(c)


def disc(c, r, z, seg, mat, a0=0.0):
    """A flat polygonal disc facing up (water surfaces)."""
    a = a0 + np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ring = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(seg, z)], 1)
    m = mk.fan_cap(ring, mat=mat)
    if np.mean(m.face_normals()[:, 2]) < 0:
        m.flip()
    m.uv0 = (m.V[:, :2] / 4.0)[m.F]
    return m


def tube(path, radii, seg=8, mat=WOOD):
    """A round timber along a 3D polyline, closed at both ends."""
    P = np.asarray(path, np.float64)
    r = np.broadcast_to(np.asarray(radii, np.float64), (len(P),))
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.linspace(0, 2 * math.pi, seg + 1)
    rows = [P[0]]
    for p, tt, rr in zip(P, t, r):
        e1 = np.cross(tt, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(tt, e1)
        rows.append(p + rr * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2))
    rows.append(P[-1])
    G_ = np.stack([np.broadcast_to(x, (seg + 1, 3)) if x.ndim == 1 else x for x in rows], 1)
    m = mk.grid(G_, mat=mat, uv_tile=0.5)
    if m.volume() < 0:
        m.flip()
    return m


# ----------------------------------------------------------------------------------------------- brooms and the lesson
def broom(rng, nose_bend=0.025):
    """A broomstick (2.05 m) along +x from its nose at the origin: a slightly bowed handle, an iron binding and a long
    twig brush flaring to a ragged end."""
    n = 9
    xs = np.linspace(0.0, 1.30, n)
    path = np.stack([xs, np.zeros(n), nose_bend * np.sin(xs / 1.30 * math.pi)], 1)
    parts = [tube(path, 0.026 - 0.009 * (1 - xs / 1.30) ** 2, seg=8, mat=WOOD)]
    band = solid([(0.001, 1.20), (0.045, 1.20), (0.048, 1.23), (0.048, 1.33), (0.045, 1.36), (0.001, 1.36)], 12, LEAD)
    brush = solid([(0.001, 1.25), (0.036, 1.25), (0.044, 1.40), (0.07, 1.58), (0.10, 1.78), (0.125, 1.95), (0.13, 2.02), (0.06, 2.05),
                   (0.001, 2.05)], 12, WOOD)
    k = brush.V[:, 2] > 1.4                               # ragged twigs: jitter the brush radially, more towards its end
    rad = np.hypot(brush.V[:, 0], brush.V[:, 1])
    j = 1.0 + np.where(k, rng.normal(0.0, 0.14, len(rad)) * np.clip((brush.V[:, 2] - 1.4) / 0.6, 0, 1), 0.0)
    brush.V[:, 0] *= j
    brush.V[:, 1] *= j
    for m in (band, brush):
        m.rotate_y(math.pi / 2)                          # the revolve axis (z) becomes the broom's axis (+x)
        parts.append(m)
    out = mk.merge(parts)
    out.uv_box(0.5, only_missing=False)
    return out


def broom_lying(rng, pos, yaw):
    """A broom laid on the grass: the brush rests on the ground, the nose too."""
    b = broom(rng)
    b.rotate_x(rng.uniform(0, 2 * math.pi))               # the bow lies any way round
    b.rotate_y(-math.atan2(0.105, 1.95))                    # the thick brush lifts the tail
    b.translate((0.0, 0.0, 0.03))
    return at(b, pos, yaw)


def broom_rack(rng, n=8):
    """A timber rack for n brooms standing brush-down against its top rail (local: the rack along y, open towards -x)."""
    L = 0.42 * n + 0.5
    parts = []
    for y in (-L / 2, L / 2):
        parts.append(G.beam((0.0, y, -0.1), (0.0, y, 1.55), 0.12, 0.12))
        parts.append(G.beam((-0.45, y, 0.05), (0.45, y, 0.05), 0.1, 0.1))
        parts.append(G.beam((0.0, y, 1.1), (0.4, y, 0.1), 0.08, 0.08))           # a brace behind
    parts.append(G.beam((0.0, -L / 2 - 0.06, 1.45), (0.0, L / 2 + 0.06, 1.45), 0.11, 0.11))
    parts.append(G.beam((-0.62, -L / 2, 0.25), (-0.62, L / 2, 0.25), 0.08, 0.08))        # the foot rail in front of the brushes
    parts.append(G.beam((-0.62, -L / 2, 0.25), (0.0, -L / 2, 0.25), 0.08, 0.08))
    parts.append(G.beam((-0.62, L / 2, 0.25), (0.0, L / 2, 0.25), 0.08, 0.08))
    for i in range(n):
        if rng.random() < 0.15:                           # a gap or two: someone has taken a broom
            continue
        b = broom(rng)
        b.rotate_x(rng.uniform(0, 2 * math.pi))
        b.rotate_y(math.pi / 2)                           # nose up, brush down
        b.translate((0.0, 0.0, 2.06))
        b.rotate_y(math.radians(rng.uniform(11.5, 13.5)))  # leaning on the top rail
        b.translate((-0.36, -L / 2 + 0.45 + 0.42 * i, 0.0))
        parts.append(b)
    return mk.merge(parts)


def ball_trunk(rng):
    """The Quidditch ball trunk, its lid open: the Quaffle, two Bludgers strapped in, the Snitch in its little bed."""
    Lx, Ly, H, t = 1.05, 0.58, 0.46, 0.04
    parts = [G.prism([(-Lx / 2, -Ly / 2), (Lx / 2, -Ly / 2), (-Lx / 2, Ly / 2), (Lx / 2, Ly / 2)], 0.0, 0.06, WOOD)]
    for (x0, x1, y0, y1) in ((-Lx / 2, Lx / 2, -Ly / 2, -Ly / 2 + t), (-Lx / 2, Lx / 2, Ly / 2 - t, Ly / 2),
                             (-Lx / 2, -Lx / 2 + t, -Ly / 2, Ly / 2), (Lx / 2 - t, Lx / 2, -Ly / 2, Ly / 2),
                             (-0.03, -0.03 + t, -Ly / 2, Ly / 2)):
        parts.append(G.prism([(x0, y0), (x1, y0), (x0, y1), (x1, y1)], 0.0, H, WOOD))
    for x in (-0.38, 0.38):                               # iron bands on the front and the back
        for y0, y1 in ((-Ly / 2 - 0.012, -Ly / 2 + 0.004), (Ly / 2 - 0.004, Ly / 2 + 0.012)):
            parts.append(G.prism([(x - 0.03, y0), (x + 0.03, y0), (x - 0.03, y1), (x + 0.03, y1)], -0.01, H + 0.012, LEAD))
    lid = G.prism([(-Lx / 2, 0.0), (Lx / 2, 0.0), (-Lx / 2, 0.12), (Lx / 2, 0.12)], 0.0, Ly, WOOD)   # hinged at the back edge
    lid.rotate_x(math.radians(-14.0))
    lid.translate((0.0, Ly / 2 - 0.02, H))
    parts.append(lid)
    parts.append(G.colour(sphere((0.22, 0.0, H - 0.02), 0.155, CLOTH), G.RED))          # the Quaffle
    for x in (-0.39, -0.15):
        parts.append(sphere((x - 0.0, 0.0, H - 0.08), 0.12, LEAD))                       # the Bludgers
        parts.append(G.prism([(x - 0.02, -Ly / 2), (x + 0.02, -Ly / 2), (x - 0.02, Ly / 2), (x + 0.02, Ly / 2)], H + 0.01, H + 0.03, LEAD))
    parts.append(G.colour(sphere((0.43, -0.16, H - 0.02), 0.028, CLOTH, seg=10, rings=6), G.GOLD))   # the Snitch
    return mk.merge(parts)


def hoop_post(height, R=1.25):
    pole = solid([(0.001, -0.2), (0.55, -0.2), (0.55, 0.2), (0.32, 0.35), (0.16, 0.7), (0.1, height - R - 0.1), (0.13, height - R + 0.04),
                  (0.001, height - R + 0.04)], 16, LEAD)
    return mk.merge([pole, G.torus((0.0, 0.0, height), (1.0, 0.0, 0.0), R, 0.09, seg=32, tube=8)])


def bench(stone=False):
    """A plank bench on two trestles (local: along x), or a stone bench on two blocks."""
    if stone:
        return mk.merge([G._bx((0.0, 0.0, 0.42), (2.2, 0.55, 0.12), 0.0, TRIM), G._bx((-0.8, 0.0, 0.2), (0.3, 0.45, 0.4), 0.0, STONE),
                         G._bx((0.8, 0.0, 0.2), (0.3, 0.45, 0.4), 0.0, STONE)])
    parts = [G._bx((0.0, 0.0, 0.45), (2.3, 0.36, 0.06), 0.0, WOOD)]
    for x in (-0.85, 0.85):
        for s in (-1, 1):
            parts.append(G.beam((x, s * 0.2, 0.0), (x, s * 0.1, 0.43), 0.07, 0.07))
        parts.append(G.beam((x, -0.15, 0.2), (x, 0.15, 0.2), 0.05, 0.05))
    parts.append(G.beam((-0.85, 0.0, 0.2), (0.85, 0.0, 0.2), 0.05, 0.05))
    return mk.merge(parts)


def flying_lawn(rng, lights):
    """The flying lesson on the lawn beside the stadium: two rows of brooms laid on the grass facing each other, the
    broom racks and the ball trunk where the teacher stands, practice hoops at the far end, benches along one side."""
    cx, cy = W.FLYING_LAWN
    yaw = math.radians(W.FLYING_LAWN_YAW)
    ca, sa = math.cos(yaw), math.sin(yaw)

    def P(u, v):
        x, y = cx + u * ca - v * sa, cy + u * sa + v * ca
        return np.array([x, y, ground(x, y)])
    parts = []
    for side in (-1, 1):                                  # the two rows, handles towards the middle
        for i in range(11):
            if rng.random() < 0.08:
                continue
            u = -10.0 + 2.0 * i + rng.normal(0, 0.12)
            p = P(u, side * 4.6 + rng.normal(0, 0.15))
            parts.append(broom_lying(rng, p, yaw + side * math.pi / 2 + rng.normal(0, 0.07)))
    for u in (1.0, 5.2):                                  # the teacher's corner beside the south row: racks, the ball trunk
        p = P(u, -11.0)
        parts.append(at(broom_rack(rng, 8), p - np.array([0, 0, 0.02]), yaw - math.pi / 2))
    p = P(3.2, -7.6)
    parts.append(at(ball_trunk(rng), p, yaw - math.pi))                   # its lid towards the racks
    q = P(-1.5, -1.0)
    parts.append(G.colour(sphere(q + np.array([0, 0, 0.15]), 0.155, CLOTH), G.RED))      # a Quaffle left in the aisle
    for k in range(4):                                    # brooms dropped by the racks
        p = P(rng.uniform(-2.0, 8.0), rng.uniform(-9.5, -6.5))
        parts.append(broom_lying(rng, p, rng.uniform(0, 2 * math.pi)))
    for u, v, a in ((17.5, -2.6, 2.3), (19.2, 2.9, -0.5)):      # two more left out at the end of the aisle
        parts.append(broom_lying(rng, P(u, v), yaw + a))
    for u, v, h in ((-26.0, -7.0, 7.0), (-29.0, 0.0, 9.5), (-26.0, 7.0, 7.0)):
        parts.append(at(hoop_post(h), P(u, v), yaw))
    for u in (-6.0, 0.0, 6.0):
        p = P(u, 10.5)
        parts.append(at(bench(), p, yaw))
    return mk.merge(parts)


# ----------------------------------------------------------------------------------------------- lights, fountain, statues
def brazier(pos, lights):
    """A stone pedestal with an iron fire bowl and its flames (a warm light)."""
    x, y, z = pos
    parts = [solid([(0.001, 0.0), (0.42, 0.0), (0.42, 0.18), (0.3, 0.28), (0.2, 0.4), (0.17, 1.0), (0.26, 1.1), (0.001, 1.1)], 12, TRIM),
             solid([(0.001, 1.08), (0.18, 1.08), (0.45, 1.2), (0.6, 1.42), (0.56, 1.45), (0.001, 1.36)], 16, LEAD)]
    for k in range(5):
        a = 2 * math.pi * k / 5
        r = 0.0 if k == 0 else 0.22
        h = 0.75 if k == 0 else 0.5
        fl = solid([(0.001, 0.0), (0.13, 0.1), (0.11, h * 0.5), (0.04, h * 0.85), (0.001, h)], 8, LANTERN)
        fl.translate((r * math.cos(a), r * math.sin(a), 1.36))
        parts.append(fl)
    lights.append((x, y, z + 2.1))
    return mk.merge(parts).translate(pos)


def lamp_post(pos, lights, yaw=0.0):
    """A wrought-iron lamp post with a glazed lantern (a warm light)."""
    parts = [solid([(0.001, 0.0), (0.2, 0.0), (0.2, 0.25), (0.11, 0.4), (0.07, 0.6), (0.06, 3.2), (0.1, 3.28), (0.001, 3.3)], 10, LEAD)]
    parts.append(G._bx((0.0, 0.0, 3.55), (0.34, 0.34, 0.5), 0.0, LANTERN))
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(G.beam((sx * 0.18, sy * 0.18, 3.28), (sx * 0.18, sy * 0.18, 3.82), 0.03, 0.03, LEAD))
    cap = G.ck._pyramid_roof((0.0, 0.0), 0.5, 0.5, 3.8, 0.32, 0.0)
    cap.mat[:] = LEAD
    parts.append(cap)
    parts.append(G.ck._pyramid_roof((0.0, 0.0), 0.12, 0.12, 4.1, 0.25, 0.0))
    parts[-1].mat[:] = LEAD
    lights.append((pos[0], pos[1], pos[2] + 3.55))
    return at(mk.merge(parts), pos, yaw)


def fountain(c, rng):
    """A tiered fountain: an octagonal basin with a moulded rim, a column carrying a large and a small bowl, a pine-cone
    finial, water in every bowl and thin falling sheets from the bowls' rims."""
    x, y, z = c
    R_o, R_i = 4.7, 4.2
    a0 = math.pi / 8
    parts = [solid([(0.001, -0.4), (R_i + 0.05, -0.4), (R_i + 0.05, 0.12), (0.001, 0.12)], 8, STONE)]
    for k in range(8):                                     # the octagonal rim, one block per side
        a, b = a0 + k * math.pi / 4, a0 + (k + 1) * math.pi / 4
        pa_o, pb_o = R_o * np.array([math.cos(a), math.sin(a)]), R_o * np.array([math.cos(b), math.sin(b)])
        pa_i, pb_i = R_i * np.array([math.cos(a), math.sin(a)]), R_i * np.array([math.cos(b), math.sin(b)])
        parts.append(G.prism([pa_i, pb_i, pa_o, pb_o], -0.4, 0.62, STONE))
        po, pi_ = (R_o + 0.12) / R_o, (R_i - 0.08) / R_i
        parts.append(G.prism([pa_i * pi_, pb_i * pi_, pa_o * po, pb_o * po], 0.62, 0.78, TRIM))
    parts.append(disc((0.0, 0.0), R_i + 0.02, 0.5, 8, WATER, a0))
    parts.append(solid([(0.001, 0.1), (1.05, 0.1), (1.05, 0.42), (0.72, 0.58), (0.48, 0.8), (0.36, 0.95), (0.33, 1.55), (0.42, 1.62),
                        (0.001, 1.62)], 20, TRIM))
    parts.append(solid([(0.001, 1.55), (0.45, 1.58), (1.1, 1.8), (1.75, 2.08), (2.12, 2.3), (2.12, 2.4), (1.95, 2.4), (1.6, 2.22),
                        (0.001, 2.12)], 32, TRIM))
    parts.append(disc((0.0, 0.0), 1.98, 2.3, 24, WATER))
    parts.append(solid([(0.001, 2.15), (0.3, 2.15), (0.24, 2.4), (0.2, 3.0), (0.3, 3.08), (0.001, 3.08)], 16, TRIM))
    parts.append(solid([(0.001, 3.0), (0.32, 3.02), (0.72, 3.22), (1.02, 3.42), (1.02, 3.5), (0.9, 3.5), (0.7, 3.38), (0.001, 3.32)],
                       24, TRIM))
    parts.append(disc((0.0, 0.0), 0.92, 3.44, 16, WATER))
    parts.append(solid([(0.001, 3.35), (0.14, 3.38), (0.1, 3.6), (0.2, 3.75), (0.24, 3.95), (0.16, 4.2), (0.05, 4.38), (0.001, 4.42)], 12,
                       TRIM))
    # falling water: streams pouring from lips round the bowls' rims, curving out and down (v runs down the fall)
    for r0, z0_, z1_, spread, n_s, wid in ((2.14, 2.38, 0.52, 0.6, 8, 0.42), (1.04, 3.46, 2.32, 0.32, 4, 0.3)):
        for k in range(n_s):
            a_c = 2 * math.pi * (k + 0.5) / n_s
            parts.append(G._bx((r0 * math.cos(a_c), r0 * math.sin(a_c), z0_ - 0.02), (0.36, wid + 0.12, 0.1), a_c, TRIM))   # the lip
            t = np.linspace(0, 1, 9)
            rr = r0 + 0.12 + spread * np.sqrt(t)
            zz = z0_ - (z0_ - z1_) * t * t
            a = a_c + np.linspace(-0.5, 0.5, 4)[:, None] * (wid + 0.06 * t[None]) / rr[None]
            Gs = np.stack([np.cos(a) * rr[None], np.sin(a) * rr[None], np.broadcast_to(zz, a.shape)], -1)
            f = mk.grid(Gs, mat=FALL, uv_tile=1.0)
            uv = np.stack(np.meshgrid(np.linspace(0, wid, 4), t * (z0_ - z1_) / 2.0, indexing="ij"), -1).reshape(-1, 2)
            f.uv0 = uv[f.F]
            nrm = f.face_normals()
            cen = f.V[f.F].mean(1)
            if np.mean(np.sum(nrm[:, :2] * cen[:, :2], 1)) < 0:      # facing outwards
                f.flip()
            parts.append(f)
    return mk.merge(parts).translate((x, y, z))


def statue(c, yaw, kind, lights=None):
    """A robed, hooded wizard (2.5 m) on a moulded plinth, facing +x: holding up a glowing orb, leaning on a staff, or
    reading a book; wide sleeves, the robe flaring to the ground."""
    parts = [G._bx((0.0, 0.0, 0.14), (1.6, 1.6, 0.28), 0.0, TRIM), G._bx((0.0, 0.0, 0.95), (1.22, 1.22, 1.35), 0.0, STONE),
             G._bx((0.0, 0.0, 1.72), (1.5, 1.5, 0.2), 0.0, TRIM)]
    z = 1.82
    s = 1.25
    robe = solid([(0.001, 0.0), (0.5, 0.0), (0.48, 0.15), (0.42, 0.5), (0.36, 0.95), (0.31, 1.2), (0.34, 1.45), (0.4, 1.66), (0.36, 1.77),
                  (0.17, 1.85), (0.08, 1.9), (0.001, 1.92)], 20, TRIM)
    robe.V[:, 0] *= 0.7                                   # shoulders wide across (y), the body thin front to back (x)
    parts.append(robe.scale((s, s, s)).translate((0.0, 0.0, z)))
    parts.append(sphere((0.025 * s, 0.0, z + 2.0 * s), 0.115 * s, TRIM, seg=12, rings=7))
    hood = sphere((0.0, 0.0, 0.0), 0.15 * s, TRIM, seg=14, rings=8)
    hood.V[:, 0] = np.where(hood.V[:, 0] > 0.0, hood.V[:, 0] * 0.35, hood.V[:, 0] * 1.25)   # open at the face, deep at the back
    parts.append(hood.translate((-0.02 * s, 0.0, z + 2.03 * s)))

    def arm(*pts, r=(0.085, 0.1, 0.13)):
        P = np.array([[x * s, y * s, z + h * s] for x, y, h in pts])
        return tube(P, np.array(r) * s, seg=10, mat=TRIM)
    if kind == "orb":
        parts.append(arm((0.0, -0.33, 1.66), (0.16, -0.42, 1.95), (0.22, -0.36, 2.3), r=(0.085, 0.095, 0.12)))
        parts.append(sphere((0.24 * s, -0.36 * s, z + 2.5 * s), 0.15 * s, LANTERN, seg=14, rings=7))
        parts.append(arm((0.0, 0.33, 1.66), (0.06, 0.39, 1.25), (0.16, 0.3, 1.05)))
    elif kind == "staff":
        parts.append(arm((0.0, -0.33, 1.66), (0.14, -0.4, 1.35), (0.34, -0.32, 1.38)))
        parts.append(G.beam((0.36 * s, -0.32 * s, z), (0.36 * s, -0.32 * s, z + 2.45 * s), 0.07, 0.07, TRIM))
        parts.append(sphere((0.36 * s, -0.32 * s, z + 2.5 * s), 0.09 * s, TRIM, seg=10, rings=6))
        parts.append(arm((0.0, 0.33, 1.66), (0.06, 0.39, 1.25), (0.16, 0.3, 1.05)))
    else:
        for side in (-1, 1):
            parts.append(arm((0.0, side * 0.33, 1.66), (0.12, side * 0.36, 1.3), (0.3, side * 0.14, 1.42)))
        for side in (-1, 1):                              # the open book
            pg = G._bx((0.0, 0.0, 0.0), (0.26 * s, 0.2 * s, 0.03 * s), 0.0, TRIM)
            pg.translate((0.0, side * 0.1 * s, 0.0))
            pg.rotate_x(side * 0.25)
            parts.append(pg.translate((0.36 * s, 0.0, z + 1.48 * s)))
    m = mk.merge(parts)
    return at(m, c, yaw)


def sundial(c):
    parts = [solid([(0.001, 0.0), (0.6, 0.0), (0.6, 0.15), (0.32, 0.3), (0.22, 0.45), (0.2, 0.95), (0.42, 1.05), (0.48, 1.12),
                    (0.001, 1.12)], 16, TRIM)]
    gn = Mesh(np.array([[-0.35, -0.012, 1.12], [0.3, -0.012, 1.12], [-0.35, -0.012, 1.5], [-0.35, 0.012, 1.12], [0.3, 0.012, 1.12],
                        [-0.35, 0.012, 1.5]]), np.array([[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [0, 2, 5], [0, 5, 3]]),
              LEAD)
    if gn.volume() < 0:
        gn.flip()
    parts.append(gn)
    return mk.merge(parts).translate(c)


# ----------------------------------------------------------------------------------------------- odds and ends
def barrel(c, rng):
    h = rng.uniform(0.85, 0.95)
    m = solid([(0.001, 0.0), (0.27, 0.0), (0.33, h * 0.25), (0.35, h * 0.5), (0.33, h * 0.75), (0.27, h), (0.001, h)], 14, WOOD)
    hoops = [solid([(0.001, z - 0.03), (r + 0.012, z - 0.03), (r + 0.012, z + 0.03), (0.001, z + 0.03)], 14, LEAD)
             for z, r in ((h * 0.12, 0.29), (h * 0.36, 0.34), (h * 0.64, 0.34), (h * 0.88, 0.29))]
    return mk.merge([m] + hoops).translate(c)


def crate(c, yaw, s=0.7):
    parts = [G._bx((0.0, 0.0, s / 2), (s, s, s), 0.0, WOOD)]
    for sx in (-1, 1):                                    # corner battens
        for sy in (-1, 1):
            parts.append(G._bx((sx * s / 2, sy * s / 2, s / 2), (0.08, 0.08, s + 0.02), 0.0, WOOD))
    return at(mk.merge(parts), c, yaw)


def woodpile(c, yaw, rng):
    parts = []
    for row in range(4):
        for k in range(7 - row):
            y = (k - (6 - row) / 2) * 0.34
            r = rng.uniform(0.13, 0.17)
            parts.append(G.beam((-0.8, y, 0.16 + row * 0.29), (0.8 + rng.uniform(-0.1, 0.1), y, 0.16 + row * 0.29), 2 * r, 2 * r, WOOD))
    return at(mk.merge(parts), c, yaw)


def court_spots():
    """Open places in the courts (from the plan): the Transfiguration court, the Viaduct court, the clock court, the
    north-west rock's garden terrace."""
    return dict(transfiguration=(-65.0, 69.0), viaduct=(103.0, 43.0), clock=(-36.0, -150.0), terrace=(-134.5, 87.5))


def build(rng):
    lights = []
    parts = [flying_lawn(rng, lights)]
    # lamp posts along the road from the castle to the gates, on alternate sides
    path = W.ROAD_PATH
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    k = 0
    for u in np.arange(20.0, s[-1], 32.0):
        i = min(int(np.searchsorted(s, u)), len(path) - 2)
        p, d = path[i], path[i + 1] - path[i]
        if p[1] > W.GATES[1] - 8.0:
            break
        d = d / (np.linalg.norm(d) + 1e-9)
        side = 1 if k % 2 == 0 else -1
        q = p + np.array([-d[1], d[0]]) * side * 5.8
        parts.append(lamp_post((q[0], q[1], ground(q[0], q[1]) - 0.05), lights))
        k += 1
    # braziers at the castle's door to the grounds and at the gates
    for q in ((43.0, 236.0), (57.0, 234.0), (W.GATES[0] - 10.5, W.GATES[1] - 4.0), (W.GATES[0] + 10.5, W.GATES[1] - 4.0)):
        parts.append(brazier((q[0], q[1], ground(q[0], q[1]) - 0.05), lights))
    # the courts
    S = court_spots()
    fx, fy = S["transfiguration"]
    fz = ground(fx, fy)
    parts.append(fountain((fx, fy, fz), rng))
    for k in range(4):                                    # benches round the fountain, statues and braziers further out
        a = math.radians(-20.0) + k * math.pi / 2 + math.pi / 4
        q = (fx + 7.6 * math.cos(a), fy + 7.6 * math.sin(a))
        parts.append(at(bench(stone=True), (q[0], q[1], ground(*q)), a + math.pi / 2))
    for k, kind in enumerate(("orb", "book")):
        a = math.radians(-20.0) + k * math.pi
        q = (fx + 12.5 * math.cos(a), fy + 12.5 * math.sin(a))
        parts.append(statue((q[0], q[1], ground(*q)), a + math.pi, kind))
    for k in range(2):
        a = math.radians(-20.0) + math.pi / 2 + k * math.pi
        q = (fx + 11.0 * math.cos(a), fy + 11.0 * math.sin(a))
        parts.append(brazier((q[0], q[1], ground(*q)), lights))
    vx, vy = S["viaduct"]
    for k, kind in enumerate(("staff", "orb")):
        q = (vx + (-7.0 if k == 0 else 7.0), vy + 2.0)
        parts.append(statue((q[0], q[1], ground(*q)), math.pi if k else 0.0, kind))
    for q in ((vx - 4.0, vy - 7.5), (vx + 4.0, vy - 7.5)):
        parts.append(brazier((q[0], q[1], ground(*q)), lights))
    cx_, cy_ = S["clock"]
    parts.append(sundial((cx_, cy_, ground(cx_, cy_))))
    tx, ty = S["terrace"]
    parts.append(statue((tx, ty, ground(tx, ty)), math.radians(160.0), "staff"))
    for k in range(3):
        a = math.radians(160.0) + (k - 1) * 0.5
        q = (tx + 6.0 * math.cos(a), ty + 6.0 * math.sin(a))
        parts.append(at(bench(stone=True), (q[0], q[1], ground(*q)), a + math.pi / 2))
    # Herbology: barrels and crates by the greenhouses' doors; the gamekeeper's woodpile
    gx, gy = W.GREENHOUSES
    for i, dy in enumerate((-18.0, 0.0, 18.0)):
        for j in range(3):
            q = (gx + 19.0 + rng.uniform(0.0, 2.5), gy + dy + rng.uniform(-3.5, 3.5))
            if j == 0:
                parts.append(barrel((q[0], q[1], ground(*q) - 0.03), rng))
            else:
                parts.append(crate((q[0], q[1], ground(*q) - 0.03), rng.uniform(0, math.pi), rng.uniform(0.55, 0.8)))
    hx, hy = W.HUT
    q = (hx - 6.5, hy + 3.0)
    parts.append(woodpile((q[0], q[1], ground(*q) - 0.05), 0.4, rng))
    for k in range(3):
        q = (hx + 6.0 + k * 0.8, hy - 4.0 + rng.uniform(-0.3, 0.3))
        parts.append(barrel((q[0], q[1], ground(*q) - 0.03), rng))
    return mk.merge(parts), np.array(lights, np.float64)
