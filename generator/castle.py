"""castle - builds the castle of castle_plan.py: unique shells + instance records of the detail modules of detail_lib.py.

Unique geometry (per building, grouped by area into OUT/geo/castle_<zone>.npz):
    walls / stages as solids with the window and door openings really cut (manifold booleans; the reveals become dressed
    stone), plinths down to the actual rock, buttresses with set-offs, string courses, parapet walls, gable walls with
    copings, slate roofs / cones / spires with modelled courses, viaduct and bridge masonry, stairs, glasshouse frames
Instances (OUT/geo/instances/castle_<module>.npz -> Nanite instanced static meshes in Unreal):
    the window inserts (tracery, mullions, glass), doors, pinnacles, crockets, finials, merlons, corbels, machicolation
    arches, balusters, pierced parapet panels, lucarnes, dormers, chimneys, ridge cresting, gargoyles, bartizans,
    arcade columns, clock faces

    python castle.py OUT/geo
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict

import manifold3d as m3d
import numpy as np

import castle_kit as ck
import castle_plan as CP
import detail_lib as DL
import instances as ins
import meshkit as mk
import world as W
from castle_plan import (ArchBridge, Arcade, Boathouse, CliffWalls, CoveredBridge, Gatehouse, Glasshouse, Hall, Range, Stair,
                         SuspensionBridge, Tower, Viaduct, Wall)

STONE, TRIM, SLATE, LEAD, GLASS, WOOD = 20, 21, 22, 23, 24, 25
PAVE = ck.PAVE                  # walkway paving (finished as trim with little moss)
UP = np.array([0.0, 0.0, 1.0])
Z = W.CASTLE_Z


# ----------------------------------------------------------------------------------------------------- the rock
_TOP = {}


def surface_top():
    """1 m height map of the top of the core terrain mesh (granite included), cached; None if the core is missing."""
    if "grid" in _TOP:
        return _TOP["grid"]
    import glob

    from scipy import ndimage

    files = sorted(glob.glob("OUT/geo/terrain_core_*.npz"))
    if not files:
        _TOP["grid"] = None
        return None
    cache = "OUT/geo/core_top.npz"
    if os.path.isfile(cache) and all(os.path.getmtime(cache) > os.path.getmtime(f) for f in files):
        z = np.load(cache)
        _TOP["grid"] = (z["top"], float(z["x0"]), float(z["y0"]))
        return _TOP["grid"]
    x0, y0, n = -610.0, -530.0, 1221
    top = np.full((n, n), -np.inf)
    for f in files:
        V = mk.Mesh.load(f).V
        i = np.clip((V[:, 0] - x0).astype(int), 0, n - 1)
        j = np.clip((V[:, 1] - y0).astype(int), 0, n - 1)
        np.maximum.at(top, (i, j), V[:, 2])
    bad = ~np.isfinite(top)
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        top = top[tuple(idx)]
    np.savez_compressed(cache, top=top.astype(np.float32), x0=x0, y0=y0)
    _TOP["grid"] = (top, x0, y0)
    return _TOP["grid"]


def ground(x, y):
    """Rock / ground height under (x, y): the core terrain mesh if built, else the analytic height field."""
    x = np.atleast_1d(np.asarray(x, np.float64))
    y = np.atleast_1d(np.asarray(y, np.float64))
    g = surface_top()
    if g is None:
        return W.Fields(x, y).height()
    top, x0, y0 = g
    i = np.clip((x - x0).astype(int), 0, top.shape[0] - 1)
    j = np.clip((y - y0).astype(int), 0, top.shape[1] - 1)
    return top[i, j].astype(np.float64)


def foot_along(pts, base, deep=1.2, max_drop=60.0):
    """Foundation level for a wall along `pts` (n, 2): the lowest rock under it minus `deep`, never above base - 0.5."""
    P = np.asarray(pts, np.float64)
    seg = [P[i] + (P[i + 1] - P[i]) * t for i in range(len(P) - 1) for t in np.linspace(0, 1, max(2, int(np.linalg.norm(P[i + 1] - P[i]) / 2.0)))]
    seg = np.array(seg) if seg else P
    g = ground(seg[:, 0], seg[:, 1])
    return float(np.clip(g.min() - deep, base - max_drop, base - 0.5))


# ----------------------------------------------------------------------------------------------------- the sink
class Sink:
    """Collects the unique meshes per zone and the module placements per module."""

    def __init__(self):
        self.parts = defaultdict(list)
        self.inst = defaultdict(lambda: ([], [], []))
        self.zone = "castle_misc"

    def set_zone(self, x, y):
        self.zone = f"castle_{'nw' if y > 0.55 * x + 40.0 else 'se'}_{int(np.clip((x + 200.0) // 110.0, 0, 4))}{int(np.clip((y + 240.0) // 110.0, 0, 4))}"

    def add(self, m):
        if m is None:
            return
        if isinstance(m, m3d.Manifold):
            if m.is_empty():
                return
            m = ck.man_to_mesh(m)
        if isinstance(m, (list, tuple)):
            for x in m:
                self.add(x)
            return
        if m.nf:
            self.parts[self.zone].append(m)

    def place(self, module, pos, R, scale=(1.0, 1.0, 1.0)):
        P, Rs, S = self.inst[module]
        P.append(np.asarray(pos, np.float64))
        Rs.append(np.asarray(R, np.float64))
        S.append(np.asarray(scale, np.float64) * np.ones(3))


def frame_R(along, inward):
    """Rotation whose columns are (along, inward, up) - the local frame of every wall module."""
    a = np.asarray(along, np.float64)
    a = a / np.linalg.norm(a)
    i = np.asarray(inward, np.float64)
    i = i - a * float(a @ i)
    i = i / np.linalg.norm(i)
    u = np.cross(a, i)
    return np.stack([a, i, u], 1)


def to_world(m: m3d.Manifold, origin, along, inward):
    """Transform a manifold built in a wall's local frame (x along, y inward, z up) to the world."""
    R = frame_R(along, inward)
    M = np.zeros((3, 4))
    M[:, :3] = R
    M[:, 3] = origin
    return m.transform(M)


def tag(m, mat):
    return ck._orig(m, mat)


def box_between(p0, p1, depth, z0, z1, mat=STONE, offset=0.0):
    """Box along p0 -> p1 (horizontal), `depth` across (centred, shifted by `offset` along the left normal), z0..z1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / max(L, 1e-9)
    n = np.array([-a[1], a[0]])
    c = (p0 + p1) / 2 + n * offset
    b = m3d.Manifold.cube([L, depth, z1 - z0], True).rotate([0, 0, math.degrees(math.atan2(a[1], a[0]))])
    return tag(b.translate([c[0], c[1], (z0 + z1) / 2]), mat)


def ring(cx, cy, r_in, r_out, z0, z1, seg=48, mat=TRIM):
    o = m3d.Manifold.cylinder(z1 - z0, r_out, r_out, seg)
    i = m3d.Manifold.cylinder(z1 - z0 + 0.2, r_in, r_in, seg).translate([0, 0, -0.1])
    return tag((o - i).translate([cx, cy, z0]), mat)


def prism(shape, size, z0, z1, x, y, yaw=0.0, depth=None, mat=STONE, seg=48, r_top=None):
    if z1 <= z0 + 1e-3:
        return None
    if shape == "round":
        m = m3d.Manifold.cylinder(z1 - z0, size, size if r_top is None else r_top, seg)
    elif shape == "octagon":
        k = 1.0 / math.cos(math.pi / 8)
        m = m3d.Manifold.cylinder(z1 - z0, size * k, (size if r_top is None else r_top) * k, 8).rotate([0, 0, 22.5])
    else:
        d = size if depth is None else depth
        m = m3d.Manifold.cube([2 * size, 2 * d, z1 - z0], True).translate([0, 0, (z1 - z0) / 2])
    if yaw:
        m = m.rotate([0, 0, yaw])
    return tag(m.translate([x, y, z0]), mat)


# ----------------------------------------------------------------------------------------------------- openings
def opening_cutter(module, w, h, depth=1.6):
    """The hole a window / door module needs, in the module's local frame (x along, y inward, z up), sill at z = 0."""
    kind = DL.OPENING_KIND.get(module, "pointed")
    if module in ("c_rose", "c_clock"):
        P = DL.circle(w / 2 - 0.05, 48, 0.0, w / 2)
    else:
        P = DL.arch_outline(w - 0.1, h - 0.05, kind, 12)
    return DL.extrude_xz(P, -0.3, depth)


def window(sink, cutters, module, origin, along, inward, w=None, h=None, set_back=0.0):
    """Place a window / door module at `origin` (sill centre on the wall face) and collect its opening cutter."""
    ow, oh = DL.OPENING[module]
    w = ow if w is None else w
    h = oh if h is None else h
    o = np.asarray(origin, np.float64) + np.asarray(inward) / np.linalg.norm(inward) * set_back
    cutters.append(tag(to_world(opening_cutter(module, w, h), o, along, inward), TRIM))
    sink.place(module, o, frame_R(along, inward), (w / ow, 1.0, h / oh))


def apply_cuts(body, cutters):
    if not cutters:
        return body
    return body - m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)


# ----------------------------------------------------------------------------------------------------- facades
def facade_windows(sink, cutters, p0, p1, n_out, z0, z1, floor_h, module, bay=4.4, first=1.2, margin=2.2, door_mid=False,
                   rng=None, stagger=False):
    """Rows of windows on a straight facade from p0 to p1 (outward normal n_out), one per bay per floor."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 2 * margin + 1.0:
        return
    a = (p1 - p0) / L
    inward = -np.array([n_out[0], n_out[1], 0.0])
    along = np.cross(inward, UP)
    if float(along[:2] @ a) < 0:                                 # keep x along the facade, left -> right from outside
        p0, p1 = p1, p0
        a = -a
    nb = max(1, int((L - 2 * margin) / bay))
    ow, oh = DL.OPENING[module]
    floors = max(1, int((z1 - z0 - first) / floor_h))
    for f in range(floors):
        zf = z0 + first + f * floor_h
        hh = min(oh * 1.15, floor_h - 1.3)
        if hh < 1.2:
            continue
        ww = min(ow * 1.1, bay - 1.4) * (hh / oh) ** 0.25
        for k in range(nb):
            t = margin + (L - 2 * margin) * (k + 0.5 + (0.25 if (stagger and f % 2) else 0.0)) / nb
            if t > L - margin + 0.1:
                continue
            q = p0 + a * t
            if door_mid and f == 0 and k == nb // 2:
                window(sink, cutters, "c_door", (q[0], q[1], z0), along, inward, 3.2, 4.8)
                continue
            window(sink, cutters, module, (q[0], q[1], zf), along, inward, ww, hh)


def corbel_table(sink, p0, p1, n_out, z_top, spacing=1.35, arches=True):
    """Corbels (and machicolation arches between them) under an overhanging parapet along a straight wall."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / max(L, 1e-9)
    inward = -np.array([n_out[0], n_out[1], 0.0])
    along = np.cross(inward, UP)
    R = frame_R(along, inward)
    n = max(1, int(L / spacing))
    for k in range(n + 1):
        q = p0 + a * (L * k / n)
        sink.place("c_corbel", (q[0], q[1], z_top - 1.3), R)
        if arches and k < n:
            qm = p0 + a * (L * (k + 0.5) / n)
            sink.place("c_macharch", (qm[0], qm[1], z_top - 0.62), R, (L / n / 1.3, 1.0, 1.0))


def parapet_line(sink, p0, p1, n_out, z, kind="crenel", over=0.0, thick=0.7):
    """A parapet along a straight edge (outward normal n_out), set `over` m out from the wall face: a low wall with
    merlons, or a run of pierced panels."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 0.5:
        return None
    a = (p1 - p0) / L
    no = np.array([n_out[0], n_out[1]])
    inward = -np.array([no[0], no[1], 0.0])
    along = np.cross(inward, UP)
    R = frame_R(along, inward)
    q0, q1 = p0 + no * (over - thick / 2), p1 + no * (over - thick / 2)
    if kind == "pierced":
        n = max(1, int(round(L / 2.0)))
        for k in range(n):
            q = p0 + a * (L * (k + 0.5) / n) + no * over
            sink.place("c_panel", (q[0], q[1], z), R, (L / n / 2.0, 1.0, 1.0))
        return box_between(q0, q1, thick, z - 0.3, z + 0.02, TRIM)
    base = box_between(q0, q1, thick, z - 0.3, z + 0.55, STONE)
    n = max(1, int(round(L / 2.4)))
    for k in range(n):
        q = p0 + a * (L * (k + 0.5) / n) + no * over
        sink.place("c_merlon", (q[0], q[1], z + 0.55), R, (min(1.1, L / n / 2.0), thick / 0.75, 1.0))
    return base


def ridge_items(sink, c, axis, length, z_ridge, cresting=False, chimneys=0, rng=None):
    a = np.array([axis[0], axis[1], 0.0]) / np.linalg.norm(axis)
    b = np.cross(UP, a)
    R = frame_R(a, b)
    if cresting:
        n = int(length)
        for k in range(n):
            q = np.asarray(c, float) + a[:2] * (-length / 2 + k + 0.5)
            sink.place("c_cresting", (q[0], q[1], z_ridge + 0.32), R)
    for k in range(chimneys):
        t = (k + 0.5) / chimneys - 0.5
        q = np.asarray(c, float) + a[:2] * t * length * 0.85 + b[:2] * (rng.uniform(-1.5, 1.5) if rng is not None else 0.0)
        sink.place("c_chimney", (q[0], q[1], z_ridge - 2.2), R, (1.0, 1.0, 1.0 + (rng.uniform(0, 0.4) if rng is not None else 0.0)))


def roof_dormers(sink, c, axis, length, width, z_eave, pitch, module="c_dormer", spacing=7.5, up_frac=0.22, scale=1.0, rows=1):
    """Dormers on both slopes of a gable roof, set into the slope (their bodies run back into the roof)."""
    a = np.array([axis[0], axis[1], 0.0]) / np.linalg.norm(axis)
    b = np.cross(UP, a)
    k = math.tan(math.radians(pitch))
    for row in range(rows):
        frac = up_frac + row * 0.32
        for side in (1.0, -1.0):
            n_out = b * side
            d_in = (width / 2) * frac
            zb = z_eave + d_in * k - 0.6
            n = max(1, int((length - 4.0) / spacing))
            for i in range(n):
                t = -length / 2 + 2.0 + (length - 4.0) * (i + 0.5 + 0.5 * row) / n
                if t > length / 2 - 2.0:
                    continue
                q = np.asarray(c, float) + a[:2] * t + n_out[:2] * (width / 2 - d_in)
                sink.place(module, (q[0], q[1], zb), frame_R(np.cross(-n_out, UP), -n_out), (scale, scale, scale))


def gable_crockets(sink, origin, b, a_out, half_w, rise, n=6):
    """Crockets up both copings of a gable and a pinnacle at its apex.  origin = eave-level centre of the gable face."""
    o = np.asarray(origin, float)
    for s in (-1.0, 1.0):
        for k in range(1, n):
            t = k / n
            p = o + b * s * half_w * (1 - t) + UP * rise * t + a_out * 0.2
            sink.place("c_crocket", p, frame_R(-b * s, -a_out), (1.4, 1.4, 1.4))
    sink.place("c_pinnacle", o + UP * (rise - 0.2) + a_out * 0.05, frame_R(b, -a_out), (0.75, 0.75, 0.75))


# ----------------------------------------------------------------------------------------------------- roofs
def slate_roof(c, axis, length, width, z_eave, pitch, overhang=0.45, ends=("gable", "gable"), course=0.42, step=0.035):
    """Slate roof over a rectangle (centre c, long axis `axis`): two side slopes and, per end, a hip ("hip"), a verge over
    a gable wall ("gable") or a plain cut against the neighbour ("none").  Modelled courses (every course stands proud
    of the one above it), a soffit under each eave, lead rolls on the ridge and the hips.
    Returns (mesh, z_ridge, r0, r1) - the ridge runs from r0 to r1 along the axis (measured from c)."""
    a = np.array([axis[0], axis[1], 0.0])
    a /= np.linalg.norm(a)
    b = np.array([-a[1], a[0], 0.0])
    k = math.tan(math.radians(pitch))
    half = width / 2 + overhang
    z_r = z_eave + (width / 2) * k
    z_e = z_eave - overhang * k
    c3 = np.array([c[0], c[1], 0.0])
    ov = [overhang + (0.15 if e == "gable" else 0.0) if e != "none" else 0.0 for e in ends]
    u0, u1 = -(length / 2 + ov[0]), length / 2 + ov[1]
    r0 = u0 + (half if ends[0] == "hip" else 0.0)
    r1 = u1 - (half if ends[1] == "hip" else 0.0)
    if r0 > r1:
        r0 = r1 = (r0 + r1) / 2

    def P(u, v, z):
        return c3 + a * u + b * v + np.array([0.0, 0.0, z])

    T0, T1 = P(r0, 0.0, z_r), P(r1, 0.0, z_r)
    planes = [(P(u0, s * half, z_e), P(u1, s * half, z_e), T0, T1, b * s) for s in (1.0, -1.0)]
    for end, (ue, T, sg) in enumerate(((u0, T0, -1.0), (u1, T1, 1.0))):
        if ends[end] == "hip":
            planes.append((P(ue, -half, z_e), P(ue, half, z_e), T, T, a * sg))
    parts = []
    for E0, E1, Ta, Tb, out in planes:
        slant = float(np.linalg.norm((E0 + E1) / 2 - (Ta + Tb) / 2))
        n_c = max(4, int(slant / course))
        sv = np.linspace(0.0, 1.0, n_c + 1)
        rows = np.repeat(sv, 2)[1:-1]
        G = np.empty((2, len(rows), 3))
        G[0] = E0[None] + (Ta - E0)[None] * rows[:, None]
        G[1] = E1[None] + (Tb - E1)[None] * rows[:, None]
        G[:, 0::2] += out * step
        m = mk.grid(G, uv_tile=1.0, mat=SLATE)
        e = (E1 - E0) / max(float(np.linalg.norm(E1 - E0)), 1e-9)
        uu = np.einsum("ijk,k->ij", G - E0[None, None], e)
        vv = rows[None] * slant * np.ones((2, 1))
        m.uv0 = np.stack([uu, vv], -1).reshape(-1, 2)[m.F]
        if np.mean(m.face_normals()[:, 2]) < 0:
            m.flip()
        parts.append(m)
        # soffit under the eave: from the eave line back to the wall face
        if overhang > 0.2:
            q0, q1 = E0[:2] - out[:2] * (overhang / 2), E1[:2] - out[:2] * (overhang / 2)
            if Ta is Tb:                                          # hip end: the soffit runs between the side soffits
                q0, q1 = q0 + e[:2] * overhang, q1 - e[:2] * overhang
            parts.append(ck.man_to_mesh(box_between(q0, q1, overhang + 0.05, z_e - 0.28, z_e - 0.02, TRIM)))
    if r1 > r0 + 0.1:
        parts.append(ck._mesh_box((T0 + T1) / 2 + np.array([0.0, 0.0, 0.06]), (r1 - r0 + 0.3, 0.42, 0.30), a, b, LEAD))
    for E0, E1, Ta, Tb, out in planes:
        if Ta is Tb:                                              # the hip lines: from both eave corners to the apex
            for E in (E0, E1):
                path = np.array([E + np.array([0, 0, 0.1]), Ta + np.array([0, 0, 0.1])])
                ex = np.repeat(np.cross(Ta - E, np.array([0, 0, 1.0]))[None], 2, 0)
                ex /= np.linalg.norm(ex[0]) + 1e-9
                parts.append(mk.sweep(mk.ngon(0.16, 6), path, ex, mat=LEAD))
    return mk.merge(parts), z_r, r0, r1


# ----------------------------------------------------------------------------------------------------- towers
def tower_windows_round(sink, cutters, x, y, r, z0, z1, floor_h, n, module, phase=0.0, stagger=False):
    """Windows round a drum, one per bay per floor, the bays lined up vertically (staggered for stair turrets);
    big drums get paired lights every third floor."""
    floors = max(0, int((z1 - z0 - 1.5) / floor_h))
    for f in range(floors):
        mod = "c_double" if (r >= 9.0 and f % 3 == 1) else module
        ow, oh = DL.OPENING[mod]
        hh = min(oh, floor_h - 1.6)
        if hh < 1.0:
            continue
        ww = min(ow * (hh / oh) ** 0.4, 2 * math.pi * r / n - 1.2)
        zf = z0 + 1.2 + f * floor_h
        off = phase + ((0.5 if f % 2 else 0.0) if stagger else 0.0)
        for k in range(n):
            th = 2 * math.pi * (k + off) / n
            nrm = np.array([math.cos(th), math.sin(th), 0.0])
            p = np.array([x, y, zf]) + nrm * (r - 0.06)
            window(sink, cutters, mod, p, np.cross(-nrm, UP), -nrm, ww, hh)


def round_band(sink, kind, x, y, r, z_top):
    """Corbel table + parapet / gallery / crenellations round the top of a round (or octagonal) stage."""
    parts = []
    circ = 2 * math.pi * r
    if kind in ("corbel", "gallery"):
        n = max(8, int(circ / 1.35))
        for k in range(n):
            th = 2 * math.pi * k / n
            nrm = np.array([math.cos(th), math.sin(th), 0.0])
            p = np.array([x, y, z_top - 1.3]) + nrm * r
            sink.place("c_corbel", p, frame_R(np.cross(-nrm, UP), -nrm))
            if kind == "corbel":
                thm = 2 * math.pi * (k + 0.5) / n
                nm = np.array([math.cos(thm), math.sin(thm), 0.0])
                sink.place("c_macharch", np.array([x, y, z_top - 0.62]) + nm * r, frame_R(np.cross(-nm, UP), -nm), (circ / n / 1.3, 1.0, 1.0))
    if kind == "corbel":
        parts.append(ring(x, y, r + 0.25, r + 0.95, z_top - 0.25, z_top + 0.6, mat=STONE))
        n = max(6, int(2 * math.pi * (r + 0.6) / 2.4))
        for k in range(n):
            th = 2 * math.pi * (k + 0.5) / n
            nrm = np.array([math.cos(th), math.sin(th), 0.0])
            sink.place("c_merlon", np.array([x, y, z_top + 0.6]) + nrm * (r + 0.95), frame_R(np.cross(-nrm, UP), -nrm),
                       (min(1.0, 2 * math.pi * (r + 0.6) / n / 2.0), 0.9, 1.0))
    elif kind == "gallery":
        parts.append(ring(x, y, r - 0.5, r + 1.7, z_top - 0.25, z_top + 0.12, mat=TRIM))
        rb = r + 1.5
        n = max(12, int(2 * math.pi * rb / 0.34))
        for k in range(n):
            th = 2 * math.pi * k / n
            sink.place("c_baluster", (x + rb * math.cos(th), y + rb * math.sin(th), z_top + 0.12), np.eye(3))
        parts.append(ring(x, y, rb - 0.14, rb + 0.14, z_top + 1.02, z_top + 1.16, mat=TRIM))
    elif kind == "crenel":
        parts.append(ring(x, y, r - 0.6, r + 0.15, z_top - 0.1, z_top + 0.55, mat=STONE))
        n = max(6, int(2 * math.pi * r / 2.4))
        for k in range(n):
            th = 2 * math.pi * (k + 0.5) / n
            nrm = np.array([math.cos(th), math.sin(th), 0.0])
            sink.place("c_merlon", np.array([x, y, z_top + 0.55]) + nrm * (r + 0.15), frame_R(np.cross(-nrm, UP), -nrm),
                       (min(1.0, 2 * math.pi * r / n / 2.0), 0.9, 1.0))
    elif kind == "string":
        parts.append(ring(x, y, r - 0.1, r + 0.28, z_top - 0.4, z_top, mat=TRIM))
    return parts


def square_band(sink, kind, x, y, half, depth, yaw, z_top):
    """The same for a square stage: corbel tables + parapets along its four sides, bartizans / gargoyles at corners."""
    parts = []
    ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    ax, ay = np.array([ca, sa]), np.array([-sa, ca])
    c = np.array([x, y])
    d = half if depth is None else depth
    corners = [c + ax * sx * half + ay * sy * d for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        e = (p1 - p0) / np.linalg.norm(p1 - p0)
        n_out = np.array([e[1], -e[0]])
        if kind in ("corbel", "gallery"):
            corbel_table(sink, p0, p1, n_out, z_top)
            parts.append(parapet_line(sink, p0 - e * 0.95, p1 + e * 0.95, n_out, z_top, "pierced" if kind == "gallery" else "crenel", over=0.95))
        elif kind == "crenel":
            parts.append(parapet_line(sink, p0, p1, n_out, z_top, "crenel", over=0.0))
        elif kind == "string":
            parts.append(box_between(p0, p1, 0.5, z_top - 0.4, z_top, TRIM, offset=-0.2 if False else 0.0))
    for i, p in enumerate(corners):
        dirv = (p - c) / np.linalg.norm(p - c)
        nrm = np.array([dirv[0], dirv[1], 0.0])
        sink.place("c_gargoyle", (p[0], p[1], z_top - 0.4), frame_R(np.cross(-nrm, UP), -nrm), (0.8, 0.8, 0.8))
    return parts


def stage_gables(sink, cutters, x, y, half, yaw, z_top, module="c_lancet"):
    """A steep gable (wimperg) with a window on each face of a square stage."""
    parts = []
    ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    for k in range(4):
        th = math.radians(yaw) + k * math.pi / 2 - math.pi / 2
        n_out = np.array([math.cos(th), math.sin(th), 0.0])
        b = np.cross(UP, n_out)
        o = np.array([x, y, z_top]) + n_out * (half - 0.6)
        w = half * 1.05
        rise = w * 1.5
        tri = np.array([[-w, 0.0], [w, 0.0], [0.0, rise]])
        g = DL.extrude_xz(tri, -0.6, 0.8)
        parts.append(tag(to_world(g, o, b, -n_out), STONE))
        window(sink, cutters, module, o + UP * 0.8 + n_out * 0.62, b, -n_out, min(1.1, w * 0.55), min(3.0, rise * 0.55))
        gable_crockets(sink, o + n_out * 0.62, b, n_out, w, rise, 5)
    return parts


def cone_lucarnes(sink, x, y, r_base, z0, h, rows):
    for row in range(rows):
        t = 0.16 + 0.24 * row
        rr = r_base * (1 - t) - 0.15
        if rr < 1.2:
            continue
        n = max(3, int(2 * math.pi * rr / (7.0 + 2.0 * row)))
        sc = max(0.55, min(1.2, rr / 7.0))
        for k in range(n):
            th = 2 * math.pi * (k + 0.5 * row) / n
            nrm = np.array([math.cos(th), math.sin(th), 0.0])
            p = np.array([x, y, z0 + h * t - 0.5 * sc]) + nrm * rr
            sink.place("c_lucarne", p, frame_R(np.cross(-nrm, UP), -nrm), (sc, sc, sc))


def spire_crockets(sink, x, y, r_base, z0, h, edges=8, yaw=0.0, every=1.6):
    for e in range(edges):
        th = math.radians(yaw) + 2 * math.pi * e / edges + (math.pi / edges if edges == 8 else math.pi / 4)
        rr0 = r_base / math.cos(math.pi / edges)
        L = math.hypot(rr0, h)
        n = int(L / every)
        for k in range(1, n):
            t = k / n
            if t > 0.92:
                continue
            nrm = np.array([math.cos(th), math.sin(th), 0.0])
            p = np.array([x, y, z0 + h * t]) + nrm * (rr0 * (1 - t) + 0.05)
            sink.place("c_crocket", p, frame_R(np.cross(-nrm, UP), -nrm), (1.2 * (1 - 0.5 * t),) * 3)


def belfry(sink, shape, st, x, y, yaw, z0, z1, bell=False):
    """An open stage (belfry / belvedere): a hollow drum or box pierced on every side by tall pointed arches over a sill
    wall, engaged shafts between the arches, a floor and a ceiling slab, a bell hung inside square belfries."""
    h = z1 - z0
    t = 0.9 if st.size < 8 else 1.2
    outer = prism(shape, st.size, z0, z1, x, y, yaw, st.depth, mat=STONE)
    inner = prism(shape, st.size - t, z0 + 0.6, z1 - 1.4, x, y, yaw, None if st.depth is None else st.depth - t, mat=STONE)
    cut = []
    sill = 1.1
    hh = h - sill - 2.4
    if shape in ("round", "octagon"):
        n = st.windows or max(6, int(2 * math.pi * st.size / 4.2))
        w = min(2 * math.pi * st.size / n - 1.1, 4.2)
        for k in range(n):
            th = math.radians(yaw) + 2 * math.pi * (k + 0.5) / n
            nrm = np.array([math.cos(th), math.sin(th), 0.0])
            o = np.array([x, y, z0 + 0.6 + sill]) + nrm * (st.size + 0.3)
            cut.append(tag(to_world(DL.extrude_xz(DL.arch_outline(w, hh, "pointed", 12), -0.5, t + 1.0), o, np.cross(-nrm, UP), -nrm), TRIM))
            thm = math.radians(yaw) + 2 * math.pi * k / n
            nm = np.array([math.cos(thm), math.sin(thm), 0.0])
            q = np.array([x, y, z0 + 0.6]) + nm * (st.size + 0.12)
            sink.place("c_column", q, frame_R(np.cross(-nm, UP), -nm), (0.8, 0.8, (h - 2.4 - 0.6) / 4.0))
    else:
        d = st.size if st.depth is None else st.depth
        ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ax, ay = np.array([ca, sa, 0.0]), np.array([-sa, ca, 0.0])
        for nrm, half_len, off in ((ay, st.size, d), (-ay, st.size, d), (ax, d, st.size), (-ax, d, st.size)):
            along = np.cross(-nrm, UP)
            n = st.windows or max(1, int(2 * half_len / 4.5))
            span = (2 * half_len - 1.6) / n
            w = min(span - 1.0, 4.2)
            for k in range(n):
                u = -half_len + 0.8 + span * (k + 0.5)
                o = np.array([x, y, z0 + 0.6 + sill]) + nrm * (off + 0.3) + along * u
                cut.append(tag(to_world(DL.extrude_xz(DL.arch_outline(w, hh, "pointed", 12), -0.5, t + 1.0), o, along, -nrm), TRIM))
                if k:
                    q = np.array([x, y, z0 + 0.6]) + nrm * (off + 0.1) + along * (-half_len + 0.8 + span * k)
                    sink.place("c_column", q, frame_R(along, -nrm), (0.8, 0.8, (h - 2.4 - 0.6) / 4.0))
    shell = apply_cuts(outer - inner, cut)
    parts = [shell, ring(x, y, st.size - 0.05, st.size + 0.3, z0 + 0.6 + sill - 0.2, z0 + 0.6 + sill + 0.05) if shape != "square" else
             tag(prism("square", st.size + 0.25, z0 + 0.6 + sill - 0.2, z0 + 0.6 + sill + 0.05, x, y, yaw,
                       None if st.depth is None else st.depth + 0.25) - prism("square", st.size - 0.5, z0, z1, x, y, yaw), TRIM)]
    if bell:
        rb = min(st.size * 0.42, 2.4)
        zb = z1 - 2.2
        prof = [(0.05, zb), (rb * 0.42, zb - 0.05), (rb * 0.5, zb - rb * 0.5), (rb * 0.62, zb - rb * 1.05), (rb * 0.95, zb - rb * 1.35),
                (rb, zb - rb * 1.45), (rb * 0.9, zb - rb * 1.42), (rb * 0.55, zb - rb * 1.1), (0.05, zb - rb * 1.0)]
        sink.add(mk.revolve(prof, 32, mat=LEAD).translate((x, y, 0.0)))
        sink.add(box_between((x - st.size + t, y), (x + st.size - t, y), 0.5, zb - 0.1, zb + 0.4, WOOD))
    return parts


def build_roof(sink, tw, x, y, z, last, shape):
    kind, h = tw.roof.kind, tw.roof.h
    size = last.size
    if kind == "flat" or h <= 0:
        if kind == "flat":
            sink.add(prism("round" if shape == "round" else shape, size + 0.2, z - 0.3, z + 0.05, x, y, tw.yaw, last.depth, mat=LEAD))
        return
    grow = 0.9 if last.band == "corbel" else (0.5 if last.band in ("crenel", "gallery") else 0.3)
    if kind == "dome":
        r = size + grow
        sink.add(mk.revolve([(r, z), (r * 0.98, z + h * 0.3), (r * 0.85, z + h * 0.62), (r * 0.55, z + h * 0.88), (0.001, z + h)], 48,
                            mat=LEAD).translate((x, y, 0.0)))
        sink.place("c_finial", (x, y, z + h - 0.2), np.eye(3), (0.8, 0.8, 0.8))
        return
    if shape == "round" and kind == "cone":
        rb = size + grow
        m = ck.cone_roof((x, y), size + grow - 0.55, z + 0.2, h, seg=64, eave=0.55, flare=tw.roof.flare, finial=False)
        sink.add(m)
        cone_lucarnes(sink, x, y, rb, z + 0.2, h, tw.roof.lucarnes)
        sink.place("c_finial", (x, y, z + h - 0.15), np.eye(3), (max(0.8, rb / 8.0),) * 3)
        return
    if kind in ("spire", "cone") or shape == "octagon":
        rb = (size + grow) if shape != "square" else (size + grow) * 1.0
        m = ck.cone_roof((x, y), rb - 0.4, z + 0.2, h, seg=8, eave=0.4, flare=0.06, finial=False)
        if shape == "square":
            m.rotate_z(0.0)
        sink.add(m)
        spire_crockets(sink, x, y, rb, z + 0.2, h, 8)
        cone_lucarnes(sink, x, y, rb * 0.95, z + 0.2, h, tw.roof.lucarnes)
        sink.place("c_finial", (x, y, z + h - 0.1), np.eye(3), (max(0.7, rb / 9.0),) * 3)
        return
    # pyramid / helm (square)
    w = 2 * (size + grow)
    d = 2 * ((last.depth if last.depth is not None else size) + grow)
    sink.add(ck._pyramid_roof((x, y), w, d, z + 0.1, h, math.radians(tw.yaw)))
    spire_crockets(sink, x, y, w / 2 * 0.8, z + 0.1, h, 4, tw.yaw + 45.0, every=1.8)
    sink.place("c_finial", (x, y, z + h + 1.0), np.eye(3), (max(0.7, w / 14.0),) * 3)


def build_tower(sink, t: Tower, rng, cuts=()):
    """A tower and its crown / attached towers; `cuts` (manifolds) are taken out of the root tower's body (passages)."""
    for tw, (x, y), base in CP.walk(t):
        sink.set_zone(x, y)
        shape0 = tw.shape
        z = base
        is_root = tw is t
        st0 = tw.stages[0]
        foot = base
        if is_root or base <= Z + 0.5:
            r0 = st0.size
            ring_pts = [(x + r0 * math.cos(a), y + r0 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 13)]
            foot = foot_along(ring_pts, base, max_drop=90.0)
        cutters = list(cuts) if is_root else []
        solids = []
        for si, st in enumerate(tw.stages):
            shape = st.shape or shape0
            z0 = foot if si == 0 else z
            z1 = z + st.h
            if st.band == "arcade":
                solids += belfry(sink, shape, st, x, y, tw.yaw, z, z1, bell=(shape == "square"))
                z = z1
                continue
            body = prism(shape, st.size, z0, z1, x, y, tw.yaw, st.depth, mat=STONE)
            if si == 0 and foot < base - 1.0:              # battered plinth down to the rock
                solids.append(prism(shape, st.size * 1.08, foot, base + 1.0, x, y, tw.yaw, None if st.depth is None else st.depth * 1.08,
                                    mat=STONE, r_top=st.size * 1.01))
            solids.append(body)
            # windows
            if shape in ("round", "octagon") and st.windows < 0:
                pass
            elif shape in ("round", "octagon"):
                n = st.windows or max(3, int(2 * math.pi * st.size / (5.0 if st.size < 9 else 6.0)))
                module = "c_smallwin" if st.size < 4.5 else "c_lancet"
                tower_windows_round(sink, cutters, x, y, st.size, z, z1 - (1.5 if st.band != "string" else 0.5), st.floor_h, n, module,
                                    phase=rng.uniform(0, 1), stagger=st.size < 4.5)
                for zz in np.arange(z + st.floor_h * 2, z1 - 2.0, st.floor_h * 2):
                    solids.append(ring(x, y, st.size - 0.1, st.size + 0.25, zz - 0.2, zz + 0.15, seg=48))
            else:
                d = st.size if st.depth is None else st.depth
                ca, sa = math.cos(math.radians(tw.yaw)), math.sin(math.radians(tw.yaw))
                ax, ay = np.array([ca, sa]), np.array([-sa, ca])
                c = np.array([x, y])
                corners = [c + ax * sx * st.size + ay * sy * d for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                for i in range(4):
                    quoins(sink, corners[i], corners[(i + 1) % 4] - corners[i], corners[i - 1] - corners[i],
                           max(z0, base - 6.0), z1 - 1.6)
                for i in range(4):
                    p0, p1 = corners[i], corners[(i + 1) % 4]
                    e = (p1 - p0) / np.linalg.norm(p1 - p0)
                    n_out = np.array([e[1], -e[0], 0.0])
                    L = float(np.linalg.norm(p1 - p0))
                    module = "c_double" if L > 10.0 else "c_lancet"
                    if st.windows >= 0:
                        facade_windows(sink, cutters, p0 + n_out[:2] * 0.0, p1, n_out, z, z1 - 1.5, st.floor_h * 1.2, module,
                                       bay=max(4.0, L / max(1, int(L / 5.0))), margin=1.8)
                    if st.clock and any(abs(((math.degrees(math.atan2(n_out[1], n_out[0])) - f + 180) % 360) - 180) < 46 for f in tw.faces):
                        q = (p0 + p1) / 2 + n_out[:2] * 0.0
                        dsz = min(6.4, L * 0.7)
                        sink.place("c_clock", (q[0], q[1], z + st.h * 0.42), frame_R(np.cross(-n_out, UP), -n_out), (dsz / 6.4,) * 3)
                        cutters.append(tag(to_world(opening_cutter("c_clock", dsz - 0.6, dsz - 0.6, 0.5),
                                                    np.array([q[0], q[1], z + st.h * 0.42 + 0.3]), np.cross(-n_out, UP), -n_out), TRIM))
                    if si == 0:                                    # corner buttress strips
                        pass
            if st.gables and (st.shape or shape0) == "square":
                solids += stage_gables(sink, cutters, x, y, st.size, tw.yaw, z1)
            # band at the top of the stage
            if shape in ("round", "octagon"):
                solids += round_band(sink, st.band, x, y, st.size, z1)
            else:
                solids += square_band(sink, st.band, x, y, st.size, st.depth, tw.yaw, z1)
            z = z1
        main = [s for s in solids if s is not None]
        if main:
            whole = m3d.Manifold.batch_boolean(main, m3d.OpType.Add)
            sink.add(apply_cuts(whole, cutters))
        last = tw.stages[-1]
        build_roof(sink, tw, x, y, z + (0.6 if last.band == "corbel" else 0.0), last, last.shape or shape0)


# ----------------------------------------------------------------------------------------------------- halls & ranges
def long_block(sink, rng, name, p0, p1, width, wall, base, *, pitch, floor_h, module, parapet, ends, roof="gable", bays=None,
               buttresses=False, tiers=1, dormers=True, chimneys=True, cresting=False, lucarne_rows=0, end_windows=(False, False),
               gable_turrets=(False, False), turret_h=0.0, fleche=0.0, door=False, water_gate=False, end_doors=(False, False)):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / L
    b = np.array([-a[1], a[0]])
    c = (p0 + p1) / 2
    sink.set_zone(*c)
    corners = [p0 - b * width / 2, p1 - b * width / 2, p1 + b * width / 2, p0 + b * width / 2]
    foot = foot_along(corners + [corners[0]], base, max_drop=70.0)
    z_e = base + wall
    body = box_between(p0, p1, width, foot, z_e, STONE)
    solids = [body]
    if foot < base - 1.5:                                         # battered plinth
        solids.append(box_between(p0 - a * 0.4, p1 + a * 0.4, width + 0.8, foot, base + 0.8, STONE))
    cutters = []
    # string courses
    for zz in np.arange(base + floor_h, z_e - 1.0, floor_h):
        solids.append(box_between(p0 - a * 0.2, p1 + a * 0.2, width + 0.5, zz - 0.18, zz + 0.12, TRIM))
    solids.append(box_between(p0 - a * 0.25, p1 + a * 0.25, width + 0.7, base - 0.6, base + 0.2, TRIM))
    # buttresses with set-offs (halls), else windows straight on the facade
    nb = bays or max(1, int(round(L / 4.6)))
    if buttresses:
        for k in range(nb + 1):
            q = p0 + a * (L * k / nb)
            for s in (-1.0, 1.0):
                for j, (dz, dd) in enumerate(((0.0, 2.4), (0.45, 1.7), (0.75, 1.1))):
                    z0b = foot if j == 0 else base + wall * dz
                    z1b = base + wall * (0.45, 0.75, 1.02)[j]
                    solids.append(box_between(q - a * 0.95, q + a * 0.95, dd, z0b, z1b, STONE, offset=s * (width / 2 + dd / 2 - 0.05)))
                    solids.append(box_between(q - a * 1.0, q + a * 1.0, dd + 0.15, z1b - 0.25, z1b, TRIM, offset=s * (width / 2 + dd / 2 - 0.05)))
                qq = q + b * s * (width / 2 + 0.55)
                sink.place("c_pinnacle", (qq[0], qq[1], base + wall * 1.02 - 0.05), frame_R(a3(a), -a3(b * s)), (1.15, 1.15, 1.25))
        # windows: per bay, per tier, both long sides
        for s in (-1.0, 1.0):
            n_out = np.array([b[0] * s, b[1] * s, 0.0])
            for k in range(nb):
                q = p0 + a * (L * (k + 0.5) / nb) + b * s * (width / 2)
                along = np.cross(-n_out, UP)
                bw = L / nb - 2.2
                for tier in range(tiers):
                    if tiers == 1:
                        zt, ht = base + wall * 0.18, wall * 0.62
                    else:
                        zt, ht = base + wall * (0.1 if tier == 0 else 0.52), wall * 0.36
                    mod = "c_double" if bw < 3.4 else "c_hallwin"
                    ww = min(bw, DL.OPENING[mod][0] * (ht / DL.OPENING[mod][1]) ** 0.5 * 1.05)
                    if door and tier == 0 and k == nb // 2 and s < 0:
                        window(sink, cutters, "c_door", (q[0], q[1], base), along, -n_out, 3.4, 5.2)
                        continue
                    window(sink, cutters, mod, (q[0], q[1], zt), along, -n_out, ww, ht)
    else:
        for s in (-1.0, 1.0):
            n_out = np.array([b[0] * s, b[1] * s, 0.0])
            facade_windows(sink, cutters, p0 + b * s * width / 2, p1 + b * s * width / 2, n_out, base, z_e - 0.6, floor_h, module,
                           bay=4.4, door_mid=door and s < 0)
    # parapets / eaves
    for s in (-1.0, 1.0):
        n_out = np.array([b[0] * s, b[1] * s])
        e0, e1 = p0 + b * s * width / 2, p1 + b * s * width / 2
        if parapet in ("crenel", "pierced"):
            corbel_table(sink, e0, e1, n_out, z_e, arches=(parapet == "crenel"))
            solids.append(parapet_line(sink, e0 - a * 0.6, e1 + a * 0.6, n_out, z_e, parapet, over=0.75))
        else:
            solids.append(box_between(e0 - a * 0.3, e1 + a * 0.3, 0.9, z_e - 0.5, z_e + 0.05, TRIM, offset=0.25 * s))
    # ends: a gable wall (copings, crockets, an apex pinnacle), a hipped end (windows only), or an abutting end - a plain
    # gable that closes the roof (hidden where the neighbour is taller, a gable over its roof where it is lower)
    rise = (width / 2) * math.tan(math.radians(pitch))
    for end, (pe, sg) in enumerate(((p0, -1.0), (p1, 1.0))):
        a_out = np.array([a[0] * sg, a[1] * sg, 0.0])
        if ends[end] in ("gable", "none") and roof != "flat":
            gw = ck.gable_wall((pe[0], pe[1], 0.0), a * sg, width, z_e, pitch, thick=1.4, end=-1.0, coping=ends[end] == "gable")
            solids += [g for g in gw if g is not None]
        if ends[end] == "none":
            continue
        if ends[end] == "gable" and roof != "flat":
            gable_crockets(sink, np.array([pe[0], pe[1], z_e]) + a_out * 0.05, np.array([b[0], b[1], 0.0]) * sg, a_out, width / 2 + 0.25,
                           rise + 0.35, 7)
        along = np.cross(-a_out, UP)
        if end == 0 and water_gate:
            # the water gate: a great pointed arch from below the water, the slip running on into the building
            gw_, gh_ = min(width - 4.0, 7.5), 12.5
            P = DL.arch_outline(gw_, gh_, "pointed", 20)
            cutters.append(tag(to_world(DL.extrude_xz(P, -1.0, L * 0.7), np.array([pe[0], pe[1], base - 4.0]), along, -a_out), TRIM))
            ring_m = DL.frame_ring(DL.arch_outline(gw_ + 1.4, gh_ + 0.7, "pointed", 20), 0.7, 0.35, -0.3)
            solids.append(tag(to_world(ring_m, np.array([pe[0], pe[1], base - 4.0]), along, -a_out), TRIM))
            window(sink, cutters, "c_rose", np.array([pe[0], pe[1], z_e + rise * 0.12]), along, -a_out, min(width * 0.3, 4.0),
                   min(width * 0.3, 4.0))
        elif end_doors[end]:
            window(sink, cutters, "c_door", (pe[0], pe[1], base), along, -a_out, 3.2, 4.8)
            window(sink, cutters, "c_rose", np.array([pe[0], pe[1], z_e + rise * 0.12]), along, -a_out, min(width * 0.3, 4.0),
                   min(width * 0.3, 4.0))
        elif end_windows[end]:
            if width >= 24:
                window(sink, cutters, "c_hallwin", (pe[0], pe[1], base + wall * 0.22) + a_out * 0.0, along, -a_out,
                       min(width * 0.36, 7.0), min(wall * 0.95, 24.0))
                window(sink, cutters, "c_rose", np.array([pe[0], pe[1], z_e + rise * 0.18]), along, -a_out, min(width * 0.32, 8.0), min(width * 0.32, 8.0))
            else:
                window(sink, cutters, "c_double", (pe[0], pe[1], base + wall * 0.3), along, -a_out, min(width * 0.3, 3.0), wall * 0.55)
        else:
            facade_windows(sink, cutters, pe - b * width / 2, pe + b * width / 2, a_out, base, z_e - 0.6, floor_h, module, bay=4.4, margin=2.0)
        if gable_turrets[end]:
            th = turret_h or wall * 1.9
            for s in (-1.0, 1.0):
                q = pe + a * sg * 1.2 + b * s * (width / 2 + 0.8)
                tt = Tower(f"{name}_GT", (q[0], q[1]), [CP.Stage(th, 3.0, "string", floor_h=4.0), CP.Stage(3.0, 3.0, "crenel")],
                           CP.Roof("spire", 14.0, 0, 2.0), "octagon", base=base)
                build_tower(sink, tt, rng)
                sink.set_zone(*c)
    for end, (pe, sg) in enumerate(((p0, -1.0), (p1, 1.0))):     # quoins on the free corners
        if ends[end] == "none":
            continue
        for s_ in (-1.0, 1.0):
            quoins(sink, pe + b * s_ * width / 2, -a * sg, -b * s_, max(foot, base - 6.0), z_e - 0.3)
    whole = m3d.Manifold.batch_boolean([s for s in solids if s is not None], m3d.OpType.Add)
    sink.add(apply_cuts(whole, cutters))
    # roof
    if roof == "flat":
        sink.add(box_between(p0, p1, width - 0.4, z_e - 0.4, z_e - 0.1, LEAD))
        return
    rm, z_r, r0, r1 = slate_roof(c, a, L, width, z_e, pitch, overhang=0.45 if parapet == "none" else 0.15, ends=ends)
    sink.add(rm)
    ua = -L / 2 + (width / 2 if ends[0] == "hip" else 0.0)       # dormers stay off the hipped ends
    ub = L / 2 - (width / 2 if ends[1] == "hip" else 0.0)
    if ub - ua > 6.0:
        cd = c + a * (ua + ub) / 2
        if dormers:
            roof_dormers(sink, cd, a, ub - ua, width, z_e, pitch, "c_dormer", spacing=8.5, up_frac=0.22, scale=0.9 if width < 20 else 1.0)
        if lucarne_rows:
            roof_dormers(sink, cd, a, ub - ua, width, z_e, pitch, "c_lucarne", spacing=6.5, up_frac=0.45, scale=1.2, rows=lucarne_rows)
    if r1 - r0 > 2.0:
        ridge_items(sink, c + a * (r0 + r1) / 2, a, r1 - r0, z_r, cresting=cresting,
                    chimneys=(max(1, int((r1 - r0) / 18.0)) if chimneys else 0), rng=rng)
    if fleche:
        tt = Tower(f"{name}_Fleche", (c[0], c[1]), [CP.Stage(5.0, 1.6, "string", floor_h=2.5)], CP.Roof("spire", fleche, 0, 2.0), "octagon",
                   base=z_r - 2.0)
        build_tower(sink, tt, rng)
        sink.set_zone(*c)


def a3(v):
    return np.array([v[0], v[1], 0.0])


def quoins(sink, corner, e1, e2, z0, z1):
    """Dressed corner stones up a convex corner from z0 to z1; e1, e2 run along the two faces away from the corner."""
    e1, e2 = a3(e1) / np.linalg.norm(e1[:2]), a3(e2) / np.linalg.norm(e2[:2])
    if np.cross(e1, e2)[2] < 0:
        e1, e2 = e2, e1
    R = np.stack([e1, e2, UP], 1)
    for k in range(max(0, int((z1 - z0) / 0.9))):
        sink.place("c_quoin", (corner[0], corner[1], z0 + 0.9 * k), R)


def build_hall(sink, h: Hall, rng):
    long_block(sink, rng, h.name, h.p0, h.p1, h.width, h.wall, h.base, pitch=h.pitch, floor_h=h.wall / 2.0, module="c_double",
               parapet=h.parapet, ends=("gable", "gable"), bays=h.bays, buttresses=True, tiers=h.tiers, dormers=False, chimneys=False,
               cresting=True, lucarne_rows=h.lucarnes, end_windows=h.end_window, gable_turrets=h.gable_turrets, turret_h=h.turret_h,
               fleche=h.fleche, door=True)


def build_range(sink, r: Range, rng):
    long_block(sink, rng, r.name, r.p0, r.p1, r.depth, r.wall, r.base, pitch=r.pitch, floor_h=r.floor_h,
               module="c_tudor" if r.win != "lancet" else ("c_tudor" if r.wall < 26 else "c_lancet"), parapet=r.parapet,
               ends=tuple(e if e in ("gable", "hip") else "none" for e in r.ends), roof=r.roof, dormers=r.dormers, chimneys=r.chimneys)


def build_gate(sink, g: Gatehouse, rng):
    ya = math.radians(g.yaw)
    ax = np.array([math.cos(ya), math.sin(ya)])
    nx = np.array([-ax[1], ax[0]])
    c = np.asarray(g.at, float)
    long_block(sink, rng, g.name, c - nx * g.d / 2, c + nx * g.d / 2, g.w, g.h, g.base, pitch=50.0, floor_h=5.0, module="c_lancet",
               parapet="crenel", ends=("none", "none"), roof="flat", dormers=False, chimneys=False)
    # the passage: doors on both faces
    sink.set_zone(*c)
    th = g.turret_h or g.h + 8.0
    for s in (-1, 1):
        q = c - ax * g.w / 2 + nx * s * g.d / 2
        build_tower(sink, CP.round_tower(f"{g.name}_T", (q[0], q[1]), g.turret_r, th, g.turret_r * 2.8, base=g.base, band="corbel"), rng)
    for s in (-1.0, 1.0):
        q = c + ax * s * (g.w / 2)
        sink.place("c_door", (q[0], q[1], g.base), frame_R(a3(nx) * s, -a3(ax) * s), (1.4, 1.0, 1.4))


def build_wall(sink, w: Wall, rng):
    P = np.asarray(w.path, float)
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        sink.set_zone(*((p0 + p1) / 2))
        foot = foot_along([p0, p1], w.base, max_drop=60.0)
        e = (p1 - p0) / np.linalg.norm(p1 - p0)
        n_out = np.array([e[1], -e[0]])
        solids = [box_between(p0, p1, w.thick, foot, w.base + w.h, STONE)]
        if w.machicolations:
            corbel_table(sink, p0, p1, n_out, w.base + w.h)
        solids.append(parapet_line(sink, p0, p1, n_out, w.base + w.h, "crenel", over=0.8 if w.machicolations else 0.0))
        sink.add(m3d.Manifold.batch_boolean([s for s in solids if s is not None], m3d.OpType.Add))


def build_viaduct(sink, v: Viaduct, rng):
    """The grand viaduct: per segment two tiers of arches (the lower spanning two upper ones) on piers with cutwaters,
    string courses, an arcaded parapet with balusters, lamp turrets over the piers."""
    pts = [np.asarray(p, float) for p in v.path]
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        sink.set_zone(*((p0 + p1) / 2))
        L = float(np.linalg.norm(p1 - p0))
        a = (p1 - p0) / L
        b = np.array([-a[1], a[0]])
        samples = np.array([p0 + a * L * t for t in np.linspace(0, 1, 41)])
        zlow = float(ground(samples[:, 0], samples[:, 1]).min()) - 3.0
        z_t = zlow + (v.deck - zlow) * 0.5
        body = box_between(p0 - a * 0.5, p1 + a * 0.5, v.width, zlow, v.deck, STONE)
        cut = []
        yaw = math.degrees(math.atan2(a[1], a[0]))
        nu = max(1, int(round(L / v.span)))
        span = L / nu
        for k in range(nu):
            cc = p0 + a * span * (k + 0.5)
            w = (span - v.pier) / 2
            spring = v.deck - 3.6 - w
            arch = m3d.Manifold.cylinder(v.width + 2, w, w, 40, True).rotate([90, 0, 0]).rotate([0, 0, yaw]).translate([cc[0], cc[1], spring])
            bx = m3d.Manifold.cube([2 * w, v.width + 2, spring - z_t], True).rotate([0, 0, yaw]).translate([cc[0], cc[1], (spring + z_t) / 2])
            cut.append(tag(arch + bx, TRIM))
        nl = max(1, nu // 2)
        spl = L / nl
        for k in range(nl):
            cc = p0 + a * spl * (k + 0.5)
            w = (spl - v.pier * 1.7) / 2
            spring = z_t - 2.2 - w
            arch = m3d.Manifold.cylinder(v.width + 2, w, w, 48, True).rotate([90, 0, 0]).rotate([0, 0, yaw]).translate([cc[0], cc[1], spring])
            bx = m3d.Manifold.cube([2 * w, v.width + 2, spring - zlow + 2], True).rotate([0, 0, yaw]).translate([cc[0], cc[1], (spring + zlow - 2) / 2])
            cut.append(tag(arch + bx, TRIM))
        solids = [body]
        # cutwaters (pointed) on the lower piers, both faces
        for k in range(nl + 1):
            cc = p0 + a * spl * k
            for s in (-1.0, 1.0):
                tri = m3d.CrossSection([np.array([[-1.6, 0.0], [1.6, 0.0], [0.0, 2.2]])]).extrude(z_t - zlow - 1.0)
                M = np.zeros((3, 4))
                M[:, 0] = a3(a)
                M[:, 1] = a3(b * s)
                M[:, 2] = UP
                M[:, 3] = [cc[0] + b[0] * s * v.width / 2, cc[1] + b[1] * s * v.width / 2, zlow]
                if np.linalg.det(M[:, :3]) < 0:
                    M[:, 0] = -M[:, 0]
                solids.append(tag(tri.transform(M), STONE))
        for zz in (z_t, v.deck - 0.3):
            solids.append(box_between(p0 - a * 0.5, p1 + a * 0.5, v.width + 0.6, zz - 0.25, zz + 0.25, TRIM))
        solids.append(box_between(p0 - a * 0.5, p1 + a * 0.5, v.width - 0.3, v.deck - 0.1, v.deck + 0.06, PAVE))   # the paved deck
        whole = apply_cuts(m3d.Manifold.batch_boolean(solids, m3d.OpType.Add), cut)
        sink.add(whole)
        # arcaded parapet: small arches on columns between a coping and a plinth course
        for s in (-1.0, 1.0):
            e0, e1 = p0 + b * s * (v.width / 2 - 0.3), p1 + b * s * (v.width / 2 - 0.3)
            n_cols = max(2, int(L / 1.5))
            for k in range(n_cols + 1):
                q = e0 + (e1 - e0) * (k / n_cols)
                sink.place("c_column", (q[0], q[1], v.deck + 0.25), np.eye(3), (0.55, 0.55, 0.42))
            sink.add(box_between(e0, e1, 0.55, v.deck, v.deck + 0.25, TRIM))
            sink.add(box_between(e0, e1, 0.65, v.deck + 1.95, v.deck + 2.25, TRIM))
        for k in range(1, nu):
            q = p0 + a * span * k
            for s in (-1.0, 1.0):
                qq = q + b * s * (v.width / 2 + 0.5)
                sink.place("c_bartizan", (qq[0], qq[1], v.deck - 2.6), np.eye(3), (0.6, 0.6, 0.55))


def build_archbridge(sink, br: ArchBridge, rng):
    p0, p1 = np.asarray(br.p0, float), np.asarray(br.p1, float)
    sink.set_zone(*((p0 + p1) / 2))
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / L
    b = np.array([-a[1], a[0]])
    yaw = math.degrees(math.atan2(a[1], a[0]))
    zlow = min(float(ground([p0[0]], [p0[1]])[0]), float(ground([p1[0]], [p1[1]])[0]), br.deck - 30.0)
    body = box_between(p0 - a * 2.0, p1 + a * 2.0, br.width, zlow, br.deck, STONE)
    cut = []
    span = L / br.arches
    for k in range(br.arches):
        cc = p0 + a * span * (k + 0.5)
        r = span * 0.46
        arch = m3d.Manifold.cylinder(br.width + 2, r, r, 48, True).rotate([90, 0, 0]).rotate([0, 0, yaw]).translate([cc[0], cc[1], br.deck - 3.2 - r])
        bx = m3d.Manifold.cube([2 * r, br.width + 2, br.deck - 3.2 - r - zlow + 1], True).rotate([0, 0, yaw]).translate(
            [cc[0], cc[1], (br.deck - 3.2 - r + zlow - 1) / 2])
        cut.append(tag(arch + bx, TRIM))
    solids = [body, box_between(p0 - a * 2.0, p1 + a * 2.0, br.width + 0.6, br.deck - 0.5, br.deck - 0.1, TRIM),
              box_between(p0 - a * 2.0, p1 + a * 2.0, br.width - 0.2, br.deck - 0.15, br.deck + 0.02, PAVE)]
    sink.add(apply_cuts(m3d.Manifold.batch_boolean(solids, m3d.OpType.Add), cut))
    for s in (-1.0, 1.0):
        n_out = b * s
        e0, e1 = p0 - a * 2.0 + b * s * (br.width / 2 - 0.2), p1 + a * 2.0 + b * s * (br.width / 2 - 0.2)
        sink.add(parapet_line(sink, e0, e1, n_out, br.deck + 0.3, "pierced", over=0.0, thick=0.4))


def build_suspension(sink, br: SuspensionBridge, rng):
    """A covered timber walkway hung on iron chains between two stone pylons."""
    p0, p1 = np.asarray(br.p0, float), np.asarray(br.p1, float)
    sink.set_zone(*((p0 + p1) / 2))
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / L
    b = np.array([-a[1], a[0]])
    sag = L * 0.035
    n = 24
    ts = np.linspace(0, 1, n + 1)
    deck_z = br.deck - sag * 4 * ts * (1 - ts)
    for i in range(n):
        q0, q1 = p0 + a * L * ts[i], p0 + a * L * ts[i + 1]
        z0 = (deck_z[i] + deck_z[i + 1]) / 2
        sink.add(box_between(q0, q1 + a * 0.05, br.width, z0 - 0.35, z0, WOOD))
        for s in (-1.0, 1.0):
            sink.add(box_between(q0, q1 + a * 0.05, 0.12, z0 + 1.0, z0 + 1.12, WOOD, offset=s * br.width / 2))
            sink.add(box_between(q0, q0 + a * 0.15, 0.15, z0, z0 + 2.6, WOOD, offset=s * br.width / 2))
        rm, _ = ck.gable_roof(((q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2, 0.0), a, L / n + 0.06, br.width + 0.4, z0 + 2.6, 35.0, overhang=0.3,
                              ridge=False)
        sink.add(rm)
    for q in (p0, p1):
        foot = float(ground([q[0]], [q[1]])[0]) - 2.0
        tt = Tower("SuspensionPylon", (q[0], q[1]), [CP.Stage(br.deck + br.pylon_h - foot, 3.2, "crenel", floor_h=5.0)], CP.Roof("pyramid", 7.0, 0, 1.5),
                   "square", base=foot, yaw=math.degrees(math.atan2(a[1], a[0])))
        build_tower(sink, tt, rng)
    # the chains: catenaries from pylon top to pylon top, with hangers
    for s in (-1.0, 1.0):
        for i in range(n):
            t0, t1 = ts[i], ts[i + 1]
            zc0 = br.deck + br.pylon_h - 2.0 - (br.pylon_h + 1.0) * 4 * t0 * (1 - t0)
            zc1 = br.deck + br.pylon_h - 2.0 - (br.pylon_h + 1.0) * 4 * t1 * (1 - t1)
            q0 = p0 + a * L * t0 + b * s * (br.width / 2 + 0.3)
            q1 = p0 + a * L * t1 + b * s * (br.width / 2 + 0.3)
            seg = mk.sweep(mk.ngon(0.12, 6), np.array([[q0[0], q0[1], zc0], [q1[0], q1[1], zc1]]), np.array([a3(b), a3(b)]), mat=LEAD)
            sink.add(seg)
            if 0 < i:
                sink.add(box_between(q0, q0 + a * 0.06, 0.06, deck_z[i], zc0, LEAD))


def build_stair(sink, st: Stair, rng):
    """The entry stairs, after the Great Wall: walled flights on their benches up the bay's north face (world.py cuts
    the benches), the outer wall a battered retaining wall down to the rock with a crenellated parapet, a plain inner
    wall with lanterns, a square watchtower astride every flight (the stair runs through it), a pointed arch where each
    flight bridges the stream's cleft (the water falls through it), round towers at the turns, a paved quay below."""
    Wd = st.width
    t = st.wall_t
    off = Wd / 2 + t / 2
    span = 9.0                                                       # the arch over the cleft
    for fi, (xa, xb, yc, za, zb) in enumerate(st.flights):
        sink.set_zone((xa + xb) / 2, yc)
        sx = 1.0 if xb > xa else -1.0
        L = abs(xb - xa)
        rise = zb - za

        def deck(x, xa=xa, za=za, sx=sx, L=L, rise=rise):
            return za + rise * np.clip((x - xa) * sx / L, 0.0, 1.0)
        on_cleft = st.cleft_x is not None and min(xa, xb) < st.cleft_x < max(xa, xb)

        def over_cleft(x0, x1):
            return on_cleft and min(x0, x1) - 1.0 < st.cleft_x + span / 2 + 1.5 and max(x0, x1) + 1.0 > st.cleft_x - span / 2 - 1.5
        solids = []
        n = max(4, int(abs(rise) / 0.17))
        tread = L / n
        for k in range(n):                                           # the steps
            x0, x1 = xa + sx * tread * k, xa + sx * tread * (k + 1)
            z = za + rise * (k + 1) / n
            solids.append(box_between((x0, yc), (x1 + sx * 0.03, yc), Wd, z - 0.75, z, PAVE))
        nb = max(2, int(round(L / 3.0)))
        for k in range(nb):                                          # the walls, bay by bay
            x0, x1 = xa + sx * L * k / nb, xa + sx * L * (k + 1) / nb
            zlo, zhi = min(deck(x0), deck(x1)), max(deck(x0), deck(x1))
            xs_ = np.linspace(min(x0, x1), max(x0, x1), 4)
            bridge = over_cleft(x0, x1)
            # outer wall: battered retaining wall in two set-offs down to the rock, a parapet with merlons
            yo = yc - off
            g = float(ground(xs_, np.full(4, yo - 0.8)).min())
            foot = zlo - 1.0 if bridge else min(g, zlo - 0.8) - 1.0
            solids.append(box_between((x0, yo), (x1 + sx * 0.02, yo), t, foot, zhi + 1.3, STONE))
            if not bridge and zlo - foot > 3.0:
                zm = foot + 0.45 * (zlo - 0.6 - foot)
                solids.append(box_between((x0, yo - 0.6), (x1 + sx * 0.02, yo - 0.6), t + 1.2, foot, zm, STONE))
                solids.append(box_between((x0, yo - 0.6), (x1 + sx * 0.02, yo - 0.6), t + 1.4, zm - 0.3, zm, TRIM))
                solids.append(box_between((x0, yo - 0.3), (x1 + sx * 0.02, yo - 0.3), t + 0.6, zm, zlo - 0.6, STONE))
                if zlo - foot > 7.0 and k % 2 == 0:                  # counterforts on the tall retaining walls
                    xb = x0
                    for (z0_, z1_, pr) in ((foot, zm + 0.4, 2.6), (zm + 0.4, zlo - 1.6, 1.7)):
                        yb = yo - t / 2 - pr / 2 + 0.3
                        solids.append(box_between((xb - 0.95, yb), (xb + 0.95, yb), pr, z0_, z1_, STONE))
                        solids.append(box_between((xb - 1.05, yb), (xb + 1.05, yb), pr + 0.2, z1_ - 0.25, z1_, TRIM))
            solids.append(box_between((x0, yo - 0.15), (x1 + sx * 0.02, yo - 0.15), t + 0.5, zlo - 0.9, zlo - 0.5, TRIM))
            solids.append(box_between((x0, yo), (x1 + sx * 0.02, yo), t + 0.16, zhi + 1.3, zhi + 1.45, TRIM))
            for j in (0.25, 0.75):                                   # two merlons per bay
                xm = x0 + (x1 - x0) * j
                sink.place("c_merlon", (xm, yo - t / 2, deck(xm) + 1.45), np.eye(3), (0.8, t / 0.75, 0.75))
            # inner wall against the bank, a coping, a lantern every fourth bay
            yi = yc + off
            g = float(ground(xs_, np.full(4, yi + 0.6)).min())
            foot_i = zlo - 1.0 if bridge else min(g, zlo - 0.8) - 0.5
            solids.append(box_between((x0, yi), (x1 + sx * 0.02, yi), t, foot_i, zhi + 1.5, STONE))
            solids.append(box_between((x0, yi), (x1 + sx * 0.02, yi), t + 0.16, zhi + 1.5, zhi + 1.65, TRIM))
            if k % 4 == 2:
                xm = (x0 + x1) / 2
                sink.place("c_lantern", (xm, yi, deck(xm) + 1.65), np.eye(3))
        # under the flight: masonry where the bench falls short
        nseg = max(2, int(L / 4.0))
        for k in range(nseg):
            x0, x1 = xa + sx * L * k / nseg, xa + sx * L * (k + 1) / nseg
            if over_cleft(x0, x1):
                continue
            xs_ = np.linspace(min(x0, x1), max(x0, x1), 4)
            g = float(ground(xs_, np.full(4, yc)).min())
            zlo = min(deck(x0), deck(x1)) - 0.7
            if g < zlo - 0.3:
                solids.append(box_between((x0, yc), (x1 + sx * 0.02, yc), Wd + 0.1, g - 1.0, zlo, STONE))
        # the bridge over the cleft: a pointed arch with a moulded ring on both faces, the stream falls through it
        if on_cleft:
            cx = st.cleft_x
            zd = float(deck(cx))
            bed = float(W.fall_bed(yc))
            crown = zd - 1.8
            wide = Wd + 2 * t + 1.6
            body = box_between((cx - span / 2 - 2.5, yc), (cx + span / 2 + 2.5, yc), wide, bed - 2.0, zd - 0.6, STONE)
            P = DL.arch_outline(span, crown - (bed - 3.0), "pointed", 20)
            cut = tag(to_world(DL.extrude_xz(P, -1.0, wide + 2.0), np.array([cx, yc - wide / 2 - 1.0, bed - 3.0]),
                               np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])), TRIM)
            solids.append(apply_cuts(body, [cut]))
            for s_ in (-1.0, 1.0):
                yf = yc + s_ * wide / 2
                ring_m = DL.frame_ring(DL.arch_outline(span + 1.8, crown - (bed - 3.0) + 0.9, "pointed", 20), 0.9, 0.4, -0.35)
                solids.append(tag(to_world(ring_m, np.array([cx, yf, bed - 3.0]), np.array([-s_, 0.0, 0.0]), np.array([0.0, -s_, 0.0])),
                                  TRIM))
                for e in (-1.0, 1.0):                                # buttresses flanking the arch
                    xq = cx + e * (span / 2 + 1.6)
                    solids.append(box_between((xq - 1.0, yf + s_ * 0.6), (xq + 1.0, yf + s_ * 0.6), 1.2, bed - 2.0, zd - 2.2, STONE))
        sink.add(m3d.Manifold.batch_boolean([q for q in solids if q is not None], m3d.OpType.Add))
        # the watchtower astride the flight, away from the cleft: the stair runs through it under pointed arches
        if L > 60.0:
            xt = xa + sx * L * (0.3 if fi % 2 == 0 else 0.7)
            if on_cleft and abs(xt - st.cleft_x) < 22.0:
                xt = xa + sx * L * (0.7 if fi % 2 == 0 else 0.3)
            zt = float(deck(xt))
            half = Wd / 2 + t + 0.6
            P = DL.arch_outline(Wd - 0.2, 5.4, "pointed", 16)
            tunnel = tag(to_world(DL.extrude_xz(P, -half - 1.5, half + 1.5), np.array([xt, yc, zt - 1.0]), np.array([0.0, -1.0, 0.0]),
                                  np.array([1.0, 0.0, 0.0])), TRIM)
            wt = Tower(f"{st.name}_Watch{fi}", (xt, yc), [CP.Stage(6.6, half, "string", windows=-1),
                                                          CP.Stage(8.4, half, "corbel", floor_h=3.6)],
                       CP.Roof("pyramid", 8.5, 0, 2.0), "square", base=zt - 0.6)
            build_tower(sink, wt, rng, cuts=[tunnel])
            for e in (-1.0, 1.0):                                    # arch rings round the passage on both faces
                ring_m = DL.frame_ring(DL.arch_outline(Wd + 1.0, 6.0, "pointed", 16), 0.55, 0.3, -0.25)
                sink.add(tag(to_world(ring_m, np.array([xt + e * half, yc, zt - 1.0]), np.array([0.0, e, 0.0]), np.array([-e, 0.0, 0.0])),
                             TRIM))
    # landings: bastions from the rock up, crenellated parapets on the open sides, a round tower on the outer corner
    for li, (x0, x1, y0, y1, z, side) in enumerate(st.landings):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sink.set_zone(cx, cy)
        if z < 3.0:                                                  # the boathouse quay: a paved stone platform
            foot = foot_along([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], z, max_drop=12.0)
            sink.add(box_between((x0, cy), (x1, cy), y1 - y0, foot, z - 0.2, STONE))
            sink.add(box_between((x0 - 0.2, cy), (x1 + 0.2, cy), y1 - y0 + 0.4, z - 0.2, z, PAVE))
            for k in range(int((x1 - x0) / 4.0) + 1):                # bollards and lanterns along the quay edge
                sink.place("c_finial", (x0 + 2.0 + 4.0 * k, y0 + 0.6, z), np.eye(3), (0.35, 0.35, 0.3))
                if k % 2 == 1:
                    sink.place("c_lantern", (x0 + 2.0 + 4.0 * k, y0 + 1.6, z), np.eye(3))
            continue
        foot = foot_along([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], z, max_drop=60.0)
        solids = [box_between((x0, cy), (x1, cy), y1 - y0, foot, z - 0.15, STONE),
                  box_between((x0 - 0.3, cy), (x1 + 0.3, cy), y1 - y0 + 0.6, z - 0.15, z + 0.05, PAVE)]
        solids.append(box_between((x0 - 1.0, cy), (x1 + 1.0, cy), y1 - y0 + 2.0, foot, foot + min(8.0, z - foot) * 0.6, STONE))
        edges = {"S": ((x0, y0), (x1, y0), (0.0, -1.0)), "N": ((x1, y1), (x0, y1), (0.0, 1.0)),
                 "W": ((x0, y1), (x0, y0), (-1.0, 0.0)), "E": ((x1, y0), (x1, y1), (1.0, 0.0))}
        for k, (e0, e1, nrm) in edges.items():
            if k == side:
                continue
            e0, e1, nrm = np.array(e0), np.array(e1), np.array(nrm)
            corbel_table(sink, e0, e1, nrm, z + 0.0, spacing=1.4)
            solids.append(parapet_line(sink, e0, e1, nrm, z + 1.0, "crenel", over=0.4, thick=0.8))
            solids.append(box_between(e0, e1, 0.8, z - 0.1, z + 0.8, STONE))
            for j in (0.3, 0.7):
                q = e0 + (e1 - e0) * j - nrm * 1.2
                sink.place("c_lantern", (q[0], q[1], z), np.eye(3))
        sink.add(m3d.Manifold.batch_boolean([q for q in solids if q is not None], m3d.OpType.Add))
        tx = x0 if side == "E" else x1                               # the round tower on the outer lake-side corner
        q = (tx + (-2.5 if side == "E" else 2.5), y0 - 1.5)
        tt = Tower(f"{st.name}_T{li}", q, [CP.Stage(z + 4.0 - foot, 5.0, "string", floor_h=5.0),
                                           CP.Stage(9.0, 5.0, "corbel", floor_h=4.0)], CP.Roof("cone", 13.0, 1, 2.0),
                   "round", base=foot)
        build_tower(sink, tt, rng)


def build_boathouse(sink, bh: Boathouse, rng):
    """The boathouse in its cove, half over the water beside its quay: a buttressed Gothic hall, a great pointed water gate
    in the lake gable (the boats glide in under it, the slip runs on inside), lancets, a door to the quay at the back,
    a bellcote spire on the ridge."""
    ya = math.radians(bh.yaw)
    ax = np.array([math.cos(ya), math.sin(ya)])
    c = np.asarray(bh.at, float)
    p0, p1 = c - ax * bh.length / 2, c + ax * bh.length / 2          # p0: the lake end
    sink.set_zone(*c)
    long_block(sink, rng, bh.name, p0, p1, bh.width, 9.0, bh.base, pitch=58.0, floor_h=4.5, module="c_lancet", parapet="none",
               ends=("gable", "gable"), bays=5, buttresses=True, tiers=1, dormers=False, chimneys=False, cresting=True,
               water_gate=True, end_doors=(False, True))
    tt = Tower("BoathouseBellcote", (p1[0] - ax[0] * 4.0, p1[1] - ax[1] * 4.0), [CP.Stage(6.0, 1.8, "string", floor_h=3.0)],
               CP.Roof("spire", 10.0, 0, 1.5), "square", base=bh.base + 9.0 + (bh.width / 2) * math.tan(math.radians(58.0)) * 0.7,
               yaw=bh.yaw)
    build_tower(sink, tt, rng)


def build_glasshouse(sink, g: Glasshouse, rng):
    p0, p1 = np.asarray(g.p0, float), np.asarray(g.p1, float)
    sink.set_zone(*((p0 + p1) / 2))
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / L
    b = np.array([-a[1], a[0]])
    foot = foot_along([p0, p1], g.base, max_drop=6.0)
    sink.add(box_between(p0, p1, g.width, foot, g.base + 1.0, STONE))
    zw = g.base + g.h * 0.55
    for s in (-1.0, 1.0):                                         # glazed walls with iron mullions
        sink.add(box_between(p0, p1, 0.06, g.base + 1.0, zw, 24, offset=s * g.width / 2))
        n = int(L / 1.2)
        for k in range(n + 1):
            q = p0 + a * (L * k / n)
            sink.add(box_between(q, q + a * 0.08, 0.12, g.base + 1.0, zw, LEAD, offset=s * g.width / 2))
    rm, z_r = ck.gable_roof(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0.0), a, L, g.width, zw, 38.0, overhang=0.2, ridge=True)
    rm.mat[rm.mat == SLATE] = 24
    sink.add(rm)
    for s in (-1.0, 1.0):
        n = int(L / 1.2)
        for k in range(n + 1):
            q = p0 + a * (L * k / n)
            sink.add(box_between(q, q + a * 0.08, 0.1, zw, zw + 0.3, LEAD, offset=s * g.width / 2))
    for pe in (p0, p1):
        sink.add(box_between(pe - b * g.width / 2, pe + b * g.width / 2, 0.06, g.base + 1.0, zw, 24))


def beam3(pa, pb, w, d, mat=WOOD):
    """A timber of section w x d between two 3D points (w horizontal, d in the vertical plane through the timber)."""
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    e = pb - pa
    L = float(np.linalg.norm(e))
    e /= L
    h = np.cross(UP, e) if abs(e[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    h /= np.linalg.norm(h)
    v = np.cross(e, h)
    M = np.zeros((3, 4))
    M[:, 0], M[:, 1], M[:, 2], M[:, 3] = e, h, v, (pa + pb) / 2
    return tag(m3d.Manifold.cube([L, w, d], True).transform(M), mat)


def build_covered_bridge(sink, br: CoveredBridge, rng):
    """The covered timber bridge: a gallery (planked lower walls with braces, a band of open bays between posts, a planked
    frieze, tie beams, board gables and a slate roof) on battered stone piers in the gorge and X-braced timber trestles
    on the banks; lanterns along it."""
    p0, p1 = np.asarray(br.p0, float), np.asarray(br.p1, float)
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / L
    b = np.array([-a[1], a[0]])
    a3_, b3_ = a3(a), a3(b)
    sink.set_zone(*((p0 + p1) / 2))
    Wd = br.width
    zd = max(br.z0, br.z1)
    ts = np.linspace(0, 1, int(L) + 1)
    P = p0[None] + (p1 - p0)[None] * ts[:, None]
    g = np.minimum(ground(P[:, 0] + b[0] * Wd / 2, P[:, 1] + b[1] * Wd / 2), ground(P[:, 0] - b[0] * Wd / 2, P[:, 1] - b[1] * Wd / 2))
    drop = zd - g
    solids = []
    nb = max(2, int(round(L / 2.0)))
    for k in range(nb):                                              # the gallery, bay by bay
        q0, q1 = p0 + a * L * k / nb, p0 + a * L * (k + 1) / nb
        solids.append(box_between(q0, q1 + a * 0.02, Wd + 0.9, zd - 0.5, zd, WOOD))
        for s_ in (-1.0, 1.0):
            o = s_ * (Wd / 2 + 0.2)
            solids.append(box_between(q0, q1 + a * 0.02, 0.12, zd, zd + 1.1, WOOD, offset=o))
            solids.append(box_between(q0, q1 + a * 0.02, 0.26, zd + 1.1, zd + 1.24, WOOD, offset=o))
            solids.append(box_between(q0, q1 + a * 0.02, 0.12, zd + 2.35, zd + 2.9, WOOD, offset=o))
            solids.append(box_between(q0, q1 + a * 0.02, 0.32, zd + 2.9, zd + 3.12, WOOD, offset=o))
            solids.append(box_between(q0, q0 + a * 0.26, 0.3, zd - 0.5, zd + 2.95, WOOD, offset=o))
            ob = b3_ * (o + s_ * 0.1)                                # a brace across the planking
            if k % 2 == 0:
                solids.append(beam3(a3(q0) + ob + UP * (zd + 0.1), a3(q1) + ob + UP * (zd + 1.05), 0.08, 0.2))
            else:
                solids.append(beam3(a3(q0) + ob + UP * (zd + 1.05), a3(q1) + ob + UP * (zd + 0.1), 0.08, 0.2))
        if k % 2 == 0:
            solids.append(box_between(q0 - b * (Wd / 2 + 0.45), q0 + b * (Wd / 2 + 0.45), 0.24, zd + 2.95, zd + 3.2, WOOD))
        if k % 4 == 1:
            qm = (q0 + q1) / 2 + b * (Wd / 2 - 0.2) * (1 if k % 8 == 1 else -1)
            sink.place("c_lantern", (qm[0], qm[1], zd + 1.24), np.eye(3), (0.55, 0.55, 0.55))
    for pe, sg in ((p0, -1.0), (p1, 1.0)):                          # board gables
        gw = ck.gable_wall((pe[0], pe[1], 0.0), a * sg, Wd + 0.6, zd + 3.12, 42.0, thick=0.2, end=-1.0, coping=False, mat=WOOD)
        solids += [q for q in gw if q is not None]
    # supports: stone piers in the gorge, trestles on the banks, abutments at the ends
    deep = np.where(drop > 12.0)[0]
    piers = []
    if len(deep):
        ta, tb = ts[deep[0]], ts[deep[-1]]
        piers = [ta + (tb - ta) * f for f in ((1 / 3, 2 / 3) if (tb - ta) * L > 24.0 else (0.5,))]
    for tp in piers:
        q = p0 + (p1 - p0) * tp
        gq = float(ground([q[0]], [q[1]])[0])
        h = zd - 0.5 - gq
        pts = []
        for zz, grow in ((gq - 1.5, 0.06 * h), (zd - 0.5, 0.0)):
            for u in (-1.3 - grow, 1.3 + grow):
                for v in (-(Wd / 2 + 0.8 + grow), Wd / 2 + 0.8 + grow):
                    c = q + a * u + b * v
                    pts.append([c[0], c[1], zz])
        solids.append(tag(m3d.Manifold.hull_points(np.array(pts)), STONE))
        solids.append(box_between(q - a * 1.6, q + a * 1.6, Wd + 2.2, zd - 1.1, zd - 0.5, TRIM))
        for zz in np.arange(gq + 6.0, zd - 3.0, 6.0):
            grow = 0.06 * (zd - 0.5 - zz)
            solids.append(box_between(q - a * (1.45 + grow), q + a * (1.45 + grow), Wd + 1.9 + 2 * grow, zz - 0.25, zz + 0.05, TRIM))
    for k in range(int(L / 6.0) + 1):
        tk = min(1.0, k * 6.0 / L)
        if any(abs(tk - tp) * L < 5.0 for tp in piers) or (len(deep) and ts[deep[0]] - 0.02 < tk < ts[deep[-1]] + 0.02):
            continue
        q = p0 + (p1 - p0) * tk
        if k == 0 or tk >= 1.0:                                      # abutments
            gq = float(ground([q[0]], [q[1]])[0])
            solids.append(box_between(q - a * 1.8, q + a * 1.8, Wd + 1.6, gq - 1.5, zd - 0.5, STONE))
            continue
        gq = float(min(ground([q[0] + b[0] * 2.5, q[0] - b[0] * 2.5], [q[1] + b[1] * 2.5, q[1] - b[1] * 2.5])))
        if zd - gq < 0.8:
            continue
        solids.append(box_between(q - a * 0.8, q + a * 0.8, Wd + 2.4, gq - 0.8, gq + 0.4, STONE))     # footing
        top = zd - 0.5
        posts = [(-1.0, Wd / 2 + 0.2, Wd / 2 + 0.2), (1.0, Wd / 2 + 0.2, Wd / 2 + 0.2)]
        if top - gq > 5.0:                                           # tall bents get raking outer posts
            posts += [(-1.0, Wd / 2 + 1.1, Wd / 2 + 0.3), (1.0, Wd / 2 + 1.1, Wd / 2 + 0.3)]
        for s_, vb, vt in posts:
            solids.append(beam3(a3(q) + b3_ * s_ * vb + UP * (gq + 0.3), a3(q) + b3_ * s_ * vt + UP * top, 0.32, 0.32))
        for zz0 in np.arange(gq + 0.4, top - 0.6, 3.2):              # X-bracing across the bent
            zz1 = min(zz0 + 3.2, top)
            solids.append(beam3(a3(q) - b3_ * (Wd / 2 + 0.2) + UP * zz0, a3(q) + b3_ * (Wd / 2 + 0.2) + UP * zz1, 0.1, 0.24))
            solids.append(beam3(a3(q) + b3_ * (Wd / 2 + 0.2) + UP * zz0, a3(q) - b3_ * (Wd / 2 + 0.2) + UP * zz1, 0.1, 0.24))
        solids.append(box_between(q - b * (Wd / 2 + 1.0), q + b * (Wd / 2 + 1.0), 0.4, top - 0.35, top, WOOD))
    sink.add(m3d.Manifold.batch_boolean([q for q in solids if q is not None], m3d.OpType.Add))
    rm, _, _, _ = slate_roof((p0 + p1) / 2, a, L, Wd + 0.6, zd + 3.12, 42.0, overhang=0.55, ends=("gable", "gable"))
    sink.add(rm)


def _occupancy(xs, ys):
    """Building footprints of the plan on a grid (True = something stands there), for the cliff walls."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    occ = np.zeros(X.shape, bool)

    def rect(p0, p1, w):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-6:
            return
        a = (p1 - p0) / L
        u = (X - p0[0]) * a[0] + (Y - p0[1]) * a[1]
        v = -(X - p0[0]) * a[1] + (Y - p0[1]) * a[0]
        occ[(u > -1.0) & (u < L + 1.0) & (np.abs(v) < w / 2 + 1.0)] = True

    T, B = CP.plan()
    for t in T:
        for tw, (x, y), _ in CP.walk(t):
            r = tw.stages[0].size * (1.42 if tw.shape == "square" else 1.0) + 1.0
            occ[(X - x) ** 2 + (Y - y) ** 2 < r * r] = True
    for e in B:
        if isinstance(e, (Hall, Range, Glasshouse)):
            rect(e.p0, e.p1, getattr(e, "width", None) or getattr(e, "depth", 10.0))
        elif isinstance(e, Gatehouse):
            ya = math.radians(e.yaw)
            ax = np.array([math.cos(ya), math.sin(ya)])
            rect(np.asarray(e.at) - ax * e.w / 2, np.asarray(e.at) + ax * e.w / 2, e.d + 4.0)
        elif isinstance(e, (Viaduct, Wall)):
            w = getattr(e, "width", 3.0)
            for q0, q1 in zip(e.path[:-1], e.path[1:]):
                rect(q0, q1, w + 2.0)
        elif isinstance(e, (ArchBridge, SuspensionBridge, CoveredBridge)):
            rect(e.p0, e.p1, e.width + 4.0)
    return occ


def build_cliff_walls(sink, cw: CliffWalls, rng):
    """The deep foundations: along every cliff rim of the castle plateau, a battered masonry wall clads the upper cliff
    (its depth wanders along the rim, its foot steps down onto the rock), counterforts with set-offs, a string course,
    rows of small lit windows low on the rock, a crenellated parapet with lanterns where no building stands at the edge,
    and towers rooted deep on the cliff at the sharpest corners of the rocks."""
    from skimage import measure

    step = 1.0
    xs = np.arange(-215.0, 236.0, step)
    ys = np.arange(-255.0, 246.0, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cd = W.Fields(X.ravel(), Y.ravel()).crag().reshape(X.shape)
    occ = _occupancy(xs, ys)

    def occupied(pt, r=0.0):
        i = int(round((pt[0] - xs[0]) / step))
        j = int(round((pt[1] - ys[0]) / step))
        rr = int(math.ceil(r / step))
        return bool(occ[max(0, i - rr):i + rr + 1, max(0, j - rr):j + rr + 1].any())

    def crag_at(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        return W.Fields(pts[:, 0], pts[:, 1]).crag()

    lo, hi = cw.depth
    tower_cands = []
    for C in measure.find_contours(cd, 0.0):
        P = np.stack([xs[0] + C[:, 0] * step, ys[0] + C[:, 1] * step], 1)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        Ltot = float(seg.sum())
        if Ltot < 40.0:
            continue
        closed = bool(np.linalg.norm(P[0] - P[-1]) < 1.5)
        s_ = np.concatenate([[0.0], np.cumsum(seg)])
        u = np.arange(0.0, Ltot, 1.0)
        Q = np.stack([np.interp(u, s_, P[:, 0]), np.interp(u, s_, P[:, 1])], 1)
        k = 7                                                        # smooth the grid's stair steps away
        if closed:
            Qp = np.concatenate([Q[-k:], Q, Q[:k]])
            Q = np.stack([np.convolve(Qp[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in (0, 1)], 1)
        else:
            Q = np.stack([np.convolve(np.pad(Q[:, i], k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in (0, 1)], 1)
        idx = np.arange(0, len(Q), 5)
        V = Q[idx]
        n_v = len(V)
        T_ = np.gradient(V, axis=0)
        if closed:
            T_ = np.roll(V, -1, 0) - np.roll(V, 1, 0)
        T_ /= np.linalg.norm(T_, axis=1, keepdims=True) + 1e-9
        N = np.stack([T_[:, 1], -T_[:, 0]], 1)
        flip = crag_at(V + N * 3.0) < crag_at(V - N * 3.0)
        N[flip] *= -1.0
        # depth of the wall at each vertex: as deep as the cliff allows, wandering between lo and hi along the rim
        drop = np.array([Z - float(ground(np.array([v[0] + n[0] * kk for kk in (2, 4, 6, 8, 10)]),
                                          np.array([v[1] + n[1] * kk for kk in (2, 4, 6, 8, 10)])).min()) for v, n in zip(V, N)])
        sv = idx.astype(float)
        ph = rng.uniform(0, 2 * math.pi, 3)
        wob = 0.5 + 0.5 * (0.45 * np.sin(sv / 37.0 + ph[0]) + 0.35 * np.sin(sv / 13.0 + ph[1]) + 0.2 * np.sin(sv / 5.0 + ph[2]))
        D = np.clip(drop - 3.0, 0.0, lo + (hi - lo) * wob)
        D[drop < 8.0] = 0.0
        # the wall must stand on rock: where the cliff falls away under the planned foot, go on down (up to 12 m more)
        # until the rock comes out to the wall's battered face
        for i in np.where(D > 0)[0]:
            for dd in np.arange(D[i], min(D[i] + 12.01, drop[i] - 2.0), 2.0):
                q = V[i] + N[i] * (1.0 + dd / 11.0 + 0.6)
                if Z - dd <= float(ground([q[0]], [q[1]])[0]) + 0.5:
                    D[i] = dd
                    break
            else:
                D[i] = max(D[i], min(D[i] + 12.0, drop[i] - 2.0))
        # the foot steps down onto the rock in courses, in runs of a few segments (never shallower than the rock needs)
        Dq = np.ceil(D / 3.5) * 3.5
        for r0 in range(0, n_v, 3):
            run = Dq[r0:r0 + 3]
            if (run > 0).all():
                Dq[r0:r0 + 3] = run.max()
        pairs = [(i, i + 1) for i in range(n_v - 1)] + ([(n_v - 1, 0)] if closed else [])
        chunk = []
        cutters = []

        def flush():
            if chunk:
                cen = np.mean([V[i] for i, _ in chunk], 0)
                sink.set_zone(cen[0], cen[1])
                body = m3d.Manifold.batch_boolean([q for _, q in chunk if q is not None], m3d.OpType.Add)
                sink.add(apply_cuts(body, cutters))
            chunk.clear()
            cutters.clear()

        for ci, (i, j) in enumerate(pairs):
            if D[i] < 4.0 or D[j] < 4.0:
                flush()
                continue
            A, Bp, nA, nB = V[i], V[j], N[i], N[j]
            pts = []
            dseg = float(Dq[i] if Dq[i] > 0 else Dq[j])
            for q, nq, d in ((A, nA, dseg), (Bp, nB, dseg)):
                for zz, out in ((Z + 0.3, 1.0), (Z - d, 1.0 + d / 11.0)):
                    pts.append([q[0] + nq[0] * out, q[1] + nq[1] * out, zz])
                    pts.append([q[0] - nq[0] * 3.0, q[1] - nq[1] * 3.0, zz])
            parts = [tag(m3d.Manifold.hull_points(np.array(pts)), STONE)]
            band = []                                                # the string course under the rim
            for q, nq in ((A, nA), (Bp, nB)):
                for zz in (Z - 1.75, Z - 1.4):
                    band.append([q[0] + nq[0] * 1.35, q[1] + nq[1] * 1.35, zz])
                    band.append([q[0] - nq[0] * 0.5, q[1] - nq[1] * 0.5, zz])
            parts.append(tag(m3d.Manifold.hull_points(np.array(band)), TRIM))
            plinth = []                                              # a footing course where the wall meets the rock
            for q, nq in ((A, nA), (Bp, nB)):
                out = 1.0 + dseg / 11.0
                for zz, extra in ((Z - dseg - 0.8, 0.9), (Z - dseg + 1.6, 0.55)):
                    plinth.append([q[0] + nq[0] * (out + extra), q[1] + nq[1] * (out + extra), zz])
                    plinth.append([q[0] - nq[0] * 1.0, q[1] - nq[1] * 1.0, zz])
            parts.append(tag(m3d.Manifold.hull_points(np.array(plinth)), STONE))
            if i % 3 == 0 and D[i] > 12.0:                           # a counterfort with two set-offs
                t_ = np.array([-nA[1], nA[0]])
                zb = Z - D[i] - 1.0
                for (z0_, z1_, p0_, p1_) in ((zb, zb + D[i] * 0.45, 2.8, 2.0), (zb + D[i] * 0.45, zb + D[i] * 0.8, 2.0, 1.3),
                                             (zb + D[i] * 0.8, Z - 2.5, 1.3, 0.7)):
                    bp = []
                    for zz, pr in ((z0_, p0_), (z1_, p1_)):
                        f = A + nA * (1.0 + (Z - zz) / 11.0)
                        for tt in (-1.3, 1.3):
                            for pp in (-0.6, pr):
                                c = f + t_ * tt + nA * pp
                                bp.append([c[0], c[1], zz])
                    parts.append(tag(m3d.Manifold.hull_points(np.array(bp)), STONE))
            for dd in (6.0, 13.0, 20.0, 27.0):                       # small windows low on the rock (lit at night)
                if i % 2 == 0 and dseg > dd + 5.0 and rng.random() < 0.7:
                    o = A + nA * (1.05 + dd / 11.0)
                    inward = -a3(nA)
                    window(sink, cutters, "c_lancet", (o[0], o[1], Z - dd - 1.5), np.cross(inward, UP), inward, 1.0, 2.6)
            chunk.append((i, m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)))
            mid = (A + Bp) / 2
            nm = (nA + nB) / np.linalg.norm(nA + nB)
            if not any(occupied(mid - nm * kk, 1.0) for kk in (1.0, 3.0, 5.0, 7.0)):
                q0_, q1_ = A + nA * 1.0, Bp + nB * 1.0                 # a terrace parapet, a lantern now and then
                sink.add(parapet_line(sink, q0_, q1_, nm, Z + 0.9, "crenel", over=-0.1, thick=0.8))
                sink.add(box_between(q0_ - nm * 0.5, q1_ - nm * 0.5, 0.8, Z - 0.3, Z + 0.7, STONE))
                if i % 5 == 0:
                    lp = A - nA * 1.2
                    sink.place("c_lantern", (lp[0], lp[1], Z), np.eye(3))
            if len(chunk) >= 12:
                flush()
            # tower candidates: sharp convex corners where the wall is deep; along straight runs every ~70 m
            if D[i] > 20.0 and 3 <= i < n_v - 3:
                score = -float((V[i - 3] - A) @ nA + (V[i + 3] - A) @ nA)
                tower_cands.append((score, A, nA, D[i]))
            elif D[i] > 18.0 and i % 14 == 7:
                tower_cands.append((1.6, A, nA, D[i]))
        flush()
    # towers rooted deep on the cliff at the sharpest corners
    tower_cands.sort(key=lambda c: -c[0])
    placed = []
    for score, A, nA, d in tower_cands:
        if len(placed) >= cw.towers or score < 1.5:
            break
        r = float(rng.uniform(5.0, 6.8))
        c = A + nA * (r * 0.55)
        if any(np.linalg.norm(c - q) < 55.0 for q in placed) or occupied(c - nA * r * 0.5, r * 0.6):
            continue
        placed.append(c)
        kind = ("cone", "stepped", "crenel", "cone", "spire", "stepped")[len(placed) % 6]
        up = float(rng.uniform(6.0, 22.0))                           # how far the tower rises above the plateau
        body = CP.Stage(d + 3.0 + up, r, "string", floor_h=5.5)
        if kind == "stepped":                                         # a narrower stage on a corbelled gallery
            st = [body, CP.Stage(1.0, r, "gallery"), CP.Stage(float(rng.uniform(6.0, 10.0)), r * 0.72, "corbel", floor_h=4.0)]
            roof = CP.Roof("cone", r * 0.72 * 2.8, 1, 2.0)
        elif kind == "crenel":
            st = [body, CP.Stage(4.5, r, "corbel", floor_h=4.0)]
            roof = CP.Roof("flat", 0.0, 0, 0.0)
        elif kind == "spire":
            st = [CP.Stage(d + 3.0 + up, r * 0.92, "string", floor_h=5.5, shape="octagon"),
                  CP.Stage(4.0, r * 0.92, "crenel", floor_h=4.0, shape="octagon")]
            roof = CP.Roof("spire", r * 3.2, 1, 2.5)
        else:
            st = [body, CP.Stage(5.0, r, "corbel", floor_h=4.0)]
            roof = CP.Roof("cone", r * float(rng.uniform(2.1, 2.9)), 1, 2.5)
        tw = Tower(f"CliffTower{len(placed)}", (c[0], c[1]), st, roof, "octagon" if kind == "spire" else "round", base=Z - d - 3.0)
        build_tower(sink, tw, rng)
    print(f"    cliff walls: {len(placed)} cliff towers", flush=True)


BUILDERS = {Hall: build_hall, Range: build_range, Gatehouse: build_gate, Wall: build_wall, Viaduct: build_viaduct,
            ArchBridge: build_archbridge, SuspensionBridge: build_suspension, Stair: build_stair, Boathouse: build_boathouse,
            Glasshouse: build_glasshouse, CoveredBridge: build_covered_bridge, CliffWalls: build_cliff_walls}


def build(out_dir, seed=21, only=None):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/instances", exist_ok=True)
    rng = np.random.default_rng(seed)
    sink = Sink()
    T, B = CP.plan()
    for b in B:
        if only and b.name not in only:
            continue
        t = time.time()
        BUILDERS[type(b)](sink, b, rng)
        print(f"  {type(b).__name__:16s} {b.name:24s} {time.time() - t:5.1f}s", flush=True)
    for tw in T:
        if only and tw.name not in only:
            continue
        t = time.time()
        build_tower(sink, tw, rng)
        print(f"  {'Tower':16s} {tw.name:24s} {time.time() - t:5.1f}s", flush=True)
    # unique meshes per zone
    for f in os.listdir(out_dir):
        if f.startswith("castle_") and f.endswith(".npz"):
            os.remove(os.path.join(out_dir, f))
    tot = 0
    for zone, parts in sorted(sink.parts.items()):
        m = ck.finish(mk.merge(parts), Z)
        m.save(f"{out_dir}/{zone}.npz")
        tot += m.nf
        print(f"  {zone:18s} {m.nf:10,d} tris", flush=True)
    # instance sets per module
    for f in os.listdir(f"{out_dir}/instances"):
        if f.startswith("castle_"):
            os.remove(os.path.join(out_dir, "instances", f))
    n_inst = 0
    lib_tris = {}
    for module, (P, R, S) in sorted(sink.inst.items()):
        ins.save_set(f"{out_dir}/instances/castle_{module[2:]}.npz", np.array(P), np.array(R), np.array(S), np.zeros(len(P), np.int16))
        n_inst += len(P)
        lp = f"{out_dir}/lib/lib_{module}.npz"
        lib_tris[module] = mk.Mesh.load(lp).nf if os.path.isfile(lp) else 0
        print(f"  instances {module:14s} {len(P):8,d}  x {lib_tris[module]:6,d} tris", flush=True)
    inst_tris = sum(len(sink.inst[m][0]) * lib_tris[m] for m in sink.inst)
    print(f"castle: {tot:,} unique tris in {len(sink.parts)} zones + {n_inst:,} instances ({inst_tris:,} instanced tris) "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo", only=set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None)
