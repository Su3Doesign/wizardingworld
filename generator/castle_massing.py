"""castle_massing - a quick block-out of castle_plan (simple solids: stages, bands, roofs, halls, ranges, the viaduct's
arches, stairs) on the analytic terrain, to judge the composition against the references before the detailed build.

    python castle_massing.py OUT/massing          # -> massing.npz (castle) + terrain_*.npz, then massing_view.py renders
"""
from __future__ import annotations

import math
import os
import sys
import time

import manifold3d as m3d
import numpy as np

import castle_plan as CP
import meshkit as mk
import world as W
from castle_plan import (ArchBridge, Arcade, Boathouse, Gatehouse, Glasshouse, Hall, Range, Stair, SuspensionBridge, Tower,
                         Viaduct, Wall)

WALL, ROOF, TRIM, GLASS = 20, 22, 21, 24     # castle material ids (export_world.MAT_NAMES)


def to_mesh(man, mat):
    mm = man.to_mesh()
    V = np.asarray(mm.vert_properties, np.float64)[:, :3]
    F = np.asarray(mm.tri_verts, np.int64)
    return mk.Mesh(V, F, np.full(len(F), mat, np.int16))


def ground_min(xs, ys):
    return float(W.Fields(np.asarray(xs, np.float64), np.asarray(ys, np.float64)).height().min())


def footprint_foot(x, y, r):
    a = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    return ground_min(np.r_[x, x + r * np.cos(a)], np.r_[y, y + r * np.sin(a)]) - 2.0


# ------------------------------------------------------------------------------------------------ primitive solids
def prism_shape(shape, size, z0, z1, x, y, yaw=0.0, depth=None, r_top=None):
    if z1 <= z0 + 1e-3:
        return None
    if shape == "round":
        m = m3d.Manifold.cylinder(z1 - z0, size, size if r_top is None else r_top, 40)
    elif shape == "octagon":
        m = m3d.Manifold.cylinder(z1 - z0, size / math.cos(math.pi / 8), (size if r_top is None else r_top) / math.cos(math.pi / 8), 8)
        m = m.rotate([0.0, 0.0, 22.5])
    else:
        d = size if depth is None else depth
        if r_top is not None:
            m = m3d.Manifold.cylinder(z1 - z0, size * math.sqrt(2), r_top * math.sqrt(2), 4).rotate([0.0, 0.0, 45.0])
            if depth is not None:
                m = m.scale([1.0, d / size, 1.0])
        else:
            m = m3d.Manifold.cube([2 * size, 2 * d, z1 - z0], True).translate([0.0, 0.0, (z1 - z0) / 2])
    if yaw:
        m = m.rotate([0.0, 0.0, yaw])
    return m.translate([x, y, z0])


def roof_solid(shape, kind, size, z0, h, x, y, yaw=0.0, depth=None):
    if kind == "flat" or h <= 0:
        return None
    eave = size * 1.08 + 0.4
    if kind == "dome":
        s = m3d.Manifold.sphere(1.0, 40).scale([eave, eave, h])
        return s.trim_by_plane([0, 0, 1], 0.0).translate([x, y, z0])
    seg = {"round": 40, "octagon": 8, "square": 4}[shape]
    if kind in ("spire",) and shape == "round":
        seg = 8
    r = eave / (math.cos(math.pi / seg) if seg <= 8 else 1.0)
    m = m3d.Manifold.cylinder(h, r, 0.02, seg)
    if seg == 4:
        m = m.rotate([0.0, 0.0, 45.0])
        if depth is not None:
            m = m.scale([1.0, depth / size, 1.0])
    elif seg == 8:
        m = m.rotate([0.0, 0.0, 22.5])
    if yaw:
        m = m.rotate([0.0, 0.0, yaw])
    return m.translate([x, y, z0])


def gable_roof(p0, p1, width, z_eave, pitch, overhang=0.6, hip=(False, False)):
    """Triangular prism roof along p0 -> p1 (optionally hipped ends)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    a = (p1 - p0) / L
    half = width / 2 + overhang
    h = half * math.tan(math.radians(pitch))
    tri = m3d.CrossSection([np.array([[-half, 0.0], [half, 0.0], [0.0, h]])])
    m = tri.extrude(L + 2 * overhang).translate([0.0, 0.0, -overhang])          # along local z
    # local (x=across, y=up, z=along) -> world
    m = m.rotate([90.0, 0.0, 0.0]).rotate([0.0, 0.0, 90.0])                       # along -> +x? fix with yaw below
    yaw = math.degrees(math.atan2(a[1], a[0]))
    m = m.rotate([0.0, 0.0, yaw])
    # after the two rotations the prism runs along +x from 0 to L+2o (shifted by -o); place it
    m = m.translate([p0[0], p0[1], z_eave])
    for end, flag in ((0, hip[0]), (1, hip[1])):
        if flag:
            n = -a if end == 0 else a
            pe = p0 if end == 0 else p1
            # cut the end with a plane leaning in at the roof pitch
            nrm = np.array([n[0], n[1], 0.0]) * math.sin(math.radians(pitch)) + np.array([0, 0, 1.0]) * math.cos(math.radians(pitch))
            off = float(np.dot(nrm, [pe[0] + n[0] * overhang, pe[1] + n[1] * overhang, z_eave]))
            m = m.trim_by_plane((-nrm).tolist(), -off)
    return m


def oriented_box(p0, p1, depth, z0, z1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    c = (p0 + p1) / 2
    yaw = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    return m3d.Manifold.cube([L, depth, z1 - z0], True).rotate([0, 0, yaw]).translate([c[0], c[1], (z0 + z1) / 2])


# ------------------------------------------------------------------------------------------------------- builders
def build_tower(t: Tower, parts):
    for tw, (x, y), base in CP.walk(t):
        z = base
        if tw is t:
            foot = footprint_foot(x, y, tw.stages[0].size)
            if foot < z:
                parts[WALL].append(prism_shape(tw.shape, tw.stages[0].size * 1.02, foot, z + 0.5, x, y, tw.yaw, tw.stages[0].depth))
        for st in tw.stages:
            shape = st.shape or tw.shape
            body = prism_shape(shape, st.size, z, z + st.h, x, y, tw.yaw, st.depth)
            if st.band == "arcade":                          # open lantern: a thinner core + piers read as a lighter mass
                body = prism_shape(shape, st.size * 0.72, z, z + st.h, x, y, tw.yaw, st.depth)
                n = max(6, st.windows or 8)
                for k in range(n):
                    a = 2 * math.pi * (k + 0.5) / n
                    parts[TRIM].append(m3d.Manifold.cylinder(st.h, 0.55, 0.55, 8).translate(
                        [x + st.size * 0.95 * math.cos(a), y + st.size * 0.95 * math.sin(a), z]))
            parts[WALL].append(body)
            if st.band in ("corbel", "gallery", "crenel"):
                grow = {"corbel": 0.9, "gallery": 1.5, "crenel": 0.5}[st.band]
                bh = {"corbel": 3.0, "gallery": 1.6, "crenel": 2.0}[st.band]
                parts[TRIM].append(prism_shape(shape, st.size + grow, z + st.h - (0.0 if st.band == "gallery" else bh * 0.6),
                                               z + st.h + (bh if st.band == "gallery" else bh * 0.4), x, y, tw.yaw,
                                               None if st.depth is None else st.depth + grow))
            if st.gables and shape == "square":                # a steep gable on each face
                for k in range(4):
                    yaw = tw.yaw + 90.0 * k
                    a = math.radians(yaw)
                    w = st.size * 1.1
                    tri = m3d.CrossSection([np.array([[-w, 0.0], [w, 0.0], [0.0, w * 1.6]])]).extrude(1.2)
                    tri = tri.rotate([90.0, 0.0, 0.0]).rotate([0.0, 0.0, yaw + 90.0])
                    parts[WALL].append(tri.translate([x + math.cos(a) * (st.size - 0.6), y + math.sin(a) * (st.size - 0.6), z + st.h]))
            z += st.h
        last = tw.stages[-1]
        shape = last.shape or tw.shape
        r = roof_solid(shape, tw.roof.kind, last.size, z, tw.roof.h, x, y, tw.yaw, last.depth)
        if r is not None:
            parts[ROOF].append(r)
            parts[TRIM].append(m3d.Manifold.cylinder(tw.roof.finial, 0.25, 0.02, 6).translate([x, y, z + tw.roof.h]))


def build_hall(h: Hall, parts):
    p0, p1 = np.asarray(h.p0, float), np.asarray(h.p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-a[1], a[0]])
    L = float(np.linalg.norm(p1 - p0))
    foot = min(ground_min([p[0] + n[0] * s * h.width / 2 for p in (p0, p1, (p0 + p1) / 2) for s in (-1, 1)],
                          [p[1] + n[1] * s * h.width / 2 for p in (p0, p1, (p0 + p1) / 2) for s in (-1, 1)]) - 2.0, h.base)
    parts[WALL].append(oriented_box(p0, p1, h.width, foot, h.base + h.wall))
    parts[ROOF].append(gable_roof(p0, p1, h.width, h.base + h.wall, h.pitch))
    # buttresses + pinnacles along both long sides
    for k in range(h.bays + 1):
        c = p0 + a * (L * k / h.bays)
        for s in (-1, 1):
            q = c + n * s * (h.width / 2 + 1.0)
            parts[WALL].append(oriented_box(q - a * 1.0, q + a * 1.0, 2.2, h.base, h.base + h.wall + 1.0))
            parts[TRIM].append(m3d.Manifold.cylinder(6.5, 0.9, 0.05, 4).rotate([0, 0, 45]).translate([q[0], q[1], h.base + h.wall + 1.0]))
    # parapet
    for s in (-1, 1):
        q0, q1 = p0 + n * s * h.width / 2, p1 + n * s * h.width / 2
        parts[TRIM].append(oriented_box(q0, q1, 0.8, h.base + h.wall - 0.2, h.base + h.wall + 1.6))
    # gable turrets
    tz = h.turret_h or 1.9 * h.wall
    for flag, pe, sgn in ((h.gable_turrets[0], p0, -1), (h.gable_turrets[1], p1, 1)):
        if not flag:
            continue
        for s in (-1, 1):
            q = pe + a * sgn * 1.5 + n * s * (h.width / 2 + 0.5)
            parts[WALL].append(prism_shape("octagon", 3.0, h.base, h.base + tz, q[0], q[1]))
            parts[ROOF].append(roof_solid("octagon", "spire", 3.0, h.base + tz, 13.0, q[0], q[1]))
    if h.fleche:
        c = (p0 + p1) / 2
        zr = h.base + h.wall + (h.width / 2 + 0.6) * math.tan(math.radians(h.pitch))
        parts[TRIM].append(prism_shape("octagon", 1.4, zr - 3.0, zr + 3.0, c[0], c[1]))
        parts[ROOF].append(roof_solid("octagon", "spire", 1.4, zr + 3.0, h.fleche, c[0], c[1]))


def build_range(r: Range, parts):
    p0, p1 = np.asarray(r.p0, float), np.asarray(r.p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-a[1], a[0]])
    foot = min(ground_min([p[0] + n[0] * s * r.depth / 2 for p in (p0, p1, (p0 + p1) / 2) for s in (-1, 1)],
                          [p[1] + n[1] * s * r.depth / 2 for p in (p0, p1, (p0 + p1) / 2) for s in (-1, 1)]) - 2.0, r.base)
    parts[WALL].append(oriented_box(p0, p1, r.depth, foot, r.base + r.wall))
    if r.roof == "flat":
        parts[TRIM].append(oriented_box(p0, p1, r.depth + 0.6, r.base + r.wall - 0.3, r.base + r.wall + 1.4))
        return
    parts[ROOF].append(gable_roof(p0, p1, r.depth, r.base + r.wall, r.pitch, hip=(r.ends[0] == "hip", r.ends[1] == "hip")))
    if r.parapet != "none":
        for s in (-1, 1):
            parts[TRIM].append(oriented_box(p0 + n * s * r.depth / 2, p1 + n * s * r.depth / 2, 0.7, r.base + r.wall - 0.2,
                                            r.base + r.wall + 1.6))
    if r.chimneys:
        L = float(np.linalg.norm(p1 - p0))
        zr = r.base + r.wall + (r.depth / 2) * math.tan(math.radians(r.pitch))
        for t in np.arange(10.0, L - 6.0, 17.0):
            c = p0 + a * t + n * r.depth * 0.18
            parts[WALL].append(oriented_box(c - a * 1.2, c + a * 1.2, 1.4, zr - 6.0, zr + 3.0))


def build_arcade(c: Arcade, parts):
    p0, p1 = np.asarray(c.p0, float), np.asarray(c.p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-a[1], a[0]])
    foot = ground_min([p0[0], p1[0], (p0[0] + p1[0]) / 2], [p0[1] - c.depth / 2, p1[1] - c.depth / 2, p0[1] - c.depth / 2]) - 2.0
    body = oriented_box(p0, p1, c.depth, min(foot, c.base), c.base + c.h + c.upper)
    L = float(np.linalg.norm(p1 - p0))
    span = L / c.arches
    cut = []
    for k in range(c.arches):
        q = p0 + a * span * (k + 0.5) - n * (c.depth / 2 - 1.2)
        w = span * 0.36
        arch = m3d.Manifold.cylinder(3.0, w, w, 24).rotate([90, 0, 0]).rotate([0, 0, math.degrees(math.atan2(a[1], a[0]))])
        arch = arch.translate([q[0], q[1], c.base + c.h - w])
        col = oriented_box(q - a * w, q + a * w, 3.0, c.base, c.base + c.h - w)
        cut.append(arch + col)
    parts[WALL].append(body - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add))
    parts[TRIM].append(oriented_box(p0, p1, c.depth + 0.4, c.base + c.h + c.upper - 0.2, c.base + c.h + c.upper + 1.3))


def build_gate(g: Gatehouse, parts):
    yaw = g.yaw
    foot = footprint_foot(g.at[0], g.at[1], max(g.w, g.d) / 2) if g.base >= W.CASTLE_Z - 1 else g.base
    parts[WALL].append(prism_shape("square", g.w / 2, min(foot, g.base), g.base + g.h, g.at[0], g.at[1], yaw, g.d / 2))
    parts[TRIM].append(prism_shape("square", g.w / 2 + 0.6, g.base + g.h - 1.0, g.base + g.h + 1.4, g.at[0], g.at[1], yaw, g.d / 2 + 0.6))
    th = g.turret_h or g.h + 8.0
    ya = math.radians(yaw)
    ax = np.array([math.cos(ya), math.sin(ya)])
    nx = np.array([-ax[1], ax[0]])
    for s in (-1, 1):
        q = np.asarray(g.at) - ax * g.w / 2 + nx * s * g.d / 2
        parts[WALL].append(prism_shape("round", g.turret_r, min(foot, g.base), g.base + th, q[0], q[1]))
        parts[ROOF].append(roof_solid("round", "cone", g.turret_r, g.base + th, g.turret_r * 2.6, q[0], q[1]))


def build_wall(w: Wall, parts):
    pts = np.asarray(w.path, float)
    for i in range(len(pts) - 1):
        foot = ground_min([pts[i][0], pts[i + 1][0], (pts[i][0] + pts[i + 1][0]) / 2],
                          [pts[i][1], pts[i + 1][1], (pts[i][1] + pts[i + 1][1]) / 2]) - 2.0
        parts[WALL].append(oriented_box(pts[i], pts[i + 1], w.thick, min(foot, w.base), w.base + w.h))


def build_viaduct(v: Viaduct, parts):
    """Tiers of arches along every segment of the polyline: the lower tier (spanning two upper arches) stands in the
    water / on the rock, the upper tier carries the deck; arcaded parapet."""
    pts = [np.asarray(p, float) for p in v.path]
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        a = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-a[1], a[0]])
        L = float(np.linalg.norm(p1 - p0))
        samples = [p0 + a * L * t for t in np.linspace(0, 1, 41)]
        zlow = min(ground_min([s_[0]], [s_[1]]) for s_ in samples) - 4.0
        body = oriented_box(p0, p1, v.width, zlow, v.deck)
        yaw = math.degrees(math.atan2(a[1], a[0]))
        z_t = zlow + (v.deck - zlow) * 0.52 if v.tiers > 1 else zlow
        cut = []
        nu = max(1, int(round(L / v.span)))
        span = L / nu
        for k in range(nu):                                     # upper tier
            c = p0 + a * span * (k + 0.5)
            w = (span - v.pier) / 2
            spring = v.deck - 3.4 - w
            arch = m3d.Manifold.cylinder(v.width + 2, w, w, 28, True).rotate([90, 0, 0]).rotate([0, 0, yaw]).translate([c[0], c[1], spring])
            bx = m3d.Manifold.cube([2 * w, v.width + 2, spring - z_t], True).rotate([0, 0, yaw]).translate([c[0], c[1], (spring + z_t) / 2])
            cut.append(arch + bx)
        if v.tiers > 1:
            nl = max(1, nu // 2)
            spl = L / nl
            for k in range(nl):                                 # lower tier
                c = p0 + a * spl * (k + 0.5)
                w = (spl - v.pier * 1.6) / 2
                spring = z_t - 2.5 - w
                arch = m3d.Manifold.cylinder(v.width + 2, w, w, 32, True).rotate([90, 0, 0]).rotate([0, 0, yaw]).translate([c[0], c[1], spring])
                bx = m3d.Manifold.cube([2 * w, v.width + 2, spring - zlow + 1], True).rotate([0, 0, yaw]).translate(
                    [c[0], c[1], (spring + zlow - 1) / 2])
                cut.append(arch + bx)
        parts[WALL].append(body - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add))
        for s_ in (-1, 1):
            parts[TRIM].append(oriented_box(p0 + n * s_ * (v.width / 2 - 0.3), p1 + n * s_ * (v.width / 2 - 0.3), 0.6, v.deck, v.deck + 1.6))


def build_archbridge(b: ArchBridge, parts):
    p0, p1 = np.asarray(b.p0, float), np.asarray(b.p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    L = float(np.linalg.norm(p1 - p0))
    yaw = math.degrees(math.atan2(a[1], a[0]))
    zlow = min(ground_min([p0[0]], [p0[1]]), ground_min([p1[0]], [p1[1]]), b.deck - 30.0)
    body = oriented_box(p0, p1, b.width, zlow, b.deck)
    cut = []
    span = L / b.arches
    for k in range(b.arches):
        c = p0 + a * span * (k + 0.5)
        r = span * 0.42
        arch = m3d.Manifold.cylinder(b.width + 2, r, r, 32, True).rotate([90, 0, 0]).rotate([0, 0, yaw]).translate([c[0], c[1], b.deck - 3.0 - r])
        bx = m3d.Manifold.cube([2 * r, b.width + 2, b.deck - 3.0 - r - zlow + 1], True).rotate([0, 0, yaw]).translate(
            [c[0], c[1], (b.deck - 3.0 - r + zlow - 1) / 2])
        cut.append(arch + bx)
    parts[WALL].append(body - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add))
    n = np.array([-a[1], a[0]])
    for s_ in (-1, 1):
        parts[TRIM].append(oriented_box(p0 + n * s_ * (b.width / 2 - 0.3), p1 + n * s_ * (b.width / 2 - 0.3), 0.5, b.deck, b.deck + 1.2))


def build_suspension(b: SuspensionBridge, parts):
    p0, p1 = np.asarray(b.p0, float), np.asarray(b.p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-a[1], a[0]])
    parts[TRIM].append(oriented_box(p0, p1, b.width, b.deck - 0.6, b.deck))
    for q in (p0, p1):
        foot = ground_min([q[0]], [q[1]]) - 2.0
        parts[WALL].append(oriented_box(q - a * 2.5, q + a * 2.5, b.width + 3.0, min(foot, b.deck - 8.0), b.deck + b.pylon_h))
        parts[ROOF].append(roof_solid("square", "pyramid", (b.width + 3.0) / 2, b.deck + b.pylon_h, 6.0, q[0], q[1],
                                      math.degrees(math.atan2(a[1], a[0]))))
    mid = (p0 + p1) / 2
    for s_ in (-1, 1):
        for q in (p0, p1):
            parts[TRIM].append(oriented_box(q + n * s_ * b.width / 2, mid + n * s_ * b.width / 2, 0.3, b.deck + 1.0, b.deck + 1.3))


def build_glasshouse(g: Glasshouse, parts):
    parts[WALL].append(oriented_box(g.p0, g.p1, g.width, g.base - 1.0, g.base + 1.2))
    parts[TRIM].append(oriented_box(g.p0, g.p1, g.width - 0.4, g.base + 1.2, g.base + g.h * 0.55))
    parts[TRIM].append(gable_roof(g.p0, g.p1, g.width - 0.4, g.base + g.h * 0.55, 40.0, overhang=0.2))


def build_stair(s: Stair, parts):
    P = np.asarray(s.path, float)
    for i in range(len(P) - 1):
        q0, q1 = P[i], P[i + 1]
        d = q1[:2] - q0[:2]
        L = float(np.linalg.norm(d))
        a = d / L
        steps = max(2, int(L / 1.2))
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            z = q0[2] + (q1[2] - q0[2]) * t1
            c0 = q0[:2] + d * t0
            c1 = q0[:2] + d * t1
            foot = ground_min([c0[0], c1[0]], [c0[1], c1[1]]) - 2.0
            parts[WALL].append(oriented_box(c0, c1 + a * 0.05, s.width + 0.8, min(foot, z - 0.4), z + s.wall_h))


def build_boathouse(b: Boathouse, parts):
    ya = math.radians(b.yaw)
    ax = np.array([math.cos(ya), math.sin(ya)])
    p0 = np.asarray(b.at) - ax * b.length / 2
    p1 = np.asarray(b.at) + ax * b.length / 2
    parts[WALL].append(oriented_box(p0, p1, b.width, b.base - 3.0, b.base + 9.0))
    parts[ROOF].append(gable_roof(p0, p1, b.width, b.base + 9.0, 55.0))
    q = p1 - ax * 3.0
    parts[WALL].append(prism_shape("square", 1.8, b.base + 9.0, b.base + 16.0, q[0], q[1], b.yaw))
    parts[ROOF].append(roof_solid("square", "spire", 1.8, b.base + 16.0, 9.0, q[0], q[1], b.yaw))


def build(out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    T, B = CP.plan()
    parts = {WALL: [], ROOF: [], TRIM: []}
    for t in T:
        build_tower(t, parts)
    fns = {Hall: build_hall, Range: build_range, Arcade: build_arcade, Gatehouse: build_gate, Wall: build_wall,
           Viaduct: build_viaduct, Stair: build_stair, Boathouse: build_boathouse, ArchBridge: build_archbridge,
           SuspensionBridge: build_suspension, Glasshouse: build_glasshouse}
    for b in B:
        fns[type(b)](b, parts)
    meshes = []
    for mat, lst in parts.items():
        lst = [m for m in lst if m is not None]
        meshes.append(to_mesh(m3d.Manifold.compose(lst), mat))
    m = mk.merge(meshes)
    m.save(f"{out_dir}/massing.npz")
    print(f"massing: {m.nf:,} tris in {time.time() - t0:.1f}s", flush=True)
    terrain(out_dir)


def terrain(out_dir):
    """Analytic height field: 2 m round the castle, 25 m for the backdrop, as two meshes."""
    t0 = time.time()
    for name, (x0, x1, y0, y1, step) in {"near": (-360.0, 640.0, -420.0, 420.0, 2.0), "far": (-4200.0, 4200.0, -4200.0, 4200.0, 25.0)}.items():
        xs = np.arange(x0, x1 + 1e-6, step)
        ys = np.arange(y0, y1 + 1e-6, step)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        H = W.Fields(X.ravel(), Y.ravel()).height().reshape(X.shape)
        if name == "far":                                      # leave a hole for the near patch (lowered under it)
            inside = (X > x0 + 0) & (X > -350) & (X < 630) & (Y > -410) & (Y < 410)
            H = np.where(inside, H - 3.0, H)
        nx, ny = X.shape
        idx = np.arange(nx * ny).reshape(nx, ny)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        V = np.stack([X.ravel(), Y.ravel(), H.ravel()], 1)
        mk.Mesh(V, F, np.zeros(len(F), np.int16)).save(f"{out_dir}/terrain_{name}.npz")
    print(f"terrain in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/massing")
