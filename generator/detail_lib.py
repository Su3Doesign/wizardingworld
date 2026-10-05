"""detail_lib - the castle's high-detail modules, built once and placed thousands of times as Nanite instances
(instance records, like the forests): traceried windows, pinnacles with crockets, finials, merlons, corbels, machicolation
arches, balusters, lucarnes, dormers, chimneys, gargoyles, bartizans, columns, the clock face, ridge cresting, pierced
parapet panels.

Every module is modelled at its real size in a local frame:
    x  along the wall (left -> right seen from outside)       y  into the wall (0 = the wall's outer face)
    z  up (0 = the module's base: a window's sill, a pinnacle's foot, a merlon's bed)
so a builder places it with  position = point on the wall face,  R = columns (along, into-the-wall, up),  scale ~1
(windows are stretched a little to fit their opening).  Roof-mounted pieces (lucarnes, dormers, chimneys) use
x along the eave, y up the roof slope (into the roof), z up.

    python detail_lib.py OUT/geo/lib        # -> lib_c_<name>.npz
"""
from __future__ import annotations

import math
import os
import sys

import manifold3d as m3d
import numpy as np

import meshkit as mk
from meshkit import Mesh

STONE, TRIM, SLATE, LEAD, GLASS, WOOD = 20, 21, 22, 23, 24, 25
GLASS_I = 27          # glass of instanced windows (the night glow is picked per instance in the material)
CLOCK = 29            # the clock dial (glows at night)
LANTERN = 34          # lantern glass (the boats' M_Lantern: glows at night)


# --------------------------------------------------------------------------------------------- geometry helpers
def man(m: m3d.Manifold, mat) -> Mesh:
    mm = m.to_mesh()
    V = np.asarray(mm.vert_properties, np.float64)[:, :3]
    F = np.asarray(mm.tri_verts, np.int64)
    return Mesh(V, F, np.full(len(F), mat, np.int16))


def cube(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def ccw(poly):
    """Counter-clockwise copy of a polygon (manifold's cross sections treat clockwise loops as holes)."""
    P = np.asarray(poly, np.float64)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if area > 0 else P[::-1].copy()


def CS(poly):
    return m3d.CrossSection([ccw(poly)])


def extrude_xz(poly, y0, y1):
    """Extrude a polygon given in the wall plane (x, z) along y from y0 to y1."""
    cs = CS(poly)
    return cs.extrude(y1 - y0).rotate([90.0, 0.0, 0.0]).scale([1.0, -1.0, 1.0]).translate([0.0, y0, 0.0])


def arch_outline(w, h, kind="pointed", n=10, x0=0.0, z0=0.0):
    """Closed outline (x, z) of an opening w wide, h tall: straight jambs, pointed (equilateral-ish), round or flat head."""
    r = w / 2
    if kind == "flat":
        return np.array([[x0 - r, z0], [x0 + r, z0], [x0 + r, z0 + h], [x0 - r, z0 + h]])
    if kind == "round":
        spring = z0 + h - r
        a = np.linspace(0, math.pi, n + 1)
        head = np.stack([x0 + r * np.cos(a), spring + r * np.sin(a)], 1)
        return np.vstack([[[x0 - r, z0], [x0 + r, z0]], head[:-1], [[x0 - r, spring]]])
    # pointed: two arcs of radius R = 0.85 w centred inside the opening
    R = 0.85 * w
    rise = math.sqrt(R * R - (R - r) ** 2)
    spring = z0 + h - rise
    a1 = math.acos((R - r) / R)
    right = np.array([[x0 - (R - r) + R * math.cos(t), spring + R * math.sin(t)] for t in np.linspace(0, a1, n)])
    left = np.array([[x0 + (R - r) - R * math.cos(t), spring + R * math.sin(t)] for t in np.linspace(a1, 0, n)])
    return np.vstack([[[x0 - r, z0], [x0 + r, z0]], right[:-1], [[x0, z0 + h]], left[1:]])


def offset_outline(P, d):
    """Inset (d > 0) / outset a closed polygon via manifold's CrossSection offset."""
    cs = CS(P).offset(-d, m3d.JoinType.Miter)
    polys = cs.to_polygons()
    return np.asarray(max(polys, key=len), np.float64) if polys else np.asarray(P)


def frame_ring(P, width, depth, y0=0.0, chamfer=0.04):
    """A moulded frame following outline P (in x, z): the band between P and P inset by `width`, from y0 to y0 + depth,
    with a small chamfer step (two bands) - built with manifold extrusions."""
    outer = CS(P)
    inner = outer.offset(-width, m3d.JoinType.Miter)
    mid = outer.offset(-width * 0.45, m3d.JoinType.Miter)
    band1 = (outer - inner).extrude(depth * 0.55)
    band2 = (mid - inner).extrude(depth * 0.45).translate([0, 0, depth * 0.55])
    m = (band1 + band2).rotate([90.0, 0.0, 0.0]).scale([1.0, -1.0, 1.0]).translate([0.0, y0, 0.0])
    return m


def glass_pane(P, y, inset=0.0, mat=GLASS_I):
    Q = offset_outline(P, inset) if inset else np.asarray(P)
    c = Q.mean(0)
    V = np.vstack([np.stack([Q[:, 0], np.full(len(Q), y), Q[:, 1]], 1), [[c[0], y, c[1]]]])
    k = len(Q)
    F = np.array([[i, k, (i + 1) % k] for i in range(k)])
    m = Mesh(V, F, mat)
    if m.face_normals()[:, 1].mean() > 0:            # the glass faces outward (-y)
        m.flip()
    m.uv_box(1.0, only_missing=False)
    return m


def cusp_pair(x0, z_spring, w, depth, y0, mat=TRIM):
    """Two small cusps (trefoil points) inside a pointed head."""
    out = []
    for s in (-1, 1):
        tri = np.array([[x0 + s * w * 0.5, z_spring], [x0 + s * w * 0.18, z_spring + w * 0.18], [x0 + s * w * 0.5, z_spring + w * 0.42]])
        out.append(extrude_xz(tri, y0, y0 + depth))
    return out


def circle(r, n=24, cx=0.0, cz=0.0):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cz + r * np.sin(a)], 1)


def foil(cx, cz, r, lobes=4, n=48, rot=0.0):
    """A quatrefoil / trefoil outline."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rr = r * (0.72 + 0.28 * np.abs(np.cos(lobes * (a + rot) / 2)))
    return np.stack([cx + rr * np.cos(a), cz + rr * np.sin(a)], 1)


def finish(parts, uv_tile=1.0):
    m = mk.merge([p for p in parts if p is not None and p.nf])
    m.uv_box(uv_tile, only_missing=False)
    return m


def crocket_mesh(size=0.35, mat=TRIM, seg=6):
    """A curled leaf: a tapering tube swept along a hook."""
    t = np.linspace(0, 1, 9)
    path = np.stack([np.zeros_like(t), -size * (0.15 + 0.85 * np.sin(t * math.pi * 0.85)) * 0.9, size * (t * 1.1 - 0.35 * t ** 3)], 1)
    ex = np.tile([1.0, 0.0, 0.0], (len(t), 1))
    prof = mk.ngon(size * 0.18, seg)
    sc = 1.0 - 0.75 * t
    m = mk.sweep(prof, path, ex, caps=True, mat=mat, scale=sc)
    leaf = mk.sweep(mk.ngon(size * 0.1, 5), path * np.array([1, 1.1, 1.0]) + np.array([0.0, 0.0, size * 0.05]), ex, caps=True, mat=mat,
                    scale=1.4 - 1.2 * t)
    return mk.merge([m, leaf])


# --------------------------------------------------------------------------------------------- windows & doors
def win_lancet(w=1.1, h=3.0, depth=0.42):
    """Single lancet: moulded frame, cusped head, sill, leaded glass set back."""
    P = arch_outline(w, h, "pointed", 12)
    parts = [man(frame_ring(P, 0.16, depth), TRIM)]
    R = 0.85 * w
    spring = h - math.sqrt(R * R - (R - w / 2) ** 2)
    parts += [man(c, TRIM) for c in cusp_pair(0.0, spring, w - 0.32, 0.12, depth * 0.5)]
    parts.append(man(cube(-w / 2 - 0.12, w / 2 + 0.12, -0.08, depth, -0.18, 0.0), TRIM))           # sill
    parts.append(glass_pane(P, depth * 0.8, 0.16))
    return finish(parts)


def win_double(w=2.3, h=4.8, depth=0.5):
    """Two-light lancet: outer frame, central mullion, two cusped sub-arches, an oculus with a quatrefoil in the head."""
    P = arch_outline(w, h, "pointed", 16)
    parts = [man(frame_ring(P, 0.2, depth), TRIM)]
    sw = (w - 0.4 - 0.22) / 2
    R = 0.85 * w
    spring = h - math.sqrt(R * R - (R - w / 2) ** 2)
    sh = spring + 0.05
    for s in (-1, 1):
        cx = s * (sw / 2 + 0.11)
        Q = arch_outline(sw, sh - 0.2, "pointed", 10, cx, 0.0)
        parts.append(man(frame_ring(Q, 0.1, depth * 0.6, depth * 0.2), TRIM))
        parts += [man(c, TRIM) for c in cusp_pair(cx, sh - 0.2 - 0.85 * sw * 0.45, sw - 0.2, 0.1, depth * 0.4)]
        parts.append(glass_pane(Q, depth * 0.75, 0.1))
    parts.append(man(cube(-0.11, 0.11, depth * 0.1, depth * 0.85, 0.0, sh), TRIM))                  # mullion
    oc = circle(sw * 0.62, 32, 0.0, sh + (h - sh) * 0.38)
    parts.append(man(frame_ring(oc, 0.09, depth * 0.6, depth * 0.2), TRIM))
    qf = CS(foil(0.0, sh + (h - sh) * 0.38, sw * 0.5))
    parts.append(man((qf - qf.offset(-0.07, m3d.JoinType.Round)).extrude(0.1).rotate([90, 0, 0]).scale([1, -1, 1])
                     .translate([0, depth * 0.45, 0]), TRIM))
    parts.append(glass_pane(oc, depth * 0.8, 0.08))
    # spandrel infill between the sub-arches and the oculus (solid tracery)
    head = CS(offset_outline(P, 0.18))
    holes = CS(oc) + CS(arch_outline(sw, sh - 0.2, "pointed", 10, -(sw / 2 + 0.11), 0.0)) \
        + CS(arch_outline(sw, sh - 0.2, "pointed", 10, sw / 2 + 0.11, 0.0))
    band = CS(np.array([[-w, sh - 1.2], [w, sh - 1.2], [w, h + 1], [-w, h + 1]]))
    sp = ((head ^ band) - holes).extrude(0.14).rotate([90, 0, 0]).scale([1, -1, 1]).translate([0, depth * 0.5, 0])
    parts.append(man(sp, TRIM))
    parts.append(man(cube(-w / 2 - 0.15, w / 2 + 0.15, -0.1, depth, -0.2, 0.0), TRIM))
    return finish(parts)


def win_hall(w=3.8, h=9.5, depth=0.6):
    """The great hall's tall three-light window: two mullions, a transom, three cusped sub-heads, a sexfoil rose and
    two trefoils in the head."""
    P = arch_outline(w, h, "pointed", 20)
    parts = [man(frame_ring(P, 0.26, depth), TRIM)]
    R = 0.85 * w
    spring = h - math.sqrt(R * R - (R - w / 2) ** 2)
    lw = (w - 0.52 - 2 * 0.18) / 3
    xs = [-(lw + 0.18), 0.0, lw + 0.18]
    sub_h = spring - 0.1
    holes = []
    for cx in xs:
        Q = arch_outline(lw, sub_h, "pointed", 10, cx, 0.0)
        holes.append(Q)
        parts.append(man(frame_ring(Q, 0.08, depth * 0.55, depth * 0.2), TRIM))
        parts.append(glass_pane(Q, depth * 0.78, 0.08))
        parts += [man(c, TRIM) for c in cusp_pair(cx, sub_h - 0.85 * lw * 0.45, lw - 0.16, 0.1, depth * 0.4)]
    for xm in (-(lw / 2 + 0.09), lw / 2 + 0.09):                                                    # mullions
        parts.append(man(cube(xm - 0.09, xm + 0.09, depth * 0.1, depth * 0.85, 0.0, sub_h), TRIM))
    parts.append(man(cube(-w / 2 + 0.2, w / 2 - 0.2, depth * 0.1, depth * 0.8, sub_h * 0.48, sub_h * 0.48 + 0.18), TRIM))   # transom
    cz = spring + (h - spring) * 0.42
    rose = circle(lw * 0.95, 40, 0.0, cz)
    parts.append(man(frame_ring(rose, 0.1, depth * 0.55, depth * 0.2), TRIM))
    parts.append(man((CS(foil(0.0, cz, lw * 0.85, 6)) - CS(foil(0.0, cz, lw * 0.85, 6)).offset(-0.08))
                     .extrude(0.12).rotate([90, 0, 0]).scale([1, -1, 1]).translate([0, depth * 0.45, 0]), TRIM))
    parts.append(glass_pane(rose, depth * 0.8, 0.08))
    tre = []
    for s in (-1, 1):
        t = foil(s * lw * 1.15, spring + 0.25, lw * 0.42, 3, 36, math.pi / 2)
        tre.append(t)
        parts.append(man(frame_ring(t, 0.06, depth * 0.5, depth * 0.25), TRIM))
        parts.append(glass_pane(t, depth * 0.8, 0.05))
    head = CS(offset_outline(P, 0.24))
    band = CS(np.array([[-w, sub_h - 1.0], [w, sub_h - 1.0], [w, h + 1], [-w, h + 1]]))
    hl = CS(rose) + CS(tre[0]) + CS(tre[1])
    for Q in holes:
        hl = hl + CS(Q)
    parts.append(man(((head ^ band) - hl).extrude(0.16).rotate([90, 0, 0]).scale([1, -1, 1]).translate([0, depth * 0.5, 0]), TRIM))
    parts.append(man(cube(-w / 2 - 0.2, w / 2 + 0.2, -0.12, depth, -0.25, 0.0), TRIM))
    return finish(parts)


def win_rose(d=6.0, depth=0.6):
    """Rose window: outer moulded ring, 12 radiating lights with cusped heads, inner ring, central sexfoil.  z = 0 at
    the bottom of the circle."""
    R = d / 2
    cz = R
    outer = circle(R, 72, 0.0, cz)
    parts = [man(frame_ring(outer, 0.3, depth), TRIM)]
    inner = circle(R * 0.32, 40, 0.0, cz)
    parts.append(man(frame_ring(inner, 0.12, depth * 0.6, depth * 0.2), TRIM))
    for k in range(12):
        a = 2 * math.pi * k / 12
        c, s = math.cos(a), math.sin(a)
        # a spoke (radial mullion)
        r0, r1 = R * 0.32, R - 0.28
        spoke = np.array([[c * r0 - s * 0.07, cz + s * r0 + c * 0.07], [c * r1 - s * 0.07, cz + s * r1 + c * 0.07],
                          [c * r1 + s * 0.07, cz + s * r1 - c * 0.07], [c * r0 + s * 0.07, cz + s * r0 - c * 0.07]])
        parts.append(man(extrude_xz(spoke, depth * 0.2, depth * 0.75), TRIM))
        # a small cusped arch head between spokes (a trefoil near the rim)
        am = a + math.pi / 12
        t = foil(math.cos(am) * R * 0.78, cz + math.sin(am) * R * 0.78, R * 0.12, 3, 30, am)
        parts.append(man(frame_ring(t, 0.05, depth * 0.4, depth * 0.3), TRIM))
    parts.append(man((CS(foil(0.0, cz, R * 0.26, 6)) - CS(foil(0.0, cz, R * 0.26, 6)).offset(-0.06))
                     .extrude(0.1).rotate([90, 0, 0]).scale([1, -1, 1]).translate([0, depth * 0.45, 0]), TRIM))
    parts.append(glass_pane(outer, depth * 0.82, 0.28))
    return finish(parts)


def win_tudor(w=1.7, h=2.5, depth=0.36):
    """Square-headed mullion-and-transom window (2 x 2 lights) with a label (hood) mould: the ranges' windows."""
    P = arch_outline(w, h, "flat")
    parts = [man(frame_ring(P, 0.14, depth), TRIM)]
    parts.append(man(cube(-0.07, 0.07, depth * 0.1, depth * 0.85, 0.0, h), TRIM))                   # mullion
    parts.append(man(cube(-w / 2, w / 2, depth * 0.1, depth * 0.85, h * 0.62, h * 0.62 + 0.12), TRIM))    # transom
    for s in (-1, 1):                                                                                 # cusped light heads
        cx = s * (w / 4)
        parts.append(man(extrude_xz(np.array([[cx - w / 4 + 0.15, h - 0.14], [cx, h - 0.32], [cx + w / 4 - 0.15, h - 0.14]]),
                                    depth * 0.2, depth * 0.4), TRIM))
    # label mould: a projecting drip over the head returning down the sides
    lab = np.array([[-w / 2 - 0.26, h - 0.55], [-w / 2 - 0.14, h - 0.55], [-w / 2 - 0.14, h + 0.12], [w / 2 + 0.14, h + 0.12],
                    [w / 2 + 0.14, h - 0.55], [w / 2 + 0.26, h - 0.55], [w / 2 + 0.26, h + 0.26], [-w / 2 - 0.26, h + 0.26]])
    parts.append(man(extrude_xz(lab, -0.16, 0.02), TRIM))
    parts.append(man(cube(-w / 2 - 0.1, w / 2 + 0.1, -0.08, depth, -0.15, 0.0), TRIM))
    parts.append(glass_pane(P, depth * 0.8, 0.14))
    return finish(parts)


def win_small(w=0.8, h=1.9, depth=0.32, kind="round"):
    """Small tower window (round or pointed head) with a chamfered surround."""
    P = arch_outline(w, h, kind, 10)
    parts = [man(frame_ring(P, 0.12, depth), TRIM), glass_pane(P, depth * 0.8, 0.12)]
    parts.append(man(cube(-w / 2 - 0.08, w / 2 + 0.08, -0.06, depth, -0.12, 0.0), TRIM))
    return finish(parts)


def door_gothic(w=2.6, h=4.4, depth=0.9):
    """Pointed doorway: three receding moulded orders, plank doors with iron straps and studs."""
    P = arch_outline(w + 0.9, h + 0.6, "pointed", 16)
    parts = []
    for k in range(3):
        Q = offset_outline(P, 0.15 * k)
        parts.append(man(frame_ring(Q, 0.16, 0.3, 0.3 * k), TRIM))
    D = offset_outline(P, 0.45)
    leaf = CS(D).extrude(0.12).rotate([90, 0, 0]).scale([1, -1, 1]).translate([0, depth - 0.1, 0])
    parts.append(man(leaf, WOOD))
    for zz in (0.5, 1.6, 2.7):
        parts.append(man(cube(-w / 2 + 0.1, w / 2 - 0.1, depth - 0.16, depth - 0.1, zz, zz + 0.1), LEAD))
    parts.append(man(cube(-0.03, 0.03, depth - 0.18, depth - 0.1, 0.0, h * 0.85), LEAD))
    rng = np.random.default_rng(3)
    for zz in np.arange(0.35, 3.2, 0.55):
        for xx in np.linspace(-w / 2 + 0.3, w / 2 - 0.3, 5):
            parts.append(mk.box((xx, depth - 0.17, zz), (0.06, 0.04, 0.06), mat=LEAD))
    parts.append(man(cube(-w / 2 - 0.6, w / 2 + 0.6, -0.2, depth, -0.3, 0.0), STONE))     # threshold
    return finish(parts)


# --------------------------------------------------------------------------------------------- pinnacles & finials
def pinnacle(h=6.5, b=1.0):
    """Square pinnacle: chamfered shaft, gablets with crockets on the four faces, crocketed spirelet, fleuron finial."""
    parts = []
    shaft_h = h * 0.38
    sh = cube(-b / 2, b / 2, -b / 2, b / 2, 0.0, shaft_h)
    # chamfer the corners
    cut = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cut.append(m3d.Manifold.cube([0.18, 0.18, shaft_h * 3], True).rotate([0, 0, 45]).translate([sx * b / 2, sy * b / 2, shaft_h]))
    sh = sh - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add)
    parts.append(man(sh, TRIM))
    # gablets
    for k in range(4):
        tri = np.array([[-b / 2 - 0.05, 0.0], [b / 2 + 0.05, 0.0], [0.0, b * 0.95]])
        g = extrude_xz(tri, -0.08, 0.25).translate([0, -b / 2, shaft_h]).rotate([0, 0, 90 * k])
        parts.append(man(g, TRIM))
        for t in (0.3, 0.65):
            for s in (-1, 1):
                c = crocket_mesh(0.22)
                c.rotate_z(math.radians(90 * k))
                ang = math.radians(90 * k)
                lx, lz = s * (b / 2) * (1 - t), b * 0.95 * t
                c.translate((lx * math.cos(ang) + (b / 2 + 0.05) * math.sin(ang), lx * math.sin(ang) - (b / 2 + 0.05) * math.cos(ang),
                             shaft_h + lz))
                parts.append(c)
    # spirelet
    sp_h = h - shaft_h - 0.8
    spire = m3d.Manifold.cylinder(sp_h, b * 0.5 * math.sqrt(2), 0.04, 4).rotate([0, 0, 45]).translate([0, 0, shaft_h + 0.25])
    parts.append(man(spire, TRIM))
    for e in range(4):
        a = math.radians(45 + 90 * e)
        for t in np.linspace(0.15, 0.8, 4):
            r = b * 0.5 * math.sqrt(2) * (1 - t) + 0.04
            c = crocket_mesh(0.2 * (1 - 0.5 * t))
            c.rotate_z(a + math.pi / 2)
            c.translate((r * math.cos(a), r * math.sin(a), shaft_h + 0.25 + sp_h * t))
            parts.append(c)
    # fleuron finial: a knop + four leaves + a bud
    top = shaft_h + 0.25 + sp_h
    parts.append(mk.revolve([(0.001, top - 0.1), (0.12, top), (0.16, top + 0.12), (0.06, top + 0.25), (0.12, top + 0.35),
                             (0.04, top + 0.55), (0.001, top + 0.62)], 12, mat=TRIM))
    for k in range(4):
        c = crocket_mesh(0.18)
        c.rotate_z(math.radians(90 * k))
        c.translate((0.0, 0.0, top + 0.12))
        parts.append(c)
    return finish(parts)


def finial_orb(h=4.0):
    """Cone tip: a lead sleeve, an orb, a spike with a small weathervane."""
    parts = [mk.revolve([(0.001, 0.0), (0.32, 0.0), (0.26, 0.6), (0.14, 1.0), (0.3, 1.2), (0.3, 1.45), (0.14, 1.62), (0.05, 1.7),
                         (0.04, h * 0.8), (0.001, h)], 16, mat=LEAD)]
    vane = np.array([[0.0, h * 0.62], [0.9, h * 0.66], [0.75, h * 0.7], [0.95, h * 0.74], [0.0, h * 0.76]])
    parts.append(man(extrude_xz(vane, -0.01, 0.01), LEAD))
    for k in range(4):
        a = math.radians(90 * k)
        parts.append(mk.box((0.35 * math.cos(a), 0.35 * math.sin(a), h * 0.55), (0.05 + 0.25 * abs(math.cos(a)), 0.05 + 0.25 * abs(math.sin(a)), 0.04),
                            mat=LEAD))
    return finish(parts)


# --------------------------------------------------------------------------------------------- parapets, corbels
def merlon(w=1.4, d=0.75, h=1.5):
    """A merlon on a parapet: chamfered coping, an arrow slit through it."""
    body = cube(-w / 2, w / 2, 0.0, d, 0.0, h)
    cap = CS(np.array([[-w / 2 - 0.06, h - 0.02], [w / 2 + 0.06, h - 0.02], [w / 2 + 0.02, h + 0.12], [0.0, h + 0.24],
                                      [-w / 2 - 0.02, h + 0.12]])).extrude(d + 0.12).rotate([90, 0, 0]).scale([1, -1, 1]).translate([0, -0.06, 0])
    slit = cube(-0.05, 0.05, -0.1, d + 0.1, 0.35, h - 0.3) + cube(-0.22, 0.22, -0.1, d + 0.1, 0.7, 0.8)
    return finish([man((body - slit), STONE), man(cap, TRIM)])


def corbel(w=0.55, d=0.95, h=1.3):
    """Three-step corbel (machicolation bracket), projecting along -y from the wall face."""
    parts = []
    for k in range(3):
        z0 = h * k / 3
        y0 = -d * (k + 1) / 3
        parts.append(cube(-w / 2, w / 2, y0, 0.05, z0, z0 + h / 3))
    m = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    # round the soffits a little: chamfer cut under each step
    for k in range(3):
        y0 = -d * (k + 1) / 3
        m = m - m3d.Manifold.cube([w * 2, 0.2, 0.2], True).rotate([45, 0, 0]).translate([0, y0, h * k / 3])
    return finish([man(m, TRIM)])


def mach_arch(span=1.3, d=0.95, h=0.6):
    """The small arch between two corbels (carrying the parapet), open beneath (a machicolation)."""
    slab = cube(-span / 2, span / 2, -d, -d + 0.35, 0.0, h)
    a = arch_outline(span - 0.1, h * 0.8, "pointed", 8)
    return finish([man(slab - extrude_xz(a, -d - 0.1, -d + 0.5).translate([0, 0, -0.01]), TRIM)])


def baluster(h=0.9):
    prof = [(0.001, 0.0), (0.09, 0.0), (0.09, 0.06), (0.06, 0.1), (0.075, 0.2), (0.1, 0.36), (0.075, 0.55), (0.045, 0.66), (0.06, 0.72),
            (0.045, 0.78), (0.08, 0.8), (0.08, h), (0.001, h)]
    return finish([mk.revolve(prof, 12, mat=TRIM)])


def parapet_panel(w=2.0, h=1.25, d=0.35):
    """Pierced parapet panel: two quatrefoils in a sunk frame, a moulded coping."""
    body = cube(-w / 2, w / 2, 0.0, d, 0.0, h)
    holes = [CS(foil(s * w / 4, h * 0.5, h * 0.3)).extrude(d + 0.2).rotate([90, 0, 0]).scale([1, -1, 1]).translate([0, -0.1, 0])
             for s in (-1, 1)]
    cap = cube(-w / 2 - 0.02, w / 2 + 0.02, -0.06, d + 0.06, h, h + 0.14)
    return finish([man(body - m3d.Manifold.batch_boolean(holes, m3d.OpType.Add), TRIM), man(cap, TRIM)])


# --------------------------------------------------------------------------------------------- roof pieces
def lucarne(w=1.3, d=1.6, h=1.6):
    """A small gabled dormer for cones and spires: front wall with a window, steep roof, a spike.  Base at z = 0, the
    front at y = 0 facing -y, the body runs back into the roof (+y)."""
    body = cube(-w / 2, w / 2, 0.0, d, 0.0, h)
    gable = extrude_xz(np.array([[-w / 2 - 0.05, h], [w / 2 + 0.05, h], [0.0, h + w * 0.9]]), -0.02, d)
    win = extrude_xz(arch_outline(w * 0.5, h * 0.7, "pointed", 8, 0.0, h * 0.18), -0.1, 0.25)
    roof = []
    for s in (-1, 1):
        q = np.array([[0.0, h + w * 0.9 + 0.1], [s * (w / 2 + 0.15), h - 0.05], [s * (w / 2 + 0.25), h - 0.05], [0.0, h + w * 0.9 + 0.22]])
        roof.append(extrude_xz(q, -0.15, d + 0.2))
    parts = [man(body + gable - win, STONE), man(m3d.Manifold.batch_boolean(roof, m3d.OpType.Add), SLATE)]
    parts.append(glass_pane(arch_outline(w * 0.5, h * 0.7, "pointed", 8, 0.0, h * 0.18), 0.18, 0.0))
    parts.append(mk.revolve([(0.001, h + w * 0.9), (0.05, h + w * 0.9), (0.01, h + w * 0.9 + 0.9), (0.001, h + w * 0.9 + 0.95)], 8, mat=LEAD))
    return finish(parts)


def dormer(w=2.4, d=3.2, h=2.6):
    """Stone roof dormer: gabled front with a two-light window, coping, finial; slate cheeks and roof."""
    body = cube(-w / 2, w / 2, 0.0, d, 0.0, h)
    gable = extrude_xz(np.array([[-w / 2 - 0.1, h], [w / 2 + 0.1, h], [0.0, h + w * 0.85]]), -0.05, 0.45)
    win = extrude_xz(arch_outline(w * 0.55, h * 0.8, "flat", 4, 0.0, h * 0.15), -0.2, 0.35)
    cop = []
    for s in (-1, 1):
        q = np.array([[0.0, h + w * 0.85 + 0.18], [s * (w / 2 + 0.22), h - 0.02], [s * (w / 2 + 0.06), h - 0.02], [0.0, h + w * 0.85 - 0.02]])
        cop.append(extrude_xz(q, -0.1, 0.5))
    roof = []
    for s in (-1, 1):
        q = np.array([[0.0, h + w * 0.85 - 0.05], [s * (w / 2 + 0.2), h - 0.1], [s * (w / 2 + 0.28), h - 0.1], [0.0, h + w * 0.85 + 0.06]])
        roof.append(extrude_xz(q, 0.45, d + 0.3))
    parts = [man(body + gable - win, STONE), man(m3d.Manifold.batch_boolean(cop, m3d.OpType.Add), TRIM),
             man(m3d.Manifold.batch_boolean(roof, m3d.OpType.Add), SLATE)]
    parts.append(man(cube(-0.06, 0.06, 0.1, 0.3, h * 0.15, h * 0.95), TRIM))
    parts.append(glass_pane(arch_outline(w * 0.55, h * 0.8, "flat", 4, 0.0, h * 0.15), 0.3, 0.0))
    parts.append(mk.revolve([(0.001, h + w * 0.85), (0.1, h + w * 0.85), (0.12, h + w * 0.85 + 0.2), (0.03, h + w * 0.85 + 0.7),
                             (0.001, h + w * 0.85 + 0.8)], 8, mat=TRIM))
    return finish(parts)


def chimney(w=2.2, d=1.1, h=4.0, flues=3):
    """Chimney stack: plinth band, shaft, cornice, octagonal flues with moulded caps and pots."""
    parts = [man(cube(-w / 2, w / 2, -d / 2, d / 2, 0.0, h * 0.62), STONE), man(cube(-w / 2 - 0.12, w / 2 + 0.12, -d / 2 - 0.12, d / 2 + 0.12, h * 0.62, h * 0.7), TRIM)]
    for k in range(flues):
        x = -w / 2 + w * (k + 0.5) / flues
        parts.append(man(m3d.Manifold.cylinder(h * 0.28, 0.3, 0.3, 8).translate([x, 0.0, h * 0.7]), STONE))
        parts.append(man(m3d.Manifold.cylinder(0.14, 0.38, 0.38, 8).translate([x, 0.0, h * 0.98]), TRIM))
        parts.append(mk.revolve([(0.18, h * 0.98 + 0.14), (0.2, h * 0.98 + 0.5), (0.15, h * 0.98 + 0.62), (0.12, h * 0.98 + 0.62)], 10, mat=WOOD))
    return finish(parts)


def cresting(L=1.0):
    """Iron ridge cresting: a rail with fleur-de-lis spikes (two per metre)."""
    parts = [man(cube(-L / 2, L / 2, -0.03, 0.03, 0.0, 0.06), LEAD)]
    for x in (-L / 4, L / 4):
        parts.append(man(cube(x - 0.015, x + 0.015, -0.015, 0.015, 0.0, 0.55), LEAD))
        fl = np.array([[x, 0.75], [x + 0.08, 0.55], [x + 0.03, 0.58], [x, 0.45], [x - 0.03, 0.58], [x - 0.08, 0.55]])
        parts.append(man(extrude_xz(fl, -0.015, 0.015), LEAD))
        parts.append(mk.revolve([(0.001, 0.3), (0.05, 0.33), (0.001, 0.36)], 8, mat=LEAD).translate((x, 0.0, 0.0)))
    return finish(parts, 0.5)


# --------------------------------------------------------------------------------------------- figures, turrets, columns
def gargoyle(L=2.2):
    """A beast-headed water spout projecting from a parapet (along -y)."""
    t = np.linspace(0, 1, 12)
    path = np.stack([np.zeros_like(t), -L * t, 0.25 * np.sin(t * math.pi * 0.9) - 0.15 * t], 1)
    ex = np.tile([1.0, 0.0, 0.0], (len(t), 1))
    prof = np.array([[-0.22, -0.18], [0.22, -0.18], [0.26, 0.05], [0.12, 0.22], [-0.12, 0.22], [-0.26, 0.05]])
    sc = 1.0 - 0.35 * t + 0.25 * np.exp(-((t - 0.85) / 0.1) ** 2)
    body = mk.sweep(prof, path, ex, caps=True, mat=STONE, scale=sc)
    parts = [body]
    for s in (-1, 1):                                                       # ears / horns, eyes, folded wings
        parts.append(mk.revolve([(0.001, 0.0), (0.07, 0.0), (0.001, 0.32)], 6, mat=STONE).rotate_x(math.radians(-30)).translate(
            (s * 0.13, -L * 0.88, 0.28)))
        parts.append(mk.box((s * 0.1, -L * 0.98, 0.12), (0.06, 0.04, 0.06), mat=STONE))
        wing = np.array([[0.0, 0.0], [0.6, 0.1], [0.75, 0.45], [0.35, 0.3], [0.0, 0.35]])
        w = man(extrude_xz(wing, -0.04, 0.04), STONE)
        w.rotate_z(math.radians(90.0 * s))
        w.translate((s * 0.24, -L * 0.45, 0.05))
        parts.append(w)
    parts.append(mk.revolve([(0.001, 0.0), (0.09, 0.0), (0.09, 0.25), (0.001, 0.25)], 8, mat=LEAD).rotate_x(math.radians(90)).translate(
        (0.0, -L * 1.0, 0.0)))
    return finish(parts)


def bartizan(r=1.5, h=6.5, roof_h=5.0):
    """Corbelled corner turret: corbelled cone under, body with slits, corbel ring, conical roof, finial.  Its axis at
    (0, 0); z = 0 is the bottom of the corbelling."""
    parts = [mk.revolve([(0.001, 0.0), (0.25, 0.0), (r * 0.6, 0.8), (r * 0.85, 1.6), (r, 2.2)], 24, mat=TRIM)]
    body = m3d.Manifold.cylinder(h - 2.2, r, r, 24).translate([0, 0, 2.2])
    sl = [cube(-0.05, 0.05, -r - 0.2, 0.0, 3.4, 4.6).rotate([0, 0, a]) for a in (0, 120, 240)]
    parts.append(man(body - m3d.Manifold.batch_boolean(sl, m3d.OpType.Add), STONE))
    parts.append(mk.revolve([(r - 0.05, h - 0.4), (r + 0.25, h - 0.3), (r + 0.25, h), (r - 0.1, h)], 24, mat=TRIM))
    parts.append(mk.revolve([(0.001, h), (r + 0.35, h), (r + 0.2, h + 0.15), (0.001, h + roof_h)], 24, mat=SLATE))
    parts.append(mk.revolve([(0.001, h + roof_h - 0.2), (0.07, h + roof_h - 0.2), (0.12, h + roof_h + 0.1), (0.02, h + roof_h + 1.2),
                             (0.001, h + roof_h + 1.3)], 8, mat=LEAD))
    return finish(parts)


def chamfered_block(x0, x1, y0, y1, z0, z1, c=0.03, mat=TRIM):
    """A dressed stone: a box with chamfered edges (the vertical and horizontal arrises)."""
    P = np.array([[x0 + c, y0], [x1 - c, y0], [x1, y0 + c], [x1, y1 - c], [x1 - c, y1], [x0 + c, y1], [x0, y1 - c], [x0, y0 + c]])
    body = CS(P).extrude(z1 - z0 - 2 * c).translate([0, 0, z0 + c])
    core = cube(x0 + c, x1 - c, y0 + c, y1 - c, z0, z1)
    return man(m3d.Manifold.batch_boolean([body, core], m3d.OpType.Add), mat)


def quoin_pair(long=0.95, short=0.5, h=0.45, proud=0.05):
    """Two courses of corner quoins: the building fills x >= 0, y >= 0 (faces y = 0 and x = 0 meet at the origin); the lower
    stone runs long along x and short along y, the upper one the other way; both stand `proud` out of the faces."""
    return finish([chamfered_block(-proud, long, -proud, short, 0.0, h - 0.012),
                   chamfered_block(-proud, short, -proud, long, h, 2 * h - 0.012)])


def lantern(post=1.3):
    """A Gothic lantern on a short post (for wall copings, landings, the quay): moulded foot, post, a hexagonal cage of
    glowing panes between iron bars, a pointed cap with a finial.  z = 0 at the coping."""
    parts = [man(cube(-0.22, 0.22, -0.22, 0.22, 0.0, 0.14), TRIM),
             mk.revolve([(0.001, 0.14), (0.12, 0.14), (0.07, 0.3), (0.05, post - 0.12), (0.09, post - 0.06), (0.16, post),
                         (0.001, post)], 10, mat=LEAD)]
    r, h = 0.26, 0.62
    z0 = post
    ring_ = mk.revolve([(0.001, z0), (r + 0.06, z0), (r + 0.06, z0 + 0.06), (0.001, z0 + 0.06)], 6, mat=LEAD)
    parts.append(ring_)
    a = np.linspace(0, 2 * math.pi, 7)[:-1] + math.pi / 6
    for k in range(6):
        p0 = np.array([r * math.cos(a[k]), r * math.sin(a[k])])
        p1 = np.array([r * math.cos(a[(k + 1) % 6]), r * math.sin(a[(k + 1) % 6])])
        V = np.array([[p0[0], p0[1], z0 + 0.06], [p1[0], p1[1], z0 + 0.06], [p1[0], p1[1], z0 + h], [p0[0], p0[1], z0 + h]])
        pane = Mesh(V, np.array([[0, 1, 2], [0, 2, 3]]), np.full(2, LANTERN, np.int16))
        if pane.face_normals()[0, :2] @ (p0 + p1) < 0:
            pane.flip()
        parts.append(pane)
        parts.append(mk.box((p0[0], p0[1], z0 + 0.06 + h / 2), (0.035, 0.035, h), mat=LEAD))
    parts.append(mk.revolve([(0.001, z0 + h), (r + 0.1, z0 + h), (r + 0.08, z0 + h + 0.06), (0.1, z0 + h + 0.42), (0.03, z0 + h + 0.48),
                             (0.05, z0 + h + 0.56), (0.001, z0 + h + 0.72)], 6, mat=LEAD))
    return finish(parts)


def column(h=4.0, r=0.28):
    """Arcade column: moulded base, shaft, foliate bell capital, square abacus."""
    parts = [man(cube(-r * 1.6, r * 1.6, -r * 1.6, r * 1.6, 0.0, 0.25), TRIM),
             mk.revolve([(0.001, 0.25), (r * 1.4, 0.25), (r * 1.25, 0.4), (r * 1.05, 0.48), (r, 0.6), (r, h - 0.75), (r * 1.1, h - 0.7),
                         (r * 1.5, h - 0.3), (r * 1.6, h - 0.25), (0.001, h - 0.25)], 16, mat=TRIM),
             man(cube(-r * 1.8, r * 1.8, -r * 1.8, r * 1.8, h - 0.25, h), TRIM)]
    for k in range(8):
        c = crocket_mesh(0.22)
        c.rotate_z(math.radians(45 * k))
        c.translate((0.0, 0.0, h - 0.62))
        parts.append(c)
    return finish(parts)


def clock_face(d=6.4, depth=0.5):
    """Clock: moulded stone ring, glowing dial, twelve raised numeral blocks, hands.  z = 0 at the dial's bottom."""
    R = d / 2
    cz = R
    parts = [man(frame_ring(circle(R, 72, 0.0, cz), 0.45, depth), TRIM)]
    dial = glass_pane(circle(R - 0.45, 72, 0.0, cz), depth * 0.6, 0.0, CLOCK)
    parts.append(dial)
    for k in range(12):
        a = math.radians(90 - 30 * k)
        rr = R - 0.85
        x, z = rr * math.cos(a), cz + rr * math.sin(a)
        blk = cube(-0.07, 0.07, depth * 0.45, depth * 0.58, -0.3, 0.3).rotate([0, -math.degrees(a) + 90, 0]).translate([x, 0, z])
        parts.append(man(blk, LEAD))
        if k % 3 == 0:
            parts.append(man(cube(-0.07, 0.07, depth * 0.45, depth * 0.58, -0.3, 0.3).rotate([0, -math.degrees(a) + 90, 0]).translate(
                [x + 0.2 * math.sin(a), 0, z - 0.2 * math.cos(a)]), LEAD))
    parts.append(man(cube(-0.08, 0.08, depth * 0.4, depth * 0.5, 0.0, R * 0.55).rotate([0, 50, 0]).translate([0, 0, cz]), LEAD))
    parts.append(man(cube(-0.06, 0.06, depth * 0.36, depth * 0.46, 0.0, R * 0.8).rotate([0, -95, 0]).translate([0, 0, cz]), LEAD))
    parts.append(mk.revolve([(0.001, 0.0), (0.18, 0.0), (0.12, 0.12), (0.001, 0.12)], 12, mat=LEAD).rotate_x(math.radians(90)).translate(
        (0.0, depth * 0.38, cz)))
    return finish(parts)


MODULES = {
    "c_lancet": win_lancet, "c_double": win_double, "c_hallwin": win_hall, "c_rose": win_rose, "c_tudor": win_tudor,
    "c_smallwin": win_small, "c_door": door_gothic, "c_pinnacle": pinnacle, "c_finial": finial_orb, "c_merlon": merlon,
    "c_corbel": corbel, "c_macharch": mach_arch, "c_baluster": baluster, "c_panel": parapet_panel, "c_lucarne": lucarne,
    "c_dormer": dormer, "c_chimney": chimney, "c_cresting": cresting, "c_gargoyle": gargoyle, "c_bartizan": bartizan,
    "c_column": column, "c_clock": clock_face, "c_crocket": lambda: finish([crocket_mesh(0.4)]), "c_lantern": lantern,
    "c_quoin": quoin_pair,
}

# the size of each module's "opening" (w, h) for the builders that fit windows into openings
OPENING = {"c_lancet": (1.1, 3.0), "c_double": (2.3, 4.8), "c_hallwin": (3.8, 9.5), "c_tudor": (1.7, 2.5), "c_smallwin": (0.8, 1.9),
           "c_door": (3.5, 5.0), "c_rose": (6.0, 6.0), "c_clock": (6.4, 6.4)}
OPENING_KIND = {"c_lancet": "pointed", "c_double": "pointed", "c_hallwin": "pointed", "c_tudor": "flat", "c_smallwin": "round",
                "c_door": "pointed"}


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tot = 0
    for name, fn in MODULES.items():
        m = fn()
        m.save(f"{out_dir}/lib_{name}.npz")
        tot += m.nf
        lo, hi = m.V.min(0), m.V.max(0)
        print(f"  lib_{name:12s} {m.nf:7,d} tris  size {np.round(hi - lo, 2)}", flush=True)
    print(f"castle detail library: {len(MODULES)} modules, {tot:,} tris", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo/lib")
