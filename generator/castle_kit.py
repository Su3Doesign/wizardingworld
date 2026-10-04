"""castle_kit - procedural Gothic castle architecture (towers, halls, ranges, walls, roofs, windows, viaduct).

Conventions: metres, Z up.  Solids are built with manifold3d (windows and doors are boolean cuts, so every opening has
real reveals); roofs are explicit meshes with the slate courses modelled as small steps, so each row of slates casts a
shadow line.  Every piece is returned as meshkit Meshes carrying material ids:

    STONE  rough ashlar walls            TRIM   dressed stone: window / door reveals, string courses, copings, quoins,
    SLATE  roof slates                          buttress caps, tracery, pinnacles, merlon caps
    LEAD   lead / dark metal: ridges, finials, spire tips, clock faces
    GLASS  leaded glass (UV2 = (random, lit flag): the night preset lights ~60 % of the windows)
    WOOD   doors, bridge timbers

Stone and trim use world-aligned (triplanar) texturing in Unreal, so they only carry rough box-projected UV0; slate
carries course-aligned UV0 (u along the eave, v up the slope, metres).  UV1 = (grime, moss) masks for stone.
"""
from __future__ import annotations

import math

import numpy as np
import manifold3d as m3d

import meshkit as mk
from meshkit import Mesh

STONE, TRIM, SLATE, LEAD, GLASS, WOOD = 20, 21, 22, 23, 24, 25
_ID_MAT = {}                    # manifold original id -> material


def _orig(man, mat):
    man = man.as_original()
    _ID_MAT[int(man.original_id())] = mat
    return man


def man_to_mesh(man, default=STONE):
    mm = man.to_mesh()
    V = np.asarray(mm.vert_properties, np.float64)[:, :3]
    F = np.asarray(mm.tri_verts, np.int64)
    mat = np.full(len(F), default, np.int16)
    ro = np.asarray(mm.run_original_id)
    ri = np.asarray(mm.run_index) // 3
    for k, oid in enumerate(ro):
        mat[ri[k]:ri[k + 1]] = _ID_MAT.get(int(oid), default)
    return Mesh(V, F, mat)


def frame(origin, xdir, ydir, zdir):
    """3x4 affine matrix mapping local x/y/z to the given world directions (right handed expected)."""
    M = np.zeros((3, 4))
    M[:, 0] = xdir
    M[:, 1] = ydir
    M[:, 2] = zdir
    M[:, 3] = origin
    return M


def box(c, size, yaw=0.0, mat=STONE):
    b = m3d.Manifold.cube(list(size), True)
    if yaw:
        b = b.rotate([0.0, 0.0, math.degrees(yaw)])
    return _orig(b.translate(list(c)), mat)


def cyl(c, r, z0, z1, seg=48, mat=STONE, r_top=None):
    m = m3d.Manifold.cylinder(z1 - z0, r, r if r_top is None else r_top, seg)
    return _orig(m.translate([c[0], c[1], z0]), mat)


def prism(poly, z0, z1, mat=STONE):
    cs = m3d.CrossSection([np.asarray(poly, np.float64)])
    return _orig(cs.extrude(z1 - z0).translate([0.0, 0.0, z0]), mat)


def sweep_poly(poly2d, origin, t, n, depth, mat=STONE):
    """Extrude a 2D polygon given in (u along t, v up) by `depth` along -n, starting at `origin`."""
    up = np.array([0.0, 0.0, 1.0])
    cs = m3d.CrossSection([np.asarray(poly2d, np.float64)])
    ex = cs.extrude(depth)
    return _orig(ex.transform(frame(origin, t, up, -np.asarray(n))), mat)


# ----------------------------------------------------------------------------------------------- 2D shapes
def lancet(w, h, n_arc=7, kind="pointed"):
    """Window / door outline (u, v): bottom centred at (0, 0), width w, height h; pointed (equilateral) or round arch."""
    hw = w / 2.0
    if kind == "round":
        vs = h - hw
        a = np.linspace(0.0, math.pi, 2 * n_arc + 1)
        arc = np.stack([hw * np.cos(a), vs + hw * np.sin(a)], 1)
        return np.vstack([[[-hw, 0.0], [hw, 0.0]], arc])
    if kind == "flat":
        return np.array([[-hw, 0.0], [hw, 0.0], [hw, h], [-hw, h]])
    rise = min(w * 0.866, h * 0.6)
    vs = h - rise
    R = w if rise > w * 0.8 else (hw ** 2 + rise ** 2) / (2 * hw)
    # right half: arc centred at (hw - R, vs) from angle 0 to the apex
    ang_top = math.acos(min(1.0, (hw - (hw - R)) / R)) if R > 0 else 0
    cxr = hw - R
    a_apex = math.atan2(rise, -cxr)
    a = np.linspace(0.0, a_apex, n_arc + 1)
    right = np.stack([cxr + R * np.cos(a), vs + R * np.sin(a)], 1)
    right[-1] = (0.0, h)
    left = right[::-1].copy()
    left[:, 0] *= -1
    pts = np.vstack([[[-hw, 0.0], [hw, 0.0]], right, left[1:]])
    return pts


def fan_triangulate(poly):
    """Triangulate a star-shaped polygon (all our openings are) from its centroid."""
    P = np.asarray(poly, np.float64)
    c = P.mean(0)
    V = np.vstack([P, c])
    k = len(P)
    F = np.array([[i, (i + 1) % k, k] for i in range(k)])
    return V, F


# ----------------------------------------------------------------------------------------------- windows
class Openings:
    """Collects window / door cutters for one wall solid, plus the glass, mullions and hood moulds that go with them."""

    def __init__(self, rng):
        self.cutters = []
        self.meshes = []
        self.rng = rng

    def window(self, P, n, w, h, depth=1.2, kind="pointed", mullions=0, transom=False, hood=True, glass=True, lit=None,
               reveal=0.55, sill=True, z_off=0.0):
        """P = point on the outer wall face at the window's bottom centre, n = outward normal (horizontal)."""
        n = np.asarray(n, np.float64)
        n = n / np.linalg.norm(n)
        up = np.array([0.0, 0.0, 1.0])
        t = np.cross(n, up)
        P = np.asarray(P, np.float64) + up * z_off
        poly = lancet(w, h, kind=kind)
        self.cutters.append(sweep_poly(poly, P + n * 0.6, t, n, depth + 0.6, TRIM))
        if glass:
            V2, F2 = fan_triangulate(poly)
            Vw = P[None] - n[None] * reveal + V2[:, :1] * t[None] + V2[:, 1:2] * up[None]
            g = Mesh(Vw, F2[:, ::-1] if np.dot(np.cross(Vw[F2[0, 1]] - Vw[F2[0, 0]], Vw[F2[0, 2]] - Vw[F2[0, 0]]), n) < 0 else F2, GLASS)
            uv = np.stack([(V2[:, 0] + w / 2) / w, V2[:, 1] / h], 1)
            g.uv0 = uv[g.F]
            lit_flag = (self.rng.random() < 0.62) if lit is None else lit
            g.uv2 = np.broadcast_to(np.array([self.rng.random(), 1.0 if lit_flag else 0.0]), (g.nf, 3, 2)).copy()
            self.meshes.append(g)
        # mullions / transom (dressed stone bars inside the opening)
        bar = min(0.16, w * 0.08)
        if mullions:
            for k in range(1, mullions + 1):
                u = -w / 2 + w * k / (mullions + 1)
                hh = h - (w * 0.45 if kind == "pointed" else w * 0.3)
                c = P - n * (reveal - 0.06) + t * u + up * (hh / 2)
                self.meshes.append(_mesh_box(c, (bar, 0.18, hh), t, n, TRIM))
        if transom:
            hh = (h - w * 0.866) * 0.62
            c = P - n * (reveal - 0.06) + up * hh
            self.meshes.append(_mesh_box(c, (w, 0.18, bar), t, n, TRIM))
        if sill:
            c = P + n * 0.06 - up * 0.10
            self.meshes.append(_mesh_box(c, (w + 0.35, 0.34, 0.18), t, n, TRIM))
        if hood and kind != "flat":
            self.meshes.append(_hood(P, n, t, w, h, kind))

    def door(self, P, n, w, h, depth=1.6, kind="pointed"):
        n = np.asarray(n, np.float64)
        n = n / np.linalg.norm(n)
        up = np.array([0.0, 0.0, 1.0])
        t = np.cross(n, up)
        poly = lancet(w, h, kind=kind)
        self.cutters.append(sweep_poly(poly, np.asarray(P) + n * 0.6, t, n, depth + 0.6, TRIM))
        V2, F2 = fan_triangulate(poly)
        Vw = np.asarray(P)[None] - n[None] * (depth * 0.6) + V2[:, :1] * t[None] + V2[:, 1:2] * up[None]
        d = Mesh(Vw, F2, WOOD)
        if np.dot(d.face_normals()[0], n) < 0:
            d.flip()
        d.uv0 = (np.stack([V2[:, 0], V2[:, 1]], 1) / 1.0)[d.F]
        self.meshes.append(d)
        self.meshes.append(_hood(np.asarray(P), n, t, w, h, kind, thick=0.3))

    def apply(self, solid):
        if not self.cutters:
            return solid
        cut = m3d.Manifold.batch_boolean(self.cutters, m3d.OpType.Add) if len(self.cutters) > 1 else self.cutters[0]
        return solid - cut


def _mesh_box(c, size, t, n, mat):
    """Oriented box mesh: size = (along t, along n, along z)."""
    up = np.array([0.0, 0.0, 1.0])
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    corners = []
    for z in (-sz, sz):
        for y in (-sy, sy):
            for x in (-sx, sx):
                corners.append(np.asarray(c) + t * x + n * y + up * z)
    V = np.array(corners)
    F = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    m = Mesh(V, F, mat)
    if m.volume() < 0:
        m.flip()
    m.uv_box(1.0)
    return m


def _hood(P, n, t, w, h, kind, thick=0.16):
    """Hood mould: a small square-section moulding following the arch, standing proud of the wall."""
    up = np.array([0.0, 0.0, 1.0])
    poly = lancet(w + 0.36, h + 0.2, n_arc=10, kind=kind)
    # arch part only (skip the two bottom corners and the vertical jambs below the springing)
    arc = poly[2:]
    vs = h - min(w * 0.866, h * 0.6) if kind == "pointed" else h - w / 2
    keep = arc[:, 1] >= vs - 0.05
    arc = arc[keep]
    if len(arc) < 3:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64), TRIM)
    path = np.asarray(P)[None] + n[None] * (thick / 2) + arc[:, :1] * t[None] + (arc[:, 1:2] - 0.1) * up[None]
    ex = np.repeat(n[None], len(path), 0)
    prof = mk.rect_profile(thick, thick)
    m = mk.sweep(prof, path, ex, caps=True, mat=TRIM)
    m.uv_box(1.0, only_missing=False)
    return m


# ----------------------------------------------------------------------------------------------- roofs
def cone_roof(c, r, z0, height, seg=64, eave=0.55, flare=0.18, course=0.42, step=0.035, finial=True, mat=SLATE):
    """Conical slate roof with a bell-cast eave and modelled slate courses; lead finial."""
    L = math.hypot(r + eave, height)
    n_c = max(6, int(L / course))
    s = np.linspace(0.0, 1.0, n_c + 1)
    # bell-cast: the lowest 18 % of the slant is flatter
    prof_r = (r + eave) * (1 - s)
    prof_z = z0 - 0.25 + height * (s + flare * (s - 1) * np.clip(1 - s / 0.18, 0, 1) * 0.0) - flare * (1 - np.clip(s / 0.18, 0, 1)) ** 2 * (r + eave) * 0.35
    # sawtooth: each course starts proud of the roof plane
    rings_r, rings_z = [], []
    for i in range(n_c):
        rings_r += [prof_r[i] + step, prof_r[i + 1]]
        rings_z += [prof_z[i] - step * 0.6, prof_z[i + 1]]
    rings_r.append(0.0)
    rings_z.append(z0 - 0.25 + height)
    R = np.array(rings_r)
    Z = np.array(rings_z)
    nseg = seg
    ang = np.linspace(0.0, 2 * math.pi, nseg + 1)
    P = np.empty((nseg + 1, len(R), 3))
    P[:, :, 0] = c[0] + np.cos(ang)[:, None] * R[None, :]
    P[:, :, 1] = c[1] + np.sin(ang)[:, None] * R[None, :]
    P[:, :, 2] = Z[None, :]
    m = mk.grid(P, uv_tile=1.0, mat=mat)
    # course-aligned UVs: u = arc length round the cone at the eave, v = slant distance
    uu = (ang[:, None] * (r + eave)) * np.ones_like(R)[None, :]
    vv = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(R), np.diff(Z)))])[None, :] * np.ones((nseg + 1, 1))
    UV = np.stack([uu, vv], -1).reshape(-1, 2)
    m.uv0 = UV[m.F]
    if np.mean(m.face_normals()[:, 2]) < 0:
        m.flip()
    parts = [m]
    # eave soffit (closes the roof from below)
    so = mk.revolve([(r + eave + step, z0 - 0.26 - step * 0.6), (r * 0.98, z0 - 0.26)], seg=nseg, mat=TRIM)
    so.translate((c[0], c[1], 0.0))
    if np.mean(so.face_normals()[:, 2]) > 0:
        so.flip()
    so.uv_box(1.0, only_missing=False)
    parts.append(so)
    if finial:
        ztop = z0 - 0.25 + height
        fh = max(1.6, height * 0.12)
        sp = mk.revolve([(0.22, ztop - 0.4), (0.18, ztop + fh * 0.35), (0.30, ztop + fh * 0.42), (0.18, ztop + fh * 0.5), (0.05, ztop + fh)], seg=12, mat=LEAD)
        sp.translate((c[0], c[1], 0.0))
        sp.uv_box(1.0, only_missing=False)
        parts.append(sp)
    return mk.merge(parts)


def gable_roof(c, axis, length, width, z_eave, pitch_deg=55.0, overhang=0.45, course=0.42, step=0.035, seg_len=2.0,
               mat=SLATE, ridge=True):
    """Two slate slopes over a rectangle (centre c, long axis unit vector `axis`), modelled courses, lead ridge."""
    a = np.array([axis[0], axis[1], 0.0])
    a /= np.linalg.norm(a)
    b = np.array([-a[1], a[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    half = width / 2 + overhang
    rise = (width / 2) * math.tan(math.radians(pitch_deg))
    z_r = z_eave + rise
    z_e = z_eave - overhang * math.tan(math.radians(pitch_deg))
    slant = math.hypot(half, z_r - z_e)
    n_c = max(4, int(slant / course))
    nu = max(2, int(math.ceil((length + 2 * overhang) / seg_len)))
    us = np.linspace(-(length / 2 + overhang), length / 2 + overhang, nu + 1)
    parts = []
    for side in (1.0, -1.0):
        s = np.linspace(0.0, 1.0, n_c + 1)
        dd = half * (1 - s)                     # horizontal distance from the ridge line
        zz = z_e + (z_r - z_e) * s
        nrm = np.array([0.0, 0.0, 0.0])
        rows_d, rows_z = [], []
        k = (z_r - z_e) / half
        sn = 1.0 / math.sqrt(1 + k * k)
        for i in range(n_c):
            rows_d += [dd[i] + step * k * sn, dd[i + 1]]
            rows_z += [zz[i] + step * sn, zz[i + 1]]
        D = np.array(rows_d)
        Z = np.array(rows_z)
        P = (np.asarray(c, np.float64)[None, None, :] + us[:, None, None] * a[None, None, :]
             + side * D[None, :, None] * b[None, None, :] + Z[None, :, None] * up[None, None, :])
        P[:, :, 2] = Z[None, :]
        m = mk.grid(P, uv_tile=1.0, mat=mat)
        vv = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(D), np.diff(Z)))])
        UV = np.stack(np.meshgrid(us, vv, indexing="ij"), -1).reshape(-1, 2)
        m.uv0 = UV[m.F]
        if np.mean(m.face_normals()[:, 2]) < 0:
            m.flip()
        parts.append(m)
    if ridge:
        rc = np.asarray(c, np.float64) + up * (z_r + 0.06)
        parts.append(_mesh_box(rc, (length + 2 * overhang + 0.1, 0.42, 0.30), a, b, LEAD))
    out = mk.merge(parts)
    return out, z_r


def gable_wall(c, axis, width, z_eave, pitch_deg, thick=1.2, coping=True, mat=STONE, end=1.0, length=None):
    """Triangular stone gable (closing a gable roof) at one end of the building; with a raised coping along its edges."""
    a = np.array([axis[0], axis[1], 0.0])
    a /= np.linalg.norm(a)
    b = np.array([-a[1], a[0], 0.0])
    rise = (width / 2) * math.tan(math.radians(pitch_deg))
    tri = np.array([[-width / 2 - 0.25, 0.0], [width / 2 + 0.25, 0.0], [0.0, rise + 0.35]])
    origin = np.asarray(c, np.float64) + np.array([0, 0, z_eave])
    n = a * end
    g = sweep_poly(tri, origin + n * (thick / 2), b, n, thick, mat)
    parts = [g]
    if coping:
        # copings: two thin slabs along the sloping edges
        for sgn in (1.0, -1.0):
            p0 = origin + b * sgn * (width / 2 + 0.25)
            p1 = origin + np.array([0, 0, rise + 0.35])
            mid = (p0 + p1) / 2
            L = np.linalg.norm(p1 - p0)
            ang = math.atan2(p1[2] - p0[2], -sgn * 1.0 * (width / 2 + 0.25))
            cop = m3d.Manifold.cube([L + 0.3, thick + 0.3, 0.32], True)
            # orient: x along the edge direction (in the b-z plane), y along a
            e = (p1 - p0) / L
            nn = np.cross(a, e)
            if nn[2] < 0:
                nn = -nn
            M = frame(mid + nn * 0.12, e, a, nn)
            if np.linalg.det(M[:, :3]) < 0:
                M[:, 1] = -M[:, 1]
            parts.append(_orig(cop.transform(M), TRIM))
    return parts


# ----------------------------------------------------------------------------------------------- small elements
def pinnacle(c, z0, h, r=0.6, mat=TRIM):
    """Gothic pinnacle: octagonal shaft + crocketed (stepped) spirelet."""
    shaft = mk.revolve([(r, z0), (r, z0 + h * 0.45), (r * 1.15, z0 + h * 0.47), (r * 1.15, z0 + h * 0.5), (r * 0.9, z0 + h * 0.52),
                        (0.0, z0 + h)], seg=8, mat=mat)
    shaft.translate((c[0], c[1], 0.0))
    shaft.uv_box(1.0, only_missing=False)
    return shaft


def merlons(path, z_top, thick, h=1.6, w=1.5, gap=1.0, mat=STONE, closed=False):
    """Crenellations along a polyline (merlon boxes + caps)."""
    P = np.asarray(path, np.float64)
    if closed:
        P = np.vstack([P, P[:1]])
    parts = []
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        d = p1[:2] - p0[:2]
        L = np.linalg.norm(d)
        if L < 0.5:
            continue
        a = np.array([d[0] / L, d[1] / L, 0.0])
        nb = np.array([-a[1], a[0], 0.0])
        k = max(1, int((L + gap) // (w + gap)))
        span = k * w + (k - 1) * gap
        start = (L - span) / 2 + w / 2
        for j in range(k):
            u = start + j * (w + gap)
            cc = np.array([p0[0], p0[1], 0.0]) + a * u
            cc[2] = z_top + h / 2
            parts.append(_mesh_box(cc, (w, thick, h), a, nb, mat))
            cc2 = cc.copy()
            cc2[2] = z_top + h + 0.08
            parts.append(_mesh_box(cc2, (w + 0.12, thick + 0.12, 0.16), a, nb, TRIM))
    return parts


def ring_course(c, r, z, h=0.35, proj=0.22, seg=64, mat=TRIM):
    """String course round a cylinder."""
    m = mk.revolve([(r - 0.05, z - h / 2), (r + proj, z - h / 2 + 0.06), (r + proj, z + h / 2), (r - 0.05, z + h / 2)], seg=seg, mat=mat)
    m.translate((c[0], c[1], 0.0))
    m.uv_box(1.0, only_missing=False)
    return m


def corbel_ring(c, r, z, n=None, size=0.45, mat=TRIM):
    n = n or max(8, int(2 * math.pi * r / 1.3))
    parts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        d = np.array([math.cos(a), math.sin(a), 0.0])
        t = np.array([-d[1], d[0], 0.0])
        cc = np.array([c[0], c[1], z]) + d * (r + size * 0.45)
        parts.append(_mesh_box(cc, (size * 0.8, size, size * 1.6), t, d, mat))
    return parts


# ----------------------------------------------------------------------------------------------- buildings
def _floors(z0, h_wall, floor_h=5.2, first=1.4):
    n = max(1, int((h_wall - first - 2.5) // floor_h))
    return [z0 + first + i * floor_h for i in range(n)]


def round_tower(c, r, z_base, h_wall, roof_h=None, z_foot=None, seg=64, rng=None, roof="cone", windows=True, win_w=1.3,
                win_h=3.0, per_floor=None, parapet=False, courses=True, door=None, roof_seg=None, cone_eave=0.55):
    """Round tower: battered plinth from z_foot, string courses, windows per floor, conical roof or crenellated top."""
    rng = rng or np.random.default_rng(0)
    z_foot = z_base - 18.0 if z_foot is None else z_foot
    top = z_base + h_wall
    drum = cyl(c, r, z_foot, top, seg, STONE)
    plinth = cyl(c, r + 0.8, z_foot, z_base + 1.2, seg, STONE, r_top=r + 0.05)
    op = Openings(rng)
    if windows:
        k = per_floor or max(3, int(2 * math.pi * r / 6.5))
        for fi, z in enumerate(_floors(z_base, h_wall - (3.0 if parapet else 1.0))):
            off = rng.uniform(0, 2 * math.pi) if fi == 0 else (fi * 0.5)
            for j in range(k):
                a = off + 2 * math.pi * j / k
                n = np.array([math.cos(a), math.sin(a), 0.0])
                if rng.random() < 0.12:
                    continue
                P = np.array([c[0], c[1], z]) + n * r
                op.window(P, n, win_w, win_h, depth=r * 0.45 + 0.6, mullions=0, hood=True)
    if door is not None:
        a = door
        n = np.array([math.cos(a), math.sin(a), 0.0])
        op.door(np.array([c[0], c[1], z_base]) + n * r, n, 2.4, 4.2)
    solid = op.apply(drum)
    parts = [man_to_mesh(solid), man_to_mesh(plinth)] + op.meshes
    if courses:
        for z in _floors(z_base, h_wall)[1:]:
            parts.append(ring_course(c, r, z - 0.6, seg=seg))
    if parapet:
        parts += corbel_ring(c, r, top - 0.9)
        ring = [(c[0] + (r + 0.45) * math.cos(a), c[1] + (r + 0.45) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, max(12, int(r * 2.2)), endpoint=False)]
        par = cyl(c, r + 0.7, top - 0.2, top + 1.2, seg, STONE) - cyl(c, r - 0.1, top - 0.3, top + 1.4, seg, STONE)
        parts.append(man_to_mesh(par))
        parts += merlons(ring, top + 1.2, 0.75, h=1.3, w=1.2, gap=0.8, closed=True)
    if roof == "cone":
        rh = roof_h if roof_h is not None else r * 2.6
        rr = r + (0.7 if parapet else 0.0)
        parts.append(cone_roof(c, rr if not parapet else r * 0.92, top + (1.0 if parapet else 0.0), rh, seg=roof_seg or seg, eave=cone_eave if not parapet else 0.4))
    parts.append(ring_course(c, r, top - 0.25, h=0.5, proj=0.3, seg=seg))
    return mk.merge(parts)


def square_tower(c, w, d, z_base, h_wall, yaw=0.0, z_foot=None, rng=None, roof="pyramid", roof_h=None, corner_turrets=False,
                 windows=True, win_w=1.4, win_h=3.2, clock=False, parapet=True, buttresses=True):
    rng = rng or np.random.default_rng(1)
    z_foot = z_base - 18.0 if z_foot is None else z_foot
    top = z_base + h_wall
    ca, sa = math.cos(yaw), math.sin(yaw)
    ax = np.array([ca, sa, 0.0])
    ay = np.array([-sa, ca, 0.0])
    body = box((c[0], c[1], (z_foot + top) / 2), (w, d, top - z_foot), yaw)
    op = Openings(rng)
    faces = [(ax, d, w), (-ax, d, w), (ay, w, d), (-ay, w, d)]
    if windows:
        for n, span, depth in faces:
            k = max(1, int(span // 4.2))
            for z in _floors(z_base, h_wall - 3.0):
                for j in range(k):
                    u = -span / 2 + span * (j + 0.5) / k
                    t = np.cross(n, [0, 0, 1.0])
                    P = np.array([c[0], c[1], z]) + n * (depth / 2) + t * u
                    if rng.random() < 0.1:
                        continue
                    op.window(P, n, win_w, win_h, depth=1.4, mullions=1 if win_w > 1.6 else 0)
    solid = op.apply(body)
    parts = [man_to_mesh(solid)] + op.meshes
    # quoins (alternating long / short dressed blocks at the corners)
    for sx in (-1, 1):
        for sy in (-1, 1):
            corner = np.array([c[0], c[1], 0.0]) + ax * sx * w / 2 + ay * sy * d / 2
            for i, z in enumerate(np.arange(z_base + 0.3, top - 0.5, 0.62)):
                lx = 0.9 if i % 2 == 0 else 0.5
                ly = 0.5 if i % 2 == 0 else 0.9
                cc = corner - ax * sx * (lx / 2 - 0.06) - ay * sy * (ly / 2 - 0.06)
                cc[2] = z
                parts.append(_mesh_box(cc, (lx, ly, 0.56), ax, ay, TRIM))
    if buttresses:
        for sx in (-1, 1):
            for sy in (-1, 1):
                corner = np.array([c[0], c[1], 0.0]) + ax * sx * w / 2 + ay * sy * d / 2
                for dirv in (ax * sx, ay * sy):
                    bt = np.cross(dirv, [0, 0, 1.0])
                    for st, (hz, dep) in enumerate(((0.55, 1.6), (0.32, 1.0))):
                        zt = z_base + h_wall * hz
                        cc = corner + dirv * (dep / 2) - bt * 0.0
                        cc[2] = (z_foot + zt) / 2
                        parts.append(_mesh_box(cc, (1.1, dep, zt - z_foot), bt, dirv, STONE))
    # string courses
    for z in _floors(z_base, h_wall)[1:]:
        parts.append(_mesh_box((c[0], c[1], z - 0.6), (w + 0.4, d + 0.4, 0.32), ax, ay, TRIM))
    if parapet:
        parts.append(_mesh_box((c[0], c[1], top - 0.9), (w + 0.9, d + 0.9, 0.5), ax, ay, TRIM))
        hw, hd = w / 2 + 0.3, d / 2 + 0.3
        ring = [np.array(c[:2]) + ax[:2] * sx * hw + ay[:2] * sy * hd for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        parts += merlons(ring, top - 0.6, 0.6, h=1.4, closed=True)
    if corner_turrets:
        for sx in (-1, 1):
            for sy in (-1, 1):
                cc = np.array(c[:2]) + ax[:2] * sx * (w / 2) + ay[:2] * sy * (d / 2)
                parts.append(round_tower(cc, 1.6, top - 6.0, 8.0, roof_h=6.5, z_foot=top - 7.5, seg=24, rng=rng, windows=False,
                                         courses=False, cone_eave=0.3))
    if roof == "pyramid":
        rh = roof_h if roof_h is not None else max(w, d) * 0.9
        parts.append(_pyramid_roof(c, w + 0.6, d + 0.6, top + (0.4 if parapet else 0.0), rh, yaw))
    elif roof == "spire":
        rh = roof_h if roof_h is not None else max(w, d) * 2.2
        sp = cone_roof(c, min(w, d) * 0.5, top + 0.5, rh, seg=8, eave=0.3, course=0.5)
        parts.append(sp)
    if clock:
        for n, span, depth in faces:
            t = np.cross(n, [0, 0, 1.0])
            cz = top - 4.2
            P = np.array([c[0], c[1], cz]) + n * (depth / 2 + 0.12)
            parts.append(_clock_face(P, n, t, min(span * 0.36, 3.2)))
    return mk.merge(parts)


def _pyramid_roof(c, w, d, z0, h, yaw=0.0, course=0.42, step=0.035):
    ca, sa = math.cos(yaw), math.sin(yaw)
    ax = np.array([ca, sa, 0.0])
    ay = np.array([-sa, ca, 0.0])
    apex = np.array([c[0], c[1], z0 + h])
    corners = [np.array([c[0], c[1], z0]) + ax * sx * w / 2 + ay * sy * d / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    parts = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        L = math.hypot(*(p1 - p0)[:2])
        n_c = max(4, int(math.hypot(h, L / 2) / course))
        s = np.linspace(0, 1, n_c + 1)
        rows = []
        for k in range(n_c):
            rows += [s[k], s[k + 1]]
        rows = np.array(rows)
        P = np.empty((2, len(rows), 3))
        P[0] = p0[None] + (apex - p0)[None] * rows[:, None]
        P[1] = p1[None] + (apex - p1)[None] * rows[:, None]
        # course steps: push every other row outward
        mid = (p0 + p1) / 2
        out = mid - np.array([c[0], c[1], mid[2]])
        out[2] = 0
        out /= np.linalg.norm(out) + 1e-9
        P[:, 0::2] += out * step
        m = mk.grid(P, uv_tile=1.0, mat=SLATE)
        uu = np.array([0.0, L])[:, None] * np.ones(len(rows))[None]
        vv = rows[None] * math.hypot(h, L / 2) * np.ones((2, 1))
        m.uv0 = np.stack([uu, vv], -1).reshape(-1, 2)[m.F]
        if np.mean(m.face_normals()[:, 2]) < 0:
            m.flip()
        parts.append(m)
    sp = mk.revolve([(0.18, z0 + h - 0.3), (0.12, z0 + h + 2.5), (0.0, z0 + h + 3.2)], seg=8, mat=LEAD)
    sp.translate((c[0], c[1], 0.0))
    sp.uv_box(1.0, only_missing=False)
    parts.append(sp)
    return mk.merge(parts)


def _clock_face(P, n, t, r):
    up = np.array([0.0, 0.0, 1.0])
    ring = mk.revolve([(r * 0.82, -0.05), (r, -0.05), (r, 0.12), (r * 0.82, 0.12)], seg=40, mat=TRIM)
    disc = mk.revolve([(0.0, 0.0), (r * 0.84, 0.0)], seg=40, mat=LEAD)
    hands = [_mesh_box((0, r * 0.25, 0.08), (0.12, r * 0.55, 0.06), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), TRIM),
             _mesh_box((r * 0.33, 0, 0.08), (r * 0.7, 0.10, 0.06), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), TRIM)]
    m = mk.merge([ring, disc] + hands)
    # local z (disc normal) -> n, local x -> t, local y -> up
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = t, up, n, P
    m.apply(M)
    if np.linalg.det(M[:3, :3]) < 0:
        pass
    m.uv_box(1.0, only_missing=False)
    return m


def hall(p0, p1, width, z_base, h_wall, rng=None, bays=None, pitch=58.0, z_foot=None, win_frac=0.56, buttress_d=2.4,
         end_windows=True, fleche=True, side_skip=()):
    """Great-Hall type building: buttressed bays with tall traceried lancets, steep slate roof, pinnacles, gables."""
    rng = rng or np.random.default_rng(2)
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    z_foot = z_base - 18.0 if z_foot is None else z_foot
    a = np.array([*(p1 - p0) / np.linalg.norm(p1 - p0), 0.0])
    b = np.array([-a[1], a[0], 0.0])
    L = float(np.linalg.norm(p1 - p0))
    c = np.array([*(p0 + p1) / 2, 0.0])
    top = z_base + h_wall
    yaw = math.atan2(a[1], a[0])
    body = box((c[0], c[1], (z_foot + top) / 2), (L, width, top - z_foot), yaw)
    op = Openings(rng)
    nb = bays or max(4, int(L // 9.5))
    bay = L / nb
    ww = bay * 0.42
    wh = h_wall * win_frac
    for side in (1.0, -1.0):
        if side in side_skip:
            continue
        n = b * side
        for k in range(nb):
            u = -L / 2 + bay * (k + 0.5)
            P = c + a * u + n * (width / 2) + np.array([0, 0, z_base + h_wall * 0.30])
            op.window(P, n, ww, wh, depth=1.6, mullions=2, transom=True, hood=True, reveal=0.7)
    if end_windows:
        for end in (1.0, -1.0):
            n = a * end
            P = c + n * (L / 2) + np.array([0, 0, z_base + h_wall * 0.26])
            op.window(P, n, width * 0.40, h_wall * 0.66, depth=1.6, mullions=3, transom=True, hood=True, reveal=0.7)
    solid = op.apply(body)
    parts = [man_to_mesh(solid)] + op.meshes
    # buttresses between bays (two stages, sloping caps) topped by pinnacles
    for side in (1.0, -1.0):
        n = b * side
        for k in range(nb + 1):
            u = -L / 2 + bay * k
            base = c + a * u + n * (width / 2)
            for stage, (frac, dep, wid) in enumerate(((0.62, buttress_d, 1.5), (0.94, buttress_d * 0.6, 1.2))):
                zt = z_base + h_wall * frac
                cc = base + n * (dep / 2)
                cc[2] = (z_foot + zt) / 2
                parts.append(_mesh_box(cc, (wid, dep, zt - z_foot), a, n, STONE))
                cap = base + n * (dep / 2)
                cap[2] = zt + 0.2
                parts.append(_mesh_box(cap, (wid + 0.2, dep + 0.2, 0.4), a, n, TRIM))
            pc = base + n * (buttress_d * 0.3)
            parts.append(pinnacle(pc, top - 0.2, 5.5, 0.55))
    # parapet + cornice along the eaves
    for side in (1.0, -1.0):
        cc = c + b * side * (width / 2 + 0.2)
        cc[2] = top - 0.6
        parts.append(_mesh_box(cc, (L + 0.6, 0.9, 0.5), a, b, TRIM))
    roof, z_r = gable_roof(c, a, L + 0.4, width, top + 0.2, pitch, overhang=0.3)
    parts.append(roof)
    for end in (1.0, -1.0):
        g = gable_wall(c + a * end * (L / 2 - 0.5), a, width, top + 0.2, pitch, thick=1.0, end=end)
        parts += [man_to_mesh(m) for m in g]
        # corner turrets (octagonal) at the four corners
        for side in (1.0, -1.0):
            cc = c + a * end * (L / 2) + b * side * (width / 2)
            parts.append(round_tower(cc[:2], 1.5, top - 9.0, 13.0, roof_h=7.5, z_foot=top - 10, seg=8, rng=rng, windows=False,
                                     courses=False, cone_eave=0.25, roof_seg=8))
    if fleche:
        fc = c + np.array([0, 0, z_r])
        f = round_tower(fc[:2], 1.6, z_r - 1.0, 5.0, roof_h=9.0, z_foot=z_r - 3.0, seg=8, rng=rng, windows=False, courses=False,
                        cone_eave=0.3, roof_seg=8)
        parts.append(f)
    return mk.merge(parts)


def wing(p0, p1, depth, z_base, h_wall, rng=None, pitch=52.0, z_foot=None, floor_h=4.6, roof="gable", dormers=True,
         chimneys=True, win_kind="pointed", side_skip=(), end_gables=True, crenel=False):
    """A range of rooms between towers: windows on every floor, string courses, slate roof with dormers and chimneys."""
    rng = rng or np.random.default_rng(3)
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    z_foot = z_base - 18.0 if z_foot is None else z_foot
    a = np.array([*(p1 - p0) / np.linalg.norm(p1 - p0), 0.0])
    b = np.array([-a[1], a[0], 0.0])
    L = float(np.linalg.norm(p1 - p0))
    c = np.array([*(p0 + p1) / 2, 0.0])
    top = z_base + h_wall
    yaw = math.atan2(a[1], a[0])
    body = box((c[0], c[1], (z_foot + top) / 2), (L, depth, top - z_foot), yaw)
    op = Openings(rng)
    floors = _floors(z_base, h_wall - 1.5, floor_h)
    nwin = max(1, int(L // 4.6))
    for side in (1.0, -1.0):
        if side in side_skip:
            continue
        n = b * side
        for fi, z in enumerate(floors):
            for k in range(nwin):
                u = -L / 2 + L * (k + 0.5) / nwin
                if rng.random() < 0.07:
                    continue
                P = c + a * u + n * (depth / 2) + np.array([0, 0, z])
                big = fi == 1 and rng.random() < 0.5
                op.window(P, n, 1.5 if big else 1.25, 3.3 if big else 2.6, depth=1.3, mullions=1 if big else 0, kind=win_kind)
    solid = op.apply(body)
    parts = [man_to_mesh(solid)] + op.meshes
    for z in floors[1:]:
        for side in (1.0, -1.0):
            cc = c + b * side * (depth / 2 + 0.1)
            cc[2] = z - 0.55
            parts.append(_mesh_box(cc, (L, 0.36, 0.3), a, b, TRIM))
    if crenel:
        for side in (1.0, -1.0):
            p_a = c + b * side * (depth / 2) - a * (L / 2)
            p_b = c + b * side * (depth / 2) + a * (L / 2)
            parts += merlons([p_a[:2], p_b[:2]], top - 0.2, 0.7, h=1.4)
    if roof == "gable":
        r, z_r = gable_roof(c, a, L, depth, top, pitch, overhang=0.45)
        parts.append(r)
        if end_gables:
            for end in (1.0, -1.0):
                parts += [man_to_mesh(m) for m in gable_wall(c + a * end * (L / 2 - 0.5), a, depth, top, pitch, thick=1.0, end=end)]
        if dormers:
            nd = max(1, int(L // 9.0))
            rise = (depth / 2) * math.tan(math.radians(pitch))
            for side in (1.0, -1.0):
                for k in range(nd):
                    if rng.random() < 0.35:
                        continue
                    u = -L / 2 + L * (k + 0.5) / nd
                    parts.append(_dormer(c + a * u, a, b * side, depth, top, pitch, rng))
        if chimneys:
            nch = max(1, int(L // 22.0))
            for k in range(nch):
                u = -L / 2 + L * (k + 0.5) / nch + rng.uniform(-3, 3)
                cc = c + a * u + b * rng.uniform(-depth * 0.15, depth * 0.15)
                hgt = z_r + rng.uniform(1.8, 3.2)
                cc[2] = (top + hgt) / 2
                parts.append(_mesh_box(cc, (1.6, 1.1, hgt - top), a, b, STONE))
                cap = cc.copy()
                cap[2] = hgt + 0.12
                parts.append(_mesh_box(cap, (1.85, 1.35, 0.24), a, b, TRIM))
                for pk in (-0.4, 0.4):
                    pot = cc + a * pk
                    pot[2] = hgt + 0.55
                    parts.append(_mesh_box(pot, (0.32, 0.32, 0.7), a, b, LEAD))
    return mk.merge(parts)


def _dormer(base, a, n, depth, z_eave, pitch, rng):
    """Gabled dormer window projecting from a roof slope."""
    k = math.tan(math.radians(pitch))
    off = depth / 2 - 1.2                          # distance from the ridge line toward the eave
    zc = z_eave + (depth / 2 - off) * k
    w, d, h = 2.0, 2.6, 2.6
    c = np.asarray(base, np.float64) + n * off
    c = c + np.array([0, 0, zc + h / 2 - 0.2])
    body = box(c, (w, d, h), math.atan2(a[1], a[0]))
    op = Openings(rng)
    P = c + n * (d / 2) - np.array([0, 0, h / 2 - 0.35])
    op.window(P, n, 1.0, 1.8, depth=0.6, kind="flat", hood=False, sill=False, reveal=0.25)
    parts = [man_to_mesh(op.apply(body))] + op.meshes
    r, _ = gable_roof(c + np.array([0, 0, 0]), n, d + 0.4, w, c[2] + h / 2 - 0.05, 50.0, overhang=0.2, ridge=False)
    parts.append(r)
    return mk.merge(parts)


def curtain_wall(path, z_base, h, thick=2.4, z_foot=None, rng=None, crenel=True, arrow_slits=True, buttress_every=12.0):
    """Battlemented wall along a polyline (the crag rim): wall walk, merlons, slits, buttresses into the rock."""
    rng = rng or np.random.default_rng(4)
    P = np.asarray(path, np.float64)
    z_foot = z_base - 24.0 if z_foot is None else z_foot
    parts = []
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1.0:
            continue
        a = np.array([d[0] / L, d[1] / L, 0.0])
        nb = np.array([-a[1], a[0], 0.0])
        c = np.array([*(p0 + p1) / 2, 0.0])
        seg = box((c[0], c[1], (z_foot + z_base + h) / 2), (L + thick * 0.5, thick, z_base + h - z_foot), math.atan2(a[1], a[0]))
        op = Openings(rng)
        if arrow_slits:
            ns = max(1, int(L // 6.0))
            for k in range(ns):
                u = -L / 2 + L * (k + 0.5) / ns
                for side in (1.0,):
                    Pw = c + a * u + nb * side * (thick / 2) + np.array([0, 0, z_base + h * 0.35])
                    op.window(Pw, nb * side, 0.32, 1.8, depth=thick + 0.4, kind="flat", hood=False, glass=False, sill=False, reveal=0.2)
        parts.append(man_to_mesh(op.apply(seg)))
        parts += op.meshes
        if crenel:
            parts += merlons([p0, p1], z_base + h, thick * 0.4, h=1.5, w=1.4, gap=0.9)
            cc = c.copy()
            cc[2] = z_base + h - 0.1
            parts.append(_mesh_box(cc, (L + thick * 0.5, thick + 0.25, 0.28), a, nb, TRIM))
        if buttress_every:
            nbt = int(L // buttress_every)
            for k in range(1, nbt + 1):
                u = -L / 2 + L * k / (nbt + 1)
                for side in (1.0, -1.0):
                    bc = c + a * u + nb * side * (thick / 2 + 0.7)
                    bc[2] = (z_foot + z_base + h * 0.7) / 2
                    if side < 0:
                        continue
                    parts.append(_mesh_box(bc, (1.4, 1.4, z_base + h * 0.7 - z_foot), a, nb, STONE))
    return mk.merge(parts) if parts else None


def viaduct(p0, p1, deck_z, ground_fn, width=6.0, n_arches=None, pier_w=3.0, rng=None):
    """Multi-arched stone viaduct; piers stand on the terrain sampled by ground_fn(x, y)."""
    rng = rng or np.random.default_rng(5)
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    a = np.array([*(p1 - p0) / np.linalg.norm(p1 - p0), 0.0])
    b = np.array([-a[1], a[0], 0.0])
    L = float(np.linalg.norm(p1 - p0))
    c = np.array([*(p0 + p1) / 2, 0.0])
    yaw = math.atan2(a[1], a[0])
    na = n_arches or max(3, int(L // 13.0))
    span = L / na
    gs = [float(ground_fn(*(p0 + (p1 - p0) * t))) for t in np.linspace(0, 1, 41)]
    z_low = min(gs) - 6.0
    wall = box((c[0], c[1], (z_low + deck_z) / 2), (L, width, deck_z - z_low), yaw)
    cutters = []
    for k in range(na):
        u = -L / 2 + span * (k + 0.5)
        cc = c + a * u
        g = float(ground_fn(cc[0], cc[1]))
        ar = (span - pier_w) / 2
        spring = deck_z - 2.2 - ar
        if spring - ar < g:
            spring = max(g + 1.0, deck_z - 2.2 - ar)
        # arch cutter = half cylinder above the springing + box below it
        cyl_ = m3d.Manifold.cylinder(width + 2.0, ar, ar, 40, True).rotate([90.0, 0.0, 0.0])
        cyl_ = cyl_.rotate([0.0, 0.0, math.degrees(yaw)]).translate([cc[0], cc[1], spring])
        bx = m3d.Manifold.cube([2 * ar, width + 2.0, spring - z_low + 1.0], True).rotate([0.0, 0.0, math.degrees(yaw)])
        bx = bx.translate([cc[0], cc[1], (spring + z_low - 1.0) / 2])
        cutters.append(_orig(cyl_ + bx, TRIM))
    solid = wall - m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)
    parts = [man_to_mesh(solid)]
    # parapets + string course
    for side in (1.0, -1.0):
        cc = c + b * side * (width / 2 - 0.25)
        cc[2] = deck_z + 0.6
        parts.append(_mesh_box(cc, (L, 0.5, 1.2), a, b, STONE))
        cc2 = c + b * side * (width / 2 + 0.05)
        cc2[2] = deck_z - 0.3
        parts.append(_mesh_box(cc2, (L, 0.6, 0.35), a, b, TRIM))
        cc3 = cc.copy()
        cc3[2] = deck_z + 1.28
        parts.append(_mesh_box(cc3, (L, 0.62, 0.16), a, b, TRIM))
    # little lamp-turrets on the piers
    for k in range(1, na):
        u = -L / 2 + span * k
        for side in (1.0, -1.0):
            pc = c + a * u + b * side * (width / 2 + 0.4)
            parts.append(pinnacle(pc[:2], deck_z + 0.6, 3.2, 0.42))
    return mk.merge(parts)


def finish(mesh: Mesh, z_ground=None):
    """UV0 box projection for stone / trim / wood (2 m tiles), grime + moss masks in UV1 for every face."""
    if mesh is None or mesh.nf == 0:
        return mesh
    stone = np.isin(mesh.mat, (STONE, TRIM, WOOD, LEAD))
    if mesh.uv0 is None:
        mesh.uv0 = np.full((mesh.nf, 3, 2), np.nan)
    box_uv = Mesh(mesh.V, mesh.F, mesh.mat).uv_box(2.0, only_missing=False).uv0
    sel = stone | np.isnan(mesh.uv0).any(axis=(1, 2))
    mesh.uv0[sel] = box_uv[sel]
    P = mesh.V[mesh.F]                                   # (m, 3, 3)
    fn = mesh.face_normals()
    z = P[:, :, 2]
    zg = (np.min(z) if z_ground is None else z_ground)
    grime = np.clip(0.75 - (z - zg) / 35.0, 0, 1) * 0.6 + 0.4 * mk.smoothstep(0.2, -0.6, fn[:, 2])[:, None]
    noise = mk.fbm(P.reshape(-1, 3) / 6.0, 3, seed=5).reshape(-1, 3)
    moss = np.clip(mk.smoothstep(0.5, 0.95, fn[:, 2])[:, None] * 0.7 + 0.5 * mk.smoothstep(8.0, 0.0, z - zg) + 0.35 * noise, 0, 1)
    moss = np.where((mesh.mat == SLATE)[:, None], 0.35 * mk.smoothstep(0.1, 0.8, noise + 0.2), moss)
    mesh.uv1 = np.stack([np.clip(grime + 0.15 * noise, 0, 1), moss], -1)
    if mesh.uv2 is None:
        mesh.uv2 = np.zeros((mesh.nf, 3, 2))
    mesh.uv2 = np.nan_to_num(mesh.uv2)
    return mesh


def bartizan(c, z0, r=2.3, h=7.0, roof_h=None, rng=None, seg=24):
    """Corbelled corner turret (no foundation): corbel courses below, a few slit windows, a tall cone."""
    rng = rng or np.random.default_rng(6)
    parts = [round_tower(c, r, z0, h, roof_h=roof_h if roof_h is not None else r * 4.2, z_foot=z0 - 0.5, rng=rng, seg=seg,
                         per_floor=3, win_w=0.5, win_h=1.4, courses=False, cone_eave=0.35)]
    # corbel cone under the turret
    cone = mk.revolve([(0.0, z0 - 3.2), (r * 0.35, z0 - 2.6), (r * 0.8, z0 - 1.2), (r + 0.1, z0 - 0.3), (r + 0.1, z0 + 0.1), (0.0, z0 + 0.1)],
                      seg=seg, mat=TRIM)
    cone.translate((c[0], c[1], 0.0))
    if cone.volume() < 0:
        cone.flip()
    cone.uv_box(1.0, only_missing=False)
    parts.append(cone)
    parts += corbel_ring(c, r - 0.1, z0 - 0.6, size=0.35)
    return mk.merge(parts)
