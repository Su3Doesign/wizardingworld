"""meshkit - a small numpy mesh toolkit for the flooded-rotunda generator.

Conventions
-----------
* Units are metres, Z is up, right handed.
* Meshes are triangle soups with *indexed positions* and *per-corner attributes*
  (uv0 = tiling UVs, uv1 = (moss, grime) masks, uv2 = (waterline, random id)).
  Per-corner attributes mean UV seams never force vertex splitting, and welding
  vertices for CSG never destroys UVs.
* Faces are wound counter-clockwise when seen from outside.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- noise


def _hash(ix, iy, iz, seed):
    """Integer lattice hash -> float in [0, 1)."""
    h = (ix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (
        iy.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    ) ^ (iz.astype(np.uint64) * np.uint64(0x165667B19E3779F9))
    h = h ^ np.uint64((seed * 0x27D4EB2F165667C5 + 0x1234567) & 0xFFFFFFFFFFFFFFFF)
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(29)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def value_noise(P, seed=0):
    """Smooth 3D value noise, range [-1, 1]. P: (n, 3)."""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    i = np.floor(P)
    f = P - i
    u = f * f * f * (f * (f * 6 - 15) + 10)
    ix, iy, iz = i[:, 0].astype(np.int64), i[:, 1].astype(np.int64), i[:, 2].astype(np.int64)

    def h(dx, dy, dz):
        return _hash(ix + dx, iy + dy, iz + dz, seed)

    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    c000, c100, c010, c110 = h(0, 0, 0), h(1, 0, 0), h(0, 1, 0), h(1, 1, 0)
    c001, c101, c011, c111 = h(0, 0, 1), h(1, 0, 1), h(0, 1, 1), h(1, 1, 1)
    x00 = c000 + (c100 - c000) * ux
    x10 = c010 + (c110 - c010) * ux
    x01 = c001 + (c101 - c001) * ux
    x11 = c011 + (c111 - c011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return (y0 + (y1 - y0) * uz) * 2.0 - 1.0


def fbm(P, octaves=4, lacunarity=2.0, gain=0.5, seed=0):
    """Fractal Brownian motion of value noise, roughly in [-1, 1]."""
    p = np.asarray(P, np.float64).reshape(-1, 3)
    out = np.zeros(len(p))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(p, seed + o * 101)
        tot += amp
        amp *= gain
        p = p * lacunarity + 17.31
    return out / tot


def ridged(P, octaves=4, seed=0):
    """Ridged multifractal-ish noise in [0, 1] (sharp creases)."""
    p = np.asarray(P, np.float64).reshape(-1, 3)
    out = np.zeros(len(p))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * (1.0 - np.abs(value_noise(p, seed + o * 57)))
        tot += amp
        amp *= 0.5
        p = p * 2.0 + 5.17
    return out / tot


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a + 1e-30), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- mesh

UVS = ("uv0", "uv1", "uv2", "uv3")


class Mesh:
    """Triangle mesh with indexed vertices and per-corner attributes."""

    def __init__(self, V, F, mat=0, uv0=None, uv1=None, uv2=None, uv3=None):
        self.V = np.ascontiguousarray(V, dtype=np.float64).reshape(-1, 3)
        self.F = np.ascontiguousarray(F, dtype=np.int64).reshape(-1, 3)
        m = len(self.F)
        self.mat = (
            np.full(m, mat, np.int16) if np.isscalar(mat) else np.ascontiguousarray(mat, np.int16).reshape(m)
        )
        self.uv0 = None if uv0 is None else np.ascontiguousarray(uv0, np.float64).reshape(m, 3, 2)
        self.uv1 = None if uv1 is None else np.ascontiguousarray(uv1, np.float64).reshape(m, 3, 2)
        self.uv2 = None if uv2 is None else np.ascontiguousarray(uv2, np.float64).reshape(m, 3, 2)
        self.uv3 = None if uv3 is None else np.ascontiguousarray(uv3, np.float64).reshape(m, 3, 2)

    # -- basic info --------------------------------------------------------
    @property
    def nv(self):
        return len(self.V)

    @property
    def nf(self):
        return len(self.F)

    def copy(self):
        return Mesh(
            self.V.copy(),
            self.F.copy(),
            self.mat.copy(),
            None if self.uv0 is None else self.uv0.copy(),
            None if self.uv1 is None else self.uv1.copy(),
            None if self.uv2 is None else self.uv2.copy(),
            None if self.uv3 is None else self.uv3.copy(),
        )

    def bbox(self):
        return self.V.min(0), self.V.max(0)

    def face_normals(self, normalise=True):
        P0, P1, P2 = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        n = np.cross(P1 - P0, P2 - P0)
        if normalise:
            l = np.linalg.norm(n, axis=1)
            n = n / np.maximum(l, 1e-30)[:, None]
        return n

    def face_areas(self):
        P0, P1, P2 = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(P1 - P0, P2 - P0), axis=1)

    def area(self):
        return float(self.face_areas().sum())

    def volume(self):
        """Signed volume (positive when outward-wound and closed)."""
        P0, P1, P2 = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        return float(np.einsum("ij,ij->i", P0, np.cross(P1, P2)).sum() / 6.0)

    # -- transforms --------------------------------------------------------
    def apply(self, M):
        """Apply a 4x4 matrix. Flips winding if the matrix mirrors."""
        M = np.asarray(M, np.float64)
        self.V = self.V @ M[:3, :3].T + M[:3, 3]
        if np.linalg.det(M[:3, :3]) < 0:
            self.F = self.F[:, ::-1].copy()
            for name in UVS:
                a = getattr(self, name)
                if a is not None:
                    setattr(self, name, a[:, ::-1, :].copy())
        return self

    def translate(self, t):
        self.V = self.V + np.asarray(t, np.float64)
        return self

    def scale(self, s):
        s = np.asarray(s, np.float64)
        self.V = self.V * s
        if np.ndim(s) and np.prod(np.sign(s)) < 0:
            self.F = self.F[:, ::-1].copy()
        return self

    def rotate_z(self, ang):
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.V = self.V @ R.T
        return self

    def rotate_x(self, ang):
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
        self.V = self.V @ R.T
        return self

    def rotate_y(self, ang):
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        self.V = self.V @ R.T
        return self

    def rotate_axis(self, axis, ang):
        a = np.asarray(axis, np.float64)
        a = a / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        self.V = self.V @ R.T
        return self

    def flip(self):
        self.F = self.F[:, ::-1].copy()
        for name in UVS:
            a = getattr(self, name)
            if a is not None:
                setattr(self, name, a[:, ::-1, :].copy())
        return self

    # -- cleanup -----------------------------------------------------------
    def weld(self, tol=1e-6):
        """Merge coincident vertices and drop degenerate faces (attributes follow)."""
        key = np.round(self.V / tol).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        V = self.V[first]
        F = inv[self.F]
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        return self._take(F, keep, V)

    def _take(self, F, keep, V=None):
        out = Mesh(
            self.V if V is None else V,
            F[keep],
            self.mat[keep],
            None if self.uv0 is None else self.uv0[keep],
            None if self.uv1 is None else self.uv1[keep],
            None if self.uv2 is None else self.uv2[keep],
            None if self.uv3 is None else self.uv3[keep],
        )
        return out

    def select_faces(self, mask):
        mask = np.asarray(mask, bool)
        return self._take(self.F, mask).compact()

    def compact(self):
        """Drop unreferenced vertices."""
        used = np.unique(self.F)
        remap = -np.ones(len(self.V), np.int64)
        remap[used] = np.arange(len(used))
        return Mesh(self.V[used], remap[self.F], self.mat, self.uv0, self.uv1, self.uv2, self.uv3)

    def remove_tiny(self, min_area=1e-10):
        keep = self.face_areas() > min_area
        return self._take(self.F, keep).compact()

    # -- attribute helpers ---------------------------------------------------
    def vertex_normals(self):
        """Area-weighted smooth vertex normals over position-welded vertices."""
        key = np.round(self.V / 1e-6).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        fn = self.face_normals(normalise=False)
        acc = np.zeros((inv.max() + 1, 3))
        for c in range(3):
            np.add.at(acc, inv[self.F[:, c]], fn)
        l = np.linalg.norm(acc, axis=1)
        acc = acc / np.maximum(l, 1e-30)[:, None]
        return acc[inv]

    def displace(self, amount):
        """Move every vertex along its smooth normal by amount (scalar or (n,))."""
        n = self.vertex_normals()
        amount = np.asarray(amount, np.float64)
        if amount.ndim == 0:
            amount = np.full(len(self.V), float(amount))
        self.V = self.V + n * amount[:, None]
        return self

    def corner_normals(self, angle_deg=35.0, weld_tol=1e-6, chunk=400_000):
        """Per-corner normals, smoothing across edges flatter than angle_deg."""
        V, F = self.V, self.F
        m = len(F)
        P0, P1, P2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        fn = np.cross(P1 - P0, P2 - P0)
        ar = np.linalg.norm(fn, axis=1)
        good = ar > 1e-18
        fn[good] /= ar[good, None]
        fn[~good] = (0, 0, 1)

        def ang(a, b, c):
            u, v = b - a, c - a
            cu = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-30)
            return np.arccos(np.clip(cu, -1, 1))

        w = np.stack([ang(P0, P1, P2), ang(P1, P2, P0), ang(P2, P0, P1)], 1).reshape(-1)
        key = np.round(V / weld_tol).astype(np.int64)
        _, wid = np.unique(key, axis=0, return_inverse=True)
        wid = np.asarray(wid).reshape(-1)
        cw = wid[F].reshape(-1)
        cf = np.repeat(np.arange(m), 3)
        order = np.argsort(cw, kind="stable")
        cws = cw[order]
        bounds = np.flatnonzero(np.diff(cws)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(cws)]])
        sizes = ends - starts
        # group id per sorted position
        gid = np.repeat(np.arange(len(starts)), sizes)
        cos_t = math.cos(math.radians(angle_deg))
        out = np.zeros((3 * m, 3))
        fn_sorted = fn[cf[order]]
        w_sorted = w[order]

        # process in chunks aligned to group boundaries
        pos = 0
        n = len(order)
        while pos < n:
            end = min(n, pos + chunk)
            if end < n:
                end = ends[gid[end - 1]]  # extend to the end of the group
            idx = np.arange(pos, end)
            g = gid[idx]
            rep = sizes[g]
            tot = int(rep.sum())
            a_local = np.repeat(np.arange(len(idx)), rep)
            base = np.repeat(starts[g], rep)
            offs = np.arange(tot) - np.repeat(np.cumsum(rep) - rep, rep)
            b = base + offs
            a = idx[a_local]
            dot = np.einsum("ij,ij->i", fn_sorted[a], fn_sorted[b])
            wt = w_sorted[b] * (dot > cos_t)
            contrib = fn_sorted[b] * wt[:, None]
            acc = np.stack([np.bincount(a_local, weights=contrib[:, k], minlength=len(idx)) for k in range(3)], 1)
            out[order[idx]] = acc
            pos = end
        l = np.linalg.norm(out, axis=1)
        bad = l < 1e-12
        out[~bad] /= l[~bad, None]
        out[bad] = fn[cf[bad]]
        return out.reshape(m, 3, 3)

    # -- UV generation ---------------------------------------------------------
    def uv_box(self, tile=2.0, origin=(0, 0, 0), only_missing=True):
        """Per-face dominant-axis box projection (metric, tile metres per UV unit)."""
        n = self.face_normals()
        ax = np.argmax(np.abs(n), axis=1)
        P = self.V[self.F] - np.asarray(origin)
        uv = np.zeros((self.nf, 3, 2))
        mz = ax == 2
        mx = ax == 0
        my = ax == 1
        uv[mz] = P[mz][:, :, [0, 1]]
        uv[mx] = P[mx][:, :, [1, 2]]
        uv[my] = P[my][:, :, [0, 2]]
        # keep mirrored faces unmirrored
        flipx = mx & (n[:, 0] < 0)
        uv[flipx, :, 0] *= -1
        flipy = my & (n[:, 1] > 0)
        uv[flipy, :, 0] *= -1
        flipz = mz & (n[:, 2] < 0)
        uv[flipz, :, 0] *= -1
        uv /= tile
        self._set_uv("uv0", uv, only_missing)
        return self

    def uv_cyl(self, tile=2.0, r_ref=10.0, only_missing=True):
        """Cylindrical box projection around the Z axis (seamless around the ring)."""
        P = self.V[self.F]
        n = self.face_normals()
        cen = P.mean(1)
        thc = np.arctan2(cen[:, 1], cen[:, 0])
        rh = np.stack([np.cos(thc), np.sin(thc), np.zeros_like(thc)], 1)
        th = np.stack([-np.sin(thc), np.cos(thc), np.zeros_like(thc)], 1)
        dr = np.abs(np.einsum("ij,ij->i", n, rh))
        dt = np.abs(np.einsum("ij,ij->i", n, th))
        dz = np.abs(n[:, 2])
        cls = np.argmax(np.stack([dr, dt, dz], 1), axis=1)
        uv = np.zeros((self.nf, 3, 2))
        # radial faces -> (theta * r_ref, z)
        ang = np.arctan2(P[:, :, 1], P[:, :, 0])
        d = (ang - thc[:, None] + np.pi) % (2 * np.pi) - np.pi
        thv = thc[:, None] + d
        m0 = cls == 0
        uv[m0, :, 0] = (thv[m0] * r_ref)
        uv[m0, :, 1] = P[m0][:, :, 2]
        # tangential faces -> (radius, z)
        m1 = cls == 1
        rad = np.hypot(P[:, :, 0], P[:, :, 1])
        uv[m1, :, 0] = rad[m1]
        uv[m1, :, 1] = P[m1][:, :, 2]
        # vertical faces -> (x, y)
        m2 = cls == 2
        uv[m2, :, 0] = P[m2][:, :, 0]
        uv[m2, :, 1] = P[m2][:, :, 1]
        uv /= tile
        self._set_uv("uv0", uv, only_missing)
        return self

    def _set_uv(self, name, uv, only_missing):
        cur = getattr(self, name)
        if cur is None or not only_missing:
            setattr(self, name, uv)
        else:
            miss = np.isnan(cur).any(axis=(1, 2))
            cur = cur.copy()
            cur[miss] = uv[miss]
            setattr(self, name, cur)

    def set_vertex_mask(self, name, values):
        """Store a per-vertex (n, 2) or (n,) array as a per-corner uv attribute."""
        values = np.asarray(values, np.float64)
        if values.ndim == 1:
            values = np.stack([values, np.zeros_like(values)], 1)
        setattr(self, name, values[self.F])
        return self

    # -- io ---------------------------------------------------------------------
    def save(self, path):
        d = dict(V=self.V.astype(np.float32), F=self.F.astype(np.int32), mat=self.mat)
        for name in UVS:
            a = getattr(self, name)
            if a is not None:
                d[name] = np.nan_to_num(a).astype(np.float32)
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path):
        z = np.load(path)
        return Mesh(
            z["V"].astype(np.float64),
            z["F"],
            z["mat"],
            z["uv0"] if "uv0" in z else None,
            z["uv1"] if "uv1" in z else None,
            z["uv2"] if "uv2" in z else None,
            z["uv3"] if "uv3" in z else None,
        )


def merge(meshes):
    """Concatenate meshes. Missing per-corner attributes are NaN (filled later)."""
    meshes = [m for m in meshes if m is not None and m.nf > 0]
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    V, F, mat = [], [], []
    uvs = {k: [] for k in UVS}
    off = 0
    for m in meshes:
        V.append(m.V)
        F.append(m.F + off)
        off += m.nv
        mat.append(m.mat)
        for k in uvs:
            a = getattr(m, k)
            uvs[k].append(a if a is not None else np.full((m.nf, 3, 2), np.nan))
    out = Mesh(np.concatenate(V), np.concatenate(F), np.concatenate(mat))
    for k, lst in uvs.items():
        if any(getattr(m, k) is not None for m in meshes):
            setattr(out, k, np.concatenate(lst))
    return out


# --------------------------------------------------------------------------- builders


def grid(P, flip=False, uv_tile=2.0, mat=0):
    """Quad-grid surface from points P (nu, nv, 3). Normal = dP/du x dP/dv.

    UVs are metric (arc length / uv_tile). A closed direction is expressed by
    duplicating the first row/column at the end (the seam stays indexed apart).
    """
    P = np.asarray(P, np.float64)
    nu, nv, _ = P.shape
    idx = np.arange(nu * nv).reshape(nu, nv)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    du = np.linalg.norm(np.diff(P, axis=0), axis=2)
    dv = np.linalg.norm(np.diff(P, axis=1), axis=2)
    u = np.concatenate([np.zeros((1, nv)), np.cumsum(du, axis=0)], 0)
    v = np.concatenate([np.zeros((nu, 1)), np.cumsum(dv, axis=1)], 1)
    UV = np.stack([u, v], -1).reshape(-1, 2) / uv_tile
    V = P.reshape(-1, 3)
    mesh = Mesh(V, F, mat, UV[F])
    if flip:
        mesh.flip()
    return mesh


def fan_cap(ring, flip=False, mat=0, uv_tile=2.0):
    """Cap a closed ring (k, 3) with a triangle fan around its centroid."""
    ring = np.asarray(ring, np.float64)
    c = ring.mean(0)
    k = len(ring)
    V = np.vstack([ring, c])
    F = np.array([[i, (i + 1) % k, k] for i in range(k)])
    mesh = Mesh(V, F, mat)
    if flip:
        mesh.flip()
    return mesh


def revolve(profile, seg=32, mat=0, uv_tile=2.0, flip=False, a0=0.0, a1=2 * math.pi):
    """Surface of revolution about Z. profile: (k, 2) of (r, z), bottom to top
    for outward normals."""
    prof = np.asarray(profile, np.float64)
    ph = np.linspace(a0, a1, seg + 1)
    P = np.empty((seg + 1, len(prof), 3))
    P[:, :, 0] = np.cos(ph)[:, None] * prof[None, :, 0]
    P[:, :, 1] = np.sin(ph)[:, None] * prof[None, :, 0]
    P[:, :, 2] = prof[None, :, 1]
    return grid(P, flip=flip, uv_tile=uv_tile, mat=mat)


def loft(rings, mat=0, uv_tile=2.0, flip=False, cap_bottom=True, cap_top=True):
    """Loft closed rings. rings: (n_rings, m, 3), CCW seen from +Z-ish, bottom to top."""
    R = np.asarray(rings, np.float64)
    n, m, _ = R.shape
    Rc = np.concatenate([R, R[:, :1]], axis=1)  # duplicate seam
    P = Rc.transpose(1, 0, 2)  # (m+1, n, 3): u around, v up
    parts = [grid(P, uv_tile=uv_tile, mat=mat)]
    if cap_bottom:
        parts.append(fan_cap(R[0], flip=True, mat=mat))
    if cap_top:
        parts.append(fan_cap(R[-1], flip=False, mat=mat))
    out = merge(parts)
    if flip:
        out.flip()
    return out


def sweep(profile, path, ex, closed_path=False, caps=True, mat=0, uv_tile=2.0, scale=None):
    """Sweep a closed 2D profile along a 3D path.

    profile : (k, 2) CCW polygon in (px, py).
    path    : (n, 3) points; ex : (n, 3) unit vectors perpendicular to the tangent
    giving the profile's local X axis. The local Y axis is T x ex (so ex x ey = T).
    scale   : optional (n,) per-point scale of the profile (mitre joints).
    """
    prof = np.asarray(profile, np.float64)
    path = np.asarray(path, np.float64)
    ex = np.asarray(ex, np.float64)
    n = len(path)
    # tangents
    T = np.zeros_like(path)
    T[1:-1] = path[2:] - path[:-2]
    if closed_path:
        T[0] = path[1] - path[-2]
        T[-1] = T[0]
    else:
        T[0] = path[1] - path[0]
        T[-1] = path[-1] - path[-2]
    T /= np.maximum(np.linalg.norm(T, axis=1), 1e-30)[:, None]
    ey = np.cross(T, ex)
    ey /= np.maximum(np.linalg.norm(ey, axis=1), 1e-30)[:, None]
    sc = np.ones(n) if scale is None else np.asarray(scale, np.float64)
    k = len(prof)
    prof_c = np.vstack([prof, prof[:1]])  # close ring (seam duplicate)
    P = np.empty((k + 1, n, 3))
    for i in range(k + 1):
        P[i] = path + (prof_c[i, 0] * sc)[:, None] * ex + (prof_c[i, 1] * sc)[:, None] * ey
    parts = [grid(P, uv_tile=uv_tile, mat=mat)]
    if caps and not closed_path:
        ring0 = P[:k, 0]
        ring1 = P[:k, -1]
        parts.append(fan_cap(ring0, flip=True, mat=mat))
        parts.append(fan_cap(ring1, flip=False, mat=mat))
    out = merge(parts)
    return out


def orient_outward(mesh):
    """Flip a closed mesh whose signed volume is negative."""
    if mesh.volume() < 0:
        mesh.flip()
    return mesh


def ngon(r, n=24, rot=0.0):
    a = rot + np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def rect_profile(w, h, cx=0.0, cy=0.0):
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def rounded_box(center, size, r=0.03, max_edge=0.25, bevel_steps=3, mat=0):
    """Rounded box (Minkowski sum of a smaller box and a sphere) with a tessellation
    fine enough to carry noise displacement. Closed, welded, outward wound."""
    c = np.asarray(center, np.float64)
    h = np.asarray(size, np.float64) / 2.0
    r = float(min(r, h.min() * 0.95))

    def axis_samples(half):
        # bevel band samples (angle based) + uniform interior
        t = np.tan(np.linspace(0.0, math.pi / 4, bevel_steps + 1))  # 0 .. 1
        band = r * t  # 0..r offset beyond the inner flat
        inner_half = half - r
        n_in = max(1, int(math.ceil(2 * inner_half / max_edge)))
        inner = np.linspace(-inner_half, inner_half, n_in + 1)
        lo = -inner_half - band[::-1][:-1]  # from -half .. just before -inner_half
        hi = inner_half + band[1:]
        return np.concatenate([lo, inner, hi])

    sx, sy, sz = axis_samples(h[0]), axis_samples(h[1]), axis_samples(h[2])
    inner = h - r

    def surf(face_axis, sign):
        axes = [0, 1, 2]
        axes.remove(face_axis)
        a, b = axes
        samples = [sx, sy, sz]
        ua, ub = np.meshgrid(samples[a], samples[b], indexing="ij")
        pts = np.empty(ua.shape + (3,))
        pts[..., face_axis] = sign * h[face_axis]
        pts[..., a] = ua
        pts[..., b] = ub
        # Minkowski projection
        q = np.clip(pts, -inner, inner)
        d = pts - q
        l = np.linalg.norm(d, axis=-1, keepdims=True)
        d = np.where(l > 1e-12, d / np.maximum(l, 1e-12), 0.0)
        out = q + d * r
        # restore flat interior exactly
        flat = l[..., 0] < 1e-12
        out[flat] = pts[flat]
        # orientation: (a x b) should point along sign*face_axis
        n_sign = 1.0 if ((a, b, face_axis) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]) else -1.0
        return out, (n_sign * sign) > 0

    parts = []
    for axis in range(3):
        for sign in (-1, 1):
            P, ok = surf(axis, sign)
            parts.append(grid(P, flip=not ok, mat=mat))
    m = merge(parts)
    m.V = m.V + c
    m = m.weld(1e-7)
    return m.compact()


def box(center, size, mat=0, **kw):
    """Sharp axis-aligned box (12 triangles)."""
    c = np.asarray(center, np.float64)
    s = np.asarray(size, np.float64) / 2
    V = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], np.float64) * s + c
    F = np.array(
        [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    )
    return Mesh(V, F, mat)


# --------------------------------------------------------------------------- CSG


def _to_manifold(mesh):
    import manifold3d as m3d

    m = mesh.weld(1e-7).compact()
    if m.volume() < 0:
        m.flip()
    return m3d.Manifold(m3d.Mesh(vert_properties=m.V.astype(np.float32), tri_verts=m.F.astype(np.uint32)))


def _from_manifold(man, mat=0):
    mm = man.to_mesh()
    V = np.asarray(mm.vert_properties, np.float64)[:, :3]
    F = np.asarray(mm.tri_verts, np.int64)
    return Mesh(V, F, mat)


def boolean(a, b, op="difference", mat=0):
    ma, mb = _to_manifold(a), _to_manifold(b)
    if op == "difference":
        r = ma - mb
    elif op == "union":
        r = ma + mb
    elif op == "intersection":
        r = ma ^ mb
    else:
        raise ValueError(op)
    return _from_manifold(r, mat)


def boolean_many(a, cutters, op="difference", mat=0):
    """a minus (union of cutters) in one go (faster, avoids repeated conversion)."""
    import manifold3d as m3d

    ma = _to_manifold(a)
    mc = [_to_manifold(c) for c in cutters]
    if op == "difference":
        cut = m3d.Manifold.batch_boolean(mc, m3d.OpType.Add) if len(mc) > 1 else mc[0]
        r = ma - cut
    else:
        r = m3d.Manifold.batch_boolean([ma] + mc, m3d.OpType.Add)
    return _from_manifold(r, mat)


def refine(mesh, max_edge):
    """Midpoint-subdivide long edges. Keeps material ids; drops UVs (re-project after)."""
    import trimesh

    V, F, idx = trimesh.remesh.subdivide_to_size(mesh.V, mesh.F, max_edge=max_edge, max_iter=12, return_index=True)
    return Mesh(V, F, mesh.mat[idx])


def bend_about_z(mesh, radius, y_is_radial=True):
    """Bend a flat bay (x tangential, y radial offset, z up) onto a cylinder.

    A point (x, y, z) maps to ((R + y) cos(x/R), (R + y) sin(x/R), z), then the
    result is rotated so the bay centre (x=0) sits on +X. The bay's local +y points
    outwards (away from the ring centre).

    The map's Jacobian determinant is -(R + y) / R < 0 (local x -> tangential, local y -> radial is a handedness swap),
    so the winding is reversed afterwards to keep every face outward-facing: correct normals for masks, ivy growth and
    scattering, and correct back-face culling in Unreal.
    """
    x, y, z = mesh.V[:, 0], mesh.V[:, 1], mesh.V[:, 2]
    th = x / radius
    rr = radius + y
    out = mesh.copy()
    out.V = np.stack([rr * np.cos(th), rr * np.sin(th), z], 1)
    return out.flip()


def polar(r, theta, z=0.0):
    return np.array([r * math.cos(theta), r * math.sin(theta), z])


def decimate(mesh, target_ratio=0.5):
    import fast_simplification

    V, F = fast_simplification.simplify(
        mesh.V.astype(np.float32), mesh.F.astype(np.int32), target_reduction=1.0 - target_ratio
    )
    return Mesh(V.astype(np.float64), F.astype(np.int64), 0)


def check_closed(mesh):
    """Return (watertight, volume) for a welded copy."""
    import trimesh

    m = mesh.weld(1e-7).compact()
    t = trimesh.Trimesh(m.V, m.F, process=False)
    return bool(t.is_watertight), float(t.volume)
