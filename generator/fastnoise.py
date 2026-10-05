"""fastnoise - numba-compiled noise, Worley cells, path distances and hydraulic erosion for the terrain and rock generators.

All functions are deterministic for a given seed and run in parallel over the input points.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------------------------- hashing / gradients


@njit(cache=True, inline="always")
def _hash3(ix, iy, iz, seed):
    h = (ix * 0x8DA6B343) ^ (iy * 0xD8163841) ^ (iz * 0xCB1AB31F) ^ (seed * 0x27D4EB2F)
    h = h & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
    h ^= h >> 12
    h = (h * 0x297A2D39) & 0xFFFFFFFF
    h ^= h >> 15
    return h


@njit(cache=True, inline="always")
def _grad3(h, x, y, z):
    # 12 cube-edge gradients (Perlin)
    k = h % 12
    if k == 0:
        return x + y
    if k == 1:
        return -x + y
    if k == 2:
        return x - y
    if k == 3:
        return -x - y
    if k == 4:
        return x + z
    if k == 5:
        return -x + z
    if k == 6:
        return x - z
    if k == 7:
        return -x - z
    if k == 8:
        return y + z
    if k == 9:
        return -y + z
    if k == 10:
        return y - z
    return -y - z


@njit(cache=True, inline="always")
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def perlin3(x, y, z, seed):
    """Perlin gradient noise, roughly in [-1, 1]."""
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ix = int(ix)
    iy = int(iy)
    iz = int(iz)
    u = _fade(fx)
    v = _fade(fy)
    w = _fade(fz)
    n000 = _grad3(_hash3(ix, iy, iz, seed), fx, fy, fz)
    n100 = _grad3(_hash3(ix + 1, iy, iz, seed), fx - 1, fy, fz)
    n010 = _grad3(_hash3(ix, iy + 1, iz, seed), fx, fy - 1, fz)
    n110 = _grad3(_hash3(ix + 1, iy + 1, iz, seed), fx - 1, fy - 1, fz)
    n001 = _grad3(_hash3(ix, iy, iz + 1, seed), fx, fy, fz - 1)
    n101 = _grad3(_hash3(ix + 1, iy, iz + 1, seed), fx - 1, fy, fz - 1)
    n011 = _grad3(_hash3(ix, iy + 1, iz + 1, seed), fx, fy - 1, fz - 1)
    n111 = _grad3(_hash3(ix + 1, iy + 1, iz + 1, seed), fx - 1, fy - 1, fz - 1)
    x00 = n000 + u * (n100 - n000)
    x10 = n010 + u * (n110 - n010)
    x01 = n001 + u * (n101 - n001)
    x11 = n011 + u * (n111 - n011)
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return (y0 + w * (y1 - y0)) * 0.97


@njit(parallel=True, cache=True)
def fbm3(P, scale, octaves, seed, lacunarity=2.0, gain=0.5):
    n = P.shape[0]
    out = np.empty(n)
    for i in prange(n):
        x = P[i, 0] / scale
        y = P[i, 1] / scale
        z = P[i, 2] / scale
        a = 1.0
        tot = 0.0
        s = 0.0
        for o in range(octaves):
            s += a * perlin3(x, y, z, seed + o * 1013)
            tot += a
            a *= gain
            x = x * lacunarity + 17.13
            y = y * lacunarity - 9.71
            z = z * lacunarity + 3.37
        out[i] = s / tot
    return out


@njit(parallel=True, cache=True)
def ridged3(P, scale, octaves, seed, gain=0.5, sharp=2.0):
    """Ridged multifractal in [0, 1]; successive octaves are weighted by the previous ridge (more detail on ridges)."""
    n = P.shape[0]
    out = np.empty(n)
    for i in prange(n):
        x = P[i, 0] / scale
        y = P[i, 1] / scale
        z = P[i, 2] / scale
        a = 1.0
        tot = 0.0
        s = 0.0
        w = 1.0
        for o in range(octaves):
            r = 1.0 - abs(perlin3(x, y, z, seed + o * 733))
            r = r ** sharp
            r *= w
            w = min(1.0, max(0.0, r * 1.6))
            s += a * r
            tot += a
            a *= gain
            x = x * 2.03 + 5.17
            y = y * 2.03 - 1.31
            z = z * 2.03 + 2.71
        out[i] = s / tot
    return out


def fbm2(x, y, scale, octaves=4, seed=0, gain=0.5, z=0.0):
    P = np.stack([np.ravel(x), np.ravel(y), np.full(np.size(x), z + 0.731 * seed)], 1).astype(np.float64)
    return fbm3(P, float(scale), int(octaves), int(seed), 2.0, float(gain))


def ridged2(x, y, scale, octaves=5, seed=0, gain=0.5, sharp=2.0):
    P = np.stack([np.ravel(x), np.ravel(y), np.full(np.size(x), 0.913 * seed)], 1).astype(np.float64)
    return ridged3(P, float(scale), int(octaves), int(seed), float(gain), float(sharp))


# ----------------------------------------------------------------------------------------------- Worley (cells)


@njit(cache=True, inline="always")
def _rnd(h):
    return (h & 0xFFFFFF) / 16777216.0


@njit(parallel=True, cache=True)
def worley3(P, cell, aniso, seed, jitter=0.9):
    """3D cellular noise. Coordinates are divided by cell * aniso (per axis).  Returns F1, F2 (in cell units, after the
    anisotropic scaling), the integer id of the nearest feature point and the vector from it to the point (cell units)."""
    n = P.shape[0]
    F1 = np.empty(n)
    F2 = np.empty(n)
    ID = np.empty(n, np.int64)
    VEC = np.empty((n, 3))
    for i in prange(n):
        x = P[i, 0] / (cell * aniso[0])
        y = P[i, 1] / (cell * aniso[1])
        z = P[i, 2] / (cell * aniso[2])
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        iz = int(math.floor(z))
        d1 = 1e9
        d2 = 1e9
        best = 0
        bx = 0.0
        by = 0.0
        bz = 0.0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cx = ix + dx
                    cy = iy + dy
                    cz = iz + dz
                    h = _hash3(cx, cy, cz, seed)
                    fx = cx + 0.5 + (_rnd(h) - 0.5) * jitter
                    h2 = _hash3(cx, cy, cz, seed + 7919)
                    fy = cy + 0.5 + (_rnd(h2) - 0.5) * jitter
                    h3 = _hash3(cx, cy, cz, seed + 15485)
                    fz = cz + 0.5 + (_rnd(h3) - 0.5) * jitter
                    ex = x - fx
                    ey = y - fy
                    ez = z - fz
                    d = math.sqrt(ex * ex + ey * ey + ez * ez)
                    if d < d1:
                        d2 = d1
                        d1 = d
                        best = h
                        bx = ex
                        by = ey
                        bz = ez
                    elif d < d2:
                        d2 = d
        F1[i] = d1
        F2[i] = d2
        ID[i] = best
        VEC[i, 0] = bx
        VEC[i, 1] = by
        VEC[i, 2] = bz
    return F1, F2, ID, VEC


@njit(parallel=True, cache=True)
def worley3_edge(P, cell, aniso, seed, jitter=0.9):
    """Distance (cell units) to the nearest Voronoi face (the true bisector distance, not F2-F1), plus the cell id.
    This gives clean, constant-width joints between blocks."""
    n = P.shape[0]
    E = np.empty(n)
    ID = np.empty(n, np.int64)
    for i in prange(n):
        x = P[i, 0] / (cell * aniso[0])
        y = P[i, 1] / (cell * aniso[1])
        z = P[i, 2] / (cell * aniso[2])
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        iz = int(math.floor(z))
        # pass 1: nearest feature point
        d1 = 1e9
        nx = 0.0
        ny = 0.0
        nz = 0.0
        best = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cx = ix + dx
                    cy = iy + dy
                    cz = iz + dz
                    h = _hash3(cx, cy, cz, seed)
                    fx = cx + 0.5 + (_rnd(h) - 0.5) * jitter
                    fy = cy + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 7919)) - 0.5) * jitter
                    fz = cz + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 15485)) - 0.5) * jitter
                    d = (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2
                    if d < d1:
                        d1 = d
                        nx = fx
                        ny = fy
                        nz = fz
                        best = h
        # pass 2: distance to the bisector planes with every other neighbour
        e = 1e9
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    cx = ix + dx
                    cy = iy + dy
                    cz = iz + dz
                    h = _hash3(cx, cy, cz, seed)
                    if h == best:
                        continue
                    fx = cx + 0.5 + (_rnd(h) - 0.5) * jitter
                    fy = cy + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 7919)) - 0.5) * jitter
                    fz = cz + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 15485)) - 0.5) * jitter
                    mx = 0.5 * (nx + fx)
                    my = 0.5 * (ny + fy)
                    mz = 0.5 * (nz + fz)
                    ax = fx - nx
                    ay = fy - ny
                    az = fz - nz
                    al = math.sqrt(ax * ax + ay * ay + az * az) + 1e-12
                    dd = ((mx - x) * ax + (my - y) * ay + (mz - z) * az) / al
                    if dd < e:
                        e = dd
        E[i] = e
        ID[i] = best
    return E, ID


def cell_random(ids, salt=0):
    """Deterministic float in [0, 1) per cell id."""
    h = (ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt * 0x632BE59BD9B4E019 + 1)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    h ^= h >> np.uint64(31)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(29)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# ----------------------------------------------------------------------------------------------- paths


class PathField:
    """Fast distance / arc-length / side queries against a polyline (densely resampled + KD-tree)."""

    def __init__(self, path, step=0.5, closed=False):
        P = np.asarray(path, np.float64)
        if closed:
            P = np.vstack([P, P[:1]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(math.ceil(s[-1] / step)) + 1)
        t = np.linspace(0, s[-1], n)
        self.pts = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)
        self.s = t
        tg = np.gradient(self.pts, axis=0)
        if closed:
            tg[0] = tg[-1] = self.pts[1] - self.pts[-2]
        self.tan = tg / np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
        self.tree = cKDTree(self.pts)
        self.length = float(s[-1])
        # orientation of a closed path (shoelace): +1 counter-clockwise (interior on the left), -1 clockwise
        self.orient = 1.0
        if closed:
            a = 0.5 * float(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1]))
            self.orient = 1.0 if a > 0 else -1.0

    def query(self, x, y):
        q = np.stack([np.ravel(x), np.ravel(y)], 1)
        d, i = self.tree.query(q, k=1, workers=-1)
        v = q - self.pts[i]
        side = np.sign(self.tan[i, 0] * v[:, 1] - self.tan[i, 1] * v[:, 0] + 1e-12)
        return d, self.s[i], side


def polygon_signed(field: PathField, x, y):
    """Signed distance to a closed polygon from its PathField (negative inside), for either winding."""
    d, _, side = field.query(x, y)
    return np.where(side * field.orient > 0, -d, d)


# ----------------------------------------------------------------------------------------------- hydraulic erosion


@njit(cache=True)
def _bilerp_grad(H, x, y):
    n0, n1 = H.shape
    ix = int(x)
    iy = int(y)
    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if ix > n1 - 2:
        ix = n1 - 2
    if iy > n0 - 2:
        iy = n0 - 2
    fx = x - ix
    fy = y - iy
    h00 = H[iy, ix]
    h10 = H[iy, ix + 1]
    h01 = H[iy + 1, ix]
    h11 = H[iy + 1, ix + 1]
    gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
    gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
    h = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy
    return h, gx, gy


@njit(cache=True)
def erode(H, n_drops, seed, cell_m, mask, inertia=0.06, capacity=6.0, min_slope=0.01, deposit=0.25, erode_k=0.35,
          evaporate=0.012, gravity=9.0, max_life=90, radius=3):
    """Droplet hydraulic erosion (after H. Beyer 2015 / S. Lague) on a height grid in metres.
    `mask` (0..1) scales the erosion per cell so the designed valley floor stays as authored."""
    np.random.seed(seed)
    n0, n1 = H.shape
    # erosion brush weights
    rr = radius
    bw = np.zeros((2 * rr + 1, 2 * rr + 1))
    tot = 0.0
    for a in range(-rr, rr + 1):
        for b in range(-rr, rr + 1):
            d = math.sqrt(a * a + b * b)
            if d <= rr:
                bw[a + rr, b + rr] = rr - d
                tot += rr - d
    bw /= tot
    for _ in range(n_drops):
        x = np.random.random() * (n1 - 3) + 1
        y = np.random.random() * (n0 - 3) + 1
        dx = 0.0
        dy = 0.0
        speed = 1.0
        water = 1.0
        sed = 0.0
        for life in range(max_life):
            ix = int(x)
            iy = int(y)
            fx = x - ix
            fy = y - iy
            h, gx, gy = _bilerp_grad(H, x, y)
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)
            l = math.sqrt(dx * dx + dy * dy)
            if l < 1e-9:
                break
            dx /= l
            dy /= l
            x += dx
            y += dy
            if x < 1 or y < 1 or x > n1 - 2 or y > n0 - 2:
                break
            nh, _, _ = _bilerp_grad(H, x, y)
            dh = nh - h
            cap = max(-dh / cell_m, min_slope) * speed * water * capacity
            m = mask[iy, ix]
            if sed > cap or dh > 0:
                amt = min(dh, sed) if dh > 0 else (sed - cap) * deposit
                sed -= amt
                H[iy, ix] += amt * (1 - fx) * (1 - fy)
                H[iy, ix + 1] += amt * fx * (1 - fy)
                H[iy + 1, ix] += amt * (1 - fx) * fy
                H[iy + 1, ix + 1] += amt * fx * fy
            else:
                amt = min((cap - sed) * erode_k * m, -dh)
                for a in range(-rr, rr + 1):
                    for b in range(-rr, rr + 1):
                        w = bw[a + rr, b + rr]
                        if w <= 0:
                            continue
                        yy = iy + a
                        xx = ix + b
                        if yy < 0 or xx < 0 or yy >= n0 or xx >= n1:
                            continue
                        H[yy, xx] -= amt * w
                sed += amt
            speed = math.sqrt(max(0.0, speed * speed + dh / cell_m * gravity * -1.0)) if dh < 0 else math.sqrt(max(0.0, speed * speed - dh / cell_m * gravity))
            water *= 1 - evaporate
    return H


@njit(parallel=True, cache=True)
def thermal(H, iters, talus, k, mask):
    """Simple thermal weathering: material above the talus slope (dh per cell) slides to lower neighbours."""
    n0, n1 = H.shape
    for _ in range(iters):
        D = np.zeros_like(H)
        for i in prange(1, n0 - 1):
            for j in range(1, n1 - 1):
                h = H[i, j]
                tot = 0.0
                dmax = 0.0
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if a == 0 and b == 0:
                            continue
                        d = h - H[i + a, j + b]
                        if d > talus:
                            tot += d - talus
                            if d > dmax:
                                dmax = d
                if tot > 0:
                    move = k * (dmax - talus) * 0.5 * mask[i, j]
                    D[i, j] -= move
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            if a == 0 and b == 0:
                                continue
                            d = h - H[i + a, j + b]
                            if d > talus:
                                D[i + a, j + b] += move * (d - talus) / tot
        H += D
    return H


# ----------------------------------------------------------------------------------------------- grid sampling


@njit(parallel=True, cache=True)
def bicubic(G, x0, y0, cell, X, Y):
    """Catmull-Rom bicubic interpolation of grid G (rows = y ascending from y0, cols = x ascending from x0)."""
    n0, n1 = G.shape
    out = np.empty(X.shape[0])
    for i in prange(X.shape[0]):
        gx = (X[i] - x0) / cell
        gy = (Y[i] - y0) / cell
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        tx = gx - ix
        ty = gy - iy
        acc = 0.0
        for a in range(-1, 3):
            wy = _cr(ty, a)
            yy = min(max(iy + a, 0), n0 - 1)
            row = 0.0
            for b in range(-1, 3):
                xx = min(max(ix + b, 0), n1 - 1)
                row += _cr(tx, b) * G[yy, xx]
            acc += wy * row
        out[i] = acc
    return out


@njit(cache=True, inline="always")
def _cr(t, k):
    # Catmull-Rom weights for offsets -1, 0, 1, 2
    if k == -1:
        return ((-t + 2) * t - 1) * t * 0.5
    if k == 0:
        return (((3 * t - 5) * t) * t + 2) * 0.5
    if k == 1:
        return ((-3 * t + 4) * t + 1) * t * 0.5
    return ((t - 1) * t * t) * 0.5
