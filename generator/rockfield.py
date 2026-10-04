"""rockfield - the 3D granite structure added to the height field round the castle crag (signed distance field).

d(p) = d_terrain(p) + w(p) * R(p, n)
  d_terrain   first-order distance to the height field  (z - H) / sqrt(1 + |grad H|^2)   (negative = rock)
  n           the height field's normal at p (local frame: n, 'up the face', 'along the face')
  w           rock weight: steep ground (cliffs, gorge and ravine walls, the crag rim); 0 on lawns / river bed /
              the castle plateau
  R           granite, built like jointed granite actually looks:
                * buttresses and bays, vertical gullies (couloirs)                         -- tens of metres
                * big joint blocks (anisotropic Voronoi, tall cells = vertical jointing): every block is a PLANAR
                  FACET with its own tilt and offset relative to the face -> flat slabs, sharp stepped edges,
                  ledges where a block leans back, overhangs where it leans out             -- 10-30 m
                * open joints (grooves) between blocks, slight rounding of their edges
                * sheeting joints: near-horizontal grooves that leave narrow ledges        -- every 4-9 m
                * a finer secondary block set with its own small facets                    -- 3-6 m
                * fine roughness                                                            -- < 1 m
              R > 0 carves the rock back, R < 0 builds it out.

Everything is numba-compiled; the same function is evaluated on the voxel volumes (marching cubes) and at single points
(ambient occlusion, masks, scatter placement).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fastnoise import _hash3, _rnd, perlin3

(P_SEED, P_BLOCK, P_BLOCK_AZ, P_BLOCK_OFF, P_TILT_UP, P_TILT_SIDE, P_JOINT, P_ROUND, P_SHEET, P_SHEET_SP,
 P_GULLY, P_BUTT, P_SMALL, P_SMALL_TILT, P_FINE) = range(15)
DEFAULT = np.array([
    11.0,     # seed
    15.0,     # big block cell size (m)
    1.9,      # vertical stretch of the big blocks (vertical jointing)
    2.2,      # big block offset amplitude (m): blocks stand proud / recede
    0.42,     # facet tilt 'up the face' amplitude (lean back +, lean out -)
    0.30,     # facet tilt 'along the face' amplitude
    0.95,     # open joint depth (m)
    0.35,     # edge rounding (m)
    0.85,     # sheeting groove depth (m)
    6.5,      # sheeting joint spacing (m)
    5.0,      # gully depth (m)
    7.5,      # buttress amplitude (m)
    0.75,     # small block offset amplitude (m)
    0.22,     # small block facet tilt
    0.10,     # fine roughness (m)
], np.float64)


@njit(cache=True, inline="always")
def _sst(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def _fbm(x, y, z, oct_, seed):
    a = 1.0
    s = 0.0
    tot = 0.0
    for o in range(oct_):
        s += a * perlin3(x, y, z, seed + o * 131)
        tot += a
        a *= 0.5
        x = x * 2.02 + 3.1
        y = y * 2.02 - 1.7
        z = z * 2.02 + 0.9
    return s / tot


@njit(cache=True)
def _cells(x, y, z, cell, sz, seed):
    """Anisotropic Voronoi (z stretched by sz).  Returns (distance in metres to the nearest Voronoi face, cell hash,
    feature point x, y, z in world metres)."""
    px = x / cell
    py = y / cell
    pz = z / (cell * sz)
    ix = int(math.floor(px))
    iy = int(math.floor(py))
    iz = int(math.floor(pz))
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
                fx = cx + 0.5 + (_rnd(h) - 0.5) * 0.85
                fy = cy + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 7919)) - 0.5) * 0.85
                fz = cz + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 15485)) - 0.5) * 0.85
                d = (px - fx) ** 2 + (py - fy) ** 2 + ((pz - fz) * sz) ** 2
                if d < d1:
                    d1 = d
                    nx = fx
                    ny = fy
                    nz = fz
                    best = h
    e = 1e9
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-1, 2):
                cx = ix + dx
                cy = iy + dy
                cz = iz + dz
                h = _hash3(cx, cy, cz, seed)
                if h == best:
                    continue
                fx = cx + 0.5 + (_rnd(h) - 0.5) * 0.85
                fy = cy + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 7919)) - 0.5) * 0.85
                fz = cz + 0.5 + (_rnd(_hash3(cx, cy, cz, seed + 15485)) - 0.5) * 0.85
                ax = fx - nx
                ay = fy - ny
                az = (fz - nz) * sz
                al = math.sqrt(ax * ax + ay * ay + az * az) + 1e-12
                mx = 0.5 * (nx + fx) - px
                my = 0.5 * (ny + fy) - py
                mz = (0.5 * (nz + fz) - pz) * sz
                dd = (mx * ax + my * ay + mz * az) / al
                if dd < e:
                    e = dd
    return e * cell, best, nx * cell, ny * cell, nz * cell * sz


@njit(cache=True)
def rock(x, y, z, gx, gy, P):
    """Granite displacement R (m) at a point whose height-field gradient is (gx, gy)."""
    seed = int(P[P_SEED])
    # local frame of the face: n = normal, u = 'up the face', v = 'along the face'
    il = 1.0 / math.sqrt(1.0 + gx * gx + gy * gy)
    nx = -gx * il
    ny = -gy * il
    nz = il
    ux = -nz * nx
    uy = -nz * ny
    uz = 1.0 - nz * nz
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    if ul < 1e-3:                       # flat ground: any tangent will do
        ux, uy, uz = 1.0, 0.0, 0.0
    else:
        ux /= ul
        uy /= ul
        uz /= ul
    vx = ny * uz - nz * uy
    vy = nz * ux - nx * uz
    vz = nx * uy - ny * ux
    # the big structure belongs to real cliffs; moderate (forested) slopes only get the smaller blocks
    big = _sst(0.9, 2.3, math.sqrt(gx * gx + gy * gy))
    # buttresses / bays and couloirs
    butt = P[P_BUTT] * _fbm(x / 70.0, y / 70.0, z / 140.0, 4, seed + 1)
    g = 1.0 - abs(perlin3(x / 34.0, y / 34.0, z / 190.0, seed + 2))
    g2 = 1.0 - abs(perlin3(x / 13.0, y / 13.0, z / 75.0, seed + 3))
    gully = P[P_GULLY] * (g ** 7 + 0.35 * g2 ** 9)
    # big joint blocks: planar facets
    e, bid, cx, cy, cz = _cells(x, y, z, P[P_BLOCK], P[P_BLOCK_AZ], seed + 4)
    r1 = _rnd(bid)
    r2 = _rnd(bid >> 5)
    r3 = _rnd(bid >> 11)
    tu = (r2 * 1.65 - 0.65) * P[P_TILT_UP]          # mostly leaning back, sometimes overhanging
    tv = (r3 * 2.0 - 1.0) * P[P_TILT_SIDE]
    qx = x - cx
    qy = y - cy
    qz = z - cz
    facet = (qx * ux + qy * uy + qz * uz) * tu + (qx * vx + qy * vy + qz * vz) * tv + (r1 * 2.0 - 1.0) * P[P_BLOCK_OFF]
    joint = P[P_JOINT] * (1.0 - _sst(0.05, 0.55, e)) + P[P_ROUND] * (1.0 - _sst(0.0, 1.4, e))
    # sheeting joints (near-horizontal), spacing varies per block
    zz = z + 1.8 * _fbm(x / 23.0, y / 23.0, z / 23.0, 2, seed + 5)
    sp = P[P_SHEET_SP] * (0.7 + 0.6 * r2)
    t = zz / sp + r3
    t = t - math.floor(t)
    sheet = P[P_SHEET] * (_sst(0.80, 0.94, t) * (1.0 - _sst(0.955, 1.0, t)))
    # secondary blocks with small facets
    e2, bid2, cx2, cy2, cz2 = _cells(x + 31.7, y - 12.3, z, 4.6, 1.35, seed + 6)
    s1 = _rnd(bid2)
    s2 = _rnd(bid2 >> 7)
    qx = x + 31.7 - cx2
    qy = y - 12.3 - cy2
    qz = z - cz2
    small = (qx * ux + qy * uy + qz * uz) * (s2 * 2.0 - 0.8) * P[P_SMALL_TILT] + (s1 * 2.0 - 1.0) * P[P_SMALL]
    small += 0.38 * (1.0 - _sst(0.03, 0.30, e2))
    fine = P[P_FINE] * _fbm(x / 1.6, y / 1.6, z / 1.6, 2, seed + 7)
    return big * (butt + gully + facet + sheet) + (0.45 + 0.55 * big) * joint + small + fine


@njit(parallel=True, cache=True)
def fill_volume(vol, Hc, Gc, Wc, z0, vox, P, band0, bandw, offc):
    """vol[i, j, k] = signed distance at (x_i, y_j, z0 + k vox); columns carry H, grad H, world x/y, the rock weight
    and an extra offset (border taper).  Only a band round the surface is evaluated; outside it the value is clamped."""
    nx, ny, nz = vol.shape
    for i in prange(nx):
        for j in range(ny):
            h = Hc[i, j, 0]
            gx = Hc[i, j, 1]
            gy = Hc[i, j, 2]
            nrm = math.sqrt(1.0 + Gc[i, j] * Gc[i, j])
            w = Wc[i, j]
            band = band0 + bandw * w
            xw = Hc[i, j, 3]
            yw = Hc[i, j, 4]
            oc = offc[i, j]
            for k in range(nz):
                z = z0 + k * vox
                d0 = (z - h) / nrm
                if d0 > band:
                    vol[i, j, k] = band
                    continue
                if d0 < -band:
                    vol[i, j, k] = -band
                    continue
                d = d0 + oc
                if w > 1e-4:
                    d += w * rock(xw, yw, z, gx, gy, P)
                if d > band:
                    d = band
                elif d < -band:
                    d = -band
                vol[i, j, k] = d


@njit(parallel=True, cache=True)
def eval_points(X, Y, Z, H, GX, GY, W, OFF, P):
    """Signed distance at arbitrary points (H, grad, W, OFF sampled at each point's column beforehand)."""
    n = X.shape[0]
    out = np.empty(n)
    for i in prange(n):
        nrm = math.sqrt(1.0 + GX[i] * GX[i] + GY[i] * GY[i])
        d = (Z[i] - H[i]) / nrm + OFF[i]
        if W[i] > 1e-4:
            d += W[i] * rock(X[i], Y[i], Z[i], GX[i], GY[i], P)
        out[i] = d
    return out


@njit(parallel=True, cache=True)
def fine_displacement(V, N, W, seed, amp):
    """Vertex displacement along the normal: exfoliation flakes + grain (applied after re-meshing)."""
    n = V.shape[0]
    out = np.empty((n, 3))
    for i in prange(n):
        x = V[i, 0]
        y = V[i, 1]
        z = V[i, 2]
        a = W[i] * amp
        f = 0.55 * _fbm(x / 0.9, y / 0.9, z / 0.9, 3, seed) + 0.35 * (1.0 - abs(perlin3(x / 2.3, y / 2.3, z / 3.1, seed + 9))) ** 3
        f += 0.25 * _fbm(x / 0.33, y / 0.33, z / 0.33, 2, seed + 4)
        out[i, 0] = x + N[i, 0] * f * a
        out[i, 1] = y + N[i, 1] * f * a
        out[i, 2] = z + N[i, 2] * f * a
    return out


@njit(parallel=True, cache=True)
def rock_points(X, Y, Z, GX, GY, P):
    """R at arbitrary points, each with the gradient of its nearest surface column (local face frame)."""
    n = X.shape[0]
    out = np.empty(n)
    for i in prange(n):
        out[i] = rock(X[i], Y[i], Z[i], GX[i], GY[i], P)
    return out
