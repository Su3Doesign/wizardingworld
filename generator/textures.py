"""Procedural, seamlessly tiling PBR textures (numpy / scipy / PIL).

Normal maps are written in OpenGL convention (green = +V up); the Unreal builder sets
"Flip Green Channel" on import.  Everything is deterministic given the seed.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------- noise


def fbm_tile(n, beta=2.0, seed=0, f_lo=1.0, f_hi=None):
    """Seamless 1/f^beta noise via spectral synthesis. Zero mean, unit std."""
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(n)[:, None] * n
    fy = np.fft.rfftfreq(n)[None, :] * n
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    amp[f < f_lo] = 0.0
    if f_hi is not None:
        amp[f > f_hi] = 0.0
    spec = amp * np.exp(1j * rng.uniform(0, 2 * math.pi, amp.shape))
    img = np.fft.irfft2(spec, s=(n, n))
    img -= img.mean()
    return (img / (img.std() + 1e-12)).astype(np.float32)


def norm01(a, lo=None, hi=None):
    lo = np.percentile(a, 1) if lo is None else lo
    hi = np.percentile(a, 99) if hi is None else hi
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)


def worley_tile(n, cells, seed=0, jitter=0.9):
    """Tileable cellular noise. Returns (F1, F2, id) in units of cell size."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
    px = (gx + 0.5 + (rng.random((cells, cells)) - 0.5) * jitter) / cells
    py = (gy + 0.5 + (rng.random((cells, cells)) - 0.5) * jitter) / cells
    pts = np.stack([px.ravel(), py.ravel()], 1)
    ids = np.arange(cells * cells)
    allp = np.concatenate([pts + np.array([dx, dy]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    allid = np.tile(ids, 9)
    tree = cKDTree(allp)
    u = (np.arange(n) + 0.5) / n
    UU, VV = np.meshgrid(u, u, indexing="ij")
    d, i = tree.query(np.stack([UU.ravel(), VV.ravel()], 1), k=2)
    f1 = (d[:, 0] * cells).reshape(n, n).astype(np.float32)
    f2 = (d[:, 1] * cells).reshape(n, n).astype(np.float32)
    cid = allid[i[:, 0]].reshape(n, n)
    return f1, f2, cid


def srgb_to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


def height_to_normal(h, strength=1.0, wrap=True):
    """Height (rows = -V, cols = +U) -> OpenGL tangent-space normal in [0,1] RGB."""
    if wrap:
        dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
        dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    else:
        dy, dx = np.gradient(h)
    nx = -dx * strength
    ny = dy * strength           # rows grow downwards, V grows upwards
    nz = np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / l, ny / l, nz / l], -1)
    return (n * 0.5 + 0.5).astype(np.float32)


def save_png(arr, path, mode=None):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(arr, mode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=False, compress_level=4)


def save_jpg(arr, path, q=93):
    arr = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, quality=q, subsampling=0)


# ----------------------------------------------------------------------------- stone


def _stone_surface(n, seed, strata=0.0):
    """Common stone detail layers. Returns dict of float32 fields."""
    L = fbm_tile(n, 2.6, seed + 1, 1, 24)           # large blotches
    M = fbm_tile(n, 2.2, seed + 2, 4, 140)          # mid mottling
    H = fbm_tile(n, 1.4, seed + 3, 40, n // 3)      # fine grain
    pores = norm01(fbm_tile(n, 0.9, seed + 4, 60, n // 2), 0.0, 1.0)
    pit = np.clip((pores - 0.74) * 6.0, 0, 1)       # sparse small pits
    stain = fbm_tile(n, 3.0, seed + 5, 1, 12)
    v = np.linspace(0, 1, n, endpoint=False)[:, None]
    band = np.sin(2 * math.pi * (6 * v + 0.35 * L)) * strata
    return dict(L=L, M=M, H=H, pit=pit, stain=stain, band=band)


def stone_ashlar(n=2048, seed=10, rows=4):
    """Ashlar masonry (running bond) with weathered blocks. Returns (albedo_lin, height, rough)."""
    rng = np.random.default_rng(seed)
    S = _stone_surface(n, seed, strata=0.03)
    row_h = n / rows
    yy = np.arange(n)[:, None] + np.zeros((1, n))
    xx = np.arange(n)[None, :] + np.zeros((n, 1))
    row = (yy // row_h).astype(int)
    block_id = np.zeros((n, n), np.int32)
    d_edge = np.full((n, n), 1e9, np.float32)
    edges_per_row = []
    bid = 0
    for r in range(rows):
        widths = rng.uniform(0.7, 1.3, size=int(rng.integers(2, 4)))
        widths = widths / widths.sum() * n
        starts = np.concatenate([[0], np.cumsum(widths)[:-1]]) + rng.uniform(0, n)
        starts = starts % n
        order = np.argsort(starts)
        starts = starts[order]
        wsorted = widths[order]
        ends = starts + wsorted
        # block id by x; handle wrap by testing x and x+n
        sel = row == r
        xs = xx[sel]
        bi = np.full(xs.shape, -1)
        for k in range(len(starts)):
            inside = ((xs >= starts[k]) & (xs < ends[k])) | ((xs + n >= starts[k]) & (xs + n < ends[k]))
            bi[inside] = bid + k
        block_id[sel] = bi
        # distance to vertical edges (wrap aware)
        dv = np.full(xs.shape, 1e9, np.float32)
        for e in starts:
            dd = np.abs(xs - e)
            dd = np.minimum(dd, n - dd)
            dv = np.minimum(dv, dd)
        # distance to horizontal edges (row boundaries)
        ys = yy[sel]
        dyb = ys - r * row_h
        dh = np.minimum(dyb, row_h - dyb)
        d_edge[sel] = np.minimum(dv, dh).astype(np.float32)
        bid += len(starts)
    # per-block random values
    nb = block_id.max() + 1
    tint = rng.uniform(0.86, 1.10, nb)[block_id].astype(np.float32)
    warm = rng.uniform(-0.04, 0.05, nb)[block_id].astype(np.float32)
    boff = rng.normal(0, 0.0012, nb)[block_id].astype(np.float32)
    joint = 4.0
    bevel = 14.0
    mortar = (d_edge < joint).astype(np.float32)
    mortar = ndimage.gaussian_filter(mortar, 1.2)
    t = np.clip((d_edge - joint) / bevel, 0, 1)
    edge_round = np.sqrt(1 - (1 - t) ** 2)             # chamfer / rounded arris
    height = (boff + 0.0014 * edge_round - 0.0030 * mortar
              + 0.0012 * S["M"] * 0.3 + 0.0006 * S["H"] - 0.0010 * S["pit"])
    # chipped arrises: erode edges with noise
    chip = np.clip(1.0 - d_edge / (6 + 10 * norm01(S["M"])), 0, 1) ** 2
    height = height - 0.0016 * chip * norm01(S["H"], -1, 1)
    # albedo
    cA = srgb_to_lin(np.array([0.60, 0.56, 0.49]))
    cB = srgb_to_lin(np.array([0.70, 0.66, 0.58]))
    k = norm01(S["L"] * 0.7 + S["M"] * 0.3, -2, 2)[..., None]
    alb = cA * (1 - k) + cB * k
    alb = alb * (tint[..., None] * (1 + 0.10 * S["M"][..., None] * 0.35)) * (1 + S["band"][..., None])
    alb[..., 0] *= 1 + warm
    alb[..., 2] *= 1 - warm
    alb *= (1 - 0.45 * S["pit"][..., None])
    # rust / lichen stains
    rust = np.clip((S["stain"] - 0.7) * 1.4, 0, 1)[..., None]
    alb = alb * (1 - 0.35 * rust) + srgb_to_lin(np.array([0.58, 0.40, 0.26])) * 0.35 * rust * 0.6
    gry = np.clip((-S["stain"] - 0.9) * 1.5, 0, 1)[..., None]
    alb = alb * (1 - 0.28 * gry) + srgb_to_lin(np.array([0.40, 0.42, 0.38])) * 0.28 * gry
    mort_col = srgb_to_lin(np.array([0.40, 0.38, 0.34]))
    alb = alb * (1 - mortar[..., None]) + mort_col * mortar[..., None]
    alb = np.clip(alb, 0, 1)
    rough = np.clip(0.80 + 0.07 * S["M"] + 0.12 * S["pit"] + 0.12 * mortar, 0.35, 1.0).astype(np.float32)
    return alb.astype(np.float32), height.astype(np.float32), rough


def stone_carved(n=2048, seed=20):
    """Un-coursed weathered limestone (columns, capitals, mouldings, floors)."""
    S = _stone_surface(n, seed, strata=0.045)
    cA = srgb_to_lin(np.array([0.62, 0.58, 0.50]))
    cB = srgb_to_lin(np.array([0.74, 0.70, 0.62]))
    k = norm01(S["L"] * 0.7 + S["M"] * 0.3, -2, 2)[..., None]
    alb = cA * (1 - k) + cB * k
    alb = alb * (1 + 0.12 * S["M"][..., None] * 0.4) * (1 + S["band"][..., None])
    alb *= (1 - 0.5 * S["pit"][..., None])
    rust = np.clip((S["stain"] - 0.65) * 1.5, 0, 1)[..., None]
    alb = alb * (1 - 0.3 * rust) + srgb_to_lin(np.array([0.60, 0.42, 0.28])) * 0.3 * rust * 0.6
    # hairline cracks from worley edges
    f1, f2, _ = worley_tile(n, 7, seed + 9, 0.95)
    crack = np.clip(1 - (f2 - f1) / 0.022, 0, 1) ** 2 * (norm01(S["L"], -1, 1) > 0.68)
    crack = ndimage.gaussian_filter(crack.astype(np.float32), 0.8)
    alb = alb * (1 - 0.40 * crack[..., None])
    height = (0.0016 * S["M"] * 0.5 + 0.0009 * S["H"] - 0.0014 * S["pit"] - 0.0020 * crack)
    rough = np.clip(0.78 + 0.08 * S["M"] + 0.15 * S["pit"] + 0.12 * crack, 0.35, 1.0).astype(np.float32)
    return np.clip(alb, 0, 1).astype(np.float32), height.astype(np.float32), rough


def stone_marble(n=2048, seed=30):
    """Smoother aged white-grey marble for the statue, with soft veining."""
    S = _stone_surface(n, seed, strata=0.0)
    vein_n = fbm_tile(n, 2.4, seed + 6, 1, 18)
    vein = np.clip(1 - np.abs(np.sin(vein_n * 2.4)) * 3.2, 0, 1) ** 2
    cA = srgb_to_lin(np.array([0.56, 0.55, 0.51]))
    cB = srgb_to_lin(np.array([0.68, 0.66, 0.61]))
    k = norm01(S["L"] * 0.6 + S["M"] * 0.4, -2, 2)[..., None]
    alb = cA * (1 - k) + cB * k
    alb = alb * (1 - 0.18 * vein[..., None]) * (1 - 0.35 * S["pit"][..., None] * 0.6)
    rust = np.clip((S["stain"] - 0.8) * 1.4, 0, 1)[..., None]
    alb = alb * (1 - 0.2 * rust) + srgb_to_lin(np.array([0.62, 0.50, 0.38])) * 0.2 * rust * 0.7
    height = 0.0004 * S["M"] + 0.0003 * S["H"] - 0.0005 * S["pit"]
    rough = np.clip(0.55 + 0.10 * S["M"] + 0.20 * S["pit"], 0.3, 1.0).astype(np.float32)
    return np.clip(alb, 0, 1).astype(np.float32), height.astype(np.float32), rough


def paving(n=2048, seed=40, tiles=4):
    """Large flagstones with narrow joints, worn and stained."""
    rng = np.random.default_rng(seed)
    S = _stone_surface(n, seed, strata=0.0)
    cell = n / tiles
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    row = (yy // cell).astype(int)
    off = rng.uniform(0, cell, tiles)
    xs = (xx + off[row]) % n
    dx = np.minimum(xs % cell, cell - xs % cell)
    dy = np.minimum(yy % cell, cell - yy % cell)
    d_edge = np.minimum(dx, dy)
    bid = (row * tiles + (xs // cell).astype(int))
    tint = rng.uniform(0.80, 1.08, tiles * tiles)[bid]
    boff = rng.normal(0, 0.0015, tiles * tiles)[bid]
    joint = 5.0
    mortar = ndimage.gaussian_filter((d_edge < joint).astype(np.float32), 1.5)
    t = np.clip((d_edge - joint) / 18.0, 0, 1)
    height = boff + 0.0016 * np.sqrt(1 - (1 - t) ** 2) - 0.004 * mortar + 0.001 * S["M"] * 0.4 - 0.001 * S["pit"]
    cA = srgb_to_lin(np.array([0.45, 0.43, 0.39]))
    cB = srgb_to_lin(np.array([0.58, 0.55, 0.50]))
    k = norm01(S["L"] * 0.6 + S["M"] * 0.4, -2, 2)[..., None]
    alb = (cA * (1 - k) + cB * k) * tint[..., None] * (1 - 0.4 * S["pit"][..., None])
    alb = alb * (1 - mortar[..., None]) + srgb_to_lin(np.array([0.22, 0.22, 0.20])) * mortar[..., None]
    rough = np.clip(0.62 + 0.10 * S["M"] + 0.2 * mortar, 0.3, 1.0).astype(np.float32)
    return np.clip(alb, 0, 1).astype(np.float32), height.astype(np.float32), rough


def moss(n=2048, seed=50):
    """Dense moss carpet with small clumps (tile = 1 m)."""
    f1, f2, cid = worley_tile(n, 90, seed, 1.0)
    clump = np.clip(1 - f1 / 0.8, 0, 1) ** 0.8
    fine = fbm_tile(n, 0.7, seed + 1, 80, n // 2)
    mid = fbm_tile(n, 2.0, seed + 2, 6, 60)
    rng = np.random.default_rng(seed)
    cid_val = rng.random(90 * 90)[cid]
    h = 0.5 * clump + 0.25 * norm01(fine, -2, 2) + 0.15 * norm01(mid, -2, 2)
    cA = srgb_to_lin(np.array([0.05, 0.12, 0.025]))
    cB = srgb_to_lin(np.array([0.19, 0.33, 0.06]))
    cC = srgb_to_lin(np.array([0.32, 0.38, 0.09]))
    k = np.clip(h[..., None] * 1.2 + (cid_val[..., None] - 0.5) * 0.25, 0, 1)   # (n, n, 1)
    alb = cA * (1 - k) + cB * k
    yel = np.clip((cid_val - 0.8) * 5, 0, 1)[..., None] * 0.35
    alb = alb * (1 - yel) + cC * yel
    alb *= 0.75 + 0.5 * norm01(mid, -2, 2)[..., None]
    rough = np.full((n, n), 0.92, np.float32) - 0.05 * clump
    return np.clip(alb, 0, 1).astype(np.float32), (0.004 * h).astype(np.float32), rough


def water_normal(n=2048, seed=60):
    """Seamless ripple height (tile = 8 m): broad swells + choppy detail, domain-warped."""
    a = fbm_tile(n, 2.9, seed, 3, 40)
    b = fbm_tile(n, 2.0, seed + 1, 20, 200)
    w1 = fbm_tile(n, 3.0, seed + 2, 1, 8)
    w2 = fbm_tile(n, 3.0, seed + 3, 1, 8)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    amp = 0.035 * n
    xs = (xx + w1 * amp) % n
    ys = (yy + w2 * amp) % n
    h = a + 0.35 * b
    h = ndimage.map_coordinates(h, [ys, xs], order=1, mode="wrap")
    return h.astype(np.float32)


# ----------------------------------------------------------------------------- foliage atlases


def _chaikin(pts, it=3):
    pts = np.asarray(pts, np.float64)
    for _ in range(it):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, 0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, 0)
        pts = np.empty((len(q) * 2, 2))
        pts[0::2], pts[1::2] = q, r
    return pts


def ivy_leaf_outline(rng, lobe=1.0):
    """Closed outline of an ivy leaf in unit coordinates; petiole at (0,0), tip at (0,1)."""
    right = [(0.0, 0.07), (0.10, -0.02), (0.24, 0.04), (0.33, 0.17), (0.30, 0.30), (0.20, 0.36),
             (0.25, 0.44 * 1.0), (0.40 * lobe, 0.52), (0.46 * lobe, 0.64), (0.34, 0.69), (0.23, 0.78), (0.10, 0.92)]
    right = [(x * (1 + rng.normal(0, 0.04)), y + rng.normal(0, 0.012)) for x, y in right]
    left = [(-x, y) for x, y in reversed(right)]
    pts = right + [(0.0, 1.0)] + left
    return _chaikin(pts, 3)


def ivy_atlas(n=2048, cells=4, seed=70):
    """cells x cells atlas of ivy leaves: RGB (sRGB-ish, edge-bled) + alpha."""
    rng = np.random.default_rng(seed)
    ss = 3
    cs = n // cells
    rgb = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)
    for cy in range(cells):
        for cx in range(cells):
            S = cs * ss
            im_a = Image.new("L", (S, S), 0)
            outline = ivy_leaf_outline(rng, lobe=rng.uniform(0.85, 1.15))
            scale = 0.82 * S
            ox, oy = S / 2, S * 0.93
            ang = rng.normal(0, 0.07)
            ca, sa = math.cos(ang), math.sin(ang)
            P = []
            for x, y in outline:
                xr, yr = x * ca - y * sa, x * sa + y * ca
                P.append((ox + xr * scale * 0.9, oy - yr * scale))
            ImageDraw.Draw(im_a).polygon(P, fill=255)
            # petiole
            ImageDraw.Draw(im_a).line([(ox, oy), (ox - 0.01 * S, S * 0.99)], fill=255, width=max(2, int(0.012 * S)))
            m = np.asarray(im_a.resize((cs, cs), Image.LANCZOS), np.float32) / 255.0
            # veins
            im_v = Image.new("L", (S, S), 0)
            dv = ImageDraw.Draw(im_v)
            tipy = oy - scale * 0.97
            dv.line([(ox, oy - 0.03 * S), (ox, tipy)], fill=255, width=max(2, int(0.014 * S)))
            for side in (-1, 1):
                for (t0, lenx, leny) in [(0.12, 0.26, 0.22), (0.30, 0.28, 0.25), (0.50, 0.20, 0.22), (0.70, 0.12, 0.16)]:
                    y0 = oy - scale * t0
                    dv.line([(ox, y0), (ox + side * scale * lenx, y0 - scale * leny * 0.7)], fill=255, width=max(2, int(0.008 * S)))
            v = np.asarray(im_v.resize((cs, cs), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0
            # colour
            yy, xx = np.mgrid[0:cs, 0:cs].astype(np.float32) / cs
            base_dark = srgb_to_lin(np.array([0.045, 0.16, 0.035]))
            base_lite = srgb_to_lin(np.array([0.16, 0.36, 0.07]))
            grad = np.clip(0.35 + 0.8 * np.abs(xx - 0.5) * 1.6 + rng.normal(0, 0.05), 0, 1)[..., None]
            rr = rng.uniform(0.85, 1.2)
            col = (base_dark * (1 - grad) + base_lite * grad) * rr
            vein_col = srgb_to_lin(np.array([0.34, 0.50, 0.20]))
            col = col * (1 - 0.65 * v[..., None]) + vein_col * 0.65 * v[..., None]
            tipfade = np.clip((0.45 - yy) * 0.0, 0, 1)
            # slight edge lightening
            edge = ndimage.gaussian_filter(m, 3) < 0.7
            col = col * (1 + 0.25 * edge[..., None])
            # edge bleed so mip levels do not darken: fill outside with nearest inside colour
            mask_in = m > 0.5
            idx = ndimage.distance_transform_edt(~mask_in, return_distances=False, return_indices=True)
            col_bleed = col[idx[0], idx[1]]
            col = np.where((m > 0.02)[..., None], col, col_bleed)
            y0_, x0_ = cy * cs, cx * cs
            rgb[y0_:y0_ + cs, x0_:x0_ + cs] = col
            alpha[y0_:y0_ + cs, x0_:x0_ + cs] = m
    return np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)


def leaf_sprite(rng, S=256, lobe=None):
    """One ivy leaf painted into an S x S RGBA sprite (petiole bottom-centre, tip up)."""
    ss = 3
    SS = S * ss
    im_a = Image.new("L", (SS, SS), 0)
    outline = ivy_leaf_outline(rng, lobe=lobe or rng.uniform(0.85, 1.2))
    scale = 0.82 * SS
    ox, oy = SS / 2, SS * 0.93
    ang = rng.normal(0, 0.07)
    ca, sa = math.cos(ang), math.sin(ang)
    P = [(ox + (x * ca - y * sa) * scale * 0.9, oy - (x * sa + y * ca) * scale) for x, y in outline]
    ImageDraw.Draw(im_a).polygon(P, fill=255)
    ImageDraw.Draw(im_a).line([(ox, oy), (ox - 0.01 * SS, SS * 0.99)], fill=255, width=max(2, int(0.012 * SS)))
    m = np.asarray(im_a.resize((S, S), Image.LANCZOS), np.float32) / 255.0
    im_v = Image.new("L", (SS, SS), 0)
    dv = ImageDraw.Draw(im_v)
    tipy = oy - scale * 0.97
    dv.line([(ox, oy - 0.03 * SS), (ox, tipy)], fill=255, width=max(2, int(0.016 * SS)))
    for side in (-1, 1):
        for (t0, lenx, leny) in [(0.12, 0.26, 0.22), (0.30, 0.28, 0.25), (0.50, 0.20, 0.22), (0.70, 0.12, 0.16)]:
            y0 = oy - scale * t0
            dv.line([(ox, y0), (ox + side * scale * lenx, y0 - scale * leny * 0.7)], fill=255, width=max(2, int(0.009 * SS)))
    v = np.asarray(im_v.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.9)), np.float32) / 255.0
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    base_dark = srgb_to_lin(np.array([0.07, 0.21, 0.045]))
    base_lite = srgb_to_lin(np.array([0.22, 0.46, 0.09]))
    grad = np.clip(0.30 + 0.9 * np.abs(xx - 0.5) * 1.6 + 0.35 * (1 - yy) + rng.normal(0, 0.05), 0, 1)[..., None]
    col = (base_dark * (1 - grad) + base_lite * grad) * rng.uniform(0.85, 1.2)
    vein_col = srgb_to_lin(np.array([0.42, 0.58, 0.26]))
    col = col * (1 - 0.7 * v[..., None]) + vein_col * 0.7 * v[..., None]
    edge = ndimage.gaussian_filter(m, 2) < 0.75
    col = col * (1 + 0.22 * edge[..., None])
    mask_in = m > 0.5
    idx = ndimage.distance_transform_edt(~mask_in, return_distances=False, return_indices=True)
    col = np.where((m > 0.02)[..., None], col, col[idx[0], idx[1]])
    return col.astype(np.float32), m.astype(np.float32)


def ivy_mat_atlas(n=2048, cells=2, seed=71, leaves=95):
    """Dense ivy 'mat' cards: many overlapping leaves with depth shading and an organic outline."""
    rng = np.random.default_rng(seed)
    cs = n // cells
    sprites = [leaf_sprite(rng, 256) for _ in range(14)]
    # RGBA PIL versions of the sprites (gamma-encoded so compositing is done in sRGB)
    pil = []
    for col, a in sprites:
        rgba = np.dstack([lin_to_srgb(col), a])
        pil.append(Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA"))
    rgb = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)
    for cy in range(cells):
        for cx in range(cells):
            canvas = Image.new("RGBA", (cs, cs), (0, 0, 0, 0))
            # background stems
            dr = ImageDraw.Draw(canvas)
            for _ in range(7):
                x0, y0 = rng.uniform(0.2, 0.8) * cs, rng.uniform(0.2, 0.8) * cs
                a = rng.uniform(0, 2 * math.pi)
                L = rng.uniform(0.2, 0.45) * cs
                dr.line([(x0, y0), (x0 + L * math.cos(a), y0 + L * math.sin(a))], fill=(46, 36, 24, 255), width=int(0.008 * cs))
            pts = []
            while len(pts) < leaves:
                p = rng.normal(0.5, 0.20, 2)
                if 0.05 < p[0] < 0.95 and 0.05 < p[1] < 0.95:
                    pts.append(p)
            order = rng.permutation(len(pts))
            for rank, k in enumerate(order):
                depth = rank / len(pts)
                spr = pil[int(rng.integers(0, len(pil)))]
                size = int(rng.uniform(0.17, 0.27) * cs)
                a = (rng.normal(180, 85)) % 360       # tips biased downwards
                im = spr.resize((size, size), Image.LANCZOS).rotate(a, resample=Image.BICUBIC, expand=True)
                arr = np.asarray(im, np.float32)
                b = 0.50 + 0.62 * depth + rng.normal(0, 0.06)
                arr[..., :3] = np.clip(arr[..., :3] * b, 0, 255)
                im = Image.fromarray(arr.astype(np.uint8), "RGBA")
                px = int(pts[k][0] * cs - im.width / 2)
                py = int(pts[k][1] * cs - im.height / 2)
                canvas.alpha_composite(im, (max(px, -im.width + 1), max(py, -im.height + 1))) if False else canvas.paste(im, (px, py), im)
            arr = np.asarray(canvas, np.float32) / 255.0
            col = srgb_to_lin(arr[..., :3])
            m = arr[..., 3]
            idx = ndimage.distance_transform_edt(m < 0.5, return_distances=False, return_indices=True)
            col = np.where((m > 0.02)[..., None], col, col[idx[0], idx[1]])
            rgb[cy * cs:(cy + 1) * cs, cx * cs:(cx + 1) * cs] = col
            alpha[cy * cs:(cy + 1) * cs, cx * cs:(cx + 1) * cs] = m
    return np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)


def fern_atlas(n=2048, cells=2, seed=80):
    """Pinnate fern fronds (arching), vertical cards, atlas cells x cells."""
    rng = np.random.default_rng(seed)
    ss = 2
    cs = n // cells
    rgb = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)
    for cy in range(cells):
        for cx in range(cells):
            S = cs * ss
            im = Image.new("L", (S, S), 0)
            d = ImageDraw.Draw(im)
            # rachis: gentle curve from the bottom centre upward
            t = np.linspace(0, 1, 160)
            bend = rng.uniform(-0.12, 0.12)
            xs = S * (0.5 + bend * t ** 2 + 0.02 * np.sin(5 * t))
            ys = S * (0.985 - 0.96 * t)
            for i in range(len(t) - 1):
                d.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill=255, width=max(2, int(S * 0.008 * (1 - t[i] * 0.7))))
            npair = 26
            for j in range(npair):
                tt = 0.08 + 0.9 * j / npair
                k = int(tt * (len(t) - 1))
                x0, y0 = xs[k], ys[k]
                # leaflet length: longest in the lower-middle, tapering to the tip
                ln = S * 0.30 * (math.sin(math.pi * min(1.0, tt * 1.1)) ** 0.8) * (1 - 0.55 * tt) + S * 0.02
                for side in (-1, 1):
                    a = side * rng.uniform(0.95, 1.2)           # angle from the rachis
                    droop = 0.28
                    x1 = x0 + side * ln * math.cos(a - 1.35 * side + 1.35 * side * 0.35)
                    # leaflet as a thin lens following a slight arc
                    pts_c, pts_t = [], []
                    L = ln
                    ang0 = math.radians(65) * side
                    for u in np.linspace(0, 1, 14):
                        px = x0 + side * L * u * math.cos(math.radians(20 + 18 * u))
                        py = y0 - L * u * math.sin(math.radians(35 - 40 * u)) * 0.6 + L * u * u * droop * 0.2
                        w = S * 0.0165 * math.sin(math.pi * (0.1 + 0.9 * u)) ** 0.7 * (1 - 0.5 * tt)
                        pts_c.append((px, py, w))
                    top = [(p[0], p[1] - p[2]) for p in pts_c]
                    bot = [(p[0], p[1] + p[2]) for p in pts_c][::-1]
                    d.polygon(top + bot, fill=255)
            m = np.asarray(im.resize((cs, cs), Image.LANCZOS), np.float32) / 255.0
            yy, xx = np.mgrid[0:cs, 0:cs].astype(np.float32) / cs
            lite = srgb_to_lin(np.array([0.26, 0.48, 0.10]))
            dark = srgb_to_lin(np.array([0.06, 0.20, 0.04]))
            g = np.clip(1.0 - yy * 0.55 + rng.normal(0, 0.04, (cs, cs)), 0, 1)[..., None]
            col = dark * (1 - g) + lite * g
            noise = fbm_tile(cs, 1.2, seed + cx * 7 + cy * 13, 4, cs // 4)
            col = col * (0.85 + 0.25 * norm01(noise, -2, 2)[..., None])
            mask_in = m > 0.5
            idx = ndimage.distance_transform_edt(~mask_in, return_distances=False, return_indices=True)
            col = np.where((m > 0.02)[..., None], col, col[idx[0], idx[1]])
            rgb[cy * cs:(cy + 1) * cs, cx * cs:(cx + 1) * cs] = col
            alpha[cy * cs:(cy + 1) * cs, cx * cs:(cx + 1) * cs] = m
    return np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)


def bark(n=1024, seed=90):
    f = fbm_tile(n, 1.8, seed, 2, 200)
    y = np.linspace(0, 1, n, endpoint=False)[:, None]
    stripes = fbm_tile(n, 1.4, seed + 2, 10, 120)
    streak = np.sin(2 * math.pi * 14 * np.linspace(0, 1, n, endpoint=False)[None, :] + 2.0 * stripes)
    h = 0.5 * norm01(streak) + 0.5 * norm01(f)
    cA = srgb_to_lin(np.array([0.12, 0.085, 0.055]))
    cB = srgb_to_lin(np.array([0.30, 0.22, 0.14]))
    k = h[..., None]
    alb = cA * (1 - k) + cB * k
    return alb.astype(np.float32), (0.004 * h).astype(np.float32), np.full((n, n), 0.85, np.float32)


# ----------------------------------------------------------------------------- driver


def write_set(outdir, name, alb, height, rough, ns=3.0, jpg=True):
    """Write BC (sRGB), N (OpenGL), R (grey)."""
    bc = lin_to_srgb(alb)
    (save_jpg if jpg else save_png)(bc, f"{outdir}/T_{name}_BC." + ("jpg" if jpg else "png"))
    nrm = height_to_normal(height, strength=ns * (alb.shape[0] / 2048.0) * 1000)
    save_png(nrm, f"{outdir}/T_{name}_N.png")
    save_png(rough, f"{outdir}/T_{name}_R.png", "L")


def build_all(outdir, n=2048, quick=False):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    nn = 1024 if quick else n

    def step(msg):
        print(f"  [{time.time() - t0:5.1f}s] {msg}", flush=True)

    a, h, r = stone_ashlar(nn)
    write_set(outdir, "StoneAshlar", a, h, r, ns=2.2)
    step("ashlar")
    a, h, r = stone_carved(nn)
    write_set(outdir, "StoneCarved", a, h, r, ns=2.2)
    step("carved")
    a, h, r = stone_marble(nn)
    write_set(outdir, "Marble", a, h, r, ns=2.0)
    step("marble")
    a, h, r = paving(nn)
    write_set(outdir, "Paving", a, h, r, ns=2.0)
    step("paving")
    a, h, r = moss(nn)
    write_set(outdir, "Moss", a, h, r, ns=1.6)
    step("moss")
    wh = water_normal(nn)
    save_png(height_to_normal(wh, strength=2.2 * nn / 2048 * 6), f"{outdir}/T_Water_N.png")
    step("water")
    rgb, al = ivy_atlas(nn)
    save_png(np.dstack([lin_to_srgb(rgb), al]), f"{outdir}/T_IvyLeaf_BCA.png", "RGBA")
    step("ivy")
    rgb, al = ivy_mat_atlas(nn)
    save_png(np.dstack([lin_to_srgb(rgb), al]), f"{outdir}/T_IvyMat_BCA.png", "RGBA")
    step("ivy mat")
    rgb, al = fern_atlas(nn)
    save_png(np.dstack([lin_to_srgb(rgb), al]), f"{outdir}/T_Fern_BCA.png", "RGBA")
    step("fern")
    a, h, r = bark(1024)
    write_set(outdir, "Bark", a, h, r, ns=2.0)
    step("bark")


if __name__ == "__main__":
    build_all(sys.argv[1], quick="--quick" in sys.argv)
