"""Photoreal weathered-masonry textures (numpy). Replaces the simple stone set in textures.py.

Every map tiles seamlessly. Packed maps: *_ORH.png  R = roughness, G = ambient occlusion (cavity),
B = height (0..1 over HMIN..HMAX millimetres).  Normal maps are OpenGL convention (green = +V up).
One tile of stone covers 2 m (2048 px  ->  0.977 mm / px).
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

import textures as tx
from textures import fbm_tile, norm01, srgb_to_lin, lin_to_srgb, worley_tile, save_png, save_jpg

HMIN, HMAX = -14.0, 4.0           # millimetres mapped to the packed height channel
TILE_MM = 2000.0


def sstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def blur(a, s):
    return ndimage.gaussian_filter(a, s, mode="wrap")


def fbm_aniso(n, beta, seed, ax=1.0, ay=0.12, f_lo=1.0, f_hi=None):
    """Seamless anisotropic noise: features long in Y (ay < ax) -> vertical rain streaks."""
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(n)[:, None] * n          # rows -> y
    fy = np.fft.rfftfreq(n)[None, :] * n         # cols -> x
    f = np.sqrt((fx * ay) ** 2 + (fy * ax) ** 2)
    f0 = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0
    amp[f0 < f_lo] = 0
    if f_hi is not None:
        amp[f0 > f_hi] = 0
    spec = amp * np.exp(1j * rng.uniform(0, 2 * math.pi, amp.shape))
    img = np.fft.irfft2(spec, s=(n, n))
    img -= img.mean()
    return (img / (img.std() + 1e-12)).astype(np.float32)


# ------------------------------------------------------------------------------------ shared layers
def lichen_layer(n, seed, density=0.4, px_mm=None):
    """Rosette lichen colonies. Returns (mask 0..1, colour (n,n,3) linear, height_mm)."""
    rng = np.random.default_rng(seed)
    cells = 30
    f1, f2, cid = worley_tile(n, cells, seed, 0.95)
    cell_px = n / cells
    nc = cells * cells
    active = rng.random(nc) < density
    rad = rng.uniform(0.22, 0.62, nc) * cell_px
    kind = rng.integers(0, 4, nc)
    colony_r = rad[cid]
    dist = f1 * cell_px
    edge_noise = fbm_tile(n, 1.6, seed + 3, 30, 300)
    rr = colony_r * (1.0 + 0.30 * edge_noise)
    inside = (dist < rr) & active[cid]
    t = np.clip(dist / np.maximum(rr, 1.0), 0, 1)
    rings = 0.5 + 0.5 * np.cos(t * 17.0 + rng.uniform(0, 6.28, nc)[cid])
    mask = inside.astype(np.float32) * (0.55 + 0.45 * rings) * (1.0 - 0.5 * t ** 3)
    mask = blur(mask, 0.8)
    pal = np.array([[0.60, 0.64, 0.50], [0.66, 0.66, 0.34], [0.78, 0.50, 0.18], [0.52, 0.56, 0.52]], np.float32)
    col = srgb_to_lin(pal[kind[cid]])
    h = 0.45 * mask * (0.7 + 0.3 * rings)
    return mask.astype(np.float32), col.astype(np.float32), h.astype(np.float32)


def crack_layer(n, seed, major=5, hair_density=0.55):
    """Hairline crack network (Worley edges, thinned) + a few major meandering cracks. Returns 0..1."""
    rng = np.random.default_rng(seed)
    f1, f2, _ = worley_tile(n, 9, seed + 1, 0.95)
    edge = np.clip(1 - (f2 - f1) / 0.012, 0, 1) ** 1.6
    gate = norm01(fbm_tile(n, 2.4, seed + 2, 1, 8), -1.3, 1.3) > (1 - hair_density) * 0.9
    hair = edge * gate
    big = np.zeros((n, n), np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(major):
        x, y = rng.uniform(0, n), rng.uniform(0, n)
        ang = rng.uniform(0, 2 * math.pi)
        width = rng.uniform(1.0, 2.6)
        for step in range(int(rng.uniform(140, 420))):
            ang += rng.normal(0, 0.22)
            x = (x + math.cos(ang) * 3.0) % n
            y = (y + math.sin(ang) * 3.0) % n
            w = width * (1.0 - step / 520.0)
            r = int(w + 1)
            xi, yi = int(x), int(y)
            ys = (np.arange(yi - r, yi + r + 1) % n)[:, None]
            xs = (np.arange(xi - r, xi + r + 1) % n)[None, :]
            big[ys, xs] = np.maximum(big[ys, xs], 1.0)
    big = blur(big, 0.9)
    return np.clip(np.maximum(hair * 0.8, big * 1.2), 0, 1).astype(np.float32)


def stains(n, seed):
    """Vertical rust / dark / efflorescence streak masks."""
    streak_a = fbm_aniso(n, 2.6, seed + 1, 1.0, 0.07, 1, 160)
    streak_b = fbm_aniso(n, 2.4, seed + 2, 1.0, 0.10, 1, 200)
    rust = sstep(0.55, 1.5, streak_a) * sstep(-0.2, 0.9, fbm_tile(n, 2.2, seed + 3, 1, 20))
    dark = sstep(0.6, 1.6, streak_b)
    white = sstep(1.0, 1.9, fbm_aniso(n, 2.0, seed + 4, 1.0, 0.16, 2, 260)) * 0.8
    return rust.astype(np.float32), dark.astype(np.float32), white.astype(np.float32)


def finish_maps(height_mm, albedo, rough, ao_extra=None, normal_strength=1.0):
    """height (mm) + albedo (linear) + roughness -> (BC jpg-ready srgb, N rgb, ORH rgb)."""
    n = height_mm.shape[0]
    px_mm = TILE_MM / n
    ao = np.clip(1.0 - 0.55 * (blur(height_mm, 6.0) - height_mm) / 2.2, 0.40, 1.0)
    ao = np.minimum(ao, np.clip(1.0 - 0.35 * (blur(height_mm, 22.0) - height_mm) / 5.0, 0.4, 1.0))
    if ao_extra is not None:
        ao = ao * ao_extra
    dx = (np.roll(height_mm, -1, 1) - np.roll(height_mm, 1, 1)) / (2 * px_mm)
    dy = (np.roll(height_mm, -1, 0) - np.roll(height_mm, 1, 0)) / (2 * px_mm)
    nx, ny, nz = -dx * normal_strength, dy * normal_strength, np.ones_like(dx)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = (np.stack([nx / l, ny / l, nz / l], -1) * 0.5 + 0.5).astype(np.float32)
    hn = np.clip((height_mm - HMIN) / (HMAX - HMIN), 0, 1)
    orh = np.stack([np.clip(rough, 0, 1), ao, hn], -1).astype(np.float32)
    return lin_to_srgb(np.clip(albedo * ao[..., None] ** 0.6, 0, 1)), nrm, orh


def write_set(outdir, name, bc_srgb, nrm, orh):
    save_jpg(bc_srgb, f"{outdir}/T_{name}_BC.jpg", q=94)
    save_png(nrm, f"{outdir}/T_{name}_N.png")
    save_png(orh, f"{outdir}/T_{name}_ORH.png")


# ------------------------------------------------------------------------------------ ashlar masonry
def ashlar_v2(n=2048, seed=10, rows=4):
    rng = np.random.default_rng(seed)
    px = TILE_MM / n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    # wobble the frame: wavy joints
    w1 = fbm_tile(n, 3.0, seed + 11, 1, 16)
    w2 = fbm_tile(n, 3.0, seed + 12, 1, 16)
    xs = (xx + w1 * (5.0 / px)) % n
    ys = (yy + w2 * (4.0 / px)) % n
    rh = rng.uniform(0.82, 1.18, rows)
    rh = rh / rh.sum() * n
    redge = np.concatenate([[0.0], np.cumsum(rh)])
    row = np.clip(np.searchsorted(redge, ys, side="right") - 1, 0, rows - 1)
    by0 = redge[row]
    by1 = redge[row + 1]
    dy_px = np.minimum(ys - by0, by1 - ys)
    ly = (ys - by0) / (by1 - by0)
    bid = np.zeros((n, n), np.int32)
    lx = np.zeros((n, n), np.float32)
    dx_px = np.zeros((n, n), np.float32)
    bw_px = np.zeros((n, n), np.float32)
    for r in range(rows):
        k = int(rng.integers(2, 4))
        widths = rng.uniform(0.7, 1.35, k)
        widths = widths / widths.sum() * n
        edges = np.concatenate([[0.0], np.cumsum(widths)])
        s0 = rng.uniform(0, n)
        sel = row == r
        u = (xs[sel] - s0) % n
        kk = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, k - 1)
        bx0, bx1 = edges[kk], edges[kk + 1]
        bid[sel] = r * 8 + kk
        lx[sel] = (u - bx0) / (bx1 - bx0)
        dx_px[sel] = np.minimum(u - bx0, bx1 - u)
        bw_px[sel] = bx1 - bx0
    d_edge_mm = np.minimum(dx_px, dy_px) * px
    nb = rows * 8
    P = lambda lo, hi: rng.uniform(lo, hi, nb).astype(np.float32)[bid]
    tone = np.exp(rng.normal(0, 0.11, nb)).astype(np.float32)[bid]
    hue = P(-0.05, 0.06)
    off = rng.normal(0, 0.9, nb).astype(np.float32)[bid]
    tilt_x, tilt_y = P(-0.6, 0.6), P(-0.5, 0.5)
    erosion = P(1.0, 4.5)
    pit_k = P(0.0, 1.0) ** 1.6
    strata_amp = P(0.2, 1.4)
    lichen_k = P(0.0, 1.0)
    # ---- height (mm)
    und = fbm_tile(n, 2.7, seed + 21, 3, 22)
    grain = fbm_tile(n, 1.2, seed + 22, 60, n // 2)
    mid = fbm_tile(n, 2.1, seed + 23, 12, 90)
    h = off + tilt_x * (lx - 0.5) * 2 + tilt_y * (ly - 0.5) * 2 + 1.3 * und + 0.30 * grain + 0.45 * mid
    # strata (bedding): thin recessed bands, differential erosion
    warp = fbm_tile(n, 3.0, seed + 24, 1, 12)
    band = np.cos(2 * math.pi * (ys * px / rng.uniform(34, 52) + 0.25 * warp + P(0, 6.28) / 6.28))
    bs = np.sign(band) * np.abs(band) ** 0.55
    h += strata_amp * 0.55 * bs
    # alveolar pits (honeycomb weathering) concentrated in some blocks and patches
    f1p, f2p, cidp = worley_tile(n, 110, seed + 25, 1.0)
    pit_patch = sstep(0.1, 0.8, norm01(fbm_tile(n, 2.4, seed + 26, 1, 14), -1.5, 1.5) * 0.6 + pit_k * 0.7)
    pit_r = (0.35 + 0.35 * np.random.default_rng(seed + 27).random(110 * 110).astype(np.float32))[cidp]
    pit = np.clip(1.0 - f1p / pit_r, 0, 1) ** 0.7 * pit_patch
    h -= 2.6 * pit
    # edge profile: irregular arris rounding + chipped flakes
    wedge = 3.0 + 7.0 * norm01(fbm_tile(n, 2.0, seed + 28, 4, 60), -1.5, 1.5)
    e = np.clip(d_edge_mm / wedge, 0, 1)
    h -= erosion * (1.0 - np.sqrt(1.0 - (1.0 - e) ** 2 + 1e-6)) * 0.55
    f1c, f2c, cidc = worley_tile(n, 26, seed + 29, 1.0)
    chip_r = (0.12 + 0.30 * np.random.default_rng(seed + 30).random(26 * 26).astype(np.float32))[cidc]
    chip = (f1c < chip_r) & (d_edge_mm < 38.0 * (0.4 + norm01(fbm_tile(n, 2.0, seed + 31, 3, 40), -1.5, 1.5)))
    chip_depth = (1.5 + 3.5 * np.random.default_rng(seed + 32).random(26 * 26).astype(np.float32))[cidc]
    chip_f = blur(chip.astype(np.float32), 1.2)
    h -= chip_f * chip_depth
    # mortar joints: recessed, uneven, sometimes lost
    joint_w = 2.0 + 3.0 * norm01(fbm_tile(n, 2.2, seed + 33, 3, 50), -1.5, 1.5)
    in_joint = sstep(joint_w + 0.8, joint_w - 0.4, d_edge_mm)
    lost = sstep(0.55, 1.0, norm01(fbm_tile(n, 2.4, seed + 34, 2, 30), -1.5, 1.5))
    joint_depth = 4.0 + 8.0 * lost
    mort_h = -joint_depth + 0.7 * fbm_tile(n, 1.0, seed + 35, 80, n // 2)
    h = h * (1 - in_joint) + mort_h * in_joint
    # cracks, lichen relief
    crack = crack_layer(n, seed + 40, major=5)
    h -= 1.8 * crack * (1 - in_joint)
    lmask, lcol, lh = lichen_layer(n, seed + 50, density=0.5)
    lmask = lmask * (0.35 + 0.9 * lichen_k) * (1 - in_joint)
    h += lh * lmask
    # ---- albedo
    cA = srgb_to_lin(np.array([0.50, 0.45, 0.38]))
    cB = srgb_to_lin(np.array([0.68, 0.62, 0.52]))
    k = norm01(0.7 * fbm_tile(n, 2.4, seed + 60, 1, 14) + 0.3 * mid, -2, 2)[..., None]
    alb = (cA * (1 - k) + cB * k) * tone[..., None]
    alb[..., 0] *= 1 + hue
    alb[..., 2] *= 1 - hue
    alb *= (1.0 + 0.11 * bs * strata_amp / 1.4)[..., None]
    alb *= (1.0 + 0.10 * grain)[..., None]
    alb *= (1.0 - 0.45 * pit)[..., None]
    alb = alb * (1 - 0.35 * chip_f[..., None]) + srgb_to_lin(np.array([0.80, 0.74, 0.64])) * 0.35 * chip_f[..., None]    # fresh stone in chips
    rust, dark, white = stains(n, seed + 70)
    alb = alb * (1 - 0.55 * rust[..., None]) + srgb_to_lin(np.array([0.55, 0.30, 0.15])) * 0.55 * rust[..., None] * 0.7
    alb *= (1 - 0.40 * dark[..., None])
    alb = alb * (1 - 0.7 * white[..., None]) + srgb_to_lin(np.array([0.78, 0.77, 0.72])) * 0.7 * white[..., None]
    macro_dirt = sstep(0.0, 1.2, fbm_tile(n, 2.8, seed + 80, 1, 10))
    alb *= (1 - 0.22 * macro_dirt)[..., None]
    alb *= (1 - 0.30 * np.exp(-d_edge_mm / 14.0))[..., None]                                                           # soot near joints
    alb = alb * (1 - lmask[..., None]) + lcol * lmask[..., None]
    mort_col = srgb_to_lin(np.array([0.50, 0.47, 0.41])) * (0.8 + 0.4 * fbm_tile(n, 1.0, seed + 85, 80, n // 2))[..., None]
    alb = alb * (1 - in_joint[..., None]) + mort_col * in_joint[..., None] * (1 - 0.35 * lost[..., None])
    rough = 0.80 + 0.07 * mid + 0.13 * pit + 0.10 * in_joint + 0.12 * lmask - 0.10 * chip_f - 0.22 * dark
    return h.astype(np.float32), np.clip(alb, 0, 1).astype(np.float32), np.clip(rough, 0.3, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------ worked / carved stone
def carved_v2(n=2048, seed=20, smooth=0.6):
    """Worked stone (shafts, capitals, mouldings): strata, pits, flaking, cracks, lichen, stains."""
    rng = np.random.default_rng(seed)
    px = TILE_MM / n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    und = fbm_tile(n, 2.7, seed + 1, 3, 24)
    mid = fbm_tile(n, 2.1, seed + 2, 10, 100)
    grain = fbm_tile(n, 1.2, seed + 3, 60, n // 2)
    h = (1.0 - 0.5 * smooth) * 1.6 * und + 0.35 * grain + 0.5 * mid
    warp = fbm_tile(n, 3.0, seed + 4, 1, 12)
    band = np.cos(2 * math.pi * (yy * px / 41.0 + 0.3 * warp))
    bs = np.sign(band) * np.abs(band) ** 0.55
    h += 0.6 * bs
    f1p, _, cidp = worley_tile(n, 100, seed + 5, 1.0)
    pit_patch = sstep(0.0, 0.9, norm01(fbm_tile(n, 2.4, seed + 6, 1, 12), -1.5, 1.5) * 1.1 - 0.15)
    pr = (0.3 + 0.4 * np.random.default_rng(seed + 7).random(100 * 100).astype(np.float32))[cidp]
    pit = np.clip(1 - f1p / pr, 0, 1) ** 0.7 * pit_patch
    h -= 2.2 * pit
    f1c, _, cidc = worley_tile(n, 18, seed + 8, 1.0)
    flake = (f1c < (0.10 + 0.18 * np.random.default_rng(seed + 9).random(18 * 18).astype(np.float32))[cidc]) & (norm01(fbm_tile(n, 2.2, seed + 10, 1, 10), -1.5, 1.5) > 0.62)
    flake_f = blur(flake.astype(np.float32), 1.4)
    h -= flake_f * (1.5 + 2.5 * np.random.default_rng(seed + 11).random(18 * 18).astype(np.float32))[cidc]
    crack = crack_layer(n, seed + 12, major=3, hair_density=0.4)
    h -= 1.6 * crack
    lmask, lcol, lh = lichen_layer(n, seed + 13, density=0.38)
    h += lh * lmask
    cA = srgb_to_lin(np.array([0.54, 0.49, 0.41]))
    cB = srgb_to_lin(np.array([0.72, 0.66, 0.56]))
    k = norm01(0.65 * fbm_tile(n, 2.4, seed + 14, 1, 14) + 0.35 * mid, -2, 2)[..., None]
    alb = cA * (1 - k) + cB * k
    alb *= (1.0 + 0.09 * bs)[..., None] * (1.0 + 0.10 * grain)[..., None] * (1 - 0.45 * pit)[..., None]
    alb = alb * (1 - 0.3 * flake_f[..., None]) + srgb_to_lin(np.array([0.82, 0.76, 0.66])) * 0.3 * flake_f[..., None]
    rust, dark, white = stains(n, seed + 15)
    alb = alb * (1 - 0.5 * rust[..., None]) + srgb_to_lin(np.array([0.55, 0.31, 0.16])) * 0.5 * rust[..., None] * 0.7
    alb *= (1 - 0.35 * dark[..., None])
    alb = alb * (1 - 0.6 * white[..., None]) + srgb_to_lin(np.array([0.80, 0.79, 0.74])) * 0.6 * white[..., None]
    alb *= (1 - 0.2 * sstep(0.0, 1.2, fbm_tile(n, 2.8, seed + 16, 1, 10)))[..., None]
    alb = alb * (1 - lmask[..., None]) + lcol * lmask[..., None]
    alb *= (1 - 0.45 * crack[..., None])
    rough = 0.78 + 0.08 * mid + 0.14 * pit + 0.12 * lmask - 0.08 * flake_f - 0.20 * dark + 0.1 * crack
    return h.astype(np.float32), np.clip(alb, 0, 1).astype(np.float32), np.clip(rough, 0.3, 1.0).astype(np.float32)


def marble_v2(n=2048, seed=30):
    """Aged statue marble: pale, veined, stained by rain and algae, lightly pitted."""
    h, alb, rough = carved_v2(n, seed, smooth=1.0)
    h = h * 0.35
    vein_n = fbm_tile(n, 2.4, seed + 90, 1, 18)
    vein = np.clip(1 - np.abs(np.sin(vein_n * 2.4)) * 3.4, 0, 1) ** 2
    cA = srgb_to_lin(np.array([0.66, 0.65, 0.61]))
    cB = srgb_to_lin(np.array([0.80, 0.79, 0.75]))
    k = norm01(fbm_tile(n, 2.6, seed + 91, 1, 12), -2, 2)[..., None]
    base = cA * (1 - k) + cB * k
    # keep the stains / lichen / pits of the carved set but lift the base towards marble
    lum = (alb * np.array([0.3, 0.55, 0.15])).sum(-1, keepdims=True)
    out = base * (0.55 + 0.9 * lum) * (1 - 0.16 * vein[..., None])
    greenish = sstep(0.4, 1.4, fbm_aniso(n, 2.4, seed + 92, 1.0, 0.09, 1, 180))[..., None]
    out = out * (1 - 0.35 * greenish) + srgb_to_lin(np.array([0.30, 0.38, 0.24])) * 0.35 * greenish * 0.6
    rough = np.clip(0.50 + 0.25 * (rough - 0.75) * 3 + 0.12 * greenish[..., 0], 0.3, 1.0)
    return h.astype(np.float32), np.clip(out, 0, 1).astype(np.float32), rough.astype(np.float32)


def paving_v2(n=2048, seed=40, tiles=4):
    rng = np.random.default_rng(seed)
    px = TILE_MM / n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    w1 = fbm_tile(n, 3.0, seed + 1, 1, 16)
    w2 = fbm_tile(n, 3.0, seed + 2, 1, 16)
    xs = (xx + w1 * (6.0 / px)) % n
    ys = (yy + w2 * (6.0 / px)) % n
    cell = n / tiles
    row = (ys // cell).astype(int)
    off_r = rng.uniform(0, cell, tiles)
    xo = (xs + off_r[row]) % n
    dx = np.minimum(xo % cell, cell - xo % cell)
    dy = np.minimum(ys % cell, cell - ys % cell)
    d_mm = np.minimum(dx, dy) * px
    bid = row * tiles + (xo // cell).astype(int)
    nb = tiles * tiles
    tone = np.exp(rng.normal(0, 0.12, nb)).astype(np.float32)[bid]
    off = rng.normal(0, 1.4, nb).astype(np.float32)[bid]
    und = fbm_tile(n, 2.6, seed + 3, 3, 24)
    grain = fbm_tile(n, 1.2, seed + 4, 60, n // 2)
    mid = fbm_tile(n, 2.1, seed + 5, 10, 100)
    h = off + 1.6 * und + 0.35 * grain + 0.5 * mid
    e = np.clip(d_mm / 10.0, 0, 1)
    h -= 3.0 * (1 - np.sqrt(1 - (1 - e) ** 2 + 1e-6))
    in_j = sstep(5.5, 2.5, d_mm)
    h = h * (1 - in_j) + (-9.0 + 0.8 * fbm_tile(n, 1.0, seed + 6, 80, n // 2)) * in_j
    crack = crack_layer(n, seed + 7, major=4, hair_density=0.5)
    h -= 2.0 * crack * (1 - in_j)
    f1p, _, cidp = worley_tile(n, 90, seed + 8, 1.0)
    pit = np.clip(1 - f1p / 0.5, 0, 1) ** 0.8 * sstep(0.1, 0.9, norm01(fbm_tile(n, 2.4, seed + 9, 1, 12), -1.5, 1.5))
    h -= 2.0 * pit
    lmask, lcol, lh = lichen_layer(n, seed + 10, density=0.25)
    cA = srgb_to_lin(np.array([0.40, 0.38, 0.34]))
    cB = srgb_to_lin(np.array([0.58, 0.55, 0.49]))
    k = norm01(0.6 * fbm_tile(n, 2.4, seed + 11, 1, 14) + 0.4 * mid, -2, 2)[..., None]
    alb = (cA * (1 - k) + cB * k) * tone[..., None] * (1 - 0.4 * pit[..., None])
    rust, dark, white = stains(n, seed + 12)
    alb = alb * (1 - 0.4 * rust[..., None]) + srgb_to_lin(np.array([0.50, 0.30, 0.16])) * 0.4 * rust[..., None] * 0.7
    alb *= (1 - 0.3 * dark[..., None])
    alb = alb * (1 - lmask[..., None] * 0.7) + lcol * lmask[..., None] * 0.7
    alb = alb * (1 - in_j[..., None]) + srgb_to_lin(np.array([0.25, 0.24, 0.21])) * in_j[..., None]
    rough = 0.66 + 0.10 * mid + 0.16 * pit + 0.12 * in_j - 0.15 * dark
    return h.astype(np.float32), np.clip(alb, 0, 1).astype(np.float32), np.clip(rough, 0.3, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------ moss
def moss_v2(n=2048, seed=50):
    """Cushion moss (tile = 1 m, 0.49 mm/px): domed clumps radiating fibres, colour variety, deep valleys."""
    rng = np.random.default_rng(seed)
    cells = 70
    f1, f2, cid = worley_tile(n, cells, seed, 1.0)
    nc = cells * cells
    cs = n / cells
    r_c = (0.55 + 0.45 * rng.random(nc)).astype(np.float32)[cid]
    dome = np.clip(1.0 - f1 / r_c, 0, 1) ** 0.65
    hgt = (0.6 + 0.8 * rng.random(nc)).astype(np.float32)[cid]
    # radial fibres around each clump centre
    f1b, _, cid2 = worley_tile(n, cells, seed, 1.0)
    jit = fbm_tile(n, 1.2, seed + 1, 60, n // 2)
    ang = np.cos(np.arctan2(np.sin(f1 * 40 + jit * 1.5), np.cos(f1 * 40 + jit * 1.5)) * 0 + jit * 6.0 + f1 * 55)
    fibre = 0.5 + 0.5 * np.cos(jit * 14.0 + f1 * 90.0)
    fine = fbm_tile(n, 0.9, seed + 2, 90, n // 2)
    mid = fbm_tile(n, 2.0, seed + 3, 4, 50)
    h = (3.5 * dome * hgt + 0.9 * fibre * dome + 0.5 * fine + 0.8 * mid)       # mm-ish
    deep = (1.0 - dome) ** 1.4
    cD = srgb_to_lin(np.array([0.030, 0.085, 0.020]))
    cM = srgb_to_lin(np.array([0.110, 0.240, 0.040]))
    cT = srgb_to_lin(np.array([0.330, 0.470, 0.090]))
    cY = srgb_to_lin(np.array([0.470, 0.470, 0.120]))
    t = np.clip(dome * (0.55 + 0.6 * fibre) + 0.12 * fine, 0, 1)[..., None]
    var = rng.random(nc).astype(np.float32)[cid]
    base = cD * (1 - t) + cM * np.minimum(t * 2, 1) * (1 - np.clip(t * 2 - 1, 0, 1)) + cT * np.clip(t * 2 - 1, 0, 1)
    yel = (np.clip((var - 0.72) * 4, 0, 1) * 0.55)[..., None]
    alb = base * (1 - yel) + cY * yel * (0.5 + 0.6 * t)
    alb *= (0.75 + 0.5 * norm01(mid, -2, 2))[..., None] * (1 - 0.35 * deep)[..., None]
    sparkle = (norm01(fine, 1.5, 3.0) ** 2)[..., None] * 0.25
    alb = alb * (1 + sparkle)
    rough = np.clip(0.93 - 0.06 * dome + 0.03 * fine, 0.6, 1.0)
    return h.astype(np.float32), np.clip(alb, 0, 1).astype(np.float32), rough.astype(np.float32)


def moss_maps(n=2048, seed=50):
    h, alb, rough = moss_v2(n, seed)
    px_mm = 1000.0 / n
    ao = np.clip(1.0 - 0.6 * (blur(h, 8.0) - h) / 2.5, 0.25, 1.0)
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * px_mm)
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * px_mm)
    nx, ny, nz = -dx * 0.9, dy * 0.9, np.ones_like(dx)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = (np.stack([nx / l, ny / l, nz / l], -1) * 0.5 + 0.5).astype(np.float32)
    hn = np.clip(h / 6.0, 0, 1)
    orh = np.stack([rough, ao, hn], -1).astype(np.float32)
    return lin_to_srgb(np.clip(alb * ao[..., None] ** 0.5, 0, 1)), nrm, orh


# ------------------------------------------------------------------------------------ macro / detail
def macro_map(n=1024, seed=100):
    """R colour variation, G roughness variation, B moss bias.  Tile = 16 m."""
    r = norm01(0.7 * fbm_tile(n, 2.6, seed + 1, 1, 10) + 0.3 * fbm_tile(n, 2.0, seed + 2, 8, 60), -2.2, 2.2)
    g = norm01(fbm_tile(n, 2.4, seed + 3, 1, 14), -2.2, 2.2)
    b = norm01(0.75 * fbm_tile(n, 2.8, seed + 4, 1, 8) + 0.25 * fbm_tile(n, 2.0, seed + 5, 6, 40), -2.2, 2.2)
    return np.stack([r, g, b], -1).astype(np.float32)


def detail_normal(n=1024, seed=110):
    """Micro grain normal map (tile = 0.25 m)."""
    h = fbm_tile(n, 1.1, seed, 30, n // 2) * 0.5 + fbm_tile(n, 1.7, seed + 1, 8, 100) * 0.6
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    s = 0.55
    nx, ny, nz = -gx * s, gy * s, np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    return (np.stack([nx / l, ny / l, nz / l], -1) * 0.5 + 0.5).astype(np.float32)


def build_all(outdir, n=2048, only=None):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    def stone(name, fn, **kw):
        if only and name not in only:
            return
        h, a, r = fn(n, **kw)
        bc, nrm, orh = finish_maps(h, a, r, normal_strength=1.0)
        write_set(outdir, name, bc, nrm, orh)
        print(f"  [{time.time() - t0:5.1f}s] {name}", flush=True)

    stone("StoneAshlar", ashlar_v2)
    stone("StoneCarved", carved_v2)
    stone("Marble", marble_v2)
    stone("Paving", paving_v2)
    if not only or "Moss" in only:
        bc, nrm, orh = moss_maps(n)
        write_set(outdir, "Moss", bc, nrm, orh)
        print(f"  [{time.time() - t0:5.1f}s] Moss", flush=True)
    if not only or "Macro" in only:
        save_png(macro_map(1024 if n >= 1024 else 512), f"{outdir}/T_Macro.png")
        save_png(detail_normal(1024 if n >= 1024 else 512), f"{outdir}/T_DetailN.png")
        print(f"  [{time.time() - t0:5.1f}s] Macro + Detail", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    n = 1024 if "--quick" in sys.argv else 2048
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    build_all(out, n, only)
