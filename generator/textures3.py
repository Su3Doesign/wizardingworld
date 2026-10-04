"""Photoreal weathered-masonry textures, v3 (numpy).  Supersedes textures2.py for stone / moss.

What changed against v2 (every change answers a specific "this looks CG" tell):
  * lichen is sparse, low-saturation, ragged-edged and clustered by a damp-patch mask (no confetti)
  * rain streaks grow DOWN from the bed joints (drip simulation) instead of random white splashes
  * mortar is pale sandy lime mortar, deeper recessed, sometimes lost - joints are no longer black lines
  * Romanesque diagonal chisel tooling on a share of the blocks, eroded away in patches
  * alveolar pits are patch-limited, multi-scale and variable in depth (no leopard print)
  * moss is a carpet of thousands of tiny star-shaped shoot tips over cushion relief, not pea clumps

Packed maps *_ORH.png: R = roughness, G = cavity AO, B = height (0..1 over HMIN..HMAX mm).
Normal maps are OpenGL (green = +V).  Stone tile = 2 m, moss tile = 1 m.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import textures as tx
from textures import fbm_tile, norm01, srgb_to_lin, lin_to_srgb, save_png, save_jpg
from textures2 import sstep, blur, fbm_aniso, crack_layer, finish_maps, write_set, macro_map, detail_normal, HMIN, HMAX, TILE_MM

C = lambda *rgb: srgb_to_lin(np.array(rgb, np.float32))      # sRGB triplet -> linear


# ------------------------------------------------------------------------------------ primitives
def worley_vec(n, cells, seed=0, jitter=1.0, warp=None):
    """Tileable cellular noise returning (F1, vu, vv, id): distance and vector (cell units) to the nearest feature.
    warp = (du, dv) arrays (tile fractions) bends the sampling lattice so cells are curvy instead of polygonal."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
    px = (gx + 0.5 + (rng.random((cells, cells)) - 0.5) * jitter) / cells
    py = (gy + 0.5 + (rng.random((cells, cells)) - 0.5) * jitter) / cells
    pts = np.stack([px.ravel(), py.ravel()], 1)
    allp = np.concatenate([pts + np.array([dx, dy]) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    allid = np.tile(np.arange(cells * cells), 9)
    tree = cKDTree(allp)
    u = (np.arange(n) + 0.5) / n
    UU, VV = np.meshgrid(u, u, indexing="ij")
    if warp is not None:
        UU = (UU + warp[0]) % 1.0
        VV = (VV + warp[1]) % 1.0
    q = np.stack([UU.ravel(), VV.ravel()], 1)
    d, i = tree.query(q, k=1)
    v = (q - allp[i]) * cells
    sh = (n, n)
    return ((d * cells).reshape(sh).astype(np.float32), v[:, 0].reshape(sh).astype(np.float32),
            v[:, 1].reshape(sh).astype(np.float32), allid[i].reshape(sh))


def cell_mean(cid, vals, nc):
    s = np.bincount(cid.ravel(), weights=vals.ravel().astype(np.float64), minlength=nc)
    c = np.bincount(cid.ravel(), minlength=nc)
    return (s / np.maximum(c, 1)).astype(np.float32)


def grain_noise(n, seed, sigma=0.65):
    g = np.random.default_rng(seed).standard_normal((n, n)).astype(np.float32)
    g = blur(g, sigma)
    return g / (g.std() + 1e-9)


def specks(n, seed, density, sigma=0.9):
    """Sparse tiny inclusions (iron specks / calcite crystals): 0..1."""
    r = np.random.default_rng(seed).random((n, n)) < density
    s = blur(r.astype(np.float32), sigma)
    return np.clip(s / (s.max() + 1e-9) * 1.6, 0, 1).astype(np.float32)


# ------------------------------------------------------------------------------------ lichen
def lichen_v3(n, seed, cover=0.22, cells=70):
    """Crustose lichen colonies - sparse, clustered in damp patches, ragged, desaturated.
    Returns (mask 0..1, colour linear (n,n,3), height mm)."""
    rng = np.random.default_rng(seed)
    f1, vu, vv, cid = worley_vec(n, cells, seed, 1.0)
    nc = cells * cells
    cpx = n / cells
    patch = norm01(fbm_tile(n, 2.7, seed + 1, 1, 9), -1.4, 1.4)
    pcell = cell_mean(cid, patch, nc)
    act = rng.random(nc) < cover * (0.02 + 3.0 * pcell ** 2.2)
    rad = rng.uniform(0.18, 0.58, nc).astype(np.float32)
    # ragged margins: multi-scale noise on the radius, lobed by angle
    ang = np.arctan2(vv, vu)
    lob = 0.5 + 0.5 * np.cos(ang * rng.integers(4, 9, nc)[cid] + rng.uniform(0, 6.28, nc)[cid])
    e1 = fbm_tile(n, 1.5, seed + 2, 12, n // 3)
    e2 = fbm_tile(n, 1.0, seed + 3, 60, n // 2)
    rr = rad[cid] * (1.0 + 0.34 * e1 + 0.12 * e2 + 0.18 * (lob - 0.5))
    t = f1 / np.maximum(rr, 1e-3)
    inside = act[cid] & (t < 1.0)
    crust = 0.62 + 0.38 * norm01(fbm_tile(n, 0.9, seed + 4, 80, n // 2), -2, 2)
    mask = np.where(inside, sstep(1.0, 0.72, t) * crust, 0.0).astype(np.float32)
    mask = blur(mask, 0.55) * 0.64
    # kinds: pale grey-green (common), whitish, black-grey biofilm, ochre-yellow, orange (very rare, tiny)
    pal = np.array([C(0.58, 0.62, 0.50), C(0.70, 0.70, 0.64), C(0.17, 0.18, 0.16), C(0.60, 0.54, 0.26), C(0.66, 0.40, 0.16)], np.float32)
    kind = rng.choice(5, nc, p=[0.40, 0.24, 0.24, 0.10, 0.02])
    var = (0.82 + 0.36 * rng.random(nc)).astype(np.float32)[cid][..., None]
    col = pal[kind[cid]] * var
    # tiny bright centre / darker rim gives colonies thickness
    col = col * (0.88 + 0.22 * sstep(0.9, 0.2, t)[..., None])
    h = 0.55 * mask * (0.6 + 0.4 * crust)
    return mask, col.astype(np.float32), h.astype(np.float32)


# ------------------------------------------------------------------------------------ drip streaks
def drip_streaks(n, seed, row_edges_px, n_streaks=14, length=(80, 520), width=(3, 22), amp=(0.30, 1.0)):
    """Rain / seepage staining that starts at the bed joints and runs downward (+y in the image): individual
    streaks that widen, meander and fade with length."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    for y0 in row_edges_px:
        for _ in range(n_streaks):
            x0, w, Lk, a = rng.uniform(0, n), rng.uniform(*width), int(rng.uniform(*length)), rng.uniform(*amp)
            ys = np.arange(Lk)
            t = ys / float(Lk)
            xs = np.arange(int(-3.2 * w * 1.8), int(3.2 * w * 1.8) + 1)
            meander = 0.45 * w * np.sin(ys / rng.uniform(18, 70) + rng.uniform(0, 6.28))
            sig = w * 0.45 * (1.0 + 0.8 * t)
            X = xs[None, :] - meander[:, None]
            prof = np.exp(-0.5 * (X / sig[:, None]) ** 2) * ((1.0 - t) ** 1.5)[:, None] * a
            yi = (int(y0) + ys) % n
            xi = (int(x0) + xs) % n
            sub = np.ix_(yi, xi)
            out[sub] = np.maximum(out[sub], prof.astype(np.float32))
    breakup = 0.55 + 0.45 * norm01(fbm_aniso(n, 1.8, seed + 5, 1.0, 0.10, 4, n // 3), -2, 2)
    return np.clip(out * breakup, 0, 1).astype(np.float32)


def cracks_v3(n, seed, d_edge, count=16, length=(40, 220)):
    """Short jagged cracks that start at block arrises and run inward (spalling cracks), thin, occasionally forking."""
    rng = np.random.default_rng(seed)
    gy, gx = np.gradient(blur(d_edge, 2.0))
    cand = np.argwhere((d_edge > 3) & (d_edge < 30))
    out = np.zeros((n, n), np.float32)
    if len(cand) == 0:
        return out
    stack = []
    for _ in range(count):
        y, x = cand[rng.integers(len(cand))]
        gv = math.hypot(gx[y, x], gy[y, x])
        ang = (math.atan2(gy[y, x], gx[y, x]) if gv > 1e-6 else rng.uniform(0, 6.28)) + rng.normal(0, 0.5)
        stack.append((float(x), float(y), ang, int(rng.uniform(*length)), rng.uniform(0.7, 1.5)))
    while stack:
        px_, py_, ang, L, w0 = stack.pop()
        for s_ in range(L):
            ang += rng.normal(0, 0.11)
            px_ += math.cos(ang)
            py_ += math.sin(ang)
            xi, yi = int(px_) % n, int(py_) % n
            r = 1 if (w0 * (1 - s_ / L) > 1.25 and n >= 2048) else 0          # mostly 1-px hairlines
            ys_ = (np.arange(yi - r, yi + r + 1) % n)[:, None]
            xs_ = (np.arange(xi - r, xi + r + 1) % n)[None, :]
            out[ys_, xs_] = np.maximum(out[ys_, xs_], 1.0 - 0.5 * s_ / L)
            if L - s_ > 30 and rng.random() < 0.012 and len(stack) < 60:
                stack.append((px_, py_, ang + rng.choice([-1, 1]) * rng.uniform(0.35, 0.8), int((L - s_) * rng.uniform(0.3, 0.6)), w0 * 0.6))
    return np.clip(blur(out, 0.50) * 1.5, 0, 1).astype(np.float32)


def algae_film(n, seed):
    """Dark olive-black biofilm in damp, vertical, patchy bands."""
    a = 0.8 * fbm_aniso(n, 2.3, seed + 1, 1.0, 0.22, 1, 70) + 0.55 * fbm_tile(n, 2.4, seed + 2, 2, 40)
    return sstep(0.25, 1.35, a).astype(np.float32)


# ------------------------------------------------------------------------------------ masonry layout
def block_layout(n, seed, rows=6, per_row=(3, 4), wobble_mm=(5.0, 4.0), rh_var=0.18):
    """Coursed ashlar layout. Returns dict: bid, lx, ly, d_edge_mm, row_edges_px, nb, warped coords."""
    rng = np.random.default_rng(seed)
    px = TILE_MM / n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    w1 = fbm_tile(n, 3.0, seed + 11, 1, 16)
    w2 = fbm_tile(n, 3.0, seed + 12, 1, 16)
    xs = (xx + w1 * (wobble_mm[0] / px)) % n
    ys = (yy + w2 * (wobble_mm[1] / px)) % n
    rh = rng.uniform(1 - rh_var, 1 + rh_var, rows)
    rh = rh / rh.sum() * n
    redge = np.concatenate([[0.0], np.cumsum(rh)])
    row = np.clip(np.searchsorted(redge, ys, side="right") - 1, 0, rows - 1)
    by0, by1 = redge[row], redge[row + 1]
    dy_px = np.minimum(ys - by0, by1 - ys)
    ly = (ys - by0) / (by1 - by0)
    bid = np.zeros((n, n), np.int32)
    lx = np.zeros((n, n), np.float32)
    dx_px = np.zeros((n, n), np.float32)
    for r in range(rows):
        k = int(rng.integers(per_row[0], per_row[1] + 1))
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
    return dict(bid=bid, lx=lx, ly=ly, d_edge_mm=np.minimum(dx_px, dy_px) * px, nb=rows * 8, row_edges_px=redge[:-1], ys=ys, xs=xs, px=px, rows=rows)


# ------------------------------------------------------------------------------------ stone core
def pits_layer(n, seed, patch_bias):
    """Alveolar (honeycomb) weathering: three scales, confined to patches. Returns depth mm (positive) and 0..1 mask."""
    patch = norm01(fbm_tile(n, 2.6, seed + 1, 1, 12), -1.5, 1.5)
    gate = sstep(0.52, 0.80, patch * 0.8 + patch_bias * 0.35)
    out = np.zeros((n, n), np.float32)
    for cells, depth, sd in ((38, 3.4, 2), (96, 2.2, 3), (230, 1.1, 4)):
        f1, f2, cid = tx.worley_tile(n, cells, seed + sd, 1.0)
        rr = (0.28 + 0.46 * np.random.default_rng(seed + 10 * sd).random(cells * cells).astype(np.float32))[cid]
        on = (np.random.default_rng(seed + 20 * sd).random(cells * cells) < (0.78 if cells < 200 else 0.32))[cid]
        pit = np.clip(1.0 - f1 / rr, 0, 1) ** 0.6 * on
        out = np.maximum(out, pit * depth * gate * (0.45 + 0.55 * np.random.default_rng(seed + 30 * sd).random(cells * cells).astype(np.float32)[cid]))
    return out, np.clip(out / 2.5, 0, 1)


def tooling(n, seed, bid, nb, px, lx, ly):
    """Diagonal claw-chisel marks (Romanesque), per-block angle/period, eroded away in patches. Returns mm relief."""
    rng = np.random.default_rng(seed)
    en = (rng.random(nb) < 0.55).astype(np.float32)[bid]
    ang = (np.deg2rad(rng.uniform(28, 58, nb)) * np.where(rng.random(nb) < 0.5, 1, -1)).astype(np.float32)[bid]
    per = rng.uniform(5.0, 9.5, nb).astype(np.float32)[bid]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    warp = fbm_tile(n, 2.8, seed + 1, 2, 30)
    u = (xx * np.cos(ang) + yy * np.sin(ang)) * px / per + 0.9 * warp
    prof = np.abs(np.sin(math.pi * u)) ** 0.6 - 0.5
    fade = sstep(0.1, 0.75, norm01(fbm_tile(n, 2.4, seed + 2, 1, 12), -1.5, 1.5))
    return (prof * en * fade * 0.55).astype(np.float32)


def stone_surface(n, seed, L, strength=1.0, tooled=True, pit_k=1.0, lichen_cover=0.13, mortar=True, base=((0.33, 0.30, 0.26), (0.57, 0.52, 0.44))):
    """Build (height_mm, albedo_linear, roughness) of a coursed masonry face from a block layout L."""
    rng = np.random.default_rng(seed + 999)
    bid, nb, px = L["bid"], L["nb"], L["px"]
    d_edge = L["d_edge_mm"]
    P = lambda lo, hi: rng.uniform(lo, hi, nb).astype(np.float32)[bid]
    tone = np.exp(rng.normal(0, 0.10, nb)).astype(np.float32)[bid]
    hue = P(-0.05, 0.05)
    off = rng.normal(0, 0.8, nb).astype(np.float32)[bid]
    tilt_x, tilt_y = P(-0.6, 0.6), P(-0.5, 0.5)
    erosion = P(1.0, 4.5)
    pit_block = P(0.0, 1.0) ** 1.4
    strata_amp = P(0.2, 1.3)
    lichen_k = P(0.0, 1.0)
    lx, ly = L["lx"], L["ly"]
    # ---- relief (mm)
    und = fbm_tile(n, 2.7, seed + 21, 3, 22)
    grain = fbm_tile(n, 1.2, seed + 22, 60, n // 2)
    mid = fbm_tile(n, 2.1, seed + 23, 12, 90)
    h = off + tilt_x * (lx - 0.5) * 2 + tilt_y * (ly - 0.5) * 2 + 1.2 * und + 0.28 * grain + 0.40 * mid
    warp = fbm_tile(n, 3.0, seed + 24, 1, 12)
    ys = L["ys"]
    warp2 = fbm_tile(n, 2.4, seed + 27, 1, 8)
    band = np.cos(2 * math.pi * (ys * px / rng.uniform(30, 70) + 0.85 * warp + 0.6 * warp2 + P(0, 6.28) / 6.28))
    bs = np.sign(band) * np.abs(band) ** 0.55 * (0.45 + 0.55 * norm01(warp2, -1.6, 1.6))
    h = h + strata_amp * 0.40 * bs
    if tooled:
        h = h + tooling(n, seed + 25, bid, nb, px, lx, ly)
    pit, pit_m = pits_layer(n, seed + 26, pit_block * pit_k)
    h = h - pit
    # arris rounding (irregular) + chipped flakes along the edges
    wedge = 3.0 + 7.0 * norm01(fbm_tile(n, 2.0, seed + 28, 4, 60), -1.5, 1.5)
    e = np.clip(d_edge / wedge, 0, 1)
    h = h - erosion * (1.0 - np.sqrt(1.0 - (1.0 - e) ** 2 + 1e-6)) * 0.55
    f1c, f2c, cidc = tx.worley_tile(n, 28, seed + 29, 1.0)
    rr = np.random.default_rng(seed + 30)
    chip_r = (0.12 + 0.30 * rr.random(28 * 28).astype(np.float32))[cidc]
    reach = 40.0 * (0.4 + norm01(fbm_tile(n, 2.0, seed + 31, 3, 40), -1.5, 1.5))
    chip = (f1c < chip_r) & (d_edge < reach)
    chip_depth = (1.5 + 3.5 * rr.random(28 * 28).astype(np.float32))[cidc]
    chip_f = blur(chip.astype(np.float32), 1.2)
    h = h - chip_f * chip_depth
    # mortar: recessed, uneven, sometimes lost
    if mortar:
        jw = 2.2 + 3.2 * norm01(fbm_tile(n, 2.2, seed + 33, 3, 50), -1.5, 1.5)
        in_joint = sstep(jw + 0.9, jw - 0.5, d_edge)
        lost = sstep(0.50, 0.95, norm01(fbm_tile(n, 2.4, seed + 34, 2, 30), -1.5, 1.5))
        mort_h = -(5.0 + 9.0 * lost) + 0.9 * fbm_tile(n, 1.0, seed + 35, 80, n // 2) + 0.5 * grain
        h = h * (1 - in_joint) + mort_h * in_joint
    else:
        in_joint, lost = np.zeros_like(h), np.zeros_like(h)
    crack = np.maximum(cracks_v3(n, seed + 40, d_edge), 0.35 * crack_layer(n, seed + 41, major=0, hair_density=0.3))
    h = h - 1.6 * crack * (1 - in_joint)
    lm, lcol, lh = lichen_v3(n, seed + 50, cover=lichen_cover)
    lm = lm * (0.30 + 1.0 * lichen_k) * (1 - in_joint)
    h = h + lh * lm
    # ---- albedo
    cA, cB = C(*base[0]), C(*base[1])
    k = norm01(0.7 * fbm_tile(n, 2.4, seed + 60, 1, 14) + 0.3 * mid, -2, 2)[..., None]
    alb = (cA * (1 - k) + cB * k) * tone[..., None]
    alb[..., 0] *= 1 + hue
    alb[..., 2] *= 1 - hue
    alb *= (1.0 + 0.045 * bs * strata_amp / 1.3)[..., None]
    g1 = grain_noise(n, seed + 61, 0.6)
    g2 = grain_noise(n, seed + 62, 0.6)
    alb *= (1.0 + 0.075 * g1)[..., None]                       # luminance speckle
    alb[..., 0] *= 1.0 + 0.030 * g2                            # chroma speckle (warm / cool grains)
    alb[..., 2] *= 1.0 - 0.030 * g2
    alb *= (1.0 - 0.40 * specks(n, seed + 63, 0.0006))[..., None]                       # iron / dark inclusions
    alb = alb * (1 - 0.35 * specks(n, seed + 64, 0.0005)[..., None]) + C(0.82, 0.80, 0.74) * 0.35 * specks(n, seed + 64, 0.0005)[..., None]
    alb *= (1.0 - 0.38 * (pit_m ** 0.8))[..., None]
    alb = alb * (1 - 0.40 * chip_f[..., None]) + C(0.76, 0.71, 0.62) * 0.40 * chip_f[..., None]     # fresh stone in the chips
    # drip staining from the bed joints, soot at joints, large-scale dirt
    drip = drip_streaks(n, seed + 70, L["row_edges_px"], n_streaks=int(10 * strength + 5))
    drip2 = drip_streaks(n, seed + 71, L["row_edges_px"], n_streaks=4, length=(160, 700), width=(2, 9))
    alb *= (1.0 - 0.40 * drip)[..., None]
    rusty = (drip2 * sstep(0.1, 0.9, norm01(fbm_tile(n, 2.4, seed + 72, 1, 14), -1.5, 1.5)))[..., None]
    alb = alb * (1 - 0.30 * rusty) + C(0.50, 0.32, 0.18) * 0.30 * rusty * 0.8
    alb *= (1 - 0.18 * sstep(0.0, 1.3, fbm_tile(n, 2.8, seed + 73, 1, 10)))[..., None]
    mott = fbm_tile(n, 2.5, seed + 75, 3, 26)                                  # cloudy light/dark mottling, 8-60 cm
    alb *= (1.0 + 0.13 * mott)[..., None]
    alb[..., 0] *= 1.0 + 0.025 * fbm_tile(n, 2.6, seed + 76, 2, 20)           # slow warm/cool drift
    alb *= (1 - 0.28 * np.exp(-d_edge / 12.0))[..., None]
    alg = algae_film(n, seed + 74)[..., None]
    alb = alb * (1 - 0.34 * alg) + C(0.17, 0.20, 0.12) * 0.34 * alg
    alb = alb * (1 - lm[..., None]) + lcol * lm[..., None]
    alb *= (1 - 0.30 * crack[..., None])
    # mortar: pale sandy lime mortar; where it has fallen out we see dark void
    mort = C(0.58, 0.55, 0.48) * (0.80 + 0.40 * fbm_tile(n, 1.0, seed + 85, 80, n // 2))[..., None] * (1 + 0.10 * g1)[..., None]
    mort = mort * (1 - 0.30 * alg)
    alb = alb * (1 - in_joint[..., None]) + mort * in_joint[..., None] * (1 - 0.55 * lost[..., None] ** 1.3)
    rough = 0.80 + 0.07 * mid + 0.10 * (pit_m) + 0.12 * in_joint + 0.10 * lm - 0.10 * chip_f - 0.20 * drip - 0.12 * alg[..., 0]
    return h.astype(np.float32), np.clip(alb, 0, 1).astype(np.float32), np.clip(rough, 0.3, 1.0).astype(np.float32)


def ashlar_v3(n=2048, seed=10, rows=6):
    # joints nearly straight (1 mm wobble): masonry.py cuts the same layout as real grooves into the geometry,
    # so the painted mortar must sit inside those grooves (irregularity comes from arris chips and lost mortar)
    L = block_layout(n, seed, rows=rows, per_row=(3, 4), wobble_mm=(1.0, 1.0))
    return stone_surface(n, seed, L, tooled=True, pit_k=1.0, lichen_cover=0.13)


def carved_v3(n=2048, seed=20):
    """Columns / capitals / mouldings: finer-grained, smoother, fewer joints (a single huge 'block' with weak bedding)."""
    L = block_layout(n, seed, rows=2, per_row=(1, 1), wobble_mm=(2.0, 2.0), rh_var=0.05)
    L["d_edge_mm"] = np.full((n, n), 400.0, np.float32)          # no joints / arrises inside the tile
    L["row_edges_px"] = np.array([0.0, n * 0.5])
    return stone_surface(n, seed, L, strength=0.8, tooled=False, pit_k=0.85, lichen_cover=0.15, mortar=False,
                         base=((0.34, 0.31, 0.26), (0.59, 0.54, 0.46)))


def paving_v3(n=2048, seed=40, tiles=4):
    """Large flagstones, dirtier and darker, heavy joint recess (moss grows there via the height blend)."""
    rng = np.random.default_rng(seed)
    px = TILE_MM / n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    w1 = fbm_tile(n, 3.0, seed + 1, 1, 16)
    w2 = fbm_tile(n, 3.0, seed + 2, 1, 16)
    xs = (xx + w1 * (7.0 / px)) % n
    ys = (yy + w2 * (7.0 / px)) % n
    cell = n / tiles
    row = (ys // cell).astype(int)
    off_r = rng.uniform(0, cell, tiles)
    xo = (xs + off_r[row]) % n
    dx = np.minimum(xo % cell, cell - xo % cell)
    dy = np.minimum(ys % cell, cell - ys % cell)
    d_mm = np.minimum(dx, dy) * px
    bid = (row * tiles + (xo // cell).astype(int)).astype(np.int32)
    L = dict(bid=bid, nb=tiles * tiles, px=px, d_edge_mm=d_mm, lx=((xo % cell) / cell).astype(np.float32), ly=((ys % cell) / cell).astype(np.float32),
             ys=ys, xs=xs, row_edges_px=np.arange(tiles) * cell)
    h, a, r = stone_surface(n, seed, L, strength=0.7, tooled=False, pit_k=0.5, lichen_cover=0.10, base=((0.26, 0.245, 0.21), (0.44, 0.41, 0.35)))
    return h, a, r


def marble_v3(n=2048, seed=30):
    """Aged white marble for the statue: pale, faintly veined, rain-streaked and algae-stained, almost no pits."""
    L = block_layout(n, seed, rows=2, per_row=(1, 1), wobble_mm=(1.0, 1.0), rh_var=0.05)
    L["d_edge_mm"] = np.full((n, n), 400.0, np.float32)
    L["row_edges_px"] = np.array([0.0, n * 0.34, n * 0.67])
    h, alb, rough = stone_surface(n, seed, L, strength=1.2, tooled=False, pit_k=0.12, lichen_cover=0.08, mortar=False,
                                  base=((0.60, 0.59, 0.55), (0.84, 0.83, 0.79)))
    h = h * 0.30
    vn = fbm_tile(n, 2.4, seed + 90, 1, 18)
    vein = np.clip(1 - np.abs(np.sin(vn * 2.4)) * 3.6, 0, 1) ** 2
    vein2 = np.clip(1 - np.abs(np.sin(fbm_tile(n, 2.2, seed + 93, 2, 40) * 3.0)) * 5.0, 0, 1) ** 2
    alb = alb * (1 - 0.15 * vein[..., None] - 0.07 * vein2[..., None]) + C(0.55, 0.57, 0.58) * (0.15 * vein[..., None] + 0.07 * vein2[..., None])
    green = sstep(0.4, 1.5, fbm_aniso(n, 2.4, seed + 92, 1.0, 0.09, 1, 180))[..., None]
    alb = alb * (1 - 0.30 * green) + C(0.26, 0.34, 0.21) * 0.30 * green
    rough = np.clip(0.52 + 0.30 * (rough - 0.78) * 2.5 + 0.12 * green[..., 0], 0.28, 0.95)
    return h.astype(np.float32), np.clip(alb, 0, 1).astype(np.float32), rough.astype(np.float32)


# ------------------------------------------------------------------------------------ moss
def moss_v3(n=2048, seed=50):
    """Cushion moss. Tile = 1 m (0.49 mm / px).  Three scales:
       cushions (~25-60 mm, warped + blended with fbm lumps so there are no polygon seams) give relief and
       clump-to-clump colour; thousands of star-shaped shoot tips (~3 mm) give the fuzzy velvet; 1 px hair noise
       gives sparkle.  Colour follows 'height in canopy' (dark in the gaps, lime-yellow at sun-struck tips)
       plus patchy hue shifts and a few dry brown tips."""
    rng = np.random.default_rng(seed)
    du = fbm_tile(n, 2.4, seed + 7, 1, 10) * 0.010
    dv = fbm_tile(n, 2.4, seed + 8, 1, 10) * 0.010
    # --- cushions: two warped Worley scales + irregular fbm lumps
    f1, vu, vv, cid = worley_vec(n, 30, seed, 1.0, warp=(du, dv))
    nc = 30 * 30
    r_c = (0.75 + 0.45 * rng.random(nc)).astype(np.float32)[cid]
    hgt = (0.50 + 0.50 * rng.random(nc)).astype(np.float32)[cid]
    dome = np.clip(1.0 - f1 / r_c, 0, 1) ** 0.6
    g1, wu, wv, gid = worley_vec(n, 74, seed + 9, 1.0, warp=(dv * 1.3, du * 1.3))
    dome2 = np.clip(1.0 - g1 / (0.9 + 0.3 * rng.random(74 * 74).astype(np.float32)[gid]), 0, 1) ** 0.7
    lump = norm01(fbm_tile(n, 2.0, seed + 1, 3, 60), -1.7, 1.7)
    cush = np.clip(0.42 * dome * hgt + 0.20 * dome2 + 0.55 * lump ** 1.4, 0, 1).astype(np.float32)
    cush = norm01(cush, 0.05, 0.95)
    # --- shoot tips (stars)
    cells_s = 330
    sf1, svu, svv, scid = worley_vec(n, cells_s, seed + 2, 1.0, warp=(du * 0.4, dv * 0.4))
    ns = cells_s * cells_s
    srad = (0.60 + 0.45 * rng.random(ns)).astype(np.float32)[scid]
    k_arm = rng.integers(5, 9, ns)[scid]
    ph = rng.uniform(0, 6.28, ns).astype(np.float32)[scid]
    star = 0.62 + 0.38 * np.cos(np.arctan2(svv, svu) * k_arm + ph)
    sh = np.clip(1.0 - sf1 / (srad * (0.55 + 0.45 * star)), 0, 1) ** 0.75
    sheight = (0.50 + 0.50 * rng.random(ns)).astype(np.float32)[scid]
    tips = sh * sheight
    hair = grain_noise(n, seed + 3, 0.55)
    fine = fbm_tile(n, 1.1, seed + 4, 40, n // 2)
    h = 3.0 * cush + 1.2 * (0.30 + 0.70 * cush) * tips + 0.10 * hair
    canopy = np.clip(0.10 + 0.50 * cush + 0.60 * tips * (0.35 + 0.65 * cush) + 0.08 * fine, 0, 1)
    # --- colour (physically dark albedo; the tips catch the light)
    dark = C(0.024, 0.058, 0.020)
    mid = C(0.082, 0.185, 0.042)
    light = C(0.185, 0.335, 0.075)
    tip = C(0.320, 0.470, 0.120)
    t = canopy[..., None]
    col = np.where(t < 0.45, dark + (mid - dark) * (t / 0.45), np.where(t < 0.78, mid + (light - mid) * ((t - 0.45) / 0.33), light + (tip - light) * np.clip((t - 0.78) / 0.22, 0, 1)))
    hv = norm01(fbm_tile(n, 2.6, seed + 5, 1, 8), -1.5, 1.5)[..., None]
    col = col * (1 - 0.40 * hv) + (col * np.array([1.22, 1.04, 0.55], np.float32)) * 0.40 * hv           # yellower patches
    bl = sstep(0.50, 0.95, norm01(fbm_tile(n, 2.4, seed + 6, 1, 10), -1.5, 1.5))[..., None]
    col = col * (1 - 0.40 * bl) + (col * np.array([0.68, 1.00, 1.10], np.float32)) * 0.40 * bl           # blue-green patches
    dry = sstep(0.62, 0.95, norm01(fbm_tile(n, 2.6, seed + 11, 2, 14), -1.5, 1.5)) * sstep(0.25, 0.7, tips + 0.3 * cush)
    dry = dry[..., None] * 0.55
    col = col * (1 - dry) + C(0.30, 0.24, 0.08) * dry                                                     # dry brown tips
    col = col * (1.0 + 0.10 * hair)[..., None]
    rough = np.clip(0.92 - 0.10 * tips + 0.04 * hair, 0.6, 1.0)
    return h.astype(np.float32), np.clip(col, 0, 1).astype(np.float32), rough.astype(np.float32), canopy.astype(np.float32)


def moss_maps_v3(n=2048, seed=50):
    h, alb, rough, canopy = moss_v3(n, seed)
    px_mm = 1000.0 / n
    ao = np.clip(0.25 + 0.95 * canopy ** 0.8, 0.22, 1.0)
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * px_mm)
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * px_mm)
    nx, ny, nz = -dx * 0.8, dy * 0.8, np.ones_like(dx)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = (np.stack([nx / l, ny / l, nz / l], -1) * 0.5 + 0.5).astype(np.float32)
    hn = np.clip(h / 5.0, 0, 1)
    orh = np.stack([rough, ao, hn], -1).astype(np.float32)
    return lin_to_srgb(np.clip(alb * (0.55 + 0.45 * ao[..., None]), 0, 1)), nrm, orh


# ------------------------------------------------------------------------------------ driver
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

    stone("StoneAshlar", ashlar_v3)
    stone("StoneCarved", carved_v3)
    stone("Marble", marble_v3)
    stone("Paving", paving_v3)
    if not only or "Moss" in only:
        bc, nrm, orh = moss_maps_v3(n)
        write_set(outdir, "Moss", bc, nrm, orh)
        print(f"  [{time.time() - t0:5.1f}s] Moss", flush=True)
    if not only or "Macro" in only:
        save_png(macro_map(1024 if n >= 1024 else 512), f"{outdir}/T_Macro.png")
        save_png(detail_normal(1024 if n >= 1024 else 512), f"{outdir}/T_Detail_N.png")
        print(f"  [{time.time() - t0:5.1f}s] Macro + Detail", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    n = 1024 if "--quick" in sys.argv else 2048
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    build_all(out, n, only)
