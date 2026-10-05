"""tex_world - the procedural texture library of the world (seamless, deterministic, numpy).

Packing (as in the Flooded Rotunda pipeline): *_BC.jpg sRGB base colour, *_N.png OpenGL normal (Unreal flips green on
import), *_ORH.png R roughness / G ambient occlusion / B height.  Tile sizes (metres per texture repeat):
    Granite 4 m (cliffs, triplanar)      CastleStone 2 m (6 courses, triplanar)   Trim 2 m      Moss 1 m
    Slate 2.52 m (6 courses of 0.42 m = the modelled slate courses)              Grass / ForestFloor / Shore / Dirt 2 m
    Lead 2 m, Wood 2 m, Bark 1.5 m, Glass = one window (UV 0..1), Needles / Leaf = one strip (UV 0..1)
Extra: T_Macro (16 m colour / roughness / moss-bias variation), T_Detail_N (0.25 m grain), T_Water_N, T_Water_Ripple_N,
T_Foam, T_Fall (waterfall streaks), T_Stars (equirectangular night sky), T_Moon (moon disc).
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

import textures as tx
import textures2 as t2
import textures3 as t3
from textures import fbm_tile, lin_to_srgb, norm01, save_jpg, save_png, srgb_to_lin, worley_tile


def C(r, g, b):
    return srgb_to_lin(np.array([r, g, b], np.float32))


def sstep(a, b, x):
    return t2.sstep(a, b, x)


def blur(a, s):
    return ndimage.gaussian_filter(a, s, mode="wrap")


def finish(height_mm, albedo, rough, tile_mm, normal_strength=1.0, hmin=-14.0, hmax=4.0, ao_extra=None):
    n = height_mm.shape[0]
    px_mm = tile_mm / n
    ao = np.clip(1.0 - 0.55 * (blur(height_mm, 6.0) - height_mm) / (2.2 * px_mm / 0.98), 0.35, 1.0)
    if ao_extra is not None:
        ao = ao * ao_extra
    dx = (np.roll(height_mm, -1, 1) - np.roll(height_mm, 1, 1)) / (2 * px_mm)
    dy = (np.roll(height_mm, -1, 0) - np.roll(height_mm, 1, 0)) / (2 * px_mm)
    nx, ny, nz = -dx * normal_strength, dy * normal_strength, np.ones_like(dx)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = (np.stack([nx / l, ny / l, nz / l], -1) * 0.5 + 0.5).astype(np.float32)
    hn = np.clip((height_mm - hmin) / (hmax - hmin), 0, 1)
    orh = np.stack([np.clip(rough, 0, 1), ao, hn], -1).astype(np.float32)
    return lin_to_srgb(np.clip(albedo * ao[..., None] ** 0.5, 0, 1)), nrm, orh


def write(outdir, name, bc, nrm, orh, q=92):
    save_jpg(bc, f"{outdir}/T_{name}_BC.jpg", q=q)
    save_png(nrm, f"{outdir}/T_{name}_N.png")
    save_png(orh, f"{outdir}/T_{name}_ORH.png")


# ----------------------------------------------------------------------------------------------- granite
def granite(n=2048, seed=300):
    """Weathered Highland granite, tile 4 m: mineral speckle (feldspar / quartz / biotite), a grey-brown patina,
    exfoliation flakes, fractures, crustose lichen rosettes, black rain streaks and iron stains."""
    tile_mm = 4000.0
    px = tile_mm / n
    rng = np.random.default_rng(seed)
    # minerals (2-8 mm grains)
    f1, f2, cid = worley_tile(n, int(tile_mm / 5.5), seed, 1.0)
    nc = int(tile_mm / 5.5) ** 2
    kind = rng.random(nc)
    mineral = np.where(kind < 0.52, 0, np.where(kind < 0.86, 1, 2))[cid]          # 0 feldspar, 1 quartz, 2 biotite
    tone = rng.uniform(0.85, 1.15, nc).astype(np.float32)[cid]
    feld = C(0.68, 0.62, 0.55)
    quartz = C(0.50, 0.50, 0.49)
    bio = C(0.08, 0.08, 0.08)
    alb = np.where((mineral == 0)[..., None], feld, np.where((mineral == 1)[..., None], quartz, bio)) * tone[..., None]
    # big-scale warmth / greyness drift and the weathering patina (dulls everything towards grey-brown)
    drift = norm01(fbm_tile(n, 2.6, seed + 1, 1, 10), -2, 2)[..., None]
    patina_c = C(0.40, 0.38, 0.35) * (0.85 + 0.3 * drift)
    pat = 0.55 + 0.25 * norm01(fbm_tile(n, 2.2, seed + 2, 2, 30), -2, 2)[..., None]
    alb = alb * (1 - pat) + patina_c * pat
    # relief: exfoliation flakes (curved slabs a few cm thick), fractures, grain pitting
    g1, g2, gid = worley_tile(n, 9, seed + 3, 1.0)
    flake = np.clip((g2 - g1) * 4.0, 0, 1)
    flake_h = rng.uniform(-6, 6, 81).astype(np.float32)[gid] + 4.0 * blur(flake, 3.0)
    und = 6.0 * fbm_tile(n, 2.6, seed + 4, 2, 24)
    grain_h = 0.35 * (mineral == 1) - 0.25 * (mineral == 2) + 0.25 * fbm_tile(n, 1.0, seed + 5, 100, n // 2)
    cracks = t2.crack_layer(n, seed + 6, major=6, hair_density=0.45)
    h = und + 0.35 * flake_h + grain_h - 9.0 * cracks - 2.0 * np.clip(1 - (g2 - g1) * 12, 0, 1)
    # crustose lichen rosettes (pale grey-green, yellow-green, orange) and dark water streaks
    lm, lcol, lh = t2.lichen_layer(n, seed + 7, density=0.10)
    lcol = lcol * 0.35 + C(0.55, 0.56, 0.50) * 0.65                       # mostly pale grey-green crusts
    alb = alb * (1 - 0.55 * lm[..., None]) + lcol * 0.55 * lm[..., None]
    h = h + 2.0 * lh * lm
    streak = sstep(0.4, 1.6, t2.fbm_aniso(n, 2.5, seed + 8, 1.0, 0.06, 1, 160))
    streak2 = sstep(0.8, 1.8, t2.fbm_aniso(n, 2.3, seed + 9, 1.0, 0.10, 2, 220))
    alb *= (1 - 0.45 * streak - 0.25 * streak2)[..., None]
    iron = sstep(0.9, 1.9, t2.fbm_aniso(n, 2.4, seed + 10, 1.0, 0.18, 1, 120)) * sstep(0.2, 1.0, fbm_tile(n, 2.3, seed + 11, 1, 12))
    alb = alb * (1 - 0.35 * iron[..., None]) + C(0.42, 0.26, 0.14) * 0.35 * iron[..., None]
    alb *= (1 - 0.45 * cracks)[..., None]
    rough = 0.82 + 0.08 * fbm_tile(n, 2.0, seed + 12, 4, 60) - 0.15 * streak + 0.06 * lm
    return finish(h, np.clip(alb, 0, 1), rough, tile_mm, normal_strength=0.9, hmin=-30.0, hmax=12.0)


# ----------------------------------------------------------------------------------------------- slate roof
def slate(n=2048, seed=320, courses=6, course_m=0.42):
    """Natural slate roof, tile = courses x course_m square; course rows match the modelled slate steps."""
    rng = np.random.default_rng(seed)
    tile_mm = courses * course_m * 1000.0
    px = tile_mm / n
    rows = courses
    rh = n // rows
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    h = np.zeros((n, n), np.float32)
    alb = np.zeros((n, n, 3), np.float32)
    rough = np.zeros((n, n), np.float32)
    base = [C(0.20, 0.21, 0.23), C(0.23, 0.22, 0.25), C(0.18, 0.19, 0.20), C(0.26, 0.25, 0.27), C(0.21, 0.20, 0.22)]
    for r in range(rows):
        y0 = r * rh
        # rows are numbered bottom-up in V (image rows run downwards)
        widths = []
        w_mm_total = 0.0
        while w_mm_total < tile_mm:
            w = rng.uniform(240.0, 360.0)
            widths.append(w)
            w_mm_total += w
        widths = np.array(widths) * (tile_mm / w_mm_total)
        edges = np.concatenate([[0.0], np.cumsum(widths)]) / px
        shift = (r % 2) * edges[1] * 0.5 + rng.uniform(0, n * 0.02)
        xr = (xx[y0:y0 + rh] - shift) % n
        sid = np.searchsorted(edges, xr, side="right") - 1
        sid = np.clip(sid, 0, len(widths) - 1)
        left = edges[sid]
        right = edges[np.minimum(sid + 1, len(edges) - 1)]
        dx = np.minimum(xr - left, right - xr) * px                         # mm to the vertical joint
        vy = (yy[y0:y0 + rh] - y0) * px                                     # mm from the top of the course image
        # each slate: slightly tilted, butt (lower edge = bottom of the course in V = image bottom) thicker
        u_in = 1.0 - vy / (rh * px)                                          # 0 at the course's lower edge (image bottom)
        thick = 5.0 + 2.0 * rng.random(len(widths))[sid]
        hh = thick * (1 - u_in) + rng.normal(0, 0.6, len(widths))[sid] * 1.0
        hh = hh - 3.0 * np.exp(-dx / 1.5)                                    # open vertical joints
        col = np.array(base)[rng.integers(0, len(base), len(widths))][sid] * rng.uniform(0.8, 1.2, len(widths))[sid][..., None]
        h[y0:y0 + rh] = hh
        alb[y0:y0 + rh] = col
        rough[y0:y0 + rh] = rng.uniform(0.45, 0.70, len(widths))[sid]
    h += 0.6 * fbm_tile(n, 1.4, seed + 1, 40, n // 2) + 1.2 * fbm_tile(n, 2.2, seed + 2, 6, 60)
    # slate cleavage texture, chipped corners, lichen / moss spots, run-off streaks
    alb *= (1 + 0.08 * fbm_tile(n, 1.5, seed + 3, 30, 300))[..., None]
    lm, lcol, lh = t2.lichen_layer(n, seed + 4, density=0.05)
    lcol = lcol * 0.3 + C(0.45, 0.47, 0.42) * 0.7
    alb = alb * (1 - 0.5 * lm[..., None]) + lcol * 0.5 * lm[..., None]
    streak = sstep(0.6, 1.8, t2.fbm_aniso(n, 2.4, seed + 5, 1.0, 0.08, 1, 140))
    alb *= (1 - 0.25 * streak)[..., None]
    rough = rough + 0.15 * streak + 0.2 * lm
    return finish(h, np.clip(alb, 0, 1), rough, tile_mm, normal_strength=1.0, hmin=-4.0, hmax=9.0)


# ----------------------------------------------------------------------------------------------- ground layers
def grass_ground(n=1024, seed=340):
    """Short lawn / meadow grass seen from above, tile 2 m."""
    tile_mm = 2000.0
    rng = np.random.default_rng(seed)
    alb = np.zeros((n, n, 3), np.float32)
    h = np.zeros((n, n), np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    # many short blades drawn as anisotropic speckles
    base = C(0.10, 0.17, 0.05)
    alb[...] = base
    for k in range(4):
        b = sstep(0.4, 1.6, t2.fbm_aniso(n, 1.6, seed + k, 1.0, 0.25 + 0.15 * k, 30, n // 2))
        tint = [C(0.22, 0.32, 0.09), C(0.30, 0.38, 0.12), C(0.16, 0.26, 0.06), C(0.40, 0.42, 0.20)][k]
        alb = alb * (1 - 0.5 * b[..., None]) + tint * 0.5 * b[..., None]
        h += 1.5 * b
    patches = norm01(fbm_tile(n, 2.4, seed + 9, 1, 10), -2, 2)[..., None]
    alb = alb * (0.9 + 0.18 * patches)
    dry = sstep(0.62, 0.95, norm01(fbm_tile(n, 2.6, seed + 10, 3, 30), -2, 2))[..., None] * 0.14
    alb = alb * (1 - dry) + C(0.42, 0.38, 0.22) * dry
    rough = np.full((n, n), 0.88, np.float32)
    return finish(h, np.clip(alb, 0, 1), rough, tile_mm, 0.8, hmin=-1.0, hmax=6.0)


def forest_floor(n=1024, seed=360):
    """Spruce forest floor: needle litter, cones, twigs, moss patches, dark humus; tile 2 m."""
    tile_mm = 2000.0
    rng = np.random.default_rng(seed)
    alb = np.zeros((n, n, 3), np.float32)
    alb[...] = C(0.12, 0.08, 0.05)
    h = 1.5 * fbm_tile(n, 2.4, seed, 2, 40)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    # needles: thousands of short random line segments
    canvas = np.zeros((n, n), np.float32)
    cols = np.zeros((n, n, 3), np.float32)
    for k in range(9000):
        x0, y0 = rng.uniform(0, n, 2)
        a = rng.uniform(0, math.pi)
        L = rng.uniform(5, 12)
        t = np.linspace(0, 1, 10)
        xs = ((x0 + np.cos(a) * L * t).astype(int)) % n
        ys = ((y0 + np.sin(a) * L * t).astype(int)) % n
        c = [C(0.40, 0.25, 0.12), C(0.30, 0.18, 0.09), C(0.50, 0.34, 0.18), C(0.22, 0.14, 0.08)][k % 4]
        canvas[ys, xs] = 1.0
        cols[ys, xs] = c
    canvas = np.clip(blur(canvas, 0.5) * 2.0, 0, 1)
    alb = alb * (1 - canvas[..., None]) + cols * canvas[..., None]
    h += 1.2 * canvas
    moss_p = sstep(0.6, 1.0, norm01(fbm_tile(n, 2.4, seed + 1, 1, 12), -2, 2))[..., None]
    alb = alb * (1 - 0.6 * moss_p) + C(0.14, 0.24, 0.05) * 0.6 * moss_p
    alb *= (0.8 + 0.4 * norm01(fbm_tile(n, 2.0, seed + 2, 2, 30), -2, 2))[..., None]
    rough = np.full((n, n), 0.9, np.float32)
    return finish(h, np.clip(alb, 0, 1), rough, tile_mm, 0.9, hmin=-3.0, hmax=5.0)


def shore(n=1024, seed=380):
    """Wet shingle and sand at the water's edge; tile 2 m."""
    tile_mm = 2000.0
    rng = np.random.default_rng(seed)
    f1, f2, cid = worley_tile(n, 70, seed, 1.0)
    nc = 70 * 70
    peb = np.clip(1 - f1 / rng.uniform(0.45, 0.8, nc).astype(np.float32)[cid], 0, 1) ** 0.5
    h = 6.0 * peb + 0.6 * fbm_tile(n, 1.2, seed + 1, 60, n // 2)
    pal = [C(0.30, 0.29, 0.27), C(0.22, 0.21, 0.20), C(0.38, 0.35, 0.31), C(0.18, 0.17, 0.16), C(0.42, 0.40, 0.37)]
    col = np.array(pal)[rng.integers(0, len(pal), nc)][cid] * rng.uniform(0.8, 1.2, nc).astype(np.float32)[cid][..., None]
    sand = C(0.28, 0.24, 0.19) * (0.9 + 0.2 * fbm_tile(n, 1.0, seed + 2, 80, n // 2))[..., None]
    alb = np.where((peb > 0.05)[..., None], col, sand)
    alb *= 0.75                                                         # wet
    rough = np.where(peb > 0.05, 0.25, 0.45).astype(np.float32)
    return finish(h, np.clip(alb, 0, 1), rough, tile_mm, 1.0, hmin=-2.0, hmax=8.0)


def dirt(n=1024, seed=390):
    tile_mm = 2000.0
    h = 2.0 * fbm_tile(n, 2.0, seed, 2, 80) + 0.8 * fbm_tile(n, 1.2, seed + 1, 60, n // 2)
    f1, f2, cid = worley_tile(n, 120, seed + 2, 1.0)
    grit = (f1 < 0.25).astype(np.float32)
    h += 1.0 * grit
    alb = C(0.30, 0.24, 0.17) * (0.8 + 0.4 * norm01(fbm_tile(n, 2.2, seed + 3, 2, 30), -2, 2))[..., None]
    alb = alb * (1 - 0.3 * grit[..., None]) + C(0.45, 0.42, 0.38) * 0.3 * grit[..., None]
    rough = np.full((n, n), 0.86, np.float32)
    return finish(h, np.clip(alb, 0, 1), rough, tile_mm, 0.9, hmin=-3.0, hmax=5.0)


# ----------------------------------------------------------------------------------------------- metal, glass, wood, cloth
def lead(n=1024, seed=400):
    tile_mm = 2000.0
    h = 0.6 * fbm_tile(n, 1.8, seed, 4, 100)
    pat = norm01(fbm_tile(n, 2.4, seed + 1, 1, 16), -2, 2)
    alb = C(0.10, 0.11, 0.12) * (1 - pat[..., None] * 0.4) + C(0.32, 0.34, 0.33) * pat[..., None] * 0.4
    streak = sstep(0.6, 1.6, t2.fbm_aniso(n, 2.4, seed + 2, 1.0, 0.08, 1, 120))
    alb = alb * (1 - 0.3 * streak[..., None]) + C(0.45, 0.50, 0.46) * 0.3 * streak[..., None]
    rough = 0.45 + 0.25 * pat
    return finish(h, np.clip(alb, 0, 1), rough, tile_mm, 0.6, hmin=-1.0, hmax=2.0)


def glass(n=1024, seed=410, across=6, down=12):
    """Leaded diamond-pane window (UV 0..1 over a typical 1.3 x 3 m lancet; the material may tile it).
    BC = glass tint, ORH.r = roughness, ORH.g = AO, ORH.b = came (lead) mask -> 1 on the lead, 0 on the glass."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    u = xx * across
    v = (1 - yy) * down
    a = (u + v) % 1.0
    b = (u - v) % 1.0
    came = np.minimum(np.minimum(a, 1 - a), np.minimum(b, 1 - b))
    lead_m = sstep(0.045, 0.02, came)
    pane = (np.floor(u + v) * 31 + np.floor(u - v) * 17).astype(int)
    tint = 0.7 + 0.3 * ((pane * 2654435761) % 1000 / 1000.0)
    glass_c = C(0.10, 0.12, 0.13) * tint[..., None]
    glass_c *= (1 + 0.15 * fbm_tile(n, 2.0, seed, 2, 40))[..., None]          # wavy old glass
    alb = glass_c * (1 - lead_m[..., None]) + C(0.07, 0.07, 0.07) * lead_m[..., None]
    h = 2.0 * lead_m + 0.3 * fbm_tile(n, 2.0, seed + 1, 3, 30) * (1 - lead_m)
    rough = 0.08 + 0.5 * lead_m
    bc, nrm, orh = finish(h, alb, rough, 1300.0, 1.0, hmin=-1.0, hmax=3.0)
    orh[..., 2] = lead_m
    return bc, nrm, orh


def wood(n=1024, seed=420):
    tile_mm = 2000.0
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    plank = (xx // (n / 10)).astype(int)
    grain = np.sin((yy / n) * 2 * math.pi * 30 + 3.0 * fbm_tile(n, 2.0, seed, 1, 30) + plank * 1.7) * 0.5 + 0.5
    gap = sstep(3.0, 0.5, np.minimum(xx % (n / 10), (n / 10) - xx % (n / 10)))
    tone = rng.uniform(0.7, 1.2, 10)[plank % 10][..., None]
    alb = (C(0.20, 0.13, 0.08) * (1 - grain[..., None]) + C(0.32, 0.22, 0.13) * grain[..., None]) * tone
    alb *= (1 - 0.6 * gap[..., None])
    h = -3.0 * gap + 0.4 * grain
    weather = sstep(0.3, 1.2, fbm_tile(n, 2.4, seed + 1, 1, 12))[..., None]
    alb = alb * (1 - 0.4 * weather) + C(0.30, 0.29, 0.27) * 0.4 * weather
    return finish(h, np.clip(alb, 0, 1), np.full((n, n), 0.8, np.float32), tile_mm, 1.0, hmin=-4.0, hmax=2.0)


def cloth(n=512, seed=430):
    """Canvas: weave + soft folds; the colour comes from the material (per stand), so this is near white."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    weave = 0.5 + 0.25 * (np.sin(xx * math.pi / 2) * np.sin(yy * math.pi / 2))
    fold = fbm_tile(n, 2.8, seed, 1, 6)
    h = 0.3 * weave + 4.0 * fold
    alb = C(0.82, 0.80, 0.76) * (0.9 + 0.1 * weave[..., None]) * (0.92 + 0.12 * norm01(fold, -2, 2)[..., None])
    return finish(h, alb, np.full((n, n), 0.9, np.float32), 1000.0, 0.6, hmin=-8.0, hmax=8.0)


# ----------------------------------------------------------------------------------------------- bark, needles, leaves
def spruce_bark(n=1024, seed=440):
    """Spruce bark: thin rounded scales, grey-brown with reddish fresh patches; tile 1.5 m (u round, v up)."""
    tile_mm = 1500.0
    rng = np.random.default_rng(seed)
    f1, f2, cid = worley_tile(n, 46, seed, 0.9)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    scale_h = np.clip(1 - f1 * 1.4, 0, 1) ** 0.7 * 4.0 - 2.0 * np.clip(1 - (f2 - f1) * 12, 0, 1)
    vert = t2.fbm_aniso(n, 2.2, seed + 1, 0.15, 1.0, 2, 120)              # vertical fissures
    h = scale_h + 3.0 * vert
    nc = 46 * 46
    col = np.where((rng.random(nc) < 0.12)[cid][..., None], C(0.34, 0.18, 0.10), C(0.24, 0.20, 0.17))
    col = col * rng.uniform(0.75, 1.2, nc).astype(np.float32)[cid][..., None]
    lm, lcol, _ = t2.lichen_layer(n, seed + 2, density=0.06)
    lcol = lcol * 0.3 + C(0.50, 0.52, 0.46) * 0.7
    alb = col * (1 - 0.45 * lm[..., None]) + lcol * 0.45 * lm[..., None]
    alb *= (1 - 0.35 * np.clip(1 - (f2 - f1) * 10, 0, 1))[..., None]
    return finish(h, np.clip(alb, 0, 1), np.full((n, n), 0.88, np.float32), tile_mm, 1.2, hmin=-6.0, hmax=5.0)


def birch_bark(n=1024, seed=450):
    tile_mm = 1500.0
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    alb = np.zeros((n, n, 3), np.float32) + C(0.78, 0.76, 0.72)
    h = np.zeros((n, n), np.float32)
    for k in range(260):                                              # horizontal dark lenticels
        x0, y0 = rng.uniform(0, n, 2)
        L = rng.uniform(8, 60)
        w = rng.uniform(1.0, 3.0)
        d = np.exp(-(((yy - y0 + n / 2) % n - n / 2) / w) ** 2) * np.exp(-(((xx - x0 + n / 2) % n - n / 2) / L) ** 2)
        alb = alb * (1 - 0.8 * d[..., None]) + C(0.08, 0.07, 0.07) * 0.8 * d[..., None]
        h -= 1.5 * d
    dark = sstep(0.8, 1.8, fbm_tile(n, 2.4, seed + 1, 1, 10))
    alb = alb * (1 - 0.5 * dark[..., None]) + C(0.15, 0.13, 0.12) * 0.5 * dark[..., None]
    return finish(h, np.clip(alb, 0, 1), np.full((n, n), 0.7, np.float32), tile_mm, 0.8, hmin=-4.0, hmax=2.0)


def needles(n=512, seed=460):
    """One needle spray strip: u across (0..1), v along (0..1, base -> tip).  Dense spruce needles radiating from the
    midrib towards the tip; dark gaps between them read as depth (the strip is opaque geometry)."""
    rng = np.random.default_rng(seed)
    H, Wd = n * 2, n
    yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
    u = xx / Wd
    v = 1 - yy / H
    alb = np.zeros((H, Wd, 3), np.float32) + C(0.035, 0.065, 0.028)
    h = np.zeros((H, Wd), np.float32)
    for k in range(2600):
        side = -1 if k % 2 else 1
        t0 = rng.uniform(0, 1)
        x0 = 0.5 * Wd
        y0 = (1 - t0) * H
        ang = math.radians(rng.uniform(25, 55))
        L = rng.uniform(0.10, 0.22) * Wd * 2.2
        tt = np.linspace(0, 1, 24)
        xs = (x0 + side * np.sin(ang) * L * tt).astype(int)
        ys = (y0 - np.cos(ang) * L * tt).astype(int)
        ok = (xs >= 0) & (xs < Wd) & (ys >= 0) & (ys < H)
        shade = 0.6 + 0.4 * tt[ok]
        c = [C(0.10, 0.19, 0.06), C(0.13, 0.24, 0.07), C(0.08, 0.15, 0.05), C(0.18, 0.28, 0.09)][k % 4]
        alb[ys[ok], xs[ok]] = c * shade[:, None]
        alb[ys[ok], np.clip(xs[ok] + 1, 0, Wd - 1)] = c * shade[:, None] * 0.8
        h[ys[ok], xs[ok]] = 1.0
    tip = sstep(0.75, 1.0, v)[..., None]                                # fresh light-green tips
    alb = alb * (1 - 0.35 * tip) + C(0.20, 0.30, 0.09) * 0.35 * tip
    h = blur(h, 0.7)
    bc, nrm, orh = finish(h * 0.6, np.clip(alb, 0, 1), np.full((H, Wd), 0.6, np.float32), 300.0, 1.2, hmin=-1.0, hmax=1.5)
    return bc, nrm, orh


def leaf(n=512, seed=470):
    """Birch leaf strip (u across, v along): green with a pale midrib and veins."""
    H, Wd = n, n
    yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
    u = xx / Wd
    v = 1 - yy / H
    alb = np.zeros((H, Wd, 3), np.float32) + C(0.16, 0.28, 0.06)
    mid = np.exp(-((u - 0.5) / 0.02) ** 2)
    veins = (np.abs(np.sin((v * 9 + np.abs(u - 0.5) * 6) * math.pi)) > 0.94) * (np.abs(u - 0.5) < 0.45)
    alb = alb * (1 - 0.4 * mid[..., None]) + C(0.40, 0.48, 0.18) * 0.4 * mid[..., None]
    alb *= (1 + 0.15 * veins[..., None])
    alb *= (0.85 + 0.3 * norm01(fbm_tile(n, 2.0, seed, 4, 60), -2, 2))[..., None]
    h = 0.5 * mid + 0.3 * veins
    return finish(h, np.clip(alb, 0, 1), np.full((H, Wd), 0.5, np.float32), 80.0, 0.8, hmin=-1.0, hmax=1.5)


# ----------------------------------------------------------------------------------------------- water, sky
def waterfall(n=1024, seed=480):
    """Falling water streaks (u across, v down): white aerated sheets with darker clear water between; tile 4 m."""
    s1 = t2.fbm_aniso(n, 2.0, seed, 0.06, 1.0, 2, 300)
    s2 = t2.fbm_aniso(n, 1.6, seed + 1, 0.10, 1.0, 8, 400)
    foam = np.clip(0.5 + 0.35 * s1 + 0.2 * s2, 0, 1)
    alb = (C(0.30, 0.34, 0.33) * (1 - foam[..., None]) + C(0.88, 0.90, 0.90) * foam[..., None])
    a = np.clip(0.55 + 0.45 * foam, 0, 1)
    bc = lin_to_srgb(np.clip(alb, 0, 1))
    return bc, a.astype(np.float32)


def stars(w=4096, h=2048, seed=490):
    """Equirectangular star field with a faint Milky Way band (linear radiance in [0,1], saved sRGB)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    n = 26000
    lon = rng.uniform(0, 2 * math.pi, n)
    lat = np.arcsin(rng.uniform(-1, 1, n))
    mag = rng.pareto(2.2, n) + 1.0
    x = (lon / (2 * math.pi) * w).astype(int) % w
    y = ((0.5 - lat / math.pi) * h).astype(int).clip(0, h - 1)
    bright = np.clip(0.02 * mag ** 1.6, 0, 1.0)
    temp = rng.uniform(0, 1, n)
    col = np.stack([0.85 + 0.15 * temp, 0.88 + 0.06 * temp, 1.0 - 0.2 * temp], 1)
    np.add.at(img, (y, x), col * bright[:, None])
    img = ndimage.gaussian_filter(img, (0.6, 0.6, 0))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lat_g = (0.5 - yy / h) * math.pi
    lon_g = xx / w * 2 * math.pi
    band = np.exp(-((np.sin(lat_g) - 0.45 * np.sin(lon_g + 0.7)) / 0.16) ** 2)
    neb = norm01(fbm_tile(1024, 1.8, seed + 1, 2, 200), -2, 2)
    neb = ndimage.zoom(neb, (h / 1024, w / 1024), order=1)
    img += (band * neb * 0.05)[..., None] * np.array([0.8, 0.85, 1.0])
    return lin_to_srgb(np.clip(img, 0, 1))


def moon(n=1024, seed=500):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n * 2 - 1
    r = np.sqrt(xx * xx + yy * yy)
    disc = (r < 0.98).astype(np.float32)
    maria = sstep(0.1, 0.6, norm01(fbm_tile(n, 2.8, seed, 1, 8), -2, 2))
    alb = 0.85 - 0.35 * maria
    for k in range(180):
        cx, cy = rng.uniform(-1, 1, 2)
        cr = rng.pareto(2.5) * 0.015 + 0.006
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / cr
        ring = np.exp(-((d - 1) / 0.25) ** 2) * 0.12 - np.exp(-(d / 0.8) ** 2) * 0.08
        alb += ring
    limb = np.sqrt(np.clip(1 - r * r, 0, 1)) ** 0.25
    img = np.clip(alb * limb * disc, 0, 1)
    return np.stack([img, img * 0.98, img * 0.94], -1)


# ----------------------------------------------------------------------------------------------- driver
def build_all(outdir, quick=False):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    N2 = 1024 if quick else 2048
    N1 = 512 if quick else 1024

    def done(name):
        print(f"  [{time.time() - t0:6.1f}s] {name}", flush=True)

    write(outdir, "Granite", *granite(N2))
    done("Granite")
    h, a, r = t3.stone_surface(N2, 510, t3.block_layout(N2, 510, rows=5, per_row=(3, 4), wobble_mm=(2.0, 1.5)), tooled=True, pit_k=0.8,
                               lichen_cover=0.10, base=((0.37, 0.34, 0.30), (0.60, 0.56, 0.49)))
    write(outdir, "CastleStone", *t2.finish_maps(h, a, r))
    done("CastleStone")
    h, a, r = t3.carved_v3(N1, 520)
    write(outdir, "Trim", *t2.finish_maps(h, a, r))
    done("Trim")
    write(outdir, "Moss", *t3.moss_maps_v3(N1 if quick else 2048, 50))
    done("Moss")
    write(outdir, "Slate", *slate(N2))
    done("Slate")
    write(outdir, "Grass", *grass_ground(N1))
    write(outdir, "ForestFloor", *forest_floor(N1))
    write(outdir, "Shore", *shore(N1))
    write(outdir, "Dirt", *dirt(N1))
    done("ground layers")
    write(outdir, "Lead", *lead(N1))
    write(outdir, "Glass", *glass(N1))
    write(outdir, "Wood", *wood(N1))
    write(outdir, "Cloth", *cloth(512))
    done("lead / glass / wood / cloth")
    write(outdir, "Bark", *spruce_bark(N1))
    write(outdir, "BirchBark", *birch_bark(N1))
    write(outdir, "Needles", *needles(512))
    write(outdir, "Leaf", *leaf(512))
    done("bark / needles / leaf")
    save_png(t2.macro_map(1024), f"{outdir}/T_Macro.png")
    save_png(t2.detail_normal(1024), f"{outdir}/T_Detail_N.png")
    hw = tx.water_normal(1024, 60)
    save_png(tx.height_to_normal(hw, 3.0), f"{outdir}/T_Water_N.png")
    hr = tx.water_normal(1024, 61)
    save_png(tx.height_to_normal(hr, 6.0), f"{outdir}/T_Water_Ripple_N.png")
    foam = np.clip(norm01(fbm_tile(1024, 1.6, 62, 4, 300), -1.2, 2.0), 0, 1)
    save_png(np.stack([foam] * 3, -1), f"{outdir}/T_Foam.png")
    bc, a = waterfall(1024)
    save_jpg(bc, f"{outdir}/T_Fall_BC.jpg", q=92)
    save_png(np.stack([a] * 3, -1), f"{outdir}/T_Fall_A.png")
    done("water")
    save_jpg(stars(2048 if quick else 4096, 1024 if quick else 2048), f"{outdir}/T_Stars.jpg", q=94)
    save_jpg(lin_to_srgb(moon(1024)), f"{outdir}/T_Moon.jpg", q=94)
    done("stars / moon")


if __name__ == "__main__":
    build_all(sys.argv[1] if len(sys.argv) > 1 else "OUT/tex", quick="--quick" in sys.argv)
