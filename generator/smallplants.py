"""smallplants - high-poly moss cushions, whole ferns and grass clumps (real geometry, opaque, Nanite-friendly).

Ported from the Flooded Rotunda generator (Su3Doesign/Colosseum, generator/hipoly.py) - the same library meshes that
carry 185 000 moss cushions there.  Unit space: cushion radius ~1, plant height ~1; instances carry the real size.
Material ids: 13 moss, 14 fern, 15 grass, 6 stem.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

import meshkit as mk
from meshkit import Mesh
from textures import fbm_tile, lin_to_srgb, norm01, save_jpg, srgb_to_lin

MAT_STEM, MAT_MOSS, MAT_FERN, MAT_GRASS = 6, 13, 14, 15


def _tube(points, radius, sides=5, mat=MAT_STEM, uv=(0.5, 0.05)):
    """Tapered tube along a polyline (k,3); radius scalar or (k,). Constant UV (one texel colour)."""
    P = np.asarray(points, np.float64)
    k = len(P)
    r = np.broadcast_to(np.asarray(radius, np.float64), (k,))
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    S = np.cross(T, ref)
    S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    Bn = np.cross(T, S)
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = (np.cos(ang)[None, :, None] * S[:, None, :] + np.sin(ang)[None, :, None] * Bn[:, None, :]) * r[:, None, None]
    V = (P[:, None, :] + ring).reshape(-1, 3)
    idx = np.arange(k * sides).reshape(k, sides)
    a, b = idx[:-1], idx[1:]
    a2, b2 = np.roll(a, -1, 1), np.roll(b, -1, 1)
    F = np.concatenate([np.stack([a, b, b2], -1).reshape(-1, 3), np.stack([a, b2, a2], -1).reshape(-1, 3)])
    m = Mesh(V, F, mat)
    m.uv0 = np.broadcast_to(np.asarray(uv, np.float64), (len(F), 3, 2)).copy()
    return m


# ------------------------------------------------------------------------------------ moss cushions
def _icosphere(subdiv):
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    return np.asarray(s.vertices, np.float64), np.asarray(s.faces, np.int64)


def moss_cushion(seed, subdiv=4):
    """Flattened lumpy dome sitting on z = 0 (radius ~1, height ~0.5); skirt sinks below the surface."""
    V, F = _icosphere(subdiv)
    n = V / np.linalg.norm(V, axis=1, keepdims=True)
    lump = mk.fbm(V * 1.6 + seed * 3.1, 4, seed=seed)
    fine = mk.fbm(V * 7.0 + seed * 1.7, 3, seed=seed + 1)
    tips = mk.fbm(V * 22.0 + seed, 2, seed=seed + 2)
    rad = 1.0 + 0.20 * lump + 0.05 * fine + 0.018 * tips
    V = n * rad[:, None]
    V[:, 2] = V[:, 2] * 0.52 + 0.06
    V[:, 2] = np.maximum(V[:, 2], -0.08)
    keep = V[F][:, :, 2].max(1) > -0.075
    F = F[keep]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    m = Mesh(V, F, MAT_MOSS)
    uv = V[:, :2] * 0.07 + 0.5
    m.uv0 = uv[F]
    m.uv1 = np.broadcast_to(np.array([0.5, 0.8]), (m.nf, 3, 2)).copy()
    m.uv2 = np.zeros((m.nf, 3, 2))
    return m


# ------------------------------------------------------------------------------------ fern plants
def fern_plant(seed, fronds=(9, 14), pinnae=34):
    """A whole fern (height ~1): arching fronds, serrated V-section pinnae.  UV0 -> pinna texture, UV1.x -> frond tint."""
    rng = np.random.default_rng(seed)
    up = np.array([0.0, 0.0, 1.0])
    parts = []
    nf = int(rng.integers(*fronds))
    for k in range(nf):
        yaw = 2 * math.pi * k / nf + rng.normal(0, 0.22)
        out = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        side = np.cross(up, out)
        L = rng.uniform(0.75, 1.15)
        tilt0 = math.radians(rng.uniform(10, 45))
        bend = math.radians(rng.uniform(40, 85))
        npts = 26
        pts = [np.zeros(3)]
        for i in range(npts):
            a = tilt0 + bend * (i / npts)
            pts.append(pts[-1] + (up * math.cos(a) + out * math.sin(a)) * (L / npts))
        C = np.array(pts)
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        Nf = np.cross(side, T)                                    # frond plane normal (points up / out)
        Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
        tint = rng.random()
        rach = _tube(C, np.linspace(0.010, 0.003, len(C)) * L, sides=5, mat=MAT_FERN, uv=(0.5, 0.5))
        rach.uv1 = np.broadcast_to(np.array([tint, 0.8]), (rach.nf, 3, 2)).copy()
        parts.append(rach)
        for j in range(pinnae):
            t = 0.08 + 0.90 * j / pinnae
            i0 = t * npts
            ia = int(i0)
            fa = i0 - ia
            c = C[ia] * (1 - fa) + C[min(ia + 1, npts)] * fa
            Tt = T[ia]
            Nt = Nf[ia]
            Lp = L * (0.26 * math.sin(math.pi * min(1.0, t * 1.08)) ** 0.75 * (1 - 0.45 * t) + 0.02)
            for sgn in (-1.0, 1.0):
                fwd = math.radians(rng.uniform(18, 32))
                d = side * sgn * math.cos(fwd) + Tt * math.sin(fwd)
                d /= np.linalg.norm(d)
                w_dir = np.cross(d, Nt)
                w_dir /= np.linalg.norm(w_dir)
                droop = rng.uniform(0.10, 0.30)
                m_st = 9
                s = np.linspace(0, 1, m_st)
                cen = c[None, :] + d[None, :] * (Lp * s)[:, None] - Nt[None, :] * (droop * Lp * s * s)[:, None]
                w = Lp * 0.11 * np.sin(math.pi * (0.06 + 0.94 * s)) ** 0.7 * (1 - 0.3 * t)
                w = w * (1.0 + 0.28 * np.where(np.arange(m_st) % 2 == 0, 1.0, -0.6))
                w[-1] = 0.0
                ridge = Nt[None, :] * (w * 0.30)[:, None]
                Lft = cen + w_dir[None, :] * w[:, None] - Nt[None, :] * (w * 0.15)[:, None]
                Rgt = cen - w_dir[None, :] * w[:, None] - Nt[None, :] * (w * 0.15)[:, None]
                Mid = cen + ridge
                V = np.concatenate([Lft, Mid, Rgt])
                idx = np.arange(3 * m_st).reshape(3, m_st)
                F = []
                for col in range(2):
                    a, b = idx[col], idx[col + 1]
                    for q in range(m_st - 1):
                        F += [(a[q], b[q], b[q + 1]), (a[q], b[q + 1], a[q + 1])]
                F = np.array(F)
                UVv = np.concatenate([np.stack([np.zeros(m_st), s], 1), np.stack([np.full(m_st, 0.5), s], 1), np.stack([np.ones(m_st), s], 1)])
                pm = Mesh(V, F, MAT_FERN)
                pm.uv0 = UVv[F]
                pm.uv1 = np.broadcast_to(np.array([tint, 0.8]), (pm.nf, 3, 2)).copy()
                parts.append(pm)
    out = mk.merge(parts)
    out.uv2 = np.zeros((out.nf, 3, 2))
    return out


def fern_pinna_texture(S=512, seed=88):
    """Pinna texture (u across, v base->tip): green gradient, lighter midrib, darker serrated margin, cell noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S * 2, 0:S].astype(np.float32)
    u = xx / S
    v = 1.0 - yy / (S * 2)
    base = srgb_to_lin(np.array([0.055, 0.150, 0.040]))
    tip = srgb_to_lin(np.array([0.175, 0.330, 0.085]))
    t = np.clip(v * 0.9 + 0.05, 0, 1)[..., None]
    col = base * (1 - t) + tip * t
    mid = np.exp(-((u - 0.5) / 0.035) ** 2)[..., None]
    col = col * (1 - 0.40 * mid) + srgb_to_lin(np.array([0.34, 0.44, 0.19])) * 0.40 * mid
    edge = np.clip(np.abs(u - 0.5) * 2, 0, 1)[..., None] ** 3
    col = col * (1 - 0.35 * edge)
    nz = norm01(fbm_tile(S * 2, 1.2, seed, 20, S // 2), -2, 2)
    col = col * (0.85 + 0.3 * nz[:, :S, None])
    # lateral veinlets
    vein = (np.abs(np.sin((v * 60 + np.abs(u - 0.5) * 18) * math.pi)) > 0.93)[..., None]
    col = col * (1 + 0.15 * vein)
    return np.clip(col, 0, 1).astype(np.float32)


# ------------------------------------------------------------------------------------ grass
def grass_clump(seed, blades=72):
    rng = np.random.default_rng(seed)
    parts = []
    for b in range(blades):
        ang = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.05, 0.6)
        Hh = rng.uniform(0.45, 1.0)
        base = np.array([rng.normal(0, 0.06), rng.normal(0, 0.06), 0.0])
        out = np.array([math.cos(ang), math.sin(ang), 0.0])
        side = np.array([-math.sin(ang), math.cos(ang), 0.0])
        m = 14
        s = np.linspace(0, 1, m)
        cen = base[None, :] + np.outer(s, np.array([0, 0, 1.0])) * Hh * (1 - 0.35 * lean * s)[:, None] + np.outer(s * s, out) * Hh * lean
        tw = rng.normal(0, 0.8)
        wv = side[None, :] * np.cos(tw * s)[:, None] + out[None, :] * np.sin(tw * s)[:, None]
        w = rng.uniform(0.010, 0.020) * (1 - s ** 1.5) + 0.0005
        Lf = cen + wv * w[:, None]
        Rt = cen - wv * w[:, None]
        V = np.concatenate([Lf, Rt])
        F = []
        for q in range(m - 1):
            F += [(q, m + q, m + q + 1), (q, m + q + 1, q + 1)]
        F = np.array(F)
        bm = Mesh(V, F, MAT_GRASS)
        UVv = np.concatenate([np.stack([np.zeros(m), s], 1), np.stack([np.ones(m), s], 1)])
        bm.uv0 = UVv[F]
        bm.uv1 = np.broadcast_to(np.array([rng.random(), 0.8]), (bm.nf, 3, 2)).copy()
        parts.append(bm)
    out = mk.merge(parts)
    out.uv2 = np.zeros((out.nf, 3, 2))
    return out


def grass_blade_texture(S=256, seed=91):
    yy, xx = np.mgrid[0:S * 4, 0:S].astype(np.float32)
    u = xx / S
    v = 1.0 - yy / (S * 4)
    base = srgb_to_lin(np.array([0.06, 0.16, 0.04]))
    tip = srgb_to_lin(np.array([0.36, 0.48, 0.12]))
    t = np.clip(v, 0, 1)[..., None] ** 0.8
    col = base * (1 - t) + tip * t
    stripes = 0.5 + 0.5 * np.sin(u * math.pi * 7)
    col = col * (0.9 + 0.12 * stripes[..., None])
    dry = np.clip((v - 0.85) * 6, 0, 1)[..., None] * 0.45
    col = col * (1 - dry) + srgb_to_lin(np.array([0.55, 0.50, 0.30])) * dry
    return np.clip(col, 0, 1).astype(np.float32)




def build_library(lib_dir, tex_dir=None):
    os.makedirs(lib_dir, exist_ok=True)
    for i in range(4):
        m = moss_cushion(700 + i, subdiv=5 if i < 2 else 4)
        m.save(f"{lib_dir}/lib_moss_{i:02d}.npz")
        f = fern_plant(800 + i)
        f.save(f"{lib_dir}/lib_fern_{i:02d}.npz")
        g = grass_clump(900 + i)
        g.save(f"{lib_dir}/lib_grass_{i:02d}.npz")
    print(f"  small plants: 4 moss cushions ({m.nf:,} tris), 4 ferns ({f.nf:,}), 4 grass clumps ({g.nf:,})", flush=True)
    if tex_dir:
        os.makedirs(tex_dir, exist_ok=True)
        save_jpg(lin_to_srgb(fern_pinna_texture(512)), f"{tex_dir}/T_FernPinna_BC.jpg", q=93)
        save_jpg(lin_to_srgb(grass_blade_texture()), f"{tex_dir}/T_GrassBlade_BC.jpg", q=93)


if __name__ == "__main__":
    build_library(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
