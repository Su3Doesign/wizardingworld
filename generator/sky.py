"""sky - the night-sky meshes: a star dome (inward-facing sphere, equirectangular UVs) and the moon disc (placed in the
direction of shots.MOON_DIR, facing the valley).  Both use additive emissive materials scaled by MPC_World.Night in
Unreal, so they vanish in daylight and never hide the sky atmosphere or the clouds."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

import meshkit as mk
import shots as SH
from meshkit import Mesh

STARS, MOON = 35, 36


def sky_dome(radius=30000.0, nu=96, nv=48):
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0.02, 1, nv + 1)                    # stop just below the horizon
    U, Vv = np.meshgrid(u, v, indexing="ij")
    lon = U * 2 * math.pi
    lat = (0.5 - Vv) * math.pi * 1.0
    lat = np.clip(lat, -0.12, math.pi / 2)
    P = np.stack([radius * np.cos(lat) * np.sin(lon), radius * np.cos(lat) * np.cos(lon), radius * np.sin(lat)], -1)
    m = mk.grid(P, mat=STARS)
    UV = np.stack([U, 1 - Vv], -1).reshape(-1, 2)
    m.uv0 = UV[m.F]
    # inward facing (seen from the inside)
    C = m.V[m.F].mean(1)
    if np.mean(np.einsum("ij,ij->i", m.face_normals(), C)) > 0:
        m.flip()
    return m


def moon_disc(seg=64):
    pos, dia = SH.moon_position()
    r = dia / 2
    d = -pos / np.linalg.norm(pos)                      # faces the valley
    up = np.array([0.0, 0.0, 1.0])
    t = np.cross(up, d)
    t /= np.linalg.norm(t)
    b = np.cross(d, t)
    a = np.linspace(0, 2 * math.pi, seg, endpoint=False)
    ring = pos[None] + r * (np.cos(a)[:, None] * t[None] + np.sin(a)[:, None] * b[None])
    V = np.vstack([ring, pos[None]])
    F = np.array([[i, (i + 1) % seg, seg] for i in range(seg)])
    m = Mesh(V, F, MOON)
    if np.mean(m.face_normals() @ d) < 0:
        m.flip()
    uv = np.vstack([np.stack([0.5 + 0.5 * np.cos(a), 0.5 + 0.5 * np.sin(a)], 1), [[0.5, 0.5]]])
    m.uv0 = uv[m.F]
    return m


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sky_dome().save(f"{out_dir}/sky_dome.npz")
    moon_disc().save(f"{out_dir}/sky_moondisc.npz")
    print("  sky dome + moon disc")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "OUT/geo")
