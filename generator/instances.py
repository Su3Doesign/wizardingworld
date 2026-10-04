"""Instance sets: many copies of a high-poly library mesh, each with position / rotation / scale.

On disk (generator side) a set is an .npz in GEO/instances:  pos (N,3) m, R (N,3,3) rotation matrices whose columns are
the instance's local x / y / z axes in world space, scale (N,3), variant (N,) library index, tint (N,).

For Unreal the exporter writes one little-endian float32 file per (set, variant):
    x, y, z (cm), qx, qy, qz, qw, sx, sy, sz          (10 floats = 40 bytes per instance)
already converted to Unreal's frame:  p_ue = M p * 100,  R_ue = M R M  with M = diag(1, -1, 1)
(the FBX import mirrors Blender's Y the same way, so the instanced mesh lands exactly where Blender put it).
"""
from __future__ import annotations

import math
import os

import numpy as np

M = np.diag([1.0, -1.0, 1.0])


def frames_from_normals(n, rng, up_blend=0.0):
    """Rotation matrices whose z axis is the (optionally up-blended) normal, with a random yaw about it."""
    n = np.asarray(n, np.float64)
    z = n * (1 - up_blend) + np.array([0, 0, 1.0]) * up_blend
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    ref = np.where(np.abs(z[:, 2:3]) < 0.95, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    x = np.cross(ref, z)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = np.cross(z, x)
    a = rng.uniform(0, 2 * math.pi, len(z))
    c, s = np.cos(a)[:, None], np.sin(a)[:, None]
    x2 = x * c + y * s
    y2 = -x * s + y * c
    return np.stack([x2, y2, z], axis=2)                       # columns = local axes


def save_set(path, pos, R, scale, variant, tint=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pos = np.asarray(pos, np.float32)
    n = len(pos)
    scale = np.asarray(scale, np.float32)
    if scale.ndim == 1:
        scale = np.repeat(scale[:, None], 3, 1)
    tint = np.random.default_rng(len(pos)).random(n).astype(np.float32) if tint is None else np.asarray(tint, np.float32)
    np.savez_compressed(path, pos=pos, R=np.asarray(R, np.float32), scale=scale, variant=np.asarray(variant, np.int16), tint=tint)


def load_set(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def quat_from_matrix(R):
    """Vectorised rotation matrix (N,3,3) -> quaternion (N,4) as (x, y, z, w), column-vector convention."""
    R = np.asarray(R, np.float64)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((len(R), 4))
    c0 = tr > 0
    s = np.sqrt(np.maximum(tr[c0] + 1.0, 1e-12)) * 2
    q[c0] = np.stack([(m21[c0] - m12[c0]) / s, (m02[c0] - m20[c0]) / s, (m10[c0] - m01[c0]) / s, 0.25 * s], 1)
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.maximum(1.0 + m00[c1] - m11[c1] - m22[c1], 1e-12)) * 2
    q[c1] = np.stack([0.25 * s, (m01[c1] + m10[c1]) / s, (m02[c1] + m20[c1]) / s, (m21[c1] - m12[c1]) / s], 1)
    c2 = ~c0 & ~c1 & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11[c2] - m00[c2] - m22[c2], 1e-12)) * 2
    q[c2] = np.stack([(m01[c2] + m10[c2]) / s, 0.25 * s, (m12[c2] + m21[c2]) / s, (m02[c2] - m20[c2]) / s], 1)
    c3 = ~c0 & ~c1 & ~c2
    s = np.sqrt(np.maximum(1.0 + m22[c3] - m00[c3] - m11[c3], 1e-12)) * 2
    q[c3] = np.stack([(m02[c3] + m20[c3]) / s, (m12[c3] + m21[c3]) / s, 0.25 * s, (m10[c3] - m01[c3]) / s], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def matrix_from_quat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


def ue_records(pos, R, scale):
    """(N,10) float32: UE location (cm), quaternion (x, y, z, w) and scale, in Unreal's mirrored frame."""
    pos = np.asarray(pos, np.float64)
    R = np.asarray(R, np.float64)
    loc = (pos * 100.0) * np.array([1.0, -1.0, 1.0])
    Rue = M[None] @ R @ M[None]
    q = quat_from_matrix(Rue)
    sc = np.asarray(scale, np.float64)
    if sc.ndim == 1:
        sc = np.repeat(sc[:, None], 3, 1)
    return np.concatenate([loc, q, sc], 1).astype("<f4")


def euler_xyz(R):
    """Blender 'XYZ' Euler angles (rotation = Rz @ Ry @ Rx) from rotation matrices, vectorised."""
    R = np.asarray(R, np.float64)
    sy = -R[:, 2, 0]
    y = np.arcsin(np.clip(sy, -1, 1))
    cy = np.cos(y)
    ok = np.abs(cy) > 1e-6
    x = np.where(ok, np.arctan2(R[:, 2, 1], R[:, 2, 2]), np.arctan2(-R[:, 1, 2], R[:, 1, 1]))
    z = np.where(ok, np.arctan2(R[:, 1, 0], R[:, 0, 0]), 0.0)
    return np.stack([x, y, z], 1)
