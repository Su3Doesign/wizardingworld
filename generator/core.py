"""core - the hero terrain round the castle (1.2 x 1.2 km): height field + granite SDF -> marching cubes -> cleanup ->
decimation -> per-vertex masks.  The outer valley and the mountains are a plain height-field mesh (terrain.py) that
overlaps this one by a few metres under a small downward taper (no visible seam, no z-fighting).

Masks per vertex (the Unreal terrain material and the Cycles preview read them from the UV channels):
    UV1 = (rock, moss)      UV2 = (forest floor, shore / wet)      UV3 = (road dirt, ambient occlusion)
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

import rockfield as rf
import world as W
from meshkit import Mesh

REGION = dict(x=(-600.0, 600.0), y=(-520.0, 680.0))
EDGE_TAPER = 6.0           # m: the core surface dips below the outer terrain within this distance of its border
EDGE_DIP = 0.45            # m


# ----------------------------------------------------------------------------------------------- 2D column fields
class Columns:
    """H, grad H, rock weight and border offset on the voxel-aligned 2D grid of the region."""

    def __init__(self, vox, region=REGION, pad=3):
        t = time.time()
        self.vox = vox
        x0, x1 = region["x"]
        y0, y1 = region["y"]
        self.nx = int(round((x1 - x0) / vox)) + 1 + 2 * pad
        self.ny = int(round((y1 - y0) / vox)) + 1 + 2 * pad
        self.x0 = x0 - pad * vox
        self.y0 = y0 - pad * vox
        xs = self.x0 + vox * np.arange(self.nx)
        ys = self.y0 + vox * np.arange(self.ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        F = W.Fields(X, Y)
        self.F = F
        H = F.height().reshape(self.nx, self.ny)
        H = ndimage.gaussian_filter(H, 0.6)                   # remove sub-voxel kinks of the analytic min/max operations
        gx, gy = np.gradient(H, vox, vox)
        G = np.hypot(gx, gy)
        # rock weight: steep ground, spread a little onto the rims; none deep inside the castle plateau
        steep = W.sstep(0.50, 1.30, G)
        cd = F.crag().reshape(self.nx, self.ny)
        inner = W.sstep(-12.0, -3.0, cd)                      # the castle plateau stays flat beyond ~12 m from the rim
        spread = ndimage.maximum_filter(steep, size=int(5 / vox) | 1) * 0.85 * inner
        steep = ndimage.gaussian_filter(np.maximum(steep, spread), 2.2 / vox)
        lakebed = W.sstep(-30.0, -6.0, H)                       # deep lake floor: smooth
        w = np.clip(steep, 0, 1) * lakebed
        # taper the rock and dip the surface near the region border (the outer terrain covers that band)
        de = np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y])
        w = w * W.sstep(0.0, 25.0, de)
        off = EDGE_DIP * W.sstep(EDGE_TAPER, 0.0, de)
        self.H, self.gx, self.gy, self.G, self.w, self.off = H, gx, gy, G, w, off
        self.X, self.Y = X, Y
        print(f"  columns {self.nx}x{self.ny} @ {vox} m in {time.time() - t:.1f}s; H {H.min():.1f} .. {H.max():.1f}, "
              f"rock cover {100 * (w > 0.2).mean():.1f}%", flush=True)

    def sample(self, x, y, arr):
        ci = (np.asarray(x) - self.x0) / self.vox
        cj = (np.asarray(y) - self.y0) / self.vox
        return ndimage.map_coordinates(arr, [ci, cj], order=1, mode="nearest")

    def sdf(self, P, params=rf.DEFAULT):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        return rf.eval_points(x, y, z, self.sample(x, y, self.H), self.sample(x, y, self.gx), self.sample(x, y, self.gy),
                              self.sample(x, y, self.w), self.sample(x, y, self.off), params)


# ----------------------------------------------------------------------------------------------- SDF volume + marching cubes
class Volume:
    """Global signed-distance volume over the region (float32, voxel lattice aligned with the columns).

    Built tile by tile: the base terrain solid (z < H) is voxelised, exact Euclidean distance transforms give the
    signed distance to it AND the nearest surface column of every voxel; the granite displacement R is evaluated in a
    band round the surface with the nearest column's rock weight and face frame (so cliffs recede and bulge along their
    true normal, right up to the rim).  Smooth ground keeps the analytic height-field distance (no voxel staircase)."""

    def __init__(self, cols: Columns, params=rf.DEFAULT, band0=1.6, bandw=24.0, tile=160, margin=40):
        from scipy.ndimage import distance_transform_edt as edt

        t = time.time()
        self.cols = cols
        vox = cols.vox
        self.vox = vox
        self.k0 = int(math.floor((cols.H.min() - band0 - bandw - 4.0) / vox))
        k1 = int(math.ceil((cols.H.max() + band0 + bandw + 4.0) / vox))
        self.z0 = self.k0 * vox
        self.nz = k1 - self.k0 + 1
        nx, ny = cols.nx, cols.ny
        bmax = band0 + bandw
        self.band_max = bmax
        self.vol = np.empty((nx, ny, self.nz), np.float32)
        zs_all = self.z0 + vox * np.arange(self.nz)
        nrm_all = np.sqrt(1.0 + cols.G * cols.G)
        tiles = [(i, j) for i in range(0, nx, tile) for j in range(0, ny, tile)]
        n_eval = 0
        for ti, (i0, j0) in enumerate(tiles):
            i1, j1 = min(nx, i0 + tile), min(ny, j0 + tile)
            a0, a1 = max(0, i0 - margin), min(nx, i1 + margin)
            b0, b1 = max(0, j0 - margin), min(ny, j1 + margin)
            Ht = cols.H[a0:a1, b0:b1]
            kk0 = max(0, int(math.floor((Ht.min() - bmax - 2.0 - self.z0) / vox)))
            kk1 = min(self.nz, int(math.ceil((Ht.max() + bmax + 2.0 - self.z0) / vox)) + 1)
            zs = zs_all[kk0:kk1]
            occ = zs[None, None, :] < Ht[:, :, None]
            # inner tile slice (relative to the margin block)
            si, sj = slice(i0 - a0, i1 - a0), slice(j0 - b0, j1 - b0)
            if occ.all() or not occ.any():
                inner = self.vol[i0:i1, j0:j1, :]
                inner[...] = -bmax if occ.all() else bmax
                continue
            d_out, ind_out = edt(~occ, sampling=vox, return_indices=True)
            d_in, ind_in = edt(occ, sampling=vox, return_indices=True)
            occ_i = occ[si, sj]
            full = np.where(occ, -(d_in - 0.5 * vox), d_out - 0.5 * vox).astype(np.float32)
            full = ndimage.gaussian_filter(full, 1.0)                 # soften the voxel staircase of the binary solid
            sdf_edt = full[si, sj]
            del full
            ni = np.where(occ_i, ind_in[0][si, sj], ind_out[0][si, sj]) + a0          # nearest surface column (global)
            nj = np.where(occ_i, ind_in[1][si, sj], ind_out[1][si, sj]) + b0
            del d_out, ind_out, d_in, ind_in, occ
            # analytic height-field distance on smooth ground, EDT distance where the nearest surface is steep
            Hi = cols.H[i0:i1, j0:j1][:, :, None]
            d0 = (zs[None, None, :] - Hi) / nrm_all[i0:i1, j0:j1][:, :, None]
            Gn = cols.G[ni, nj]
            alpha = W.sstep(0.8, 1.6, Gn)
            base = alpha * sdf_edt + (1.0 - alpha) * d0
            base += cols.off[i0:i1, j0:j1][:, :, None]
            wn = cols.w[ni, nj]
            band = band0 + bandw * wn
            sel = (np.abs(base) < band) & (wn > 1e-4)
            if sel.any():
                ii, jj, kk = np.nonzero(sel)
                X = cols.x0 + vox * (ii + i0)
                Y = cols.y0 + vox * (jj + j0)
                Z = zs[kk]
                R = rf.rock_points(X, Y, Z, cols.gx[ni[sel], nj[sel]], cols.gy[ni[sel], nj[sel]], params)
                base[sel] += wn[sel] * R
                n_eval += len(R)
            np.clip(base, -bmax, bmax, out=base)
            blk = np.empty((i1 - i0, j1 - j0, self.nz), np.float32)
            blk[:, :, :kk0] = -bmax
            blk[:, :, kk1:] = bmax
            blk[:, :, kk0:kk1] = base
            self.vol[i0:i1, j0:j1, :] = blk
            if ti % 10 == 0:
                print(f"    sdf tile {ti + 1}/{len(tiles)}  z {zs[0]:.0f}..{zs[-1]:.0f}  rock evals {n_eval:,}  ({time.time() - t:.0f}s)", flush=True)
        print(f"  SDF volume {nx}x{ny}x{self.nz} ({self.vol.nbytes / 1e9:.1f} GB) in {time.time() - t:.0f}s, {n_eval:,} rock evaluations", flush=True)

    def sample(self, P):
        """Trilinear SDF lookup at points (m)."""
        c = self.cols
        ci = (P[:, 0] - c.x0) / self.vox
        cj = (P[:, 1] - c.y0) / self.vox
        ck = (P[:, 2] - self.z0) / self.vox
        return ndimage.map_coordinates(self.vol, [ci, cj, ck], order=1, mode="nearest")

    def march(self, block=200):
        from skimage.measure import marching_cubes

        t = time.time()
        vox = self.vox
        c = self.cols
        Vs, Fs = [], []
        nv = 0
        for i0 in range(0, c.nx - 1, block):
            for j0 in range(0, c.ny - 1, block):
                i1, j1 = min(c.nx, i0 + block + 1), min(c.ny, j0 + block + 1)
                sub = self.vol[i0:i1, j0:j1]
                # restrict z to where the surface can be
                lay_max = sub.max(axis=(0, 1))
                lay_min = sub.min(axis=(0, 1))
                air = np.nonzero(lay_max > 0)[0]
                rock = np.nonzero(lay_min < 0)[0]
                if len(air) == 0 or len(rock) == 0:
                    continue
                k0 = max(0, air[0] - 1)                  # the surface can lie between two layers (flat ground)
                k1 = min(self.nz, rock[-1] + 2)
                if k1 - k0 < 2:
                    continue
                v, f, _, _ = marching_cubes(sub[:, :, k0:k1], level=0.0, spacing=(vox, vox, vox), allow_degenerate=False)
                v[:, 0] += c.x0 + i0 * vox
                v[:, 1] += c.y0 + j0 * vox
                v[:, 2] += self.z0 + k0 * vox
                Vs.append(v.astype(np.float64))
                Fs.append(f.astype(np.int64) + nv)
                nv += len(v)
        V = np.concatenate(Vs)
        F = np.concatenate(Fs)
        key = np.round(V / 1e-4).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        V = V[first]
        F = inv[F]
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[keep]
        print(f"  marching cubes: {len(F):,} tris, {len(V):,} verts in {time.time() - t:.0f}s", flush=True)
        return V, F


def adjacency(F, nv):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    return A


def drop_small(V, F, min_faces=1500):
    A = adjacency(F, len(V))
    n, lab = connected_components(A, directed=False)
    fl = lab[F[:, 0]]
    counts = np.bincount(fl, minlength=n)
    keep = counts[fl] >= min_faces
    F = F[keep]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    print(f"  components: {n}, kept {int((counts >= min_faces).sum())}, removed {int((~keep).sum()):,} tris", flush=True)
    return V[used], remap[F]


def taubin(V, F, iters=3, lam=0.5, mu=-0.53, fixed=None):
    A = adjacency(F, len(V))
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    Dinv = sparse.diags(1.0 / deg)
    L = Dinv @ A
    for _ in range(iters):
        for k in (lam, mu):
            d = L @ V - V
            if fixed is not None:
                d[fixed] = 0
            V = V + k * d
    return V


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def decimate(V, F, target, protect_border=True):
    import fast_simplification as fs

    red = 1.0 - target / len(F)
    if red <= 0:
        return V, F
    t = time.time()
    V2, F2 = fs.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=red, agg=7.0, verbose=False)
    print(f"  decimated {len(F):,} -> {len(F2):,} tris in {time.time() - t:.0f}s", flush=True)
    return V2.astype(np.float64), F2.astype(np.int64)


# ----------------------------------------------------------------------------------------------- masks
def masks(V, N, cols: Columns, rock_w=None, volume=None):
    """Per-vertex material masks (see module doc)."""
    F = W.Fields(V[:, 0], V[:, 1])
    z = V[:, 2]
    nz = N[:, 2]
    w = cols.sample(V[:, 0], V[:, 1], cols.w) if rock_w is None else rock_w
    # rock: steep faces + the granite region; broken by noise so rock shows through thin soil on moderate slopes
    nb = F.n(18.0, 3, 101)
    rock = np.clip(np.maximum(W.sstep(0.80, 0.55, nz + 0.12 * nb), w * W.sstep(0.93, 0.72, nz)), 0, 1)
    # moss on the rock: ledges / tops (up-facing), damp gorge walls (low, near water), patchy
    up = W.sstep(0.25, 0.75, nz)
    damp = W.sstep(60.0, 10.0, z - np.maximum(W.gorge_bed(F.gorge()[1]), 0)) * W.sstep(-0.2, 0.4, nz)
    patch = W.sstep(-0.35, 0.45, F.n(9.0, 4, 102) + 0.2)
    moss = np.clip((0.85 * up + 0.55 * damp) * patch, 0, 1) * (0.25 + 0.75 * rock)
    forest = F.forest()
    # shore: wet band at the lake and along the river
    gd, gs, _ = F.gorge()
    river_z = np.maximum(W.gorge_bed(gs) + 0.6, W.LAKE_Z)
    rd, rs, _ = F.ravine()
    stream_z = W.ravine_bed(rs) + 0.35
    wet_lake = W.sstep(2.5, 0.2, z - W.LAKE_Z)
    wet_river = W.sstep(2.0, 0.1, z - river_z) * W.sstep(30.0, 8.0, gd)
    wet_stream = W.sstep(1.2, 0.05, z - stream_z) * W.sstep(14.0, 4.0, rd)
    shore = np.clip(np.maximum.reduce([wet_lake, wet_river, wet_stream]), 0, 1)
    road = W.sstep(4.2, 2.2, F.road() + 0.8 * F.n(6.0, 2, 103))
    # ambient occlusion from the SDF: march a few steps along the normal
    ao = np.ones(len(V))
    if volume is not None:
        occ = np.zeros(len(V))
        for dist, wt in ((0.6, 0.30), (1.6, 0.30), (3.5, 0.25), (7.0, 0.15)):
            d = volume.sample(V + N * dist)
            occ += wt * np.clip((dist - d) / dist, 0, 1)
        ao = np.clip(1.0 - 1.1 * occ, 0.15, 1.0)
    return dict(rock=rock, moss=moss, forest=forest, shore=shore, road=road, ao=ao)


def pack(mesh: Mesh, m):
    F = mesh.F
    mesh.uv1 = np.stack([m["rock"], m["moss"]], 1)[F]
    mesh.uv2 = np.stack([m["forest"], m["shore"]], 1)[F]
    mesh.uv3 = np.stack([m["road"], m["ao"]], 1)[F]
    return mesh


# ----------------------------------------------------------------------------------------------- driver
def build(out_dir, vox=0.75, target=3_600_000, tiles=(3, 3)):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    cols = Columns(vox)
    vol = Volume(cols)
    V, F = vol.march()
    V, F = drop_small(V, F)
    V = taubin(V, F, iters=2)
    V, F = decimate(V, F, target)
    N = vertex_normals(V, F)
    m = masks(V, N, cols, volume=vol)
    # split into export tiles (by face centroid); vertices are duplicated along tile seams (identical positions)
    x0, x1 = REGION["x"]
    y0, y1 = REGION["y"]
    C = V[F].mean(1)
    tx = np.clip(((C[:, 0] - x0) / (x1 - x0) * tiles[0]).astype(int), 0, tiles[0] - 1)
    ty = np.clip(((C[:, 1] - y0) / (y1 - y0) * tiles[1]).astype(int), 0, tiles[1] - 1)
    out = []
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            sel = (tx == i) & (ty == j)
            if not sel.any():
                continue
            sub = Mesh(V, F[sel], 0).compact()
            idx = np.unique(F[sel])
            mm = {k: v[idx] for k, v in m.items()}
            sub.uv0 = (sub.V[:, :2] / 4.0)[sub.F]                  # planar world UVs, 4 m per tile (ground layers)
            pack(sub, mm)
            name = f"terrain_core_{i}{j}"
            sub.save(f"{out_dir}/{name}.npz")
            out.append((name, sub.nf))
    print(f"  core terrain: {sum(n for _, n in out):,} tris in {len(out)} tiles ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(f"{out_dir}/core_columns.npz", H=cols.H.astype(np.float32), gx=cols.gx.astype(np.float32), gy=cols.gy.astype(np.float32),
                        w=cols.w.astype(np.float32), off=cols.off.astype(np.float32), x0=cols.x0, y0=cols.y0, vox=cols.vox)
    return out


class ColumnsCache(Columns):
    """Columns restored from core_columns.npz (scatter / castle placement query the same surface without recomputing)."""

    def __init__(self, path):
        z = np.load(path)
        self.H, self.w, self.off = z["H"].astype(np.float64), z["w"].astype(np.float64), z["off"].astype(np.float64)
        self.gx, self.gy = z["gx"].astype(np.float64), z["gy"].astype(np.float64)
        self.G = np.hypot(self.gx, self.gy)
        self.x0, self.y0, self.vox = float(z["x0"]), float(z["y0"]), float(z["vox"])
        self.nx, self.ny = self.H.shape


if __name__ == "__main__":
    out = sys.argv[1]
    vox = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    target = int(float(sys.argv[3])) if len(sys.argv) > 3 else 3_600_000
    build(out, vox, target)
