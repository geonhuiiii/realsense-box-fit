"""Oriented Bounding Box volume estimation via Nelder-Mead simplex.

Vendored to keep this repo standalone (depends only on numpy + scipy).
Originally from the RMC-Visual-Agent-for-Moving-Quote project
(src/volume/obb_nelder_mead.py).

Given a point cloud `P` (N x 3), we want the smallest axis-aligned bounding box
`AABB(R_xyz · P)` after rotating by Euler angles (rx, ry, rz). Its volume is an
upper bound on the object's volume. We minimize this volume using `scipy.optimize.minimize`
with the Nelder-Mead simplex.

Why Nelder-Mead instead of PCA or an exact O(N^3) min-volume algorithm?
* PCA is fast but not scale-invariant and tends to overestimate when the
  point cloud is partial or noisy (common after single-view segmentation).
* The O'Rourke algorithm is exact in 3D but O(N^3) and painful to implement.
* Nelder-Mead on 3 Euler angles has a tiny search space (SO(3) is 3-dim),
  the AABB-volume objective is piecewise-smooth and deterministic, so the
  simplex converges in ~50-200 evaluations.

The optimizer is seeded from a PCA estimate, and then restarted from a few
perturbations to escape obvious local minima (face-parallel vs diagonal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _euler_to_rotmat(angles: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ Euler angles -> 3x3 rotation matrix.

    Uses the ZYX convention applied as R = Rz @ Ry @ Rx. Pure Python / numpy,
    no scipy.spatial.transform dependency so this function is easy to JIT.
    """
    rx, ry, rz = float(angles[0]), float(angles[1]), float(angles[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _aabb_extent(points: np.ndarray) -> np.ndarray:
    return points.max(axis=0) - points.min(axis=0)


def _pca_initial_angles(points: np.ndarray) -> np.ndarray:
    """Initial Euler-angle guess from PCA of the point cloud."""
    c = points.mean(axis=0)
    centered = points - c
    cov = centered.T @ centered / max(1, len(centered) - 1)
    # eigenvectors ordered by increasing eigenvalue; flip to descending.
    _, vecs = np.linalg.eigh(cov)
    R = vecs[:, ::-1]  # columns = principal axes
    # Ensure right-handed
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    # R = Rz @ Ry @ Rx  =>  recover (rx, ry, rz)
    # Using standard extraction for ZYX intrinsic
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = np.arcsin(sy)
    if np.isclose(np.cos(ry), 0.0, atol=1e-6):
        rx = 0.0
        rz = np.arctan2(-R[0, 1], R[1, 1])
    else:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    # We want the rotation that *aligns* points to axes, i.e. R^T. Pass its angles.
    return -np.array([rx, ry, rz], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class OBBResult:
    volume: float                           # meters^3
    extent: np.ndarray                      # (3,) side lengths (m)
    center: np.ndarray                      # (3,) world-space center
    rotation: np.ndarray                    # (3,3) R such that box = R @ AABB + center
    euler_xyz: np.ndarray                   # (3,) optimized Euler angles
    history: list = field(default_factory=list)   # optimization trace (optional)
    n_points: int = 0
    n_fev: int = 0

    def corners(self) -> np.ndarray:
        """Return the 8 corners of the OBB in world space, shape (8,3)."""
        hx, hy, hz = self.extent / 2.0
        local = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
        ])
        return (self.rotation @ local.T).T + self.center

    def as_dict(self) -> dict:
        return {
            "volume_m3": float(self.volume),
            "extent_m": [float(x) for x in self.extent],
            "center_m": [float(x) for x in self.center],
            "rotation": self.rotation.tolist(),
            "euler_xyz_rad": [float(x) for x in self.euler_xyz],
            "n_points": int(self.n_points),
            "n_fev": int(self.n_fev),
        }


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _aabb_volume_objective(angles: np.ndarray, points_c: np.ndarray) -> float:
    """Volume of the axis-aligned bounding box of the rotated (centered) cloud."""
    R = _euler_to_rotmat(angles)
    rotated = points_c @ R.T
    extent = _aabb_extent(rotated)
    # Add a tiny regularizer to keep the simplex numerically well-behaved when
    # the cloud is degenerate (e.g., a plane): otherwise volume can be ~0 with
    # no gradient signal.
    return float(extent[0] * extent[1] * extent[2]) + 1e-9 * float(np.sum(extent))


def _remove_nn_outliers(pts: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Remove points whose nearest-neighbor distance exceeds `factor` × mean.

    Light per-point trim used by the refit endpoint after the user clips the
    OBB to a tighter `max_extent` — there's no longer a ghost-cluster failure
    mode at that point, just stragglers near the clip boundary.
    """
    from scipy.spatial import cKDTree

    if pts.shape[0] < 10:
        return pts
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)          # k=2: self + nearest
    nn_dist = dists[:, 1]                     # skip self-distance (0)
    mean_d = float(nn_dist.mean())
    if mean_d < 1e-12:
        return pts
    keep = nn_dist <= factor * mean_d
    cleaned = pts[keep]
    if cleaned.shape[0] < 4:                  # safety: don't drop everything
        return pts
    return cleaned


def _voxel_cc_filter(
    pts: np.ndarray,
    voxel_divisor: float = 80.0,
    min_component_frac: float = 0.05,
    target_n: int = 10000,
    rng_seed: int = 0,
) -> np.ndarray:
    """Voxel-grid connected-components outlier filter for OBB fitting.

    1) Voxel-downsample with `voxel = max(bbox_diag / voxel_divisor, 2 × mean_nn)`.
       The mean_nn lower bound keeps the grid dense enough that 26-neighbor
       voxels actually connect — without it, sparse clouds fragment into a
       cloud of single-voxel islands and the filter destroys the geometry.
    2) Find 26-neighbor connected components on the voxel grid.
    3) Keep voxels whose component is ≥ `min_component_frac` × total voxels.
       Disconnected ghost blobs from completion (ODGNet/PoinTr/mirror) get
       cleanly dropped here — this was the dominant cause of inflated OBBs.
    4) Map kept voxels back to ORIGINAL points (preserves the true cloud
       boundary; voxel centroids would systematically shrink it by ~½ voxel).
    5) Randomly subsample to `target_n` to bound the per-fev cost in OBB.
       Boundary points dominate the AABB extent and are still well-represented
       at 10K from any cloud size.

    Effect on cost: replaces the O(N · n_fev) reduce loop with one shot of
    voxel-CC pre-processing (~80–100 ms for 100k pts) plus an OBB fit on at
    most `target_n` points (~150–250 ms for n_restarts=4).
    """
    from scipy.spatial import cKDTree

    if pts.shape[0] < 10:
        return pts

    extent = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(extent))
    if diag < 1e-9:
        return pts

    # Density-adaptive voxel size: 2 × mean_nn is the smallest voxel that still
    # guarantees adjacent points share or 26-neighbor adjacent voxels.
    sample_n = min(5000, len(pts))
    rng = np.random.default_rng(rng_seed)
    if len(pts) > sample_n:
        sample = pts[rng.choice(len(pts), sample_n, replace=False)]
    else:
        sample = pts
    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=2)
    mean_nn = float(d[:, 1].mean())
    voxel = max(diag / voxel_divisor, 2.0 * mean_nn)
    if voxel < 1e-9:
        return pts

    # Voxelize: each point gets a (kx, ky, kz) lattice key.
    keys = np.floor(pts / voxel).astype(np.int64)
    voxel_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    n_voxels = len(voxel_keys)

    # 26-neighbor connected components on the voxel set.
    key_to_idx = {(int(k[0]), int(k[1]), int(k[2])): i
                  for i, k in enumerate(voxel_keys)}
    labels = np.full(n_voxels, -1, dtype=np.int64)
    cur = 0
    for start in range(n_voxels):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = cur
        while stack:
            i = stack.pop()
            kx, ky, kz = (int(voxel_keys[i, 0]),
                          int(voxel_keys[i, 1]),
                          int(voxel_keys[i, 2]))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        j = key_to_idx.get((kx + dx, ky + dy, kz + dz), -1)
                        if j >= 0 and labels[j] < 0:
                            labels[j] = cur
                            stack.append(j)
        cur += 1

    comp_sizes = np.bincount(labels.astype(np.intp))
    threshold = max(1, int(min_component_frac * n_voxels))
    keep_labels = np.where(comp_sizes >= threshold)[0]
    if keep_labels.size == 0:
        keep_labels = np.array([int(np.argmax(comp_sizes))])

    voxel_kept = np.isin(labels, keep_labels)
    pts_kept_mask = voxel_kept[inv]
    cleaned = pts[pts_kept_mask]

    if cleaned.shape[0] > target_n:
        idx = rng.choice(cleaned.shape[0], target_n, replace=False)
        cleaned = cleaned[idx]

    if cleaned.shape[0] < 4:
        return pts
    return cleaned


def estimate_obb_volume(
    points: np.ndarray,
    n_restarts: int = 4,
    max_iter: int = 300,
    xatol: float = 1e-3,
    fatol: float = 1e-5,
    record_history: bool = False,
    rng_seed: int = 0,
    outlier_filter: bool = True,
    voxel_divisor: float = 80.0,
    min_component_frac: float = 0.05,
    target_n: int = 10000,
) -> OBBResult:
    """Estimate OBB volume of a point cloud using scipy's Nelder-Mead.

    Args:
        points: (N,3) float array in meters.
        n_restarts: how many random perturbations of the PCA seed to try.
            The best result across all restarts is returned.
        max_iter: Nelder-Mead `maxiter`.
        xatol, fatol: convergence tolerances for Nelder-Mead.
        record_history: record per-iteration volume for debugging / plots.
        rng_seed: seed for the restart perturbations.
        outlier_filter: run voxel-CC keep-largest pre-filter to drop
            disconnected ghost clusters from completion. See
            `_voxel_cc_filter` for the full algorithm.
        voxel_divisor, min_component_frac, target_n: tuning knobs for the
            pre-filter. Defaults are tuned on completion outputs (8k–100k pts,
            object-scale 0.5–3 m).

    Returns:
        OBBResult with volume, extent, center, rotation, and trace info.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 4:
        raise ValueError(f"Need at least 4 points; got {pts.shape[0]}")

    # --- pre-filter: drop ghost clusters + cap point count for fast OBB ----
    if outlier_filter:
        pts = _voxel_cc_filter(
            pts,
            voxel_divisor=voxel_divisor,
            min_component_frac=min_component_frac,
            target_n=target_n,
            rng_seed=rng_seed,
        )

    center = pts.mean(axis=0)
    centered = pts - center

    # Seed from PCA, then add perturbed restarts.
    rng = np.random.default_rng(rng_seed)
    seeds = [_pca_initial_angles(pts)]
    for _ in range(max(0, n_restarts - 1)):
        seeds.append(seeds[0] + rng.uniform(-np.pi / 6, np.pi / 6, size=3))

    best: Optional[tuple[float, np.ndarray, int]] = None  # (volume, angles, n_fev)
    full_history: list = []
    total_fev = 0

    for seed in seeds:
        history: list[float] = []
        callback = None
        if record_history:
            def _cb(xk, _h=history, _pts=centered):
                _h.append(_aabb_volume_objective(xk, _pts))
            callback = _cb

        res = minimize(
            _aabb_volume_objective,
            seed,
            args=(centered,),
            method="Nelder-Mead",
            options={
                "xatol": xatol,
                "fatol": fatol,
                "maxiter": max_iter,
                "adaptive": True,   # better scaling for 3-D simplex
                "disp": False,
            },
            callback=callback,
        )
        total_fev += res.nfev
        full_history.append(history)
        vol = float(res.fun)
        if best is None or vol < best[0]:
            best = (vol, np.asarray(res.x, dtype=np.float64), int(res.nfev))

    assert best is not None
    vol, angles, _ = best

    R = _euler_to_rotmat(angles)
    rotated = centered @ R.T
    min_ = rotated.min(axis=0)
    max_ = rotated.max(axis=0)
    extent = max_ - min_
    # The local-frame center of the AABB relative to the rotated cloud
    aabb_center_local = (min_ + max_) / 2.0
    # Lift back to world space
    world_center = center + (R.T @ aabb_center_local)

    return OBBResult(
        volume=float(extent.prod()),
        extent=extent.astype(np.float64),
        center=world_center.astype(np.float64),
        rotation=R.T.astype(np.float64),   # box-local -> world
        euler_xyz=angles,
        history=full_history if record_history else [],
        n_points=int(pts.shape[0]),
        n_fev=total_fev,
    )
