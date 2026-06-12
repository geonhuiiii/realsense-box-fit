"""Depth back-projection + Nelder-Mead OBB wrapper.

Uses the vendored Nelder-Mead OBB implementation (`obb_nelder_mead.py`,
numpy+scipy only) so this package is self-contained.
"""
from __future__ import annotations

import numpy as np

from obb_nelder_mead import estimate_obb_volume


def backproject(
    depth_u16: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    confidence: np.ndarray | None = None,
    conf_thresh: float = 0.0,
    z_min_m: float = 0.1,
    z_max_m: float = 12.0,
) -> np.ndarray:
    """Back-project masked, valid depth pixels into a camera-frame point cloud (m).

    Returns (N,3) float64 array. Camera/OpenCV frame: x right, y down, z forward.
    """
    H, W = depth_u16.shape
    valid = mask & (depth_u16 > 0)
    if confidence is not None and conf_thresh > 0:
        valid = valid & (confidence >= conf_thresh)

    vs, us = np.nonzero(valid)
    z = depth_u16[vs, us].astype(np.float64) * depth_scale
    keep = (z >= z_min_m) & (z <= z_max_m)
    us, vs, z = us[keep], vs[keep], z[keep]

    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def scene_background_mask(
    depth_u16: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    max_planes: int = 4,
    dist_thresh_m: float = 0.02,
    min_plane_frac: float = 0.08,
    min_diag_m: float = 1.0,
) -> np.ndarray:
    """Return HxW bool mask of pixels on large background planes (floor/walls).

    Iteratively RANSAC-segments the dominant planes of the full scene cloud and
    flags them as background. A plane is only treated as background if it holds
    >= `min_plane_frac` of points AND spans >= `min_diag_m` (so compact object
    faces like a mattress top are not stripped).
    """
    import open3d as o3d

    H, W = depth_u16.shape
    vs, us = np.nonzero(depth_u16 > 0)
    z = depth_u16[vs, us].astype(np.float64) * depth_scale
    keep = z > 0.1
    us, vs, z = us[keep], vs[keep], z[keep]
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    n_total = len(pts)

    bg = np.zeros((H, W), dtype=bool)
    if n_total < 1000:
        return bg

    work_pts = pts
    work_v = vs
    work_u = us
    for _ in range(max_planes):
        if len(work_pts) < 1000:
            break
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(work_pts)
        _, inliers = p.segment_plane(dist_thresh_m, 3, 1000)
        inliers = np.asarray(inliers)
        if len(inliers) < min_plane_frac * n_total:
            break
        plane_pts = work_pts[inliers]
        diag = float(np.linalg.norm(plane_pts.max(0) - plane_pts.min(0)))
        if diag < min_diag_m:
            break
        bg[work_v[inliers], work_u[inliers]] = True
        keep_mask = np.ones(len(work_pts), dtype=bool)
        keep_mask[inliers] = False
        work_pts = work_pts[keep_mask]
        work_v = work_v[keep_mask]
        work_u = work_u[keep_mask]
    return bg


def obb_dims_cm(points_m: np.ndarray, n_restarts: int = 4) -> tuple[list[float], float, int]:
    """Run Nelder-Mead OBB; return (sorted dims in cm, volume cm^3, n_points_used)."""
    res = estimate_obb_volume(points_m, n_restarts=n_restarts)
    extent_cm = sorted(float(e) * 100.0 for e in res.extent)
    vol_cm3 = float(res.volume) * 1e6
    return extent_cm, vol_cm3, int(res.n_points)
