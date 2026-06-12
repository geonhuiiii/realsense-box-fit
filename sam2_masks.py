"""SAM2 object segmentation — HYBRID.

Primary: GRID SAMPLING. We lay a regular grid of prompt points over the
foreground, ask SAM2 for a mask at each point, then MERGE overlapping masks
(union) instead of NMS-dropping them — so the many partial masks SAM produces
for one object collapse into a single whole object instead of fragmenting. Each
merged mask is restricted to its own depth band so background bleed can't inflate
the OBB.

Fallback: if the grid pass fails to cover the foreground (the typical failure
mode for large low-texture objects like a mattress/curtain, which SAM only grabs
a fragment of), we strip the large background planes, cluster the remaining
foreground in 3D, and prompt SAM2 per cluster — which reliably grabs big
low-texture objects and separates multi-object scenes.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

# SAM2 checkpoint/config. Default resolves to ./models/sam2/... next to the repo
# (where setup_models.sh downloads it) so it works on macOS/Windows without env
# vars. Override when running elsewhere:
#   RBF_SAM2_CKPT=/path/to/sam2_hiera_large.pt
#   RBF_SAM2_CFG=sam2_hiera_l.yaml   (a config on the sam2 hydra search path)
_DEFAULT_CKPT = Path(__file__).resolve().parent / "models" / "sam2" / "sam2_hiera_large.pt"
SAM2_CKPT = os.environ.get("RBF_SAM2_CKPT", str(_DEFAULT_CKPT))
SAM2_CFG = os.environ.get("RBF_SAM2_CFG", "sam2_hiera_l.yaml")

_MODEL = None
_PREDICTOR = None
_LOCK = threading.Lock()        # guard model build (preload thread vs run worker)


def _model():
    """Build the SAM2 model once, shared by predictor + AMG."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:                 # double-checked: only one thread builds
        if _MODEL is None:
            try:
                import torch
                from sam2.build_sam import build_sam2
            except Exception as e:  # ImportError, or a broken torch build (missing libs)
                raise RuntimeError(
                    "SAM2 (and torch) not available in this Python. Run the app with an "
                    "env that has them (install SAM2 + a PyTorch build for your platform — "
                    "CUDA on Linux, the default wheel on macOS for Apple-Silicon MPS/CPU). "
                    f"(original import error: {e})") from e
            from torch_device import pick_device
            device = pick_device(torch)   # cuda → Apple-Silicon mps → cpu
            _MODEL = build_sam2(SAM2_CFG, SAM2_CKPT, device=device,
                                apply_postprocessing=True)
    return _MODEL


def get_predictor():
    """Build (once) and return a SAM2ImagePredictor."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    model = _model()
    with _LOCK:
        if _PREDICTOR is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            _PREDICTOR = SAM2ImagePredictor(model)
    return _PREDICTOR


def preload() -> None:
    """Eagerly build the SAM2 predictor so the first run isn't slow.

    Raises RuntimeError with an actionable message if SAM2/torch/checkpoint is missing.
    """
    from env_check import require_sam2
    require_sam2()
    get_predictor()


def _grid_points(H: int, W: int, fg: np.ndarray, n: int) -> list[tuple]:
    """Regular n×n grid of interior pixel coords, kept only where `fg` is True."""
    ys = np.linspace(0, H - 1, n + 2)[1:-1]
    xs = np.linspace(0, W - 1, n + 2)[1:-1]
    pts = []
    for yy in ys:
        iy = int(round(yy))
        for xx in xs:
            ix = int(round(xx))
            if fg[iy, ix]:
                pts.append((ix, iy))
    return pts


def _merge_overlapping(segs: list[np.ndarray], overlap_thresh: float = 0.5) -> list[np.ndarray]:
    """Union masks that overlap (intersection / smaller area >= thresh).

    Iterated to a fixed point so a chain of partially-overlapping fragments of one
    object all collapse into a single mask. Two distinct objects that merely touch
    (small overlap) stay separate.
    """
    groups = [s.copy() for s in segs]
    changed = True
    while changed:
        changed = False
        merged: list[np.ndarray] = []
        for s in groups:
            sa = int(s.sum())
            hit = -1
            for i, g in enumerate(merged):
                inter = int((s & g).sum())
                if inter and inter / max(1, min(sa, int(g.sum()))) >= overlap_thresh:
                    hit = i
                    break
            if hit >= 0:
                merged[hit] = merged[hit] | s
                changed = True
            else:
                merged.append(s.copy())
        groups = merged
    return groups


def _restrict_depth_band(seg: np.ndarray, depth_u16: np.ndarray,
                         frac: float = 0.30) -> np.ndarray:
    """Keep only mask pixels within +-frac of the mask's median depth.

    Removes background bleed (e.g. a mask that leaks onto a far wall), which
    would otherwise blow up the OBB along the camera axis.
    """
    vd = seg & (depth_u16 > 0)
    if int(vd.sum()) < 50:
        return seg
    med = float(np.median(depth_u16[vd]))
    lo, hi = med * (1 - frac), med * (1 + frac)
    return seg & (depth_u16 > 0) & (depth_u16 >= lo) & (depth_u16 <= hi)


def _foreground_clusters(
    depth_u16: np.ndarray,
    bg_mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, scale: float,
    eps_m: float = 0.06,
    min_points: int = 400,
    min_cluster_frac: float = 0.03,
) -> list[dict]:
    """3D-cluster the foreground (valid depth & not background plane).

    Returns clusters (largest first), each: {'pixels': (v,u) arrays,
    'centroid_uv': (u,v), 'n': int, 'depth_med_m': float}.
    """
    import open3d as o3d

    fg = (depth_u16 > 0) & (~bg_mask)
    vs, us = np.nonzero(fg)
    if len(vs) < min_points:
        return []
    z = depth_u16[vs, us].astype(np.float64) * scale
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps_m, min_points=30))

    clusters = []
    n_fg = len(vs)
    for lab in np.unique(labels):
        if lab < 0:
            continue
        sel = labels == lab
        n = int(sel.sum())
        if n < max(min_points, int(min_cluster_frac * n_fg)):
            continue
        cu, cv = us[sel], vs[sel]
        clusters.append({
            "u": cu, "v": cv, "n": n,
            "centroid_uv": (float(cu.mean()), float(cv.mean())),
            "depth_med_m": float(np.median(z[sel])),
        })
    clusters.sort(key=lambda c: c["n"], reverse=True)
    return clusters


def _seed_points(cluster: dict, k: int = 6, rng_seed: int = 0) -> np.ndarray:
    """Pick k positive prompt points inside the cluster (centroid + interior)."""
    u, v = cluster["u"], cluster["v"]
    rng = np.random.default_rng(rng_seed)
    cu, cv = cluster["centroid_uv"]
    pts = [(cu, cv)]
    if len(u) > k:
        idx = rng.choice(len(u), k - 1, replace=False)
        pts.extend([(float(u[i]), float(v[i])) for i in idx])
    else:
        pts.extend([(float(u[i]), float(v[i])) for i in range(len(u))])
    return np.array(pts, dtype=np.float32)


def _cluster_objects(
    rgb: np.ndarray,
    depth_u16: np.ndarray,
    bg_mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, scale: float,
    max_objects: int = 8,
    containment_thresh: float = 0.85,
) -> list[dict]:
    """FALLBACK: cluster foreground in 3D, prompt SAM2 per cluster.

    Each returned item: {'segmentation','area','area_frac','n_seed',
                         'depth_median_m'}.
    """
    H, W = depth_u16.shape
    img_area = H * W
    clusters = _foreground_clusters(depth_u16, bg_mask, fx, fy, cx, cy, scale)
    if not clusters:
        return []

    import cv2

    predictor = get_predictor()
    predictor.set_image(rgb)
    fg = (depth_u16 > 0) & (~bg_mask)
    depth_m = depth_u16.astype(np.float64) * scale

    cands = []
    for ci, cl in enumerate(clusters[:max_objects]):
        # Cluster pixel mask, dilated to let SAM2 refine the boundary a little
        # but NOT jump to far regions of the image.
        cl_mask = np.zeros((H, W), dtype=bool)
        cl_mask[cl["v"], cl["u"]] = True
        cl_dil = cv2.dilate(cl_mask.astype(np.uint8),
                            np.ones((25, 25), np.uint8)).astype(bool)
        # Object depth slab from the cluster's own depth distribution.
        zc = depth_m[cl["v"], cl["u"]]
        z_lo, z_hi = np.percentile(zc, 1) - 0.10, np.percentile(zc, 99) + 0.10
        band = (depth_m >= z_lo) & (depth_m <= z_hi)

        seeds = _seed_points(cl)
        labels = np.ones(len(seeds), dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=seeds, point_labels=labels, multimask_output=True)
        # Pick the predicted mask with best cluster coverage (constrained later).
        best, best_key = None, -1.0
        for mi in range(masks.shape[0]):
            seg = masks[mi].astype(bool)
            ov = int((seg & cl_mask).sum()) / max(1, cl["n"])
            key = ov * float(scores[mi])
            if key > best_key:
                best_key, best = key, seg

        # Constrain SAM2 mask to this object's region + depth slab + foreground.
        final = best & fg & cl_dil & band if best is not None else None
        if final is None or final.sum() < 0.5 * cl["n"]:
            final = cl_mask  # fall back to the geometric cluster
        area = int(final.sum())
        if area < 200:
            continue
        cands.append({
            "segmentation": final,
            "area": area,
            "area_frac": area / img_area,
            "n_seed": cl["n"],
            "depth_median_m": cl["depth_med_m"],
        })

    cands.sort(key=lambda c: c["area"], reverse=True)
    # Containment NMS: drop masks largely nested inside a larger kept one.
    kept: list[dict] = []
    for c in cands:
        seg = c["segmentation"]
        nested = any(
            int((seg & k["segmentation"]).sum()) / max(1, c["area"]) > containment_thresh
            for k in kept
        )
        if not nested:
            kept.append(c)
    return kept


def _borders_touched(seg: np.ndarray) -> int:
    H, W = seg.shape
    return int(seg[0, :].any()) + int(seg[H - 1, :].any()) + \
        int(seg[:, 0].any()) + int(seg[:, W - 1].any())


def _grid_objects(
    rgb: np.ndarray, depth_u16: np.ndarray, bg_mask: np.ndarray, scale: float,
    points_per_side: int = 16, max_objects: int = 12,
    min_area_frac: float = 0.01, max_area_frac: float = 0.85,
    bg_plane_frac: float = 0.5, merge_overlap: float = 0.5,
) -> list[dict]:
    """PRIMARY: grid-sample prompt points, mask each, MERGE overlapping masks.

    A regular grid of foreground points is prompted into SAM2; every resulting
    mask is background-filtered + depth-banded, then masks that overlap are unioned
    (so one object's many partial masks become a single whole object).
    """
    H, W = depth_u16.shape
    img_area = H * W
    predictor = get_predictor()
    predictor.set_image(rgb)
    fg = (depth_u16 > 0) & (~bg_mask)

    segs: list[np.ndarray] = []
    for (ix, iy) in _grid_points(H, W, fg, points_per_side):
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[ix, iy]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True)
        seg = masks[int(np.argmax(scores))].astype(bool)
        if seg.shape != (H, W):
            continue
        af = int(seg.sum()) / img_area
        if af < min_area_frac or af > max_area_frac:
            continue
        if _borders_touched(seg) >= 3 and af > 0.45:        # full-frame backdrop
            continue
        vd = seg & (depth_u16 > 0)
        nvd = int(vd.sum())
        if nvd < 300:
            continue
        if float((vd & bg_mask).sum()) / nvd > bg_plane_frac:  # on a bg plane
            continue
        seg = _restrict_depth_band(seg, depth_u16)           # kill depth bleed
        if int(seg.sum()) < 0.008 * img_area:                # drop tiny scraps
            continue
        segs.append(seg)

    cands = []
    for seg in _merge_overlapping(segs, merge_overlap):      # union overlaps
        vd = seg & (depth_u16 > 0)
        area = int(seg.sum())
        if area < 0.008 * img_area or int(vd.sum()) == 0:
            continue
        cands.append({
            "segmentation": seg, "area": area, "area_frac": area / img_area,
            "depth_median_m": float(np.median(depth_u16[vd])) * scale,
        })
    cands.sort(key=lambda c: c["area"], reverse=True)
    return cands[:max_objects]


def generate_object_masks(
    rgb: np.ndarray,
    depth_u16: np.ndarray,
    bg_mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, scale: float,
    max_objects: int = 8,
    coverage_thresh: float = 0.60,
) -> list[dict]:
    """HYBRID: grid-sampling primary; fall back to 3D-cluster prompting if it under-covers.

    The grid pass is used when its objects cover >= `coverage_thresh` of the
    LARGEST foreground blob. Otherwise (big low-texture object the grid only
    fragmented) we use the cluster-prompt fallback.
    """
    from scipy.ndimage import label

    fg = (depth_u16 > 0) & (~bg_mask)

    grid = _grid_objects(rgb, depth_u16, bg_mask, scale, max_objects=max_objects)
    union = np.zeros_like(fg)
    for c in grid:
        union |= c["segmentation"]

    # Fragment test on the DOMINANT foreground blob: if the single largest
    # foreground region (a big object) is poorly covered, the grid only got a
    # fragment of it -> fall back. Multi-object scenes (each object its own blob,
    # fully covered) keep the grid result even if some distractor blob is uncovered.
    lab, n = label(fg)
    cov_big = 1.0
    if n > 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        big = lab == int(sizes.argmax())
        cov_big = int((union & big).sum()) / max(1, int(big.sum()))

    if grid and cov_big >= coverage_thresh:
        for c in grid:
            c["method"] = "grid"
        return grid

    cl = _cluster_objects(rgb, depth_u16, bg_mask, fx, fy, cx, cy, scale,
                          max_objects=max_objects)
    for c in cl:
        c["method"] = "cluster"
    return cl


def overlay_masks(rgb: np.ndarray, masks: list[dict]) -> np.ndarray:
    """Color each kept mask over the RGB image and number it (BGR for cv2 write)."""
    import cv2

    out = rgb.copy()
    rng = np.random.default_rng(0)
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        color = rng.integers(60, 255, size=3)
        out[seg] = (0.5 * out[seg] + 0.5 * color).astype(np.uint8)
        ys, xs = np.nonzero(seg)
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.putText(out, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 3, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
