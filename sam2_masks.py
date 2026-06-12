"""SAM2 object segmentation — AMG only.

SAM2AutomaticMaskGenerator over the full image (quality-tuned settings).
We keep only foreground masks that pass depth / background-plane filters and
depth-band each mask so background bleed can't inflate the OBB.
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
_AMG = None
_LOCK = threading.Lock()        # guard model build (preload thread vs run worker)


def _model():
    """Build the SAM2 model once, shared by AMG."""
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


def get_amg_generator():
    """Build (once) and return a quality-tuned SAM2AutomaticMaskGenerator."""
    global _AMG
    if _AMG is not None:
        return _AMG
    model = _model()
    with _LOCK:
        if _AMG is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            _AMG = SAM2AutomaticMaskGenerator(
                model,
                points_per_side=64,          # 4096 prompts (default 32 → 1024)
                points_per_batch=64,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.95,
                box_nms_thresh=0.7,
                crop_n_layers=1,             # multi-scale crops for small/detail regions
                crop_nms_thresh=0.7,
                min_mask_region_area=100,    # drop tiny holes / speckle
                use_m2m=True,                # mask-to-mask refinement pass
                multimask_output=True,
            )
    return _AMG


def preload() -> None:
    """Eagerly build the SAM2 AMG generator so the first run isn't slow.

    Raises RuntimeError with an actionable message if SAM2/torch/checkpoint is missing.
    """
    from env_check import require_sam2
    require_sam2()
    get_amg_generator()


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


def _borders_touched(seg: np.ndarray) -> int:
    H, W = seg.shape
    return int(seg[0, :].any()) + int(seg[H - 1, :].any()) + \
        int(seg[:, 0].any()) + int(seg[:, W - 1].any())


def _amg_objects(
    rgb: np.ndarray, depth_u16: np.ndarray, bg_mask: np.ndarray, scale: float,
    max_objects: int = 12,
    min_area_frac: float = 0.01, max_area_frac: float = 0.85,
    bg_plane_frac: float = 0.5, containment_thresh: float = 0.85,
) -> list[dict]:
    """SAM2 automatic mask generation, filtered to foreground objects."""
    H, W = depth_u16.shape
    img_area = H * W
    fg = (depth_u16 > 0) & (~bg_mask)

    raw = get_amg_generator().generate(rgb)
    cands = []
    for ann in raw:
        seg = ann["segmentation"].astype(bool)
        if seg.shape != (H, W):
            continue
        if int((seg & fg).sum()) < 300:
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
        vd = seg & (depth_u16 > 0)
        area = int(seg.sum())
        if area < 0.008 * img_area or int(vd.sum()) == 0:
            continue
        cands.append({
            "segmentation": seg, "area": area, "area_frac": area / img_area,
            "depth_median_m": float(np.median(depth_u16[vd])) * scale,
        })

    cands.sort(key=lambda c: c["area"], reverse=True)
    kept: list[dict] = []
    for c in cands:
        seg = c["segmentation"]
        nested = any(
            int((seg & k["segmentation"]).sum()) / max(1, c["area"]) > containment_thresh
            for k in kept
        )
        if not nested:
            kept.append(c)
    return kept[:max_objects]


def generate_object_masks(
    rgb: np.ndarray,
    depth_u16: np.ndarray,
    bg_mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, scale: float,
    max_objects: int = 8,
    coverage_thresh: float = 0.60,  # kept for API compat; unused
) -> list[dict]:
    """Run AMG and return foreground object masks."""
    masks = _amg_objects(rgb, depth_u16, bg_mask, scale, max_objects=max_objects)
    for c in masks:
        c["method"] = "amg"
    return masks


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
