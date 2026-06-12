"""Render result images for every folder.

Per folder produces `outputs/<folder>/result_render.png` with two panels:
  left  : RGB with SAM2 object masks overlaid (numbered)
  right : 3D point cloud of the segmented objects (real RGB colors) + the
          Nelder-Mead OBB drawn as a wireframe box per object, with the measured
          sorted dims (cm) annotated.
Also writes `outputs/_montage_overlay.png` and `outputs/_montage_3d.png`.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import backproject, scene_background_mask, estimate_obb_volume
from sam2_masks import generate_object_masks, overlay_masks
from run_box_fit import load_intrinsics

ROOT = Path(__file__).resolve().parent
DEPTH = ROOT.parent / "realsense" / "sam2_pointcloud_workspace" / "depth"
OUT = ROOT / "outputs"

_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
_OBJ_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _set_equal_3d(ax, pts):
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def render_folder(folder: str, rng_seed: int = 0):
    f = DEPTH / folder
    if not (f / "rgb.png").exists() or not (f / "depth_aligned.npy").exists():
        return None
    rgb = cv2.cvtColor(cv2.imread(str(f / "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(f / "depth_aligned.npy")
    intr, scale = load_intrinsics(f / "point_cloud_report.json")
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    H, W = depth.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    bg = scene_background_mask(depth, fx, fy, cx, cy, scale)
    masks = generate_object_masks(rgb, depth, bg, fx, fy, cx, cy, scale)

    overlay = cv2.cvtColor(overlay_masks(rgb, masks), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(15, 6))
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.imshow(overlay)
    ax0.set_title(f"{folder}  ·  SAM2 segmentation")
    ax0.axis("off")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    rng = np.random.default_rng(rng_seed)
    all_pts = []
    for i, m in enumerate(masks):
        # Replicate backproject's pixel selection so colors align with points.
        valid = m["segmentation"] & (depth > 0)
        vs, us = np.nonzero(valid)
        z = depth[vs, us].astype(np.float64) * scale
        keep_v = (z >= 0.1) & (z <= 12.0)
        vs, us = vs[keep_v], us[keep_v]
        pts = backproject(depth, m["segmentation"], fx, fy, cx, cy, scale)
        if pts.shape[0] < 50 or pts.shape[0] != len(vs):
            if pts.shape[0] < 50:
                continue
        cols = rgb[vs, us].astype(np.float64) / 255.0
        n = pts.shape[0]
        sel = rng.choice(n, min(3000, n), replace=False)
        p, c = pts[sel], cols[sel]
        ax1.scatter(p[:, 0], p[:, 2], -p[:, 1], s=1, c=np.clip(c, 0, 1), depthshade=False)
        all_pts.append(p)
        try:
            res = estimate_obb_volume(pts, n_restarts=4)
            corners = res.corners()
            col = _OBJ_COLORS[i % 10]
            for a, b in _BOX_EDGES:
                ax1.plot([corners[a, 0], corners[b, 0]],
                         [corners[a, 2], corners[b, 2]],
                         [-corners[a, 1], -corners[b, 1]], c=col, lw=1.2)
            dims = sorted(float(e) * 100 for e in res.extent)
            cen = corners.mean(0)
            ax1.text(cen[0], cen[2], -cen[1],
                     f"#{i}: {dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f}cm",
                     color=col, fontsize=8)
        except Exception:
            pass

    if all_pts:
        _set_equal_3d(ax1, np.concatenate(all_pts))
    ax1.set_title("3D point cloud + Nelder-Mead OBB")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)"); ax1.set_zlabel("Y up (m)")
    ax1.view_init(elev=18, azim=-60)

    fig.tight_layout()
    outp = OUT / folder / "result_render.png"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=110)
    plt.close(fig)
    return outp, overlay


def main():
    folders = sorted(p.name for p in DEPTH.iterdir() if p.is_dir())
    rendered = []
    for folder in folders:
        r = render_folder(folder)
        if r is None:
            print(f"skip {folder} (no rgb/depth)")
            continue
        print(f"rendered {folder} -> {r[0]}")
        rendered.append((folder, r[0]))

    # Montages
    if rendered:
        n = len(rendered)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.0))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (folder, png) in zip(axes, rendered):
            ax.imshow(plt.imread(png))
            ax.set_title(folder, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "_montage_all.png", dpi=90)
        plt.close(fig)
        print(f"\nmontage -> {OUT / '_montage_all.png'}")


if __name__ == "__main__":
    main()
