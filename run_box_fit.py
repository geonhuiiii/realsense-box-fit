"""End-to-end: RealSense capture -> SAM2 masks -> Nelder-Mead OBB -> box fit.

For each capture folder under <workspace>/depth/<folder>/:
  1) load rgb.png, depth_aligned.npy, confidence_aligned.npy, point_cloud_report.json
  2) SAM2 automatic mask generation, background-filtered to candidate objects
  3) back-project each object's masked depth -> point cloud (m)
  4) Nelder-Mead OBB -> sorted dims (cm), volume
  5) match against the moving-box catalog (smallest fitting box, sorted dims + tol)
  6) cross-check vs ground-truth dims parsed from the folder name (if present)

Outputs: per-folder result.json + masks_overlay.png, and a top-level summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from box_fit import Box, best_fit_box, catalog_from_folders, load_catalog, parse_dims
from geometry import backproject, obb_dims_cm, scene_background_mask
from sam2_masks import generate_object_masks, overlay_masks


# --- Dataset metadata corrections (see investigation notes) ---------------- #
# The capture's point_cloud_report.json carries two wrong values:
#  * depth_scale_m_per_unit=0.001 — that is the D400 default; the L515 depth
#    unit is 0.00025 m. Using 0.001 inflates every reconstruction ~4x.
#  * fy=684.63 = fx*0.75 — intrinsics were anisotropically scaled from the
#    1024x768 depth sensor. Depth is rs.align-ed to the COLOR frame (square
#    pixels), where fx=913 already gives the L515's 70 deg HFOV, so fy must
#    equal fx (~913); fy=684 would imply an impossible 56 deg VFOV.
L515_DEPTH_SCALE = 0.00025
FORCE_FY_EQ_FX = True


def load_intrinsics(report_path: Path) -> tuple[dict, float]:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    intr = dict(rep["intrinsics"])
    if FORCE_FY_EQ_FX:
        intr["fy"] = intr["fx"]
    scale = L515_DEPTH_SCALE
    return intr, scale


def process_folder(
    folder: Path,
    boxes: list[Box],
    out_dir: Path,
    save_ply: bool = False,
    nm_restarts: int = 4,
) -> dict:
    rgb_p = folder / "rgb.png"
    dep_p = folder / "depth_aligned.npy"
    rep_p = folder / "point_cloud_report.json"
    if not (rgb_p.exists() and dep_p.exists() and rep_p.exists()):
        return {"folder": folder.name, "status": "skipped_missing_data"}

    rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
    depth = np.load(dep_p)
    conf_p = folder / "confidence_aligned.npy"
    confidence = np.load(conf_p) if conf_p.exists() else None
    intr, scale = load_intrinsics(rep_p)

    # Align rgb to depth resolution if needed.
    H, W = depth.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    gt_dims = parse_dims(folder.name)
    gt_box = None
    if gt_dims is not None:
        gt_box, _ = best_fit_box(gt_dims, boxes)

    bg_mask = scene_background_mask(
        depth, intr["fx"], intr["fy"], intr["cx"], intr["cy"], scale)
    masks = generate_object_masks(
        rgb, depth, bg_mask,
        intr["fx"], intr["fy"], intr["cx"], intr["cy"], scale)
    out_folder = out_dir / folder.name
    out_folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_folder / "masks_overlay.png"), overlay_masks(rgb, masks))

    objects = []
    for i, m in enumerate(masks):
        pts = backproject(
            depth, m["segmentation"],
            intr["fx"], intr["fy"], intr["cx"], intr["cy"], scale,
            confidence=confidence,
        )
        if pts.shape[0] < 50:
            continue
        try:
            dims_cm, vol_cm3, n_used = obb_dims_cm(pts, n_restarts=nm_restarts)
        except Exception as e:
            objects.append({"object_id": i, "error": str(e)})
            continue
        box, details = best_fit_box(dims_cm, boxes)
        if save_ply:
            _save_ply(out_folder / f"object_{i}.ply", pts)
        objects.append({
            "object_id": i,
            "n_points": int(pts.shape[0]),
            "n_points_obb": n_used,
            "area_frac": round(m["area_frac"], 4),
            "depth_median_m": round(m["depth_median_m"], 3),
            "measured_dims_cm": [round(d, 1) for d in dims_cm],
            "volume_cm3": round(vol_cm3, 1),
            "matched_box": box.name if box else "NO_FIT",
            "fit_details": details,
        })

    # Pick the object whose dims best match GT (for the summary's headline row).
    best_obj = None
    if gt_dims is not None and objects:
        gt_sorted = sorted(gt_dims)
        scored = [o for o in objects if "measured_dims_cm" in o]
        if scored:
            best_obj = min(
                scored,
                key=lambda o: sum(abs(a - b) for a, b in
                                  zip(o["measured_dims_cm"], gt_sorted)),
            )

    result = {
        "folder": folder.name,
        "status": "ok",
        "gt_dims_cm": gt_dims,
        "gt_dims_sorted_cm": sorted(gt_dims) if gt_dims else None,
        "gt_expected_box": gt_box.name if gt_box else ("NO_FIT" if gt_dims else None),
        "n_objects": len([o for o in objects if "measured_dims_cm" in o]),
        "best_match_object_id": best_obj["object_id"] if best_obj else None,
        "objects": objects,
    }
    (out_folder / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _save_ply(path: Path, pts: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def write_summary(results: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "gt_dims_cm", "gt_expected_box", "object_id",
                    "measured_dims_cm", "volume_cm3", "matched_box", "is_best_match"])
        for r in results:
            if r.get("status") != "ok":
                w.writerow([r["folder"], "", "", "", "", "", r.get("status", ""), ""])
                continue
            gt = "x".join(str(int(d)) for d in r["gt_dims_cm"]) if r["gt_dims_cm"] else ""
            objs = [o for o in r["objects"] if "measured_dims_cm" in o]
            if not objs:
                w.writerow([r["folder"], gt, r["gt_expected_box"] or "", "",
                            "", "", "no_objects", ""])
            for o in objs:
                md = "x".join(str(d) for d in o["measured_dims_cm"])
                w.writerow([
                    r["folder"], gt, r["gt_expected_box"] or "", o["object_id"],
                    md, o["volume_cm3"], o["matched_box"],
                    "Y" if o["object_id"] == r.get("best_match_object_id") else "",
                ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(Path(__file__).resolve().parent.parent
                    / "realsense" / "sam2_pointcloud_workspace"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--folders", nargs="*", default=None,
                    help="specific folder names to process (default: all)")
    ap.add_argument("--save-ply", action="store_true")
    ap.add_argument("--nm-restarts", type=int, default=4)
    args = ap.parse_args()

    ws = Path(args.workspace)
    depth_root = ws / "depth"

    # The REAL box catalog = the distinct dimension triples encoded in ALL
    # capture folder names (not the 3-entry example CSV).
    all_folder_names = sorted(p.name for p in depth_root.iterdir() if p.is_dir())
    catalog = catalog_from_folders(all_folder_names)
    print("Real box catalog (from folder names):",
          [(b.name, b.sorted_cm) for b in catalog])

    names = args.folders or all_folder_names
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in names:
        folder = depth_root / name
        print(f"\n=== {name} ===")
        r = process_folder(folder, catalog, out_dir, save_ply=args.save_ply,
                           nm_restarts=args.nm_restarts)
        if r.get("status") == "ok":
            print(f"  GT={r['gt_dims_cm']} expect={r['gt_expected_box']} "
                  f"n_obj={r['n_objects']}")
            for o in r["objects"]:
                if "measured_dims_cm" in o:
                    star = "*" if o["object_id"] == r.get("best_match_object_id") else " "
                    print(f"  {star}obj{o['object_id']}: {o['measured_dims_cm']}cm "
                          f"-> {o['matched_box']}")
        else:
            print(f"  {r.get('status')}")
        results.append(r)

    write_summary(results, out_dir / "summary.csv")
    print(f"\nSummary written to {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
