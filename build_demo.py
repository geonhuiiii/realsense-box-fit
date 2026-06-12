"""Build the shipped demo (demo/web) by running the pipeline on the dataset.

Runs RGB-D → SAM2 → OBB → VLM → packing for each capture, downsamples the point
clouds to keep the repo small, and writes demo/web/<folder>.json + overlays.
The web app falls back to these cached results so the UI works after clone.

    python build_demo.py            # all dataset folders
    python build_demo.py --max-points 1200
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import pipeline_run
from box_fit import catalog_from_folders

ROOT = Path(__file__).resolve().parent
WS = ROOT.parent / "realsense" / "sam2_pointcloud_workspace"
DST = ROOT / "demo" / "web"


def _downsample(viz, max_points, rng):
    for o in viz["instances"]:
        pts, cols = o["points"], o["colors"]
        if len(pts) > max_points:
            idx = rng.choice(len(pts), max_points, replace=False)
            o["points"] = [pts[i] for i in idx]
            o["colors"] = [cols[i] for i in idx]
    return viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-points", type=int, default=1500)
    ap.add_argument("--folders", nargs="*", default=None)
    args = ap.parse_args()

    depth = WS / "depth"
    names = args.folders or sorted(
        p.name for p in depth.iterdir()
        if (p / "rgb.png").exists() and (p / "depth_aligned.npy").exists())
    cat = catalog_from_folders([p.name for p in depth.iterdir() if p.is_dir()])
    (DST / "overlays").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for name in names:
        viz = pipeline_run.run_capture(
            depth / name, catalog=cat, overlay_path=DST / "overlays" / f"{name}.png")
        viz = _downsample(viz, args.max_points, rng)
        (DST / f"{name}.json").write_text(json.dumps(viz, ensure_ascii=False),
                                          encoding="utf-8")
        nb = len(viz["packing"]["boxes"])
        print(f"demo {name}: {len(viz['instances'])} obj, {nb} box(es)")

    sz = sum(f.stat().st_size for f in DST.rglob("*")) / 1e6
    print(f"\ndemo at {DST}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
