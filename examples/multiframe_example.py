"""Example: combine several default captures into ONE multi-frame "house" scene
and run the truck bin-packing over all of them.

A house is never one photo — so this takes a few default dataset folders, lays
them out as one multi-frame upload (frame_00, frame_01, …, exactly what the web
"add frame" button produces), and computes the full result: per-frame
segmentation + OBB on the left, and ALL objects packed into moving boxes then
loaded onto the cheapest truck (with a quote) on the right.

By default it MERGES the per-folder demo results (no GPU / SAM2 needed) so you
can view the scene immediately; pass --run to instead execute the live
RGB-D → SAM2 → … → truck pipeline.

    # no GPU — merge cached demo results into a scene, view instantly
    python examples/multiframe_example.py
    python examples/multiframe_example.py --name house --folders 001_150X20X64 004_20X70X40 007_37X44X40_2 010_37X44X50

    # live pipeline (run in the env that has SAM2 + torch)
    python examples/multiframe_example.py --run

Then open the web app and pick the scene (e.g. "house") from the dropdown.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import packing as packing_mod          # noqa: E402
import trucks as trucks_mod            # noqa: E402
from box_fit import catalog_from_folders  # noqa: E402

WS = ROOT.parent / "realsense" / "sam2_pointcloud_workspace" / "depth"
DEMO = ROOT / "demo" / "web"
UPLOADS = ROOT / "outputs" / "uploads"
OUT = ROOT / "outputs" / "web"
FRAME_GAP_CM = 60.0

# Default scene: a few rooms with different item mixes.
DEFAULT_FOLDERS = ["001_150X20X64", "004_20X70X40", "007_37X44X40_2",
                   "010_37X44X50", "002_175X15X5"]


def _stage_upload(name: str, folders: list[str]) -> int:
    """Copy each source capture into outputs/uploads/<name>/frame_NN (like the UI)."""
    dest = UPLOADS / name
    if dest.exists():
        shutil.rmtree(dest)
    n = 0
    for i, f in enumerate(folders):
        src = WS / f
        if not (src / "rgb.png").exists():
            print(f"  (skip {f}: not found in workspace)")
            continue
        fd = dest / f"frame_{i:02d}"
        fd.mkdir(parents=True, exist_ok=True)
        for fn in ("rgb.png", "depth_aligned.npy", "point_cloud_report.json"):
            if (src / fn).exists():
                shutil.copy2(src / fn, fd / fn)
        n += 1
    print(f"staged {n} frame(s) -> {dest}")
    return n


def _merge_demo(name: str, folders: list[str]) -> dict:
    """Merge per-folder demo viz into one multi-frame scene + truck packing (no GPU)."""
    instances: list = []
    x_cursor = 0.0
    gid = 0
    for fi, f in enumerate(folders):
        dj = DEMO / f"{f}.json"
        if not dj.exists():
            print(f"  (skip {f}: no demo json)")
            continue
        src = json.loads(dj.read_text(encoding="utf-8"))
        xs = [p[0] for o in src["instances"] for p in o["points"]] or [0.0]
        fmin, fmax = min(xs), max(xs)
        shift = x_cursor - fmin
        for o in src["instances"]:
            o = dict(o)
            o["id"] = gid
            o["frame"] = fi
            o["points"] = [[round(p[0] + shift, 1), p[1], p[2]] for p in o["points"]]
            o["obb"] = {"corners": [[round(c[0] + shift, 1), c[1], c[2]]
                                    for c in o["obb"]["corners"]]}
            instances.append(o)
            gid += 1
        x_cursor += (fmax - fmin) + FRAME_GAP_CM

    # boxes <- kept objects (corrected dims, fold/compress) ; trucks <- boxes + loose
    cat = catalog_from_folders([p.name for p in WS.iterdir() if p.is_dir()]) \
        if WS.exists() else catalog_from_folders(folders)
    pack_objs = [{
        "instance_id": o["id"], "label": o.get("identity", o["label"]),
        "dims_cm": o.get("corrected_dims_cm", o["dims_cm"]),
        "compressible": o.get("compressible", False),
        "foldable": o.get("foldable", False),
    } for o in instances if o.get("kept", True)]
    packed = packing_mod.pack(pack_objs, cat)

    cargo = []
    for bi, b in enumerate(packed["boxes"]):
        cargo.append({"id": f"box{bi}", "label": b["name"], "kind": "box",
                      "size": [float(d) for d in b["dims_cm"]]})
    for u in packed["unplaced"]:
        cargo.append({"id": f"loose{u['instance_id']}", "label": u["label"],
                      "kind": "loose", "size": [float(s) for s in u["size_cm"]]})
    truck_plan = trucks_mod.plan(cargo, trucks_mod.default_catalog())

    allpts = np.array([p for o in instances for p in o["points"]]) if instances \
        else np.zeros((1, 3))
    return {
        "folder": name, "frames": len([f for f in folders if (DEMO / f"{f}.json").exists()]),
        "scene_bbox": {"min": allpts.min(0).round(1).tolist(),
                       "max": allpts.max(0).round(1).tolist()},
        "overlay": "", "overlays": [], "method": "merged-demo",
        "vlm_source": "heuristic", "instances": instances,
        "packing": packed, "truck_plan": truck_plan,
        "catalog": [{"name": b.name, "dims_cm": b.sorted_cm} for b in cat],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="house")
    ap.add_argument("--folders", nargs="*", default=DEFAULT_FOLDERS)
    ap.add_argument("--run", action="store_true",
                    help="run the live SAM2->...->truck pipeline (needs the SAM2 env)")
    args = ap.parse_args()

    n = _stage_upload(args.name, args.folders)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.run:
        import pipeline_run
        cat = catalog_from_folders([p.name for p in WS.iterdir() if p.is_dir()])
        viz = pipeline_run.run_capture(UPLOADS / args.name, catalog=cat,
                                       overlay_path=OUT / "overlays" / f"{args.name}.png")
    else:
        viz = _merge_demo(args.name, args.folders)

    (OUT / f"{args.name}.json").write_text(json.dumps(viz, ensure_ascii=False),
                                           encoding="utf-8")
    tp = viz["truck_plan"]
    print(f"\nscene '{args.name}': {viz['frames']} frames, {len(viz['instances'])} objects")
    print(f"  boxes: {len(viz['packing']['boxes'])}  loose: {len(viz['packing']['unplaced'])}")
    if tp.get("truck"):
        print(f"  TRUCK: {tp['truck']['name']} x{tp['count']} | "
              f"util {round(tp['utilization']*100)}% | cargo {tp['cargo_volume_m3']} m3 | "
              f"{tp['quote_krw']:,} KRW")
        print("  alternatives:", ", ".join(
            f"{o['name']} x{o['count']} {o['total_krw']:,}" for o in tp["options"]))
    print(f"\nwrote {OUT / (args.name + '.json')}")
    print(f"Open the app and pick '{args.name}' ({n}-frame upload) from the dropdown.")


if __name__ == "__main__":
    main()
