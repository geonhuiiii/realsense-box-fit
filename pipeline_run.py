"""End-to-end run for an RGB-D capture (one or many frames) → viz dict.

RGB-D → SAM2 segmentation → point cloud back-projection → per-object OBB
(Nelder-Mead) → Gemini VLM (filter/compress/fold) → greedy 3D bin-packing.

A capture is either a single frame (``<dir>/rgb.png`` + ``depth_aligned.npy``)
or a multi-frame scene (``<dir>/frame_00/...``, ``frame_01/...`` — a house is
never captured in one shot). Every frame is segmented + measured independently;
their objects are laid side-by-side in the left view and packed together in the
right view.

Returns a viz dict the Three.js dual-view reads:
  left view  : instances (segmented point cloud + OBB), one cluster per frame
  right view : packing (boxes with placed object OBBs) over ALL frames

All 3D coords are in centimetres, display frame Z-up: [x, z, -y].
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

import packing as packing_mod
import trucks as trucks_mod
import vlm
from box_fit import catalog_from_folders
from geometry import backproject, scene_background_mask, estimate_obb_volume
from run_box_fit import load_intrinsics
from sam2_masks import generate_object_masks

_PALETTE = [[0.90,0.30,0.24],[0.18,0.80,0.44],[0.20,0.60,0.86],[0.95,0.61,0.07],
            [0.61,0.35,0.71],[0.10,0.74,0.61],[0.95,0.77,0.06],[0.20,0.29,0.37],
            [0.91,0.30,0.24],[0.52,0.73,0.40]]

_FRAME_GAP_CM = 60.0   # spacing between frames laid side-by-side in the left view


def _disp(arr_m: np.ndarray) -> np.ndarray:
    """camera frame (x,y down,z fwd) meters -> display Z-up cm: [x, z, -y]*100."""
    a = np.asarray(arr_m, dtype=float)
    return np.stack([a[..., 0], a[..., 2], -a[..., 1]], axis=-1) * 100.0


# OBB corner connectivity (matches obb_nelder_mead.corners() ordering / the viewer).
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _project(corners_cam_m: np.ndarray, fx, fy, cx, cy) -> list:
    """Project camera-frame (meters) OBB corners back to image pixels (u, v)."""
    out = []
    for X, Y, Z in np.asarray(corners_cam_m, dtype=float):
        if Z <= 1e-3:
            out.append(None)            # behind the camera -> skip its edges
        else:
            out.append((int(round(fx * X / Z + cx)), int(round(fy * Y / Z + cy))))
    return out


def _annotate_for_vlm(rgb: np.ndarray, anns: list) -> np.ndarray:
    """Draw, on the RGB, each object's mask outline + projected 3D OBB + id number.

    This is the image the VLM sees: the point cloud's OBBs reprojected onto the
    photo and labelled with the same ids the prompt lists, so the model can match
    'id N' to a visible object and its box. Returns a BGR image (for cv2.imwrite).
    """
    img = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    for a in anns:
        bgr = tuple(int(c * 255) for c in a["color"][::-1])
        m = a["mask"].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, bgr, 2)
        p = a["obb2d"]
        for i, j in _EDGES:
            if p[i] is not None and p[j] is not None:
                cv2.line(img, p[i], p[j], (0, 255, 255), 1, cv2.LINE_AA)
        u, v = int(a["centroid"][1]), int(a["centroid"][0])
        cv2.putText(img, str(a["id"]), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, str(a["id"]), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    bgr, 2, cv2.LINE_AA)
    return img


def _emit(cb, stage, message, progress):
    if cb:
        cb(stage=stage, message=message, progress=progress)


def list_frames(folder_dir: str | Path) -> list[Path]:
    """Return the frame directories of a capture (single- or multi-frame)."""
    folder_dir = Path(folder_dir)
    if (folder_dir / "rgb.png").exists() and (folder_dir / "depth_aligned.npy").exists():
        return [folder_dir]
    subs = sorted(p for p in folder_dir.iterdir() if p.is_dir()
                  and (p / "rgb.png").exists() and (p / "depth_aligned.npy").exists())
    return subs


def _merge_duplicates(instances: list) -> None:
    """Fold VLM-flagged duplicate detections into their canonical object (in place).

    When the VLM says id B is `duplicate_of` id A (two masks on one physical item),
    B is dropped (kept=False) and A's corrected dims are unioned (per-axis max) so
    the surviving object covers the whole thing — no double counting in packing.
    """
    by_id = {o["id"]: o for o in instances}
    for o in instances:
        dup = o.get("duplicate_of", -1)
        if dup is None or dup < 0 or dup == o["id"] or dup not in by_id:
            continue
        canon = by_id[dup]
        if canon is o or not canon.get("kept", True):
            continue
        cd = sorted(float(x) for x in canon["corrected_dims_cm"])
        od = sorted(float(x) for x in o["corrected_dims_cm"])
        canon["corrected_dims_cm"] = [round(max(a, b), 1) for a, b in zip(cd, od)]
        o["kept"] = False
        o["reasoning"] = f"merged into id {dup} (duplicate of one object)"


def _process_frame(frame_dir: Path, fi: int, id_offset: int, max_points: int,
                   rng_seed: int, overlay_path: Path | None,
                   catalog=None) -> tuple[list, list, str, str]:
    """Segment + measure + VLM one frame. Returns (instances, display-pts, method, vlm_source).

    The VLM also picks each object's box from `catalog`. Instance ids are globally
    unique via `id_offset`. No packing here.
    """
    rgb = cv2.cvtColor(cv2.imread(str(frame_dir / "rgb.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(frame_dir / "depth_aligned.npy")
    intr, scale = load_intrinsics(frame_dir / "point_cloud_report.json")
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    H, W = depth.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    bg = scene_background_mask(depth, fx, fy, cx, cy, scale)
    masks = generate_object_masks(rgb, depth, bg, fx, fy, cx, cy, scale)

    rng = np.random.default_rng(rng_seed + fi)
    instances, all_pts, anns = [], [], []
    for i, m in enumerate(masks):
        valid = m["segmentation"] & (depth > 0)
        vs, us = np.nonzero(valid)
        z = depth[vs, us].astype(np.float64) * scale
        keepz = (z >= 0.1) & (z <= 12.0)
        vs, us = vs[keepz], us[keepz]
        pts = backproject(depth, m["segmentation"], fx, fy, cx, cy, scale)
        if pts.shape[0] < 50:
            continue
        cols = rgb[vs, us].astype(np.float64) / 255.0
        try:
            res = estimate_obb_volume(pts, n_restarts=4)
        except Exception:
            continue
        dims = sorted(float(e) * 100 for e in res.extent)
        n = pts.shape[0]
        sel = rng.choice(n, min(max_points, n), replace=False)
        dpts = _disp(pts[sel])
        gid = id_offset + len(instances)
        color = _PALETTE[gid % len(_PALETTE)]
        instances.append({
            "id": gid, "frame": fi, "label": f"f{fi}o{i}",
            "method": m.get("method", "?"),
            "color": color,
            "n_points_total": int(n),
            "points": dpts.round(1).tolist(),
            "colors": (cols[sel] if len(cols) == n else
                       np.full((len(sel), 3), 0.5)).round(3).tolist(),
            "obb": {"corners": _disp(res.corners()).round(1).tolist()},
            "dims_cm": [round(d, 1) for d in dims],
            "volume_cm3": round(float(res.volume) * 1e6, 1),
        })
        all_pts.append(dpts)
        if len(vs):                          # for the VLM annotation: reproject OBB
            anns.append({"id": gid, "color": color, "mask": m["segmentation"],
                         "obb2d": _project(res.corners(), fx, fy, cx, cy),
                         "centroid": (float(vs.mean()), float(us.mean()))})

    # The VLM sees the annotated image (mask outlines + reprojected OBBs + id labels).
    vlm_image = _annotate_for_vlm(rgb, anns) if anns else cv2.cvtColor(
        rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if overlay_path:
        Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), vlm_image)
    vlm_rgb = cv2.cvtColor(vlm_image, cv2.COLOR_BGR2RGB)

    vlm_recs = vlm.analyze(vlm_rgb, [
        {"instance_id": o["id"], "label": o["label"], "dims_cm": o["dims_cm"]}
        for o in instances], catalog=catalog)
    vlm_by_id = {r["instance_id"]: r for r in vlm_recs}
    for o in instances:
        r = vlm_by_id.get(o["id"], {})
        o["kept"] = bool(r.get("keep", True))
        o["duplicate_of"] = int(r.get("duplicate_of", -1))
        o["identity"] = r.get("identity", o["label"])
        o["material"] = r.get("material", "unknown")
        o["compressible"] = bool(r.get("compressible", False))
        o["foldable"] = bool(r.get("foldable", False))
        o["corrected_dims_cm"] = r.get("corrected_dims_cm", o["dims_cm"])
        o["box"] = r.get("box", "")            # VLM-chosen box (""=loose furniture)
        o["reasoning"] = r.get("reasoning", "")
        o["vlm_source"] = r.get("source", "heuristic")

    _merge_duplicates(instances)

    method = instances[0]["method"] if instances else "none"
    vlm_source = vlm_recs[0]["source"] if vlm_recs else "heuristic"
    return instances, all_pts, method, vlm_source


def _offset_frame(instances: list, all_pts: list, shift_x: float) -> None:
    """Translate a frame's display geometry by `shift_x` along X (in place)."""
    for o in instances:
        o["points"] = [[round(x + shift_x, 1), y, z] for x, y, z in o["points"]]
        o["obb"]["corners"] = [[round(c[0] + shift_x, 1), c[1], c[2]]
                               for c in o["obb"]["corners"]]
    for i in range(len(all_pts)):
        all_pts[i] = all_pts[i] + np.array([shift_x, 0.0, 0.0])


def run_capture(folder_dir: str | Path, catalog=None, progress=None,
                max_points: int = 4000, overlay_path: str | Path | None = None,
                rng_seed: int = 0) -> dict:
    folder_dir = Path(folder_dir)
    name = folder_dir.name
    frames = list_frames(folder_dir)
    nframes = len(frames)
    if catalog is None:
        import config as rbf_config
        from box_fit import boxes_from_items
        catalog = boxes_from_items(rbf_config.load_box_catalog())

    overlay_dir = Path(overlay_path).parent if overlay_path else None

    instances: list = []
    all_pts: list = []
    overlays: list[str] = []
    methods: list[str] = []
    sources: list[str] = []
    x_cursor = 0.0

    for fi, fr in enumerate(frames):
        frac = 0.15 + 0.70 * (fi / max(nframes, 1))
        _emit(progress, "frame", f"프레임 {fi + 1}/{nframes}: 세그·OBB·VLM", frac)
        ov = (overlay_dir / f"{name}__f{fi}.png") if overlay_dir else None
        f_inst, f_pts, method, source = _process_frame(
            fr, fi, len(instances), max_points, rng_seed, ov, catalog=catalog)
        methods.append(method)
        sources.append(source)
        if ov:
            overlays.append(f"overlays/{name}__f{fi}.png")

        if f_pts:
            fmin = min(float(p[:, 0].min()) for p in f_pts)
            fmax = max(float(p[:, 0].max()) for p in f_pts)
        else:
            fmin = fmax = 0.0
        if nframes > 1:
            _offset_frame(f_inst, f_pts, x_cursor - fmin)
            x_cursor += (fmax - fmin) + _FRAME_GAP_CM
        instances += f_inst
        all_pts += f_pts

    _emit(progress, "packing", "박스 패킹 (전체 프레임)", 0.88)
    pack_objs = [{
        "instance_id": o["id"], "label": o["identity"],
        "dims_cm": o["corrected_dims_cm"],
        "compressible": o["compressible"], "foldable": o["foldable"],
        "box": o.get("box", ""),               # VLM's chosen box (""=loose furniture)
    } for o in instances if o["kept"]]
    packed = packing_mod.pack(pack_objs, catalog)

    _emit(progress, "truck", "트럭 적재 + 견적", 0.94)
    cargo_units = []
    for bi, b in enumerate(packed["boxes"]):
        cargo_units.append({"id": f"box{bi}", "label": b["name"], "kind": "box",
                            "size": [float(d) for d in b["dims_cm"]]})
    for u in packed["unplaced"]:           # large furniture loaded loose onto the truck
        cargo_units.append({"id": f"loose{u['instance_id']}", "label": u["label"],
                            "kind": "loose", "size": [float(s) for s in u["size_cm"]]})
    truck_plan = trucks_mod.plan(cargo_units, trucks_mod.default_catalog())

    if all_pts:
        allnp = np.concatenate(all_pts)
        bbox = {"min": allnp.min(0).round(1).tolist(),
                "max": allnp.max(0).round(1).tolist()}
    else:
        bbox = {"min": [0, 0, 0], "max": [100, 100, 100]}

    method = next((m for m in methods if m != "none"), "none")
    vlm_source = next((s for s in sources if s != "heuristic"), sources[0] if sources else "heuristic")
    viz = {
        "folder": name,
        "frames": nframes,
        "scene_bbox": bbox,
        "overlay": overlays[0] if overlays else f"overlays/{name}.png",
        "overlays": overlays,
        "method": method,
        "vlm_source": vlm_source,
        "instances": instances,
        "packing": packed,
        "truck_plan": truck_plan,
        "catalog": [{"name": b.name, "dims_cm": b.sorted_cm} for b in catalog],
    }

    # 파이프라인 종료 후: VLM(또는 휴리스틱)으로 결과를 훑어 견적서 전 항목을 예상한다.
    # 백엔드는 메인 VLM과 동일한 config(vlm_backend + Gemini 키)를 따른다.
    if progress:
        progress("quote", "견적서 예상 중…", 0.97)
    try:
        import quote_estimate
        viz["quote_estimate"] = quote_estimate.estimate(viz)
    except Exception as e:  # noqa: BLE001
        log.warning("견적 예상 실패(%s)", e)
    return viz


def save_viz(viz: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{viz['folder']}.json"
    p.write_text(json.dumps(viz, ensure_ascii=False), encoding="utf-8")
    return p
