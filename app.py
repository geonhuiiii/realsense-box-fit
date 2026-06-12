"""Flask web GUI — RealSense Box-Fit (segmentation → OBB → VLM → box packing).

UI cloned from the RMC-Visual-Agent moving-quote viewer (Three.js dual view,
dark theme, SSE progress) but driven by our pipeline and box-packing instead of
pricing.

Run:
    pip install -r requirements-full.txt   # + SAM2/torch for processing
    export GEMINI_API_KEY=...              # optional; without it a heuristic is used
    python app.py                          # http://localhost:8000
"""
from __future__ import annotations

import json
import os
import queue
import threading
import zipfile
from pathlib import Path

# --- macOS native-lib safety: MUST run before numpy/cv2/open3d/torch import --- #
# Several wheels (numpy, opencv, open3d, torch) each bundle their own OpenMP
# runtime; loading two libomp/libiomp into one process is what segfaults the
# pipeline thread on macOS. Allow the duplicate and keep the native thread pools
# single-threaded so heavy ops stay safe inside Flask's background (non-main)
# worker. Override any of these by exporting your own value.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# If a native lib still crashes, dump a Python+C traceback instead of a bare
# "segmentation fault" so we can see which library/frame died.
import faulthandler
faulthandler.enable()

import numpy as np
from flask import (Flask, Response, abort, jsonify, render_template, request,
                   send_file, send_from_directory)

import config as rbf_config

rbf_config.apply_to_env()                 # model paths / API key -> env (no export needed)

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get(
    "RBF_WORKSPACE", str(ROOT.parent / "realsense" / "sam2_pointcloud_workspace")))
DEPTH_ROOT = WORKSPACE / "depth"
OUT = ROOT / "outputs" / "web"
UPLOADS = ROOT / "outputs" / "uploads"
DEMO = ROOT / "demo" / "web"
QUOTES = ROOT / "outputs" / "quotes"

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")


def _frame_count(d: Path) -> int:
    """Number of RGB-D frames in a capture dir (single root or frame_* subdirs)."""
    if (d / "rgb.png").exists() and (d / "depth_aligned.npy").exists():
        return 1
    if not d.is_dir():
        return 0
    return sum(1 for s in d.iterdir() if s.is_dir()
               and (s / "rgb.png").exists() and (s / "depth_aligned.npy").exists())


def _list_folders():
    folders = []
    for base, src in ((DEPTH_ROOT, "dataset"), (UPLOADS, "upload")):
        if not base.exists():
            continue
        for p in sorted(base.iterdir()):
            nf = _frame_count(p)
            if nf:
                folders.append({"name": p.name, "source": src, "frames": nf})
    return folders


def _resolve_dir(name: str) -> Path | None:
    for base in (DEPTH_ROOT, UPLOADS):
        d = base / name
        if _frame_count(d):
            return d
    return None


def _catalog():
    from box_fit import catalog_from_folders
    names = []
    if DEPTH_ROOT.exists():
        names = [p.name for p in DEPTH_ROOT.iterdir() if p.is_dir()]
    cat = catalog_from_folders(names)
    if not cat:                       # sensible default catalog (cm)
        from box_fit import Box
        defaults = {"box_small": [37, 40, 44], "box_medium": [37, 44, 50],
                    "box_long": [20, 64, 150]}
        cat = [Box(n, d, sorted(d)) for n, d in defaults.items()]
    return cat


def _parse_catalog(items) -> list:
    """Build Box objects from user-supplied [{name, dims_cm:[w,h,d]}, ...]."""
    from box_fit import Box
    boxes = []
    for b in items or []:
        try:
            name = (str(b.get("name") or "").strip() or "box")
            dims = [float(x) for x in b.get("dims_cm", [])][:3]
        except (TypeError, ValueError):
            continue
        if len(dims) == 3 and all(d > 0 for d in dims):
            boxes.append(Box(name, dims, sorted(dims)))
    return boxes


def _ensure_reports(folder: Path) -> None:
    """Write a default L515-like intrinsics report wherever a depth has none."""
    targets = []
    if (folder / "depth_aligned.npy").exists():
        targets.append(folder)
    if folder.is_dir():
        targets += [s for s in folder.iterdir()
                    if s.is_dir() and (s / "depth_aligned.npy").exists()]
    for t in targets:
        rp = t / "point_cloud_report.json"
        if rp.exists():
            continue
        depth = np.load(t / "depth_aligned.npy")
        h, w = depth.shape
        rp.write_text(json.dumps({
            "intrinsics": {"fx": 913.21, "fy": 913.21, "cx": w / 2, "cy": h / 2},
            "depth_scale_m_per_unit": 0.00025}), encoding="utf-8")


def _config_status() -> dict:
    cfg = rbf_config.load()
    key = cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or ""
    backend = cfg.get("vlm_backend", "auto")
    # which backend actually runs under "auto"
    if backend == "auto":
        active = "gemini" if key else ("gemma" if (cfg.get("gemma_e4b") or cfg.get("gemma_31b")) else "heuristic")
    else:
        active = backend
    return {
        "vlm_backend": backend,
        "active_backend": active,
        "sam2_ckpt": cfg.get("sam2_ckpt") or os.environ.get("RBF_SAM2_CKPT") or "",
        "sam2_cfg": cfg.get("sam2_cfg") or os.environ.get("RBF_SAM2_CFG") or "",
        "gemma_e4b": cfg.get("gemma_e4b") or "",
        "gemma_31b": cfg.get("gemma_31b") or "",
        "gemini_key_set": bool(key),
        "gemini_key_hint": ("…" + key[-4:]) if len(key) >= 4 else ("set" if key else ""),
    }


@app.route("/")
def index():
    return render_template("index.html", folders=_list_folders())


@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify(_config_status())


@app.route("/api/config", methods=["POST"])
def api_config_set():
    """Save the Gemini key / backend choice / model paths typed in the UI."""
    body = request.get_json(silent=True) or {}
    allowed = {"gemini_api_key", "vlm_backend", "sam2_ckpt", "sam2_cfg",
               "gemma_e4b", "gemma_31b", "gemini_model"}
    updates = {k: v for k, v in body.items() if k in allowed}
    if "gemini_api_key" in updates:
        updates["gemini_api_key"] = (updates["gemini_api_key"] or "").strip()
    rbf_config.save(updates)
    rbf_config.apply_to_env()             # take effect immediately, no restart
    return jsonify(_config_status())


@app.route("/api/folders")
def api_folders():
    return jsonify(_list_folders())


@app.route("/api/catalog")
def api_catalog():
    """Default box catalog (user can edit it in the sidebar before running)."""
    return jsonify([{"name": b.name, "dims_cm": [round(d, 1) for d in b.dims_cm]}
                    for b in _catalog()])


@app.route("/api/result/<name>")
def api_result(name):
    for base in (OUT, DEMO):
        p = base / f"{name}.json"
        if p.exists():
            return app.response_class(p.read_text(encoding="utf-8"),
                                      mimetype="application/json")
    abort(404)


@app.route("/data/<path:fname>")
def data_file(fname):
    for base in (OUT, DEMO):
        if (base / fname).exists():
            return send_from_directory(base, fname)
    abort(404)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept one or many RGB-D frames (a house needs several shots), or a .zip.

    Multi-frame: send N `rgb` + N `depth` files (paired by order, optional `report`
    per pair); each pair is stored as ``frame_00/``, ``frame_01/`` …  A single
    pair is stored flat. A .zip may contain either layout.
    """
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = request.form.get("name") or "upload"
    name = "".join(c for c in name if c.isalnum() or c in "._-") or "upload"
    dest = UPLOADS / name
    dest.mkdir(parents=True, exist_ok=True)

    files = request.files
    if "zip" in files:
        zpath = dest / "_u.zip"
        files["zip"].save(zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        zpath.unlink(missing_ok=True)
        # flatten if single subdir holds the frame(s)
        subs = [p for p in dest.iterdir() if p.is_dir()]
        if not (dest / "rgb.png").exists() and _frame_count(dest) == 0 and len(subs) == 1:
            for f in subs[0].iterdir():
                f.rename(dest / f.name)
    else:
        rgbs = files.getlist("rgb")
        deps = files.getlist("depth")
        reps = files.getlist("report")
        if len(rgbs) != len(deps) or not rgbs:
            return jsonify(error="need matching rgb + depth file(s)"), 400
        if len(rgbs) == 1:                                   # single frame, flat
            rgbs[0].save(dest / "rgb.png")
            deps[0].save(dest / "depth_aligned.npy")
            if reps and reps[0].filename:
                reps[0].save(dest / "point_cloud_report.json")
        else:                                                # multi frame
            for i, (r, d) in enumerate(zip(rgbs, deps)):
                fd = dest / f"frame_{i:02d}"
                fd.mkdir(parents=True, exist_ok=True)
                r.save(fd / "rgb.png")
                d.save(fd / "depth_aligned.npy")
                if i < len(reps) and reps[i].filename:       # skip empty placeholders
                    reps[i].save(fd / "point_cloud_report.json")

    if _frame_count(dest) == 0:
        return jsonify(error="need rgb.png + depth_aligned.npy (+ report json)"), 400
    _ensure_reports(dest)
    return jsonify(name=name, frames=_frame_count(dest))


def _load_viz(name: str) -> dict | None:
    for base in (OUT, DEMO):
        p = base / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


@app.route("/api/quote/<name>", methods=["POST"])
def api_quote(name):
    """Generate the moving-quote .docx from the run result + the web form fields.

    Body = the quote form (customer / schedule / cost fields); the truck and the
    detected inventory are pulled from the cached pipeline result for `name`.
    """
    viz = _load_viz(name)
    if viz is None:
        return jsonify(error=f"no result for '{name}' — run the pipeline first"), 404
    import quote_doc
    form = request.get_json(silent=True) or {}
    data = quote_doc.data_from_viz(viz, form)
    QUOTES.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in "._-") or "quote"
    out = QUOTES / f"{safe}_quote.docx"
    quote_doc.build_quote(data, str(out))
    return send_file(out, as_attachment=True, download_name=f"견적서_{safe}.docx",
                     mimetype="application/vnd.openxmlformats-officedocument."
                              "wordprocessingml.document")


@app.route("/api/run/<name>", methods=["POST"])
def api_run(name):
    folder = _resolve_dir(name)
    if folder is None:
        return jsonify(error=f"capture '{name}' not found"), 404

    body = request.get_json(silent=True) or {}
    catalog = _parse_catalog(body.get("catalog")) or _catalog()

    q: queue.Queue = queue.Queue()

    def cb(stage, message, progress):
        q.put({"type": "progress", "stage": stage, "message": message,
               "progress": progress})

    def worker():
        try:
            import pipeline_run
            OUT.mkdir(parents=True, exist_ok=True)
            viz = pipeline_run.run_capture(
                folder, catalog=catalog, progress=cb,
                overlay_path=OUT / "overlays" / f"{name}.png")
            pipeline_run.save_viz(viz, OUT)
            q.put({"type": "done", "result": viz})
        except Exception as e:  # noqa: BLE001
            import traceback
            q.put({"type": "error", "error": str(e),
                   "trace": traceback.format_exc()[-1500:]})

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            evt = q.get()
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            if evt["type"] in ("done", "error"):
                break

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _preload_models():
    """Eagerly load the heavy models so the FIRST pipeline run is fast.

    SAM2 always (segmentation runs every time); local Gemma only when it's the active
    VLM backend. Failures raise — no silent skip.
    """
    import sam2_masks
    print("[RBF] preloading SAM2 …", flush=True)
    sam2_masks.preload()
    print("[RBF] SAM2 ready", flush=True)

    st = _config_status()
    if st["active_backend"].startswith("gemma"):
        import gemma_vlm
        print(f"[RBF] preloading local Gemma ({st['vlm_backend']}) …", flush=True)
        gemma_vlm.preload(st["vlm_backend"])
        print("[RBF] Gemma ready", flush=True)


if __name__ == "__main__":
    from env_check import require_python
    try:
        require_python()
    except RuntimeError as e:
        print(f"[RBF] ERROR: {e}", flush=True)
        raise SystemExit(1) from e

    print(f"[RBF] workspace: {DEPTH_ROOT}  | gemini: {bool(os.environ.get('GEMINI_API_KEY'))}")
    if os.environ.get("RBF_PRELOAD", "1") != "0":
        try:
            _preload_models()
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[RBF] model preload failed:\n{e}", flush=True)
            traceback.print_exc()
            raise SystemExit(1) from e
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)),
            threaded=True, debug=False)
