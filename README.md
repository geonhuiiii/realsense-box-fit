# RealSense Box-Fit

Measure household moving items from a single RealSense L515 RGB-D capture, decide
how to pack them into real moving boxes, and inspect everything in an interactive
3D web UI.

**Pipeline:**
`RGB-D → SAM2 segmentation → point cloud back-projection → per-object OBB
(Nelder-Mead) → VLM (Gemini): filter / compress / fold → items into moving boxes
→ boxes + furniture onto a truck → quote by tonnage → dual-view 3D visualization
(segmentation + OBB | truck loading + quote)`

The goal is a **moving quote**: objects are boxed, the boxes plus loose furniture
are loaded onto the smallest/cheapest Korean cargo truck(s), and the move is
priced by truck tonnage (1톤 → 25톤 class).

The web UI is a Three.js dual-view (dark theme, OrbitControls, SSE progress)
cloned from the RMC-Visual-Agent moving-quote viewer, with the quoting pipeline
replaced by our segmentation → OBB → VLM → box-packing pipeline.

---

## Pipeline stages

1. **Segment** — hybrid SAM2: **grid-sampling** primary (a grid of foreground
   prompt points → a SAM2 mask each → overlapping masks are **merged/unioned** so one
   object's partial masks don't fragment), with a depth-foreground + point-prompt
   fallback for big low-texture objects. Each mask is depth-banded so background
   bleed can't inflate the OBB.
2. **Point cloud** — back-project masked depth with corrected L515 intrinsics.
3. **OBB** — Nelder-Mead simplex over 3 Euler angles → object dimensions.
4. **VLM (the decision-maker)** — SAM gives masks only (no labels). The VLM is sent
   an **annotated image**: each object's mask outline + its 3D OBB *reprojected onto
   the photo* + an id number (the same id the prompt lists with that object's size
   and volume). From this the VLM **identifies** each object, **drops mis-segments /
   background** (`keep:false`), marks `compressible` / `foldable`, gives corrected
   (packed) dimensions, and **chooses the box** from the catalog (or "" = large
   furniture, ride loose). Backend: **Gemini API** if a key is set, else **local
   Gemma 4** (downloaded by `setup_models.sh`), else a deterministic heuristic.
5. **Box packing** — places each item into the box the VLM chose (3-D shelf packer).
   The VLM is allowed to **cram**: if its chosen box can't geometrically hold the
   item, the item is forced in anyway (flagged `crammed`) — it opens more boxes of
   that type rather than rerouting to a different box. Items the VLM marked loose
   become furniture on the truck.
6. **Truck loading + quote** — the packed boxes plus loose furniture are 3-D packed
   into real Korean cargo trucks (탑차/윙바디). For each tonnage we compute how many
   trucks are needed (greedy fill) and the fare; the **cheapest plan wins** (so one
   2.5톤 can beat two 1톤). Covers 1톤 → 25톤 class (20톤급+). The right view shows
   the chosen truck(s) with cargo packed inside; the summary shows the quote and
   alternative tonnages.
7. **견적서 생성 (quote document)** — fill the supplied Word form (`quote_template.docx`)
   into a finished 견적서 `.docx`. When a run finishes, the **VLM reviews the whole result
   and predicts the entire quote** (`quote_estimate.py`): 작업유형 / 작업 인원 / 입·출고
   비용 / 계약금 / 합계 / 고객 요청사항, all pre-filled into the **견적서 생성** form for
   the human to tweak. This estimate uses the **same backend as the main VLM stage** —
   set a Gemini key + `auto`/`gemini` and it uses the Gemini API; otherwise local Gemma;
   otherwise a tonnage/volume heuristic. The chosen **truck**, the **합계**, the **세부비용**
   breakdown and the **보관짐** inventory (detected items + assigned boxes, dropped
   detections excluded) come straight from the run. Customer / schedule fields are the
   only ones left blank. One click downloads the document (`quote_doc.py`).

### Truck catalog (`trucks.py`)
Interior cargo dimensions of Korean enclosed trucks (탑차/윙바디) — the binding
constraint for a move (not the open flat-bed gate height). 1톤·1.4톤·2.5톤·3.5톤·5톤·
5톤축·8톤·11톤·14톤·18톤·25톤, each with a ballpark **all-in single-trip move fare**
— the real 이사 price that bundles truck + driver + loading/unloading crew (인부
인건비) for one metro trip, not bare vehicle hire (so it climbs with tonnage as the
crew grows). Editable. Sources: carcar.kr / sendy.ai / dabori spec tables.

### L515 metadata corrections (important)
The capture's `point_cloud_report.json` has two wrong values; the code overrides
them (`run_box_fit.py`):
- `depth_scale_m_per_unit`: `0.001` → **`0.00025`** (L515 native unit).
- `fy`: `684.63` → **`fy = fx ≈ 913`** (depth is aligned to the color frame, so
  square pixels / 70° HFOV). Verified: perpendicular walls only square up with `fy=fx`.

---

## Install

```bash
pip install -r requirements-full.txt          # web UI + pipeline (+ huggingface_hub, transformers)
# SAM2 + a PyTorch build must be installed separately.

bash setup_models.sh                          # download SAM2 + Gemma, auto-fill config
```

Runs on **Linux/CUDA, macOS (Apple-Silicon MPS or CPU), and CPU-only** — the device
is auto-detected (`torch_device.py`): CUDA → Apple-Silicon **MPS** → CPU, with
`bfloat16` on CUDA and `float32` elsewhere (CPU/MPS don't support bfloat16 well). On
Apple Silicon `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so the few ops MPS lacks fall
back to CPU instead of erroring. No CUDA is required — local Gemma is just slower on
CPU/MPS, so a **Gemini API key** is the smoother path on a Mac.

**No env vars / `export` needed.** `setup_models.sh` downloads:
- **SAM2** checkpoint (segmentation),
- **Gemma 4 E4B** (`google/gemma-4-E4B-it`) — light local VLM,
- **Gemma 4 31B** (`google/gemma-4-31B-it`) — high-quality local VLM,

into `./models/` (git-ignored) and writes their paths into `config.local.json`, so
the web UI is pre-filled. The Gemma repos are gated — accept the license on
huggingface.co once; the script prompts for your HF token if you aren't logged in.
Flags: `--skip-sam`, `--skip-gemma`.

### VLM backend (set in the UI, no env vars)
Open **Models & API key** in the sidebar:
- **Backend:** `auto` (Gemini key → local Gemma → heuristic), or force Gemini /
  Gemma E4B / Gemma 31B / heuristic.
- **Gemini API key:** paste it once → saved to `config.local.json` and **kept across
  restarts** (never an env var). Without a key the pipeline uses local Gemma; with
  no models either, a deterministic heuristic.
- **Model paths:** shown read-only, filled in by `setup_models.sh`.

(Everything is still overridable by env vars — `RBF_SAM2_CKPT`, `RBF_SAM2_CFG`,
`GEMINI_API_KEY`, `GEMINI_MODEL` — but the UI/config is the intended path.)

> **Local Gemma 4 needs `torch>=2.4`** (a recent transformers with the `gemma4`
> model requires it). If your SAM2 env is pinned to an older torch, transformers
> disables PyTorch and Gemma falls back to the heuristic — the log says so. Either
> run Gemma in an env with `torch>=2.4`, or just set a **Gemini API key** in the UI
> (works with any torch). SAM2 itself runs fine on torch 2.3.x.

---

## Run

```bash
python app.py            # http://localhost:8000
```
On startup the heavy models are **preloaded in the background** so the first run is
fast: SAM2 always, plus the local Gemma when it's the active VLM backend (skipped for
Gemini/heuristic). It's best-effort — a viewer-only env without torch/SAM2 just logs
and skips. Disable with `RBF_PRELOAD=0`.

- **Capture (scene)** dropdown → **▶ Run pipeline** → watch SSE progress.
- **Box catalog (editable):** the sidebar shows a scrollable list of boxes — edit
  name/size, add (`+ add box`) or delete (`✕`) rows. The packer chooses the
  smallest effective box(es) from *your* list each run.
- **Multi-frame upload:** a house is never one shot — add a frame row per view
  (`+ add frame`), each with its own RGB + depth (+ optional report). Every frame
  is segmented/measured independently, laid side-by-side in the left view, and all
  objects are packed together in the right view. A single pair, or a `.zip`
  (flat `rgb.png`+`depth_aligned.npy`, or `frame_00/…` subdirs), also works.
- **Left view:** segmented point clouds + Nelder-Mead OBB (per object, labelled),
  one cluster per frame.
- **Right view:** truck loading — the chosen truck cargo wireframe with the packed
  boxes (cyan) and loose furniture (amber) stacked inside; one truck per copy when
  multiple are needed.
- **Quote summary:** chosen tonnage × count, load utilisation, total cargo m³,
  estimated fare, and alternative tonnages.
- **견적서 생성:** open the **견적서 생성** panel, fill the customer / schedule / cost
  fields (truck, 합계 and item list are auto-filled from the run), and click
  **견적서 다운로드 (.docx)** to download the finished quote document.
- **Bottom table:** id, identity, material, measured vs corrected dims, assigned
  box, fold/compress flags, and the VLM's note.
- Toggles: points / OBB / labels / grid.

The data dir resolves as `outputs/web` then `demo/web`; selecting a folder loads
its cached result if present (so the shipped `demo/` renders without a GPU).

### Expected capture layout
Single-frame:
```
<workspace>/depth/<NNN_DIMS>/
    rgb.png
    depth_aligned.npy            # uint16, aligned to color
    point_cloud_report.json      # intrinsics (fy/scale auto-corrected)
```
Multi-frame (one scene, several shots):
```
<workspace>/depth/<scene>/
    frame_00/{rgb.png, depth_aligned.npy, point_cloud_report.json}
    frame_01/{...}
    ...
```
Folder names seed the default box catalog (e.g. `006_37X44X40`), but the catalog
is fully editable in the UI before each run. `RBF_WORKSPACE` overrides the path.

### Rebuild the shipped demo
```bash
python build_demo.py             # runs the pipeline → demo/web/*.json
```

---

## Files
```
app.py            Flask web app (folders, upload, SSE run)
pipeline_run.py   one capture → viz dict (instances + packing)
sam2_masks.py     hybrid SAM2 segmentation
geometry.py       depth back-projection, RANSAC bg planes, OBB wrapper
obb_nelder_mead.py Nelder-Mead OBB (vendored, numpy+scipy only)
box_fit.py        real-box catalog + sorted-dim fit
vlm.py            VLM dispatcher (Gemini → local Gemma → heuristic)
gemini_vlm.py     Gemini VLM backend (+ shared prompt/schema/heuristic)
gemma_vlm.py      local Gemma VLM backend (transformers, used without an API key)
config.py         config.local.json (model paths + Gemini key, no env vars)
torch_device.py   cross-platform device/dtype pick (CUDA / Apple-Silicon MPS / CPU)
setup_models.sh   one-shot downloader: SAM2 + Gemma → ./models, writes config
packing.py        greedy 3D bin-packing (items → boxes)
trucks.py         Korean truck catalog + truck loading + tonnage quote
quote_estimate.py VLM predicts the full quote (작업/인원/비용/요청) after a run
quote_doc.py      fill the .docx quote form from a run result + form fields
quote_template.docx  the moving-quote form (견적서 양식), filled by quote_doc.py
run_box_fit.py    batch CLI (outputs/<f>/result.json, summary.csv)
render_results.py CLI PNG renderer (segmentation + 3D + OBB)
build_demo.py     build demo/web from the dataset
web/              templates + static (Three.js dual-view UI)
demo/             cached results so the UI works after clone
```

## Limitations
- Single-view: back/thickness under-measured; mask bleed can inflate a side.
- Packing is a visualization-grade greedy heuristic (not optimal).
- VLM corrections (fold/compress/filter) are judgment-based, not measurement.

## Attribution
- UI cloned from RMC-Visual-Agent-for-Moving-Quote (Three.js dual-view).
- `obb_nelder_mead.py` vendored from the same project (numpy+scipy only).
- Segmentation: Meta **SAM 2**. VLM: Google **Gemini** (both installed/configured separately).
