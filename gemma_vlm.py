"""Local Gemma VLM backend — used when there is no Gemini API key.

Runs a local multimodal Gemma (downloaded by `setup_models.sh`) via 🤗
transformers and returns the same per-object records as the Gemini path
(reusing its prompt / schema / heuristic). Model path comes from `config.py`
(`gemma_e4b` or `gemma_31b`); selection is driven by `vlm_backend`.

Heavy (torch/transformers) imports are lazy and the model is cached, so importing
this module is cheap. Any failure degrades to the heuristic — never raises.

Gemma is run in a "think first" mode: it reasons step by step (filter walls,
catch duplicate masks, reason per-axis compression) before emitting the final
JSON, with a generous token budget. Tune via config: `gemma_think_tokens`
(default 8192) and `gemma_think` (false to skip the reasoning preface).
"""
from __future__ import annotations

import json
import logging
import re
import threading

import numpy as np

import config
from gemini_vlm import _build_prompt, _heuristic, _merge_record

log = logging.getLogger(__name__)

_PIPE = {}        # cache: model_path -> transformers pipeline
_LOCK = threading.Lock()        # guard pipeline build (preload thread vs run worker)

# Default "thinking budget" for Gemma (max new tokens). Raise via config key
# `gemma_think_tokens` to let it reason longer. Set `gemma_think` to false to skip
# the chain-of-thought preface (faster, less thorough).
_THINK_TOKENS_DEFAULT = 2048

# Appended to the prompt for Gemma only (Gemini uses structured output instead):
# make it reason step by step BEFORE emitting the final JSON, so it filters
# walls/duplicates and reasons per-axis compression more carefully.
_THINK_SUFFIX = """

먼저 <thinking> ... </thinking> 안에서 천천히 단계별로 추론하세요. 서두르지 말고 각 id마다:
- 무엇인지(옮길 짐인지, 아니면 벽/바닥/천장 같은 구조물인지) — 벽과 직물(커튼/러그)·옷더미를
  혼동하지 않았는지 근거를 적으세요.
- 다른 id와 같은 물체(중복/부분 마스크)인지.
- 재질이 단단한지 부드러운지, 포장하면 각 축이 어떻게 줄어드는지.
충분히 따져본 뒤 </thinking>을 닫고, 최종 답을 단 하나의 ```json 코드블록으로만 출력하세요
(스키마의 "objects" 객체). 최종 JSON은 반드시 ```json 과 ``` 사이에 두고 그 밖에는 JSON을 쓰지 마세요."""


def model_path_for(backend: str) -> str | None:
    """Resolve the local Gemma path for a backend name."""
    if backend == "gemma_31b":
        return config.get("gemma_31b")
    if backend in ("gemma_e4b", "gemma"):
        return config.get("gemma_e4b")
    # auto: prefer the lighter E4B, then the 31B
    return config.get("gemma_e4b") or config.get("gemma_31b")


def _get_pipe(path: str):
    if path in _PIPE:
        return _PIPE[path]
    with _LOCK:                 # double-checked: only one thread builds this model
        if path in _PIPE:
            return _PIPE[path]
        import torch
        from transformers import pipeline
        from torch_device import pick_device, pick_dtype
        device = pick_device(torch)         # cuda → Apple-Silicon mps → cpu
        kwargs = {"torch_dtype": pick_dtype(device, torch)}
        if device == "cuda":
            kwargs["device_map"] = "auto"   # accelerate sharding (CUDA only)
        else:
            kwargs["device"] = device       # explicit cpu / mps placement
        pipe = pipeline("image-text-to-text", model=path, **kwargs)
        _PIPE[path] = pipe
    return _PIPE[path]


def preload(backend: str = "auto") -> bool:
    """Eagerly load the local Gemma model for `backend` so the first run isn't slow.
    Best-effort: returns True if loaded, False if no path or torch/transformers fails."""
    path = model_path_for(backend)
    if not path:
        return False
    try:
        _get_pipe(path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _match_bracket(text: str, start: int):
    """End index (exclusive) of the {...}/[...] opening at `start`, or None."""
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == open_c) - (text[i] == close_c)
        if depth == 0:
            return i + 1
    return None


def _loads(blob: str):
    """json.loads tolerant of trailing commas (,} -> } and ,] -> ])."""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", blob))


def _scan_json_values(text: str):
    """Every top-level JSON value ({...} or [...]) found in `text`, in order."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] in "{[":
            end = _match_bracket(text, i)
            if end:
                try:
                    out.append(_loads(text[i:end]))
                except json.JSONDecodeError:
                    pass
                i = end
                continue
        i += 1
    return out


def _record_count(val) -> int:
    """How many object-records a parsed value yields (to pick the real answer over
    stray brackets like [120,10,10] that appear in reasoning prose)."""
    return sum(1 for r in _objects_from(val) if isinstance(r, dict))


def _extract_json(text: str):
    """Pull the answer JSON from a reply that may contain step-by-step thinking.

    Gemma is asked to reason first, then emit the answer. So we (1) drop any
    `<thinking>...</thinking>` block, (2) search the LAST fenced ```json block first,
    then the whole text, and (3) among all JSON values found, pick the one with the
    MOST object-records — so a stray `[120,10,10]` in the reasoning never wins over
    the real `{"objects":[...]}` answer.
    """
    text = text.strip()
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)
    fences = [f.strip() for f in
              re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
              if f.strip()]
    for blob in reversed(fences) if fences else [text]:   # final fenced block wins
        vals = _scan_json_values(blob)
        scored = [(v, _record_count(v)) for v in vals]
        usable = [v for v, c in scored if c >= 1]
        if usable:
            return max(usable, key=_record_count)
        if vals:                                          # parsed but recordless
            return vals[0]
    # no fence matched: scan the whole text and pick the richest value
    vals = _scan_json_values(text)
    usable = [v for v in vals if _record_count(v) >= 1]
    if usable:
        return max(usable, key=_record_count)
    if vals:
        return vals[0]
    raise ValueError("no JSON value in model output")


def _reply_text(out) -> str:
    """Pull the assistant's text out of a transformers pipeline output (many shapes)."""
    o = out[0] if isinstance(out, list) and out else out
    gen = o.get("generated_text", o) if isinstance(o, dict) else o
    if isinstance(gen, list) and gen:           # list of chat messages
        last = gen[-1]
        c = last.get("content", last) if isinstance(last, dict) else last
        if isinstance(c, list):                 # content as list of {type,text} parts
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c if isinstance(c, str) else str(c)
    return gen if isinstance(gen, str) else str(gen)


def _objects_from(data) -> list:
    """The local model may return {'objects':[...]}, a bare [...], or {'results':[...]}."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("objects", "results", "items", "predictions"):
            if isinstance(data.get(k), list):
                return data[k]
    return []


def _rec_id(r: dict):
    """The model labels objects with 'id' (as the prompt does) or 'instance_id'."""
    for k in ("instance_id", "id", "index", "idx"):
        if k in r:
            try:
                return int(r[k])
            except (ValueError, TypeError):
                pass
    return None


def analyze(rgb: np.ndarray, objects: list[dict], catalog=None,
            backend: str = "auto") -> list[dict] | None:
    """Run local Gemma; return per-object records (source='gemma') or None on failure."""
    path = model_path_for(backend)
    if not path:
        return None

    prompt = _build_prompt(objects, catalog)
    cfg = config.load()
    think = cfg.get("gemma_think", True)            # read raw so `false` is honoured
    think = think if isinstance(think, bool) else str(think).lower() not in ("0", "false", "no")
    if think:
        prompt = prompt + _THINK_SUFFIX
    try:
        max_tokens = int(cfg.get("gemma_think_tokens", _THINK_TOKENS_DEFAULT))
    except (TypeError, ValueError):
        max_tokens = _THINK_TOKENS_DEFAULT

    reply = ""
    try:
        from PIL import Image
        pipe = _get_pipe(path)
        img = Image.fromarray(rgb.astype(np.uint8))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        out = pipe(text=messages, max_new_tokens=max_tokens, do_sample=False)
        reply = _reply_text(out)
        data = _extract_json(reply)
        recs = {}
        for r in _objects_from(data):
            rid = _rec_id(r)                    # 'id' or 'instance_id', possibly "0"
            if rid is not None:
                recs[rid] = r
    except Exception as e:  # noqa: BLE001
        hint = ""
        if "PyTorch" in str(e) or "torch" in str(e).lower():
            import torch
            hint = (f" — local Gemma 4 needs torch>=2.4 + a transformers with gemma4 "
                    f"support, but this env has torch {torch.__version__}. Set a "
                    f"GEMINI_API_KEY (UI) to use Gemini instead, or upgrade torch.")
        log.warning("Gemma VLM failed (%s); using heuristic.%s", e, hint)
        return None

    matched = sum(1 for o in objects if int(o["instance_id"]) in recs)
    if matched == 0:                            # model replied but nothing usable
        log.warning("Gemma returned no matching objects (parsed %d; reply head: %r); "
                    "using heuristic.", len(recs), reply[:240])
        return None
    if matched < len(objects):
        log.info("Gemma matched %d/%d objects (rest -> heuristic).", matched, len(objects))

    out_recs = []
    for o in objects:
        r = recs.get(int(o["instance_id"]))
        out_recs.append(_merge_record(o, r, "gemma", catalog) if r
                        else _heuristic([o], catalog)[0])
    return out_recs
