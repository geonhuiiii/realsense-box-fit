"""VLM entry point — picks a backend and returns per-object records.

Dispatch (driven by `config.vlm_backend`, default "auto"):
  - "auto":      Gemini if an API key is set, else local Gemma if downloaded,
                 else the deterministic heuristic.
  - "gemini":    force the Gemini API (falls through to heuristic if it fails).
  - "gemma_e4b" / "gemma_31b": force that local Gemma model.
  - "heuristic": skip any model.

Always returns one record per object; never raises.
"""
from __future__ import annotations

import logging
import os

import numpy as np

import config
import gemini_vlm

log = logging.getLogger(__name__)


def _has_gemini_key() -> bool:
    return bool(config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))


def _has_gemma() -> bool:
    return bool(config.get("gemma_e4b") or config.get("gemma_31b"))


def analyze(rgb: np.ndarray, objects: list[dict], catalog=None) -> list[dict]:
    """Per-object records incl. the VLM-chosen `box`. Never raises."""
    if not objects:
        return []
    backend = (config.get("vlm_backend") or "auto").lower()

    if backend == "heuristic":
        return gemini_vlm._heuristic(objects, catalog)

    if backend == "gemini" or (backend == "auto" and _has_gemini_key()):
        recs = gemini_vlm._gemini(rgb, objects, catalog)
        if recs is not None:
            log.info("VLM via Gemini")
            return recs
        # explicit gemini request can still fall back so the run completes
        if backend == "gemini":
            return gemini_vlm._heuristic(objects, catalog)

    if backend in ("gemma_e4b", "gemma_31b", "gemma") or (backend == "auto" and _has_gemma()):
        import gemma_vlm
        recs = gemma_vlm.analyze(rgb, objects, catalog, backend=backend)
        if recs is not None:
            log.info("VLM via local Gemma (%s)", backend)
            return recs

    return gemini_vlm._heuristic(objects, catalog)
