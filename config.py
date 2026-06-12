"""Persisted runtime config so users never touch env vars or `export`.

Everything lives in `config.local.json` at the project root (git-ignored):
  - model paths written automatically by `setup_models.sh`
    (`sam2_ckpt`, `sam2_cfg`, `gemma_e4b`, `gemma_31b`)
  - the Gemini API key the user types into the web UI (`gemini_api_key`) — saved
    once, then kept across restarts
  - which VLM to use (`vlm_backend`: auto | gemini | gemma_e4b | gemma_31b | heuristic)
  - the moving-box catalog the user edits in the web UI (`box_catalog`)

`apply_to_env()` mirrors the relevant keys into the env vars the existing modules
already read (RBF_SAM2_CKPT / RBF_SAM2_CFG / GEMINI_API_KEY / GEMINI_MODEL), so the
rest of the pipeline needs no changes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("RBF_CONFIG", str(ROOT / "config.local.json")))

# config key -> env var consumed by existing code
_ENV_MAP = {
    "sam2_ckpt": "RBF_SAM2_CKPT",
    "sam2_cfg": "RBF_SAM2_CFG",
    "gemini_api_key": "GEMINI_API_KEY",
    "gemini_model": "GEMINI_MODEL",
}

_DEFAULTS = {
    "vlm_backend": "auto",
    "gemini_model": "gemini-3.5-flash",
    "gemma_think": True,
    "gemma_think_tokens": 4096,
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**_DEFAULTS, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULTS)


def save(updates: dict) -> dict:
    """Merge `updates` into the config file and return the full config."""
    cfg = load()
    cfg.update({k: v for k, v in updates.items() if v is not None})
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return cfg


def get(key: str, default=None):
    """Config value, else the mapped env var, else `default`."""
    cfg = load()
    if cfg.get(key):
        return cfg[key]
    env = _ENV_MAP.get(key)
    if env and os.environ.get(env):
        return os.environ[env]
    return default


def apply_to_env(cfg: dict | None = None) -> None:
    """Mirror config values into the env vars the existing modules read."""
    cfg = cfg or load()
    for key, env in _ENV_MAP.items():
        val = cfg.get(key)
        if val:
            os.environ[env] = str(val)


def load_box_catalog() -> list[dict]:
    """Return the persisted box catalog, or the built-in defaults."""
    from box_fit import boxes_from_items, catalog_items_from_boxes, default_box_catalog_items

    items = load().get("box_catalog")
    if isinstance(items, list) and boxes_from_items(items):
        return items
    return default_box_catalog_items()


def save_box_catalog(items: list) -> list[dict]:
    """Validate, persist, and return the box catalog."""
    from box_fit import boxes_from_items, catalog_items_from_boxes

    boxes = boxes_from_items(items)
    normalized = catalog_items_from_boxes(boxes)
    save({"box_catalog": normalized})
    return normalized
