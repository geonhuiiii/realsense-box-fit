"""Cross-platform torch device / dtype selection (CUDA, Apple-Silicon MPS, CPU).

Centralises the "what hardware do we run on" decision so SAM2 and Gemma behave on
macOS as well as Linux/CUDA:

- device: CUDA if present, else Apple-Silicon **MPS**, else CPU.
- dtype: ``bfloat16`` only on CUDA (CPU/MPS don't support it well) — float32 elsewhere.

When MPS is chosen we also enable ``PYTORCH_ENABLE_MPS_FALLBACK`` so the handful of
ops SAM2/transformers run that MPS doesn't implement transparently fall back to CPU
instead of raising. Pass an already-imported ``torch`` module (callers import it
lazily) or let the helper import it.
"""
from __future__ import annotations

import os


def _torch(torch=None):
    if torch is not None:
        return torch
    import torch as _t
    return _t


def pick_device(torch=None) -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    torch = _torch(torch)
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # Let unsupported MPS ops fall back to CPU instead of raising.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def pick_dtype(device: str, torch=None):
    """bfloat16 on CUDA (memory win, well supported); float32 on CPU/MPS."""
    torch = _torch(torch)
    return torch.bfloat16 if device == "cuda" else torch.float32
