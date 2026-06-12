"""Runtime checks for SAM2 + local Gemma — fail loudly instead of silently skipping."""
from __future__ import annotations

import sys
from pathlib import Path

_MIN_PYTHON = (3, 10)
_VENV = Path(__file__).resolve().parent / ".venv" / "bin" / "python"


def python_ok() -> bool:
    return sys.version_info[:2] >= _MIN_PYTHON


def python_hint() -> str:
    venv_py = _VENV
    if venv_py.is_file():
        return (f"Use the project venv:  {venv_py} app.py\n"
                f"  or:  source .venv/bin/activate && python app.py")
    return ("Install Python >= 3.10 and create a venv:\n"
            "  /opt/homebrew/bin/python3.10 -m venv .venv\n"
            "  .venv/bin/pip install -r requirements-full.txt\n"
            "  .venv/bin/pip install 'transformers>=5.5.0' "
            "'git+https://github.com/facebookresearch/segment-anything-2.git'")


def require_python() -> None:
    if python_ok():
        return
    raise RuntimeError(
        f"Python {'.'.join(map(str, _MIN_PYTHON))}+ is required for SAM2 and Gemma 4 "
        f"(this interpreter is {sys.version.split()[0]}).\n{python_hint()}")


def _import_err(name: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"{name} is not available in this Python ({sys.executable}).\n"
        f"  original error: {exc}\n{python_hint()}")


def require_sam2() -> None:
    require_python()
    try:
        import sam2  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise _import_err("SAM2 (segment-anything-2)", e) from e
    ckpt = Path(__import__("sam2_masks", fromlist=["SAM2_CKPT"]).SAM2_CKPT)
    if not ckpt.is_file():
        raise RuntimeError(
            f"SAM2 checkpoint not found: {ckpt}\n"
            "Run:  bash setup_models.sh")


def require_gemma_support() -> None:
    require_python()
    try:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception as e:  # noqa: BLE001
        raise _import_err("transformers", e) from e
    if "gemma4" not in CONFIG_MAPPING:
        raise RuntimeError(
            f"transformers {getattr(transformers, '__version__', '?')} does not support "
            "Gemma 4 (model_type `gemma4`). Upgrade:\n"
            "  pip install 'transformers>=5.5.0'\n"
            f"{python_hint()}")
