#!/usr/bin/env bash
# One-shot model installer for RealSense Box-Fit.
#
# Downloads everything the pipeline needs and writes the paths into
# config.local.json so the web UI is pre-filled — the user never touches an
# env var:
#   - SAM2 checkpoint (segmentation)
#   - Gemma 4 E4B  (local VLM, light)        google/gemma-4-E4B-it
#   - Gemma 4 31B   (local VLM, high quality) google/gemma-4-31B-it
#
# Usage (safe to run OR source — it never exits your shell):
#   bash setup_models.sh                 # download all into ./models, write config
#   bash setup_models.sh --skip-gemma    # SAM only
#   bash setup_models.sh --skip-sam      # Gemma only
#
# Gemma models are gated: accept the license on huggingface.co once, then either
# `huggingface-cli login` or paste your token when prompted.
#
# NOTE: deliberately NO `set -euo pipefail` / `exit` — a failed download must not
# kill the terminal. Failures print a warning and the script keeps going.

# Download one HF repo into a dir; warn and continue on failure.
_rbf_hf_download() {
  local py="$1" dest="$2" repo="$3"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "[gemma] already present: $dest"
    return 0
  fi
  echo "[gemma] downloading $repo -> $dest"
  if ! HF_DEST="$dest" HF_REPO="$repo" "$py" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["HF_REPO"], local_dir=os.environ["HF_DEST"],
                  token=os.environ.get("HF_TOKEN"))
PYEOF
  then
    echo "WARN: [gemma] $repo download failed (check HF token / license acceptance); continuing"
  fi
}

rbf_setup_models() {
  local ROOT MODELS PY a cand SKIP_SAM SKIP_GEMMA
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  MODELS="$ROOT/models"

  SKIP_SAM=0; SKIP_GEMMA=0
  for a in "$@"; do
    case "$a" in
      --skip-sam) SKIP_SAM=1 ;;
      --skip-gemma) SKIP_GEMMA=1 ;;
      *) echo "unknown arg: $a (use --skip-sam / --skip-gemma)"; return 2 ;;
    esac
  done

  # Pick a python that actually has huggingface_hub (don't assume bare `python`).
  # Set RBF_PYTHON=/path/to/python to force a specific interpreter/env.
  PY="${RBF_PYTHON:-}"
  if [ -z "$PY" ]; then
    for cand in python python3; do
      if command -v "$cand" >/dev/null 2>&1 \
         && "$cand" -c "import huggingface_hub" >/dev/null 2>&1; then
        PY="$cand"; break
      fi
    done
  fi
  [ -z "$PY" ] && PY="python"
  echo "[setup] using python: $PY"
  if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "WARN: '$PY' has no huggingface_hub. Install it:  $PY -m pip install huggingface_hub"
    echo "  or rerun with RBF_PYTHON=/path/to/python. (continuing; Gemma will be skipped on failure)"
  fi

  mkdir -p "$MODELS/sam2" "$MODELS/gemma" \
    || { echo "WARN: cannot create $MODELS"; return 1; }

  # --- SAM2 checkpoint ------------------------------------------------------ #
  local SAM2_CKPT SAM2_URL SAM2_CFG
  SAM2_CKPT="$MODELS/sam2/sam2_hiera_large.pt"
  SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
  SAM2_CFG="sam2_hiera_l.yaml"           # resolved on the sam2 hydra search path
  if [ "$SKIP_SAM" -eq 0 ]; then
    if [ -s "$SAM2_CKPT" ]; then
      echo "[sam2] already present: $SAM2_CKPT"
    else
      echo "[sam2] downloading checkpoint -> $SAM2_CKPT"
      if command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "$SAM2_CKPT" "$SAM2_URL" \
          || echo "WARN: [sam2] download failed (continuing)"
      elif command -v wget >/dev/null 2>&1; then
        wget -O "$SAM2_CKPT" "$SAM2_URL" \
          || echo "WARN: [sam2] download failed (continuing)"
      else
        echo "WARN: [sam2] neither curl nor wget found; skipping"
      fi
    fi
  fi

  # --- Gemma (HuggingFace) -------------------------------------------------- #
  local GEMMA_E4B_DIR GEMMA_31B_DIR
  GEMMA_E4B_DIR="$MODELS/gemma/gemma-4-E4B-it"
  GEMMA_31B_DIR="$MODELS/gemma/gemma-4-31B-it"
  if [ "$SKIP_GEMMA" -eq 0 ]; then
    if [ -z "${HF_TOKEN:-}" ] \
       && ! "$PY" -c "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" >/dev/null 2>&1; then
      echo "[gemma] gated models — accept the license on huggingface.co, then paste your HF token."
      read -r -s -p "        HF token (leave blank if already logged in): " HF_TOKEN
      echo
      [ -n "$HF_TOKEN" ] && export HF_TOKEN
    fi
    _rbf_hf_download "$PY" "$GEMMA_E4B_DIR" "google/gemma-4-E4B-it"
    _rbf_hf_download "$PY" "$GEMMA_31B_DIR" "google/gemma-4-31B-it"
  fi

  # --- write paths into config.local.json (only what actually downloaded) --- #
  echo "[config] writing model paths into config.local.json"
  RBF_SAM2_CKPT="$SAM2_CKPT" RBF_SAM2_CFG="$SAM2_CFG" \
  RBF_GEMMA_E4B="$GEMMA_E4B_DIR" RBF_GEMMA_31B="$GEMMA_31B_DIR" \
  RBF_SKIP_SAM="$SKIP_SAM" RBF_SKIP_GEMMA="$SKIP_GEMMA" \
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" - <<'PYEOF'
import os
import config


def _has_weights(d):
    # a real model dir has weight shards — README-only (failed gated download) doesn't
    return os.path.isdir(d) and any(
        f.endswith((".safetensors", ".bin", ".gguf", ".pt"))
        for f in os.listdir(d))


updates = {}
if os.environ["RBF_SKIP_SAM"] == "0" and os.path.isfile(os.environ["RBF_SAM2_CKPT"]):
    updates["sam2_ckpt"] = os.environ["RBF_SAM2_CKPT"]
    updates["sam2_cfg"] = os.environ["RBF_SAM2_CFG"]
if os.environ["RBF_SKIP_GEMMA"] == "0":
    for key, env in (("gemma_e4b", "RBF_GEMMA_E4B"), ("gemma_31b", "RBF_GEMMA_31B")):
        d = os.environ[env]
        if _has_weights(d):
            updates[key] = d
        else:
            print(f"  WARN: {key}: no model weights in {d} — download incomplete, NOT saved")
cfg = config.save(updates)
print("  saved:", {k: cfg.get(k) for k in
      ("sam2_ckpt", "sam2_cfg", "gemma_e4b", "gemma_31b")})
PYEOF

  echo
  echo "Done. Start the app:  $PY app.py"
  echo "(Paths are pre-filled in the UI; enter a Gemini key there if you have one.)"
  return 0
}

# Run for both `bash setup_models.sh` and `source setup_models.sh` — using a
# function + return means a failure never closes the caller's terminal.
rbf_setup_models "$@"
