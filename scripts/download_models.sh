#!/bin/bash
# Pre-fetch every model on a login node (compute nodes may not have internet access).
# HF_HOME must match the Slurm jobs ($WORK/.cache/huggingface), which slurm/generate.py sets.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${WORK:-}" ]]; then export HF_HOME="$WORK/.cache/huggingface"; fi
python - <<'PY'
from ultralytics import YOLO
from ultralytics.utils.checks import check_font

for name in ("yolo26s-depth.pt", "yolo26m-depth.pt", "yolo26x-depth.pt", "yolo26s-sem.pt"):
    YOLO(name)
check_font("Arial.ttf")

import transformers
from mono_depth.models import MODELS

for name, spec in MODELS.items():
    if spec.family == "hf":
        transformers.AutoImageProcessor.from_pretrained(spec.weights)
        transformers.AutoModelForDepthEstimation.from_pretrained(spec.weights)
        print("cached", name)
try:
    from depth_anything_3.api import DepthAnything3

    for name, spec in MODELS.items():
        if spec.family == "da3":
            DepthAnything3.from_pretrained(spec.weights)
            print("cached", name)
except ImportError:
    print("depth-anything-3 not installed; see scripts/install_da3.sh")
PY
