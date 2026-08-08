#!/bin/bash
# Depth Anything 3 (Lin et al., 2025) has no transformers port yet. Its pip package pins
# xformers, which does not build on every machine and is only an attention speed-up, so the
# package is installed without it. Needs numpy < 2 (the package's requirement).
#   bash scripts/install_da3.sh               # into the active environment
set -euo pipefail
pip install --no-deps depth-anything-3
pip install "numpy<2" einops omegaconf addict e3nn evo imageio "moviepy==1.0.3" plyfile trimesh \
    pillow-heif safetensors huggingface-hub opencv-python-headless typer pycolmap
