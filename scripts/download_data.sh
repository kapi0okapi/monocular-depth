#!/bin/bash
# KITTI Eigen test split with improved ground truth: 652 frames (~0.7 GB) read out of the raw
# drives and the annotation archive with HTTP range requests. Run on a login node.
#   bash scripts/download_data.sh            # into $DATA/kitti_eigen (cluster) or ./data/kitti_eigen
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mono_depth.datasets.kitti_eigen --out "${DATA:-data}/kitti_eigen"
