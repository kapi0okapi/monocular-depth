#!/bin/bash
# KITTI depth training set in Ultralytics' layout (improved GT + matching raw frames, Eigen
# train split; ~190 GB to download, ~140 GB converted). Only for the fine-tuning job.
# Uses Ultralytics' own preparation recipe from depth-kitti.yaml. Run on a login node.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${DATA:-data}/depth-kitti"
mkdir -p "$OUT"
python - "$OUT" <<'PY'
import sys
from pathlib import Path

import yaml
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ROOT

out = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load((ROOT / "cfg" / "datasets" / "depth-kitti.yaml").read_text())
cfg["path"] = str(out)
(out / "depth-kitti.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
check_det_dataset(str(out / "depth-kitti.yaml"), autodownload=True)
print("ready:", out / "depth-kitti.yaml")
PY
