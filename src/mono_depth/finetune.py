"""Fine-tune YOLO26-depth on KITTI.

    python -m mono_depth.finetune --root data/depth-kitti --model yolo26s-depth.pt --name kitti_yolo26s

1. Geometry. Ultralytics validates and predicts on images stretched to a square
   (``LetterBox(scale_fill=True)``); training here resizes the same way, so the network learns on
   exactly the input it sees at inference (colour and flip augmentation stay).
2. Validation. Every ``every``-th training drive is held out for validation (left camera, every
   ``frame_step``-th frame). Checkpoint selection and the output-scale calibration use these
   drives only; the Eigen test split stays untouched until the benchmark.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger("mono_depth.finetune")


def drive_of(name: str) -> str:
    """``2011_09_26_drive_0001_02_0000000005`` -> ``2011_09_26_drive_0001``."""
    return "_".join(name.split("_")[:5])


def holdout_split(root: Path, every: int = 12, frame_step: int = 5) -> Path:
    """Write train/val lists that hold out whole drives and a yaml that uses them."""
    images = sorted(p.name for p in (root / "images" / "train").iterdir() if p.suffix in (".png", ".jpg"))
    drives = sorted({drive_of(n) for n in images})
    held_out = set(drives[every // 2 :: every])
    train = [n for n in images if drive_of(n) not in held_out]
    val = [n for n in images if drive_of(n) in held_out and n.split("_")[5] == "02"][::frame_step]
    for split, names in (("train", train), ("val", val)):
        (root / f"holdout_{split}.txt").write_text("".join(f"./images/train/{n}\n" for n in names))
    cfg = yaml.safe_load((root / "depth-kitti.yaml").read_text())
    cfg.pop("download", None)
    cfg |= {"path": str(root.resolve()), "train": "holdout_train.txt", "val": "holdout_val.txt"}
    path = root / "depth-kitti-holdout.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log.info(
        "held out %d of %d drives: %d train / %d val frames", len(held_out), len(drives), len(train), len(val)
    )
    return path


def stretch_training_images() -> None:
    """Make training resize like validation and prediction: stretch to the square input."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.data.dataset import DepthDataset

    stock = DepthDataset.build_transforms

    def build_transforms(self, hyp):
        transforms = stock(self, hyp)
        if self.augment:  # replaces mosaic (off for depth) + random affine; HSV and flips follow
            transforms[0] = LetterBox(new_shape=(self.imgsz, self.imgsz), scale_fill=True)
        return transforms

    DepthDataset.build_transforms = build_transforms


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="depth-kitti folder with depth-kitti.yaml")
    parser.add_argument("--model", default="yolo26s-depth.pt")
    parser.add_argument("--name", default="kitti_yolo26s")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--fraction", type=float, default=1.0, help="share of the training set (quick runs)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--stock-geometry", action="store_true", help="keep Ultralytics' letterbox + affine")
    parser.add_argument("--eigen-val", action="store_true", help="validate on the Eigen test frames (stock)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ultralytics import YOLO

    data = args.root / "depth-kitti.yaml" if args.eigen_val else holdout_split(args.root)
    if not args.stock_geometry:
        stretch_training_images()
    YOLO(args.model).train(
        data=str(data),
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr0=args.lr0,
        warmup_epochs=0.5,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
