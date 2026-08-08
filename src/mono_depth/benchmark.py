"""Evaluate depth models on the KITTI Eigen test split (652 frames, improved ground truth).

    depth-benchmark --models da2-small depth-pro da3-metric-large --root data/kitti_eigen --out runs/benchmark
    depth-benchmark --models $RUNS/kitti_yolo26s/weights/best.pt --root data/kitti_eigen --out runs/finetuned

Besides the standard metrics, ``--by-class`` labels every image with a Cityscapes semantic
model (YOLO26s-sem) and reports AbsRel per class: how far off is the depth of a guard rail,
a pole or a car, which is what a measurement from one camera would rely on.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from mono_depth.datasets.kitti_eigen import load, samples
from mono_depth.metrics import DepthEvaluator
from mono_depth.models import DepthModel

log = logging.getLogger("mono_depth.benchmark")
ALIGNMENT = {"metric": "none", "depth": "median", "disparity": "disparity"}
CLASSES = {0: "road", 2: "building", 4: "fence", 5: "pole", 7: "traffic sign", 8: "vegetation", 13: "car"}


class ClassLabeler:
    def __init__(self, device: str):
        from ultralytics import YOLO

        self.model, self.device = YOLO("yolo26s-sem.pt"), device

    def __call__(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        result = self.model.predict(rgb[..., ::-1], imgsz=768, verbose=False, device=self.device)[0]
        labels = result.semantic_mask.data.cpu().numpy()
        return {name: labels == k for k, name in CLASSES.items()}


def evaluate(model: DepthModel, items, labeler: ClassLabeler | None = None, metric_too: bool = True) -> dict:
    mode = ALIGNMENT[model.spec.output]
    evaluators = {mode: DepthEvaluator(mode)}
    if metric_too and model.spec.output == "metric":
        evaluators["median"] = DepthEvaluator("median")  # how good is the shape, ignoring scale
    if model.spec.output == "depth":
        evaluators["log"] = DepthEvaluator("log")  # Ultralytics' alignment, for its documented numbers
    seconds = []
    for i, sample in enumerate(items):
        rgb, gt = load(sample)
        started = time.perf_counter()
        pred = model(rgb, sample.focal)
        seconds.append(time.perf_counter() - started)
        classes = labeler(rgb) if labeler is not None else None
        for name, ev in evaluators.items():
            ev.update(pred, gt, classes if name == mode else None)
        if i % 100 == 0:
            r = evaluators[mode].result()
            log.info(
                "%s %d/%d: AbsRel %.4f delta1 %.4f", model.name, i + 1, len(items), r["abs_rel"], r["delta1"]
            )
    report = {name: ev.result() for name, ev in evaluators.items()}
    return {
        "alignment": mode,
        **report[mode],
        "median_aligned": report.get("median") if mode != "median" else None,
        "log_aligned": report.get("log"),
        "seconds_per_image": float(np.median(seconds[1:] if len(seconds) > 1 else seconds)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--every", type=int, default=1, help="use every n-th image (quick runs)")
    parser.add_argument("--by-class", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    items = samples(args.root)[:: args.every]
    args.out.mkdir(parents=True, exist_ok=True)
    labeler = None
    for name in args.models:
        stem = Path(name).stem if name.endswith(".pt") else name
        target = args.out / f"{stem}.json"
        if target.exists():
            log.info("%s already done", stem)
            continue
        model = DepthModel(name, args.device)
        if args.by_class and labeler is None:
            labeler = ClassLabeler(str(model.device))
        report = {
            "model": name,
            "output": model.spec.output,
            "training": model.spec.training,
            "paper": model.spec.paper,
            "device": str(model.device),
            **evaluate(model, items, labeler),
        }
        target.write_text(json.dumps(report, indent=2))
        absrel, d1, rmse = report["abs_rel"], report["delta1"], report["rmse"]
        print(f"{stem}: AbsRel {absrel:.4f}, delta1 {d1:.4f}, RMSE {rmse:.3f}")


if __name__ == "__main__":
    main()
