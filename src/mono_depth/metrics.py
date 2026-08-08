"""Depth metrics as in the KITTI Eigen-split literature (Eigen et al., 2014; Garg et al., 2016).

Pixels count when the ground truth lies in (1e-3, 80] m and inside the Garg crop (the lower
part of the image that the LiDAR covers). Predictions are clipped to the same range. Metrics
are computed per image and averaged over images.

Four ways to read a prediction:

* ``none``: metric models are scored as they are, which is the point of metric depth;
* ``median``: relative *depth* (scale unknown) is multiplied by median(gt) / median(pred),
  per image, as Ultralytics and Monodepth2 do;
* ``disparity``: relative *inverse depth* with unknown scale and shift (MiDaS, Depth Anything V2)
  is fitted to 1/gt by least squares, then inverted;
* ``log``: log(depth) fitted to log(gt) by an affine map (scale and exponent), the alignment of
  the cross-dataset table in Ultralytics' YOLO26 overview (with test-time augmentation). It also
  undoes a compressed depth range, so it flatters.
"""

from __future__ import annotations

import numpy as np

MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0
GARG_CROP = (0.40810811, 0.99189189, 0.03594771, 0.96405229)  # top, bottom, left, right (fractions)
KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")


def valid_mask(gt: np.ndarray, garg_crop: bool = True) -> np.ndarray:
    mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if garg_crop:
        h, w = gt.shape
        top, bottom, left, right = GARG_CROP
        crop = np.zeros_like(mask)
        crop[int(top * h) : int(bottom * h), int(left * w) : int(right * w)] = True
        mask &= crop
    return mask


def align(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return pred
    if mode == "median":
        return pred * np.median(gt[mask]) / max(np.median(pred[mask]), 1e-9)
    if mode == "disparity":
        x, y = pred[mask], 1.0 / gt[mask]
        A = np.stack([x, np.ones_like(x)], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(A, y, rcond=None)
        disparity = np.maximum(scale * pred + shift, 1.0 / MAX_DEPTH)
        return 1.0 / disparity
    if mode == "log":
        x, y = np.log(np.maximum(pred[mask], 1e-6)), np.log(gt[mask])
        A = np.stack([x, np.ones_like(x)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        return np.exp(a * np.log(np.maximum(pred, 1e-6)) + b)
    raise ValueError(f"unknown alignment '{mode}'")


def image_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = np.clip(pred[mask], MIN_DEPTH, MAX_DEPTH)
    g = gt[mask]
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


class DepthEvaluator:
    """Accumulates per-image metrics; optionally also per semantic class (e.g. guard rail)."""

    def __init__(self, mode: str = "none", garg_crop: bool = True):
        self.mode, self.garg_crop = mode, garg_crop
        self.rows: list[dict[str, float]] = []
        self.by_class: dict[str, list[tuple[float, int]]] = {}

    def update(
        self, pred: np.ndarray, gt: np.ndarray, classes: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        mask = valid_mask(gt, self.garg_crop)
        if mask.sum() < 10:
            return pred
        aligned = align(pred, gt, mask, self.mode)
        self.rows.append(image_metrics(aligned, gt, mask))
        for name, region in (classes or {}).items():
            m = mask & region
            if m.sum():
                err = np.abs(np.clip(aligned[m], MIN_DEPTH, MAX_DEPTH) - gt[m]) / gt[m]
                self.by_class.setdefault(name, []).append((float(err.sum()), int(m.sum())))
        return aligned

    def result(self) -> dict:
        out = {k: float(np.mean([r[k] for r in self.rows])) for k in KEYS} if self.rows else {}
        out["images"] = len(self.rows)
        if self.by_class:  # pixel-weighted AbsRel per class
            out["abs_rel_by_class"] = {
                c: sum(s for s, _ in v) / sum(n for _, n in v) for c, v in sorted(self.by_class.items())
            }
            out["pixels_by_class"] = {c: sum(n for _, n in v) for c, v in sorted(self.by_class.items())}
        return out
