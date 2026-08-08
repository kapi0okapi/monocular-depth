"""Depth models from three generations behind one call: ``DepthModel(name)(rgb, focal)``.

Each model is run with its own published preprocessing (resolution, normalisation) and the
output is resized to the input image. What the output means differs:

* ``metric``: metres, scored as is (Depth Pro, ZoeDepth, Depth Anything V2/V3 metric),
* ``depth``: depth up to an unknown scale (YOLO26-depth, DA3 Mono), aligned by the median,
* ``disparity``: inverse depth up to scale and shift (Depth Anything V2 relative), aligned
  by least squares in inverse depth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    family: str  # yolo | hf | da3
    weights: str
    output: str  # metric | depth | disparity
    training: str  # what the published weights were trained on, for the README
    paper: tuple[float, float, str] | None = None  # KITTI Eigen (AbsRel, delta1, source)


# YOLO26-depth: delta1 from the Ultralytics KITTI dataset page (652 Eigen frames, improved ground
# truth, median alignment, 80 m cap). The 0.942 in the YOLO26 overview uses test-time augmentation
# and log alignment, which the validator does not implement.
_YOLO_TRAINING = "2.2M-image indoor/outdoor mix"
MODELS = {
    "yolo26s-depth": ModelSpec(
        "yolo", "yolo26s-depth.pt", "depth", _YOLO_TRAINING, (None, 0.879, "Ultralytics docs")
    ),
    "yolo26m-depth": ModelSpec(
        "yolo", "yolo26m-depth.pt", "depth", _YOLO_TRAINING, (None, 0.913, "Ultralytics docs")
    ),
    "yolo26x-depth": ModelSpec(
        "yolo", "yolo26x-depth.pt", "depth", _YOLO_TRAINING, (None, 0.932, "Ultralytics docs")
    ),
    "da2-small": ModelSpec(
        "hf",
        "depth-anything/Depth-Anything-V2-Small-hf",
        "disparity",
        "595K synthetic + 62M pseudo-labelled",
        (0.078, 0.936, "DA-V2 Tab. 2"),
    ),
    "da2-large": ModelSpec(
        "hf",
        "depth-anything/Depth-Anything-V2-Large-hf",
        "disparity",
        "595K synthetic + 62M pseudo-labelled",
        (0.074, 0.946, "DA-V2 Tab. 2"),
    ),
    "da2-metric-outdoor-small": ModelSpec(
        "hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "metric",
        "Virtual KITTI 2 (synthetic)",
    ),
    "da2-metric-outdoor-large": ModelSpec(
        "hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "metric",
        "Virtual KITTI 2 (synthetic)",
    ),
    "depth-pro": ModelSpec("hf", "apple/DepthPro-hf", "metric", "zero-shot, estimates its own focal length"),
    "depth-pro-known-focal": ModelSpec(
        "hf", "apple/DepthPro-hf", "metric", "as above, rescaled to the true focal"
    ),
    "da3-mono-large": ModelSpec("da3", "depth-anything/DA3MONO-LARGE", "depth", "Depth Anything 3 data mix"),
    "da3-metric-large": ModelSpec(
        "da3", "depth-anything/DA3METRIC-LARGE", "metric", "Depth Anything 3 data mix"
    ),
}


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DepthModel:
    def __init__(self, name: str, device: str = "auto", imgsz: int = 768, da3_resolution: int = 504):
        """``name`` is a key of MODELS or a fine-tuned YOLO depth checkpoint (.pt, metric output).
        ``da3_resolution``: longest side Depth Anything 3 works at (its default is 504 px)."""
        self.name = name
        self.spec = MODELS.get(name) or ModelSpec("yolo", name, "metric", "fine-tuned")
        self.device = select_device(device)
        self.imgsz, self.da3_resolution = imgsz, da3_resolution
        weights = os.path.expandvars(self.spec.weights)
        if self.spec.family == "yolo":
            from ultralytics import YOLO

            self.model = YOLO(weights)
        elif self.spec.family == "hf":
            import transformers

            self.processor = transformers.AutoImageProcessor.from_pretrained(weights)
            self.model = (
                transformers.AutoModelForDepthEstimation.from_pretrained(weights).to(self.device).eval()
            )
        else:
            from depth_anything_3.api import DepthAnything3  # optional: pip install depth-anything-3

            self.model = DepthAnything3.from_pretrained(weights).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, focal: float) -> np.ndarray:
        """(H, W) prediction in the model's output space (see ``spec.output``)."""
        h, w = rgb.shape[:2]
        family = self.spec.family
        if family == "yolo":
            # YOLO26-depth is trained and validated on images stretched to a square (Ultralytics'
            # depth dataset resizes without letterboxing), while predict() letterboxes and keeps the
            # aspect ratio. A KITTI frame then looks unlike anything seen in training (delta1 0.69
            # instead of 0.87 for YOLO26s), so the frame is stretched here and the map stretched back.
            square = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            result = self.model.predict(
                square[..., ::-1], imgsz=self.imgsz, verbose=False, device=str(self.device)
            )
            depth = result[0].depth.data.float().cpu().numpy()
            return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        if family == "hf":
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            post = self.processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])[0]
            depth = post["predicted_depth"].float().cpu().numpy()
            if self.name == "depth-pro-known-focal" and post.get("focal_length") is not None:
                # Depth Pro's metric depth scales with the focal length it estimated; use the true one
                depth = depth * focal / float(post["focal_length"])
            return depth
        prediction = self.model.inference([rgb], process_res=self.da3_resolution)  # Depth Anything 3
        depth = torch.as_tensor(np.asarray(prediction.depth[0]), dtype=torch.float32)[None, None]
        processed_w = depth.shape[-1]
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
        if self.spec.output == "metric":  # canonical depth -> metres (DA3 README: focal * output / 300)
            depth = depth * (focal * processed_w / w) / 300.0
        return depth
