"""Metrics, alignments, the KITTI Eigen loader and the benchmark loop, without downloads."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from mono_depth.benchmark import evaluate
from mono_depth.datasets.kitti_eigen import Sample, load, samples
from mono_depth.metrics import DepthEvaluator, align, image_metrics, valid_mask
from mono_depth.models import ModelSpec


def scene(h=120, w=400, seed=0):
    rng = np.random.default_rng(seed)
    depth = np.tile(np.linspace(80, 3, h)[:, None], (1, w)) * rng.uniform(0.9, 1.1, (h, w))
    gt = np.where(rng.random((h, w)) < 0.3, depth, 0.0)  # sparse LiDAR-like ground truth
    return depth, gt


def test_perfect_prediction_scores_perfectly():
    depth, gt = scene()
    m = valid_mask(gt)
    r = image_metrics(depth, gt, m)
    assert r["abs_rel"] == pytest.approx(0.0, abs=1e-9) and r["delta1"] == 1.0


def test_garg_crop_keeps_the_lower_middle():
    m = valid_mask(np.full((100, 200), 10.0))
    assert not m[:40].any() and m[41:99, 8:192].all() and not m[:, :7].any()


@pytest.mark.parametrize(
    ("mode", "transform"),
    [
        ("median", lambda d: 0.37 * d),
        ("disparity", lambda d: 2.5 / d + 0.1),  # inverse depth with scale and shift
        ("log", lambda d: 3.0 * d**0.6),  # compressed range: what log alignment can undo
    ],
)
def test_alignment_recovers_relative_predictions(mode, transform):
    depth, gt = scene()
    m = valid_mask(gt)
    aligned = align(transform(depth), gt, m, mode)
    assert image_metrics(aligned, gt, m)["abs_rel"] < 0.06  # the 10 % pixel noise remains


def test_median_cannot_undo_a_compressed_range():
    depth, gt = scene()
    m = valid_mask(gt)
    worse = image_metrics(align(3.0 * depth**0.6, gt, m, "median"), gt, m)["abs_rel"]
    better = image_metrics(align(3.0 * depth**0.6, gt, m, "log"), gt, m)["abs_rel"]
    assert worse > 3 * better


def test_per_class_abs_rel():
    depth, gt = scene()
    ev = DepthEvaluator("none")
    pred = depth.copy()
    region = np.zeros_like(depth, bool)
    region[100:, :] = True
    pred[region] *= 1.5
    ev.update(pred, gt, {"near": region, "far": ~region})
    r = ev.result()
    assert r["abs_rel_by_class"]["far"] == pytest.approx(0.0, abs=1e-9)
    assert r["abs_rel_by_class"]["near"] == pytest.approx(0.5, rel=1e-6)


def fake_eigen(tmp_path, n=3):
    lines = []
    for i in range(n):
        drive, frame = "2011_09_26_drive_0002_sync", f"{i:010d}"
        img = tmp_path / "raw" / "2011_09_26" / drive / "image_02" / "data" / f"{frame}.png"
        gt = (
            tmp_path
            / "annotated"
            / "val"
            / drive
            / "proj_depth"
            / "groundtruth"
            / "image_02"
            / f"{frame}.png"
        )
        img.parent.mkdir(parents=True, exist_ok=True)
        gt.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((120, 400, 3), 100, np.uint8)).save(img)
        depth, sparse = scene(seed=i)
        Image.fromarray((sparse * 256).astype(np.uint16)).save(gt)
        np.save(tmp_path / f"truth_{i}.npy", depth)
        lines.append(f"{drive} {frame} val")
    (tmp_path / "eigen_test.txt").write_text("\n".join(lines) + "\n")
    calib = tmp_path / "calib" / "2011_09_26" / "calib_cam_to_cam.txt"
    calib.parent.mkdir(parents=True)
    calib.write_text(
        "P_rect_02: 7.215377e+02 0 6.095593e+02 4.485728e+01 0 7.215377e+02 1.728540e+02 0 0 0 1 0\n"
    )
    return tmp_path


def test_loader_reads_depth_in_metres(tmp_path):
    root = fake_eigen(tmp_path)
    items = samples(root)
    assert len(items) == 3 and items[0].focal == pytest.approx(721.5377)
    rgb, gt = load(items[0])
    assert rgb.shape == (120, 400, 3) and gt.max() < 90.0 and (gt > 0).mean() > 0.2


class OracleModel:
    """Returns the true depth, scaled: a stand-in for a relative-depth network."""

    name = "oracle"
    spec = ModelSpec("fake", "", "depth", "test")

    def __init__(self, root):
        self.root = root
        self.calls = 0

    def __call__(self, rgb, focal):
        depth = np.load(self.root / f"truth_{self.calls}.npy")
        self.calls += 1
        return 0.2 * depth


def test_benchmark_loop_aligns_relative_models(tmp_path):
    root = fake_eigen(tmp_path)
    report = evaluate(OracleModel(root), samples(root))
    assert report["alignment"] == "median" and report["images"] == 3
    assert report["abs_rel"] < 1e-3 and report["delta1"] == 1.0  # PNG stores 1/256 m
    assert report["log_aligned"]["abs_rel"] < 1e-3


def test_sample_name():
    s = Sample(
        image=__import__("pathlib").Path(
            "raw/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png"
        ),
        depth=__import__("pathlib").Path("x.png"),
        focal=700.0,
    )
    assert s.name == "2011_09_26_drive_0002_sync_0000000069"


def test_holdout_split_keeps_whole_drives_out_of_training(tmp_path):
    from mono_depth.finetune import drive_of, holdout_split

    (tmp_path / "images" / "train").mkdir(parents=True)
    for d in range(24):
        for cam in ("02", "03"):
            for f in range(10):
                (tmp_path / "images" / "train" / f"2011_09_26_drive_{d:04d}_{cam}_{f:010d}.png").touch()
    (tmp_path / "depth-kitti.yaml").write_text("path: x\ntrain: images/train\nval: images/val\ndownload: x\n")
    cfg = yaml.safe_load(holdout_split(tmp_path, every=12, frame_step=5).read_text())
    assert cfg["val"] == "holdout_val.txt" and "download" not in cfg
    train = {drive_of(Path(p).name) for p in (tmp_path / "holdout_train.txt").read_text().split()}
    val = (tmp_path / "holdout_val.txt").read_text().split()
    assert len(train) == 22 and not train & {drive_of(Path(p).name) for p in val}
    assert len(val) == 2 * 2 and all("_02_" in p for p in val)  # 2 drives, left camera, every 5th frame
