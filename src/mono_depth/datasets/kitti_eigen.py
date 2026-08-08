"""KITTI Eigen test split with the improved ground truth (Uhrig et al., 3DV 2017).

    depth-download --out data/kitti_eigen        # 652 frames, ~0.7 GB via range requests

The Eigen et al. (NeurIPS 2014) test split lists 697 left-camera frames from 28 KITTI drives.
652 of them have the denser "improved" ground truth (LiDAR accumulated over 11 frames, with
stereo consistency checks), which is what current papers evaluate on. Only those frames are
read out of the raw-drive archives and the 14 GB annotation archive. The split list itself is
taken from the monodepth2 repository at download time (not redistributed here).

Layout after download:

    <out>/raw/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png
    <out>/annotated/{train,val}/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png
    <out>/calib/2011_09_26/calib_cam_to_cam.txt
    <out>/eigen_test.txt        one "drive frame" per line, the 652 frames with ground truth
"""

from __future__ import annotations

import argparse
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from mono_depth.remote_zip import HttpRangeFile, extract_members

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
ANNOTATED_ZIP = f"{BASE}/data_depth_annotated.zip"
SPLIT_URL = "https://raw.githubusercontent.com/nianticlabs/monodepth2/master/splits/eigen/test_files.txt"


@dataclass(frozen=True)
class Sample:
    image: Path
    depth: Path  # uint16 PNG, metres * 256, 0 = no measurement
    focal: float  # pixels, from P_rect_02

    @property
    def name(self) -> str:
        return f"{self.image.parents[2].name}_{self.image.stem}"


def load(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image and depth in metres (0 where unknown)."""
    rgb = np.array(Image.open(sample.image).convert("RGB"))
    depth = np.array(Image.open(sample.depth), dtype=np.float32) / 256.0
    return rgb, depth


def read_focal(calib: Path) -> float:
    for line in calib.read_text().splitlines():
        if line.startswith("P_rect_02:"):
            return float(line.split()[1])
    raise ValueError(f"no P_rect_02 in {calib}")


def samples(root: Path) -> list[Sample]:
    root = Path(root)
    listing = root / "eigen_test.txt"
    if not listing.exists():
        raise FileNotFoundError(f"{listing} missing; run depth-download --out {root}")
    out = []
    for line in listing.read_text().splitlines():
        drive, frame, gt_split = line.split()
        date = drive[:10]
        out.append(
            Sample(
                root / "raw" / date / drive / "image_02" / "data" / f"{frame}.png",
                root
                / "annotated"
                / gt_split
                / drive
                / "proj_depth"
                / "groundtruth"
                / "image_02"
                / f"{frame}.png",
                read_focal(root / "calib" / date / "calib_cam_to_cam.txt"),
            )
        )
    return out


def download(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wanted = [
        line.split()[:2] for line in requests.get(SPLIT_URL, timeout=60).text.splitlines() if line.strip()
    ]
    # which of the 697 frames have improved ground truth, and in which half of the archive
    annotated = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ANNOTATED_ZIP), 1 << 20))
    available = {i.filename for i in annotated.infolist()}
    by_drive: dict[str, list[tuple[str, str]]] = defaultdict(list)
    gt_names, listing = set(), []
    for drive_path, frame in wanted:
        drive = drive_path.split("/")[1]
        frame = f"{int(frame):010d}"
        for split in ("train", "val"):
            name = f"{split}/{drive}/proj_depth/groundtruth/image_02/{frame}.png"
            if name in available:
                gt_names.add(name)
                by_drive[drive].append((frame, split))
                listing.append(f"{drive} {frame} {split}")
                break
    extract_members(ANNOTATED_ZIP, out / "annotated", names=gt_names)
    for drive, frames in sorted(by_drive.items()):
        date = drive[:10]
        names = {f"{date}/{drive}/image_02/data/{frame}.png" for frame, _ in frames}
        extract_members(f"{BASE}/raw_data/{drive[:-5]}/{drive}.zip", out / "raw", names=names)
        if not (out / "calib" / date / "calib_cam_to_cam.txt").exists():
            calib = requests.get(f"{BASE}/raw_data/{date}_calib.zip", timeout=120)
            calib.raise_for_status()
            zipfile.ZipFile(io.BytesIO(calib.content)).extractall(out / "calib")
        print(f"{drive}: {len(frames)} frames")
    (out / "eigen_test.txt").write_text("\n".join(listing) + "\n")
    print(f"{len(listing)} of {len(wanted)} Eigen test frames have improved ground truth")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/kitti_eigen"))
    args = parser.parse_args(argv)
    download(args.out)


if __name__ == "__main__":
    main()
