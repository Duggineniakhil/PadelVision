"""Build a YOLO ball dataset from our video's labels plus public (Roboflow Universe) datasets.

Leakage: consecutive frames are near-identical, so train and test frames must not interleave.
The video is cut into blocks of `block_s` seconds: odd blocks are test-only, even blocks
train-only, minus a margin at each edge (`is_test_frame`, `is_train_frame`).

Layout written (Ultralytics YOLO format, one class: 0 = ball):
    <root>/images/{train,val}/<name>.jpg
    <root>/labels/{train,val}/<name>.txt     "0 cx cy w h" normalised, one line per ball
    <root>/data.yaml
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# Class names in public datasets that mean "the ball" (lower-cased, compared exactly).
BALL_NAMES = {"ball", "bola", "pelota", "padel ball", "padel-ball", "tennis ball", "tennis-ball",
              "sports ball", "balls"}  # fmt: skip
DEFAULT_BOX_PX = 16.0


def is_test_frame(frame, fps: float, block_s: float = 20.0):
    """True for frames in odd `block_s`-second blocks (works on scalars and arrays)."""
    return (np.asarray(frame) / fps // block_s).astype(int) % 2 == 1


def is_train_frame(frame, fps: float, block_s: float = 20.0, margin_s: float = 2.0):
    """True for frames in even blocks, at least `margin_s` away from the neighbouring test
    blocks (frames right at a block edge are near-duplicates of test frames)."""
    t = np.asarray(frame) / fps
    t_in = t % block_s
    return ~is_test_frame(frame, fps, block_s) & (t_in >= margin_s) & (t_in <= block_s - margin_s)


def write_video_frames(video, boxes: pd.DataFrame, root, split: str, prefix: str = "vid") -> int:
    """Write frames + labels. `boxes`: frame, u, v, size (px); several rows per frame allowed."""
    root = Path(root)
    img_dir, lab_dir = root / "images" / split, root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    by_frame = {int(f): g for f, g in boxes.groupby("frame")}
    if not by_frame:
        return 0
    cap = cv2.VideoCapture(str(video))
    written, f = 0, 0
    try:
        while f <= max(by_frame):
            ok, img = cap.read()
            if not ok:
                break
            if f in by_frame:
                h, w = img.shape[:2]
                name = f"{prefix}_{f:06d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = [
                    f"0 {r.u / w:.6f} {r.v / h:.6f} {r.size / w:.6f} {r.size / h:.6f}"
                    for r in by_frame[f].itertuples()
                ]
                (lab_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
                written += 1
            f += 1
    finally:
        cap.release()
    return written


def import_yolo_dataset(src, root, prefix: str, val_fraction: float = 0.1, seed: int = 0) -> dict:
    """Copy a downloaded YOLO-format dataset into `root`, keeping only ball classes (-> 0).

    Images whose labels contain no ball are skipped (they'd teach "no ball here" for
    images that may well contain unlabelled balls of other classes' datasets).
    """
    src, root = Path(src), Path(root)
    meta = yaml.safe_load((src / "data.yaml").read_text())
    names = meta["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    keep = {int(i) for i, n in names.items() if str(n).strip().lower() in BALL_NAMES}
    stats = {"source": str(src), "classes": {int(k): v for k, v in names.items()},
             "kept_classes": sorted(keep), "images": 0, "balls": 0}  # fmt: skip
    if not keep:
        return stats
    rng = random.Random(seed)
    for lab in sorted(src.rglob("labels/*.txt")):
        rows = []
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(float(parts[0])) in keep:
                rows.append("0 " + " ".join(parts[1:5]))
        if not rows:
            continue
        img = next((p for p in lab.parent.parent.joinpath("images").glob(lab.stem + ".*")), None)
        if img is None:
            continue
        split = "val" if rng.random() < val_fraction else "train"
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        name = f"{prefix}_{lab.stem}"[:180]
        shutil.copy(img, root / "images" / split / f"{name}{img.suffix}")
        (root / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + "\n")
        stats["images"] += 1
        stats["balls"] += len(rows)
    return stats


def write_data_yaml(root, sources: list[dict] | None = None) -> Path:
    root = Path(root)
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train", "val": "images/val",
        "names": {0: "ball"},
    }))  # fmt: skip
    if sources is not None:
        (root / "sources.json").write_text(json.dumps(sources, indent=1))
    return path


def count(root) -> dict:
    root = Path(root)
    return {
        s: len(list((root / "images" / s).glob("*"))) if (root / "images" / s).exists() else 0
        for s in ("train", "val")
    }
