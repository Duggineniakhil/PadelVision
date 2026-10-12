"""Video metadata. FPS always comes from the file, never a hardcoded constant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "duration_s": self.duration_s}


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        info = VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()
    if info.fps <= 0:
        raise ValueError(f"Video reports no FPS: {path}")
    return info


def median_background(
    path: str | Path, n: int = 40, size: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(a middle frame, the per-pixel median of `n` frames spread over the video). The median
    removes moving people and balls: a clean court to calibrate on. `size` = (w, h) to resize
    to (e.g. the analysis resolution when the file is the original 4K)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0.05 * count, 0.95 * count, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, img = cap.read()
            if ok:
                if size and (img.shape[1], img.shape[0]) != tuple(size):
                    img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
                frames.append(img)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Could not read frames from {path}")
    return frames[len(frames) // 2], np.median(np.stack(frames), axis=0).astype(np.uint8)
