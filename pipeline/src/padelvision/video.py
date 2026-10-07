"""Video metadata. FPS always comes from the file, never a hardcoded constant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


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
