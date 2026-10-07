"""Person detection + multi-object tracking (Ultralytics YOLO + ByteTrack).

The only module that imports ultralytics for players. Install with `pip install -e "pipeline[ml]"`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameDetections:
    frame: int
    boxes: np.ndarray  # (N, 4) x1, y1, x2, y2 in pixels
    conf: np.ndarray  # (N,)
    track_ids: np.ndarray  # (N,) int, -1 when the tracker hasn't confirmed an id


class PersonTracker:
    def __init__(
        self,
        weights: str | Path,
        device: str | int | None = None,
        imgsz: int = 1280,
        conf: float = 0.25,
        tracker: str = "bytetrack.yaml",
    ):
        from ultralytics import YOLO

        if device is None:
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = tracker

    def track(
        self, video_path: str | Path, stride: int = 1, max_frames: int | None = None
    ) -> Iterator[FrameDetections]:
        results = self.model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker,
            vid_stride=stride,
            verbose=False,
        )
        for i, r in enumerate(results):
            frame = i * stride
            if max_frames is not None and frame >= max_frames:
                break
            b = r.boxes
            ids = b.id.int().cpu().numpy() if b.id is not None else np.full(len(b), -1)
            yield FrameDetections(
                frame=frame,
                boxes=b.xyxy.cpu().numpy(),
                conf=b.conf.cpu().numpy(),
                track_ids=ids.astype(int),
            )
