"""Stage 2: run person detection + tracking over the video -> detections table."""

from __future__ import annotations

import time
from collections.abc import Iterable

import pandas as pd

from padelvision.models.person_tracker import FrameDetections

COLUMNS = ["frame", "t", "track_id", "conf", "x1", "y1", "x2", "y2"]


def detections_to_frame(frames: Iterable[FrameDetections], fps: float) -> pd.DataFrame:
    rows = []
    for fd in frames:
        for box, conf, tid in zip(fd.boxes, fd.conf, fd.track_ids, strict=True):
            rows.append((fd.frame, fd.frame / fps, int(tid), float(conf), *map(float, box)))
    return pd.DataFrame(rows, columns=COLUMNS)


def run_detection(
    tracker,
    video_path,
    fps: float,
    total_frames: int,
    stride: int = 1,
    max_frames: int | None = None,
    log_every: int = 500,
) -> pd.DataFrame:
    """`tracker` is anything with `.track(video_path, stride, max_frames)` (see PersonTracker)."""
    limit = min(total_frames, max_frames) if max_frames else total_frames
    start = time.perf_counter()

    def logged(frames):
        for n, fd in enumerate(frames, 1):
            if n % log_every == 0:
                rate = n / (time.perf_counter() - start)
                print(f"  detect: frame {fd.frame}/{limit}  ({rate:.1f} frames/s)", flush=True)
            yield fd

    return detections_to_frame(logged(tracker.track(video_path, stride, max_frames)), fps)
