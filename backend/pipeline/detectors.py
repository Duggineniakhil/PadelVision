"""
pipeline/detectors.py

Thin wrappers around YOLOv8 player detection and ball detection.
Each function takes video frames and returns raw detection dicts.
"""

import sys
import os

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(_BACKEND_DIR)

from trackers import PlayerTracker, BallTracker


def _resolve(path):
    """If path is relative and doesn't exist, try resolving from backend dir."""
    if os.path.exists(path):
        return path
    resolved = os.path.join(_BACKEND_DIR, path)
    if os.path.exists(resolved):
        return resolved
    return path


def detect_players(
    video_frames: list,
    model_path: str = "yolov8x.pt",
    stub_path: str | None = None,
    read_from_stub: bool = False,
) -> list[dict]:
    tracker = PlayerTracker(model_path=_resolve(model_path))
    detections = tracker.detect_frames(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
    )
    return detections, tracker


def detect_ball(
    video_frames: list,
    model_path: str = "models/padel_ball_detector.pt",
    stub_path: str | None = None,
    read_from_stub: bool = False,
) -> list[dict]:
    tracker = BallTracker(model_path=_resolve(model_path))
    detections = tracker.detect_frames(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
    )
    return detections, tracker
