"""
pipeline/detectors.py

Thin wrappers around YOLOv8 player detection and ball detection.
Each function takes video frames and returns raw detection dicts.
"""

import sys
import os

_BACKEND_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(_BACKEND_DIR)

from trackers import PlayerTracker, BallTracker


def detect_players(
    video_frames: list,
    model_path: str = "yolov8x.pt",
    stub_path: str | None = None,
    read_from_stub: bool = False,
) -> list[dict]:
    """
    Run YOLOv8 player detection + tracking on all frames.

    Returns:
        List of dicts per frame: { track_id: [x1, y1, x2, y2] }
    """
    tracker = PlayerTracker(model_path=model_path)
    detections = tracker.detect_frames(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
    )
    return detections, tracker


def detect_ball(
    video_frames: list,
    model_path: str = os.path.join(_BACKEND_DIR, "models", "padel_ball_detector.pt"),
    stub_path: str | None = None,
    read_from_stub: bool = False,
) -> list[dict]:
    """
    Run custom YOLO ball detection on all frames.

    Returns:
        List of dicts per frame: { 1: [x1, y1, x2, y2] }
    """
    tracker = BallTracker(model_path=model_path)
    detections = tracker.detect_frames(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
    )
    return detections, tracker
