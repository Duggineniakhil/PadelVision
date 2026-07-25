"""
pipeline/trackers.py

Post-processing after raw detection:
  - Filter to four active players using court keypoints
  - Interpolate missing ball positions
  - Convert bounding boxes to mini-court coordinates
  - Detect ball shot frames
"""

import sys
import os
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from court_mapper import CourtMapper

logger = logging.getLogger(__name__)


def detect_court_keypoints(
    first_frame,
    model_path: str = "models/padel_court_keypoints.pt",
) -> dict:
    """
    Predict court landmarks from the first frame as a dictionary of {class_name: (x, y)}.
    """
    detector = CourtLineDetector(model_path)
    keypoints = detector.predict(first_frame)
    return keypoints, detector


def filter_players(
    player_tracker,
    court_keypoints: list,
    player_detections: list,
) -> list[dict]:
    """
    Keep only the four tracked players most consistently near the court.
    """
    return player_tracker.choose_and_filter_players(court_keypoints, player_detections)


def interpolate_ball(
    ball_tracker,
    ball_detections: list,
) -> list[dict]:
    """
    Fill in missing ball positions using Pandas interpolation + backfill.
    """
    return ball_tracker.interpolate_ball_positions(ball_detections)


def get_shot_frames(
    ball_tracker,
    ball_detections: list,
) -> list[int]:
    """
    Detect frames where a ball shot occurred based on y-direction changes.
    """
    return ball_tracker.get_ball_shot_frames(ball_detections)


def build_mini_court(first_frame) -> MiniCourt:
    """
    Create the MiniCourt object from the first video frame.
    """
    return MiniCourt(first_frame)


def convert_to_mini_court_coordinates(
    mini_court: MiniCourt,
    player_detections: list,
    ball_detections: list,
    court_keypoints: dict,
) -> tuple[list, list]:
    logger.info("Court keypoints detected: %s",
                {k: len(v) for k, v in court_keypoints.items()} if court_keypoints else "none")

    mapper = CourtMapper(court_keypoints, mini_court)

    if not mapper.is_valid:
        logger.info("Keypoint-based homography unavailable — estimating from player positions")
        mapper.estimate_from_player_positions(player_detections)

    output_player_boxes = []
    output_ball_boxes = []

    for frame_num, player_bbox in enumerate(player_detections):
        frame_player_boxes = {}
        for player_id, bbox in player_bbox.items():
            foot_pos = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
            mini_pos = mapper.get_mini_court_coordinates(foot_pos)
            frame_player_boxes[player_id] = mini_pos
        output_player_boxes.append(frame_player_boxes)

        frame_ball_boxes = {}
        if frame_num < len(ball_detections):
            ball_dict = ball_detections[frame_num]
            for ball_id, bbox in ball_dict.items():
                ball_pos = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                mini_pos = mapper.get_mini_court_coordinates(ball_pos)
                frame_ball_boxes[ball_id] = mini_pos
        output_ball_boxes.append(frame_ball_boxes)

    player_count = sum(1 for f in output_player_boxes if f)
    ball_count = sum(1 for f in output_ball_boxes if f)
    logger.info("Mini-court positions: %d frames with players, %d frames with ball",
                player_count, ball_count)

    return output_player_boxes, output_ball_boxes
