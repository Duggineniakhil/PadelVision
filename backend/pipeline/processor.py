"""
pipeline/processor.py

Orchestrator — calls each pipeline module in sequence.
Responsible only for coordination, not computation.

Usage (API):
    from pipeline.processor import process_video
    result = process_video(job_id="abc123", input_path="uploads/abc123.mp4")

Usage (CLI):
    Still works via backend/main.py
"""

import os
import sys
import json
import logging

# Allow imports from backend root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── PyTorch 2.6 compatibility patch ───────────────────────────────────────────
# PyTorch 2.6 changed the default of weights_only to True, which breaks loading
# YOLO model checkpoints that embed Python objects (e.g. DetectionModel, PoseModel).
# We patch torch.load here unconditionally so all downstream YOLO model loads work.
import torch as _torch
_torch_load_original = _torch.load
def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_original(*args, **kwargs)
_torch.load = _torch_load_patched
# ─────────────────────────────────────────────────────────────────────────────

import cv2
from utils import get_video_fps, read_video, save_video, save_highlights_video
from utils.player_stats_drawer_utils import draw_player_stats

from pipeline.detectors import detect_players, detect_ball
from pipeline.trackers import (
    detect_court_keypoints,
    filter_players,
    interpolate_ball,
    get_shot_frames,
    build_mini_court,
    convert_to_mini_court_coordinates,
)
from pipeline.analytics import generate_statistics
from pipeline.heatmap import generate_heatmap
from pipeline.shot_map import generate_shot_map
from pipeline.highlights import generate_highlights

logger = logging.getLogger(__name__)

# Base output directory (relative to backend/)
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def process_video(
    input_path: str,
    job_id: str,
    player_stub: str | None = None,
    ball_stub: str | None = None,
    use_stubs: bool = False,
    on_progress=None,       # Optional callback: on_progress(stage: str, pct: int)
) -> dict:
    """
    Full PadelVision analysis pipeline.

    Stages:
      1. Read video
      2. Detect players + ball
      3. Court keypoint detection
      4. Filter & interpolate
      5. Mini-court coordinate conversion
      6. Statistics generation
      7. Heatmap generation
      8. Shot map generation
      9. Highlight detection
      10. Render annotated output video
      11. Save analysis.json

    Returns:
        {
            "video_path": str,
            "analysis_path": str,
            "heatmap_paths": { "player_1": str, "player_2": str },
            "shot_map_path": str,
            "stats": { "player_1": {...}, "player_2": {...} },
            "highlights": [ { start_frame, end_frame, timestamp_seconds, label } ],
        }
    """
    def _progress(stage: str, pct: int):
        logger.info("[%s] %s — %d%%", job_id, stage, pct)
        if on_progress:
            on_progress(stage, pct)

    output_dir = os.path.join(OUTPUTS_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Read video ─────────────────────────────────────────────────────────
    _progress("Reading video", 5)
    video_frames = read_video(input_path)
    total_frames = len(video_frames)
    if total_frames == 0:
        raise ValueError("Input video contains no readable frames.")
    fps = get_video_fps(input_path)
    logger.info("Loaded %d frames at %.2f FPS from %s", total_frames, fps, input_path)

    # ── 2. Detect players & ball ──────────────────────────────────────────────
    _progress("Detecting players", 10)
    player_detections, player_tracker = detect_players(
        video_frames,
        stub_path=player_stub,
        read_from_stub=use_stubs,
    )

    _progress("Detecting ball", 20)
    ball_detections, ball_tracker = detect_ball(
        video_frames,
        stub_path=ball_stub,
        read_from_stub=use_stubs,
    )

    # ── 3. Court keypoints ────────────────────────────────────────────────────
    _progress("Detecting court keypoints", 30)
    court_keypoints, court_detector = detect_court_keypoints(video_frames[0])
    logger.info("Court keypoints result: %s",
                {k: len(v) for k, v in court_keypoints.items()} if court_keypoints else "empty")

    # ── 4. Filter players, interpolate ball, detect shots ─────────────────────
    _progress("Filtering & interpolating", 35)
    player_detections = filter_players(player_tracker, court_keypoints, player_detections)
    ball_detections = interpolate_ball(ball_tracker, ball_detections)
    ball_shot_frames = get_shot_frames(ball_tracker, ball_detections)

    # ── 5. Mini-court coordinate conversion ───────────────────────────────────
    _progress("Projecting to mini court", 40)
    mini_court = build_mini_court(video_frames[0])
    player_mini_positions, ball_mini_positions = convert_to_mini_court_coordinates(
        mini_court, player_detections, ball_detections, court_keypoints
    )

    # ── 6. Statistics ─────────────────────────────────────────────────────────
    _progress("Computing statistics", 50)
    stats_result = generate_statistics(
        player_mini_positions,
        ball_mini_positions,
        ball_shot_frames,
        mini_court,
        fps=fps,
    )
    frame_stats_df = stats_result.pop("frame_stats_df")

    # ── 7. Heatmaps ───────────────────────────────────────────────────────────
    _progress("Generating heatmaps", 60)
    heatmap_paths = generate_heatmap(player_mini_positions, mini_court, output_dir)

    # ── 8. Shot map ───────────────────────────────────────────────────────────
    _progress("Generating shot map", 65)
    shot_map_path = generate_shot_map(ball_mini_positions, mini_court, output_dir)

    # ── 9. Highlights ─────────────────────────────────────────────────────────
    _progress("Detecting highlights", 70)
    highlights = generate_highlights(ball_shot_frames, ball_mini_positions, total_frames, fps=fps)

    # ── 10. Render annotated video ─────────────────────────────────────────────
    _progress("Rendering output video", 75)
    output_frames = player_tracker.draw_bboxes(list(video_frames), player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_detector.draw_keypoints_on_video(output_frames, court_keypoints)
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_positions)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_positions, color=(0, 255, 255))
    output_frames = draw_player_stats(output_frames, frame_stats_df)

    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _progress("Saving video", 90)
    video_out_path = os.path.join(output_dir, "video.avi")
    save_video(output_frames, video_out_path, fps=fps)

    _progress("Saving highlights reel", 92)
    highlights_out_path = os.path.join(output_dir, "highlights.avi")
    if highlights:
        save_highlights_video(output_frames, highlights, highlights_out_path, fps=fps)
    else:
        highlights_out_path = None

    # ── 11. Save analysis.json ────────────────────────────────────────────────
    _progress("Saving analysis", 95)
    analysis = {
        "player_1": stats_result.get("player_1", {}),
        "player_2": stats_result.get("player_2", {}),
        "player_3": stats_result.get("player_3", {}),
        "player_4": stats_result.get("player_4", {}),
        "highlights": highlights,
        "total_frames": total_frames,
        "fps": fps,
    }
    analysis_path = os.path.join(output_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    _progress("Done", 100)

    return {
        "video_path": video_out_path,
        "analysis_path": analysis_path,
        "heatmap_paths": heatmap_paths,
        "shot_map_path": shot_map_path,
        "highlights_video_path": highlights_out_path,
        "stats": analysis,
        "highlights": highlights,
    }
