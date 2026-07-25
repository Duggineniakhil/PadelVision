import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import measure_distance

# Tuning thresholds
FAST_SHOT_SPEED_THRESHOLD = 0.4   # Normalized ball speed (fraction of max)
LONG_RALLY_MIN_SHOTS = 5          # Consecutive shots to count as a long rally
CLIP_PADDING_FRAMES = 60          # Extra frames to include before/after a highlight

def generate_highlights(
    ball_shot_frames: list[int],
    ball_mini_court_positions: list[dict],
    total_frames: int,
    fps: float = 24.0,
) -> list[dict]:
    """
    Identify Padel highlight moments based on shot speed, rally length, and rebounds.
    """
    highlights = []

    if len(ball_shot_frames) < 2:
        return highlights

    # Compute per-shot ball speeds
    shot_speeds = []
    for i in range(len(ball_shot_frames) - 1):
        sf = ball_shot_frames[i]
        ef = ball_shot_frames[i + 1]
        if 1 in ball_mini_court_positions[sf] and 1 in ball_mini_court_positions[ef]:
            d = measure_distance(
                ball_mini_court_positions[sf][1],
                ball_mini_court_positions[ef][1],
            )
            dt = (ef - sf) / fps
            speed = d / dt if dt > 0 else 0
        else:
            speed = 0
        shot_speeds.append(speed)

    max_speed = max(shot_speeds) if shot_speeds else 1.0

    # ── Highlight: Fast Padel Smashes ──────────────────────────────────────────
    for i, speed in enumerate(shot_speeds):
        norm_speed = speed / max(max_speed, 1)
        if norm_speed >= FAST_SHOT_SPEED_THRESHOLD:
            center_frame = ball_shot_frames[i]
            start = max(0, center_frame - CLIP_PADDING_FRAMES)
            end = min(total_frames - 1, center_frame + CLIP_PADDING_FRAMES)
            highlights.append({
                "start_frame": start,
                "end_frame": end,
                "timestamp_seconds": round(center_frame / fps, 2),
                "label": f"Fast Smash — {round(norm_speed * 100)}% max speed",
            })

    # ── Highlight: Long Padel Rallies ──────────────────────────────────────────
    rally_start_idx = 0
    for i in range(1, len(ball_shot_frames)):
        rally_length = i - rally_start_idx
        if rally_length >= LONG_RALLY_MIN_SHOTS:
            # Check if the next shot ends the rally (gap > 3s)
            gap = ball_shot_frames[i] - ball_shot_frames[i - 1]
            if gap > fps * 3 or i == len(ball_shot_frames) - 1:
                sf = ball_shot_frames[rally_start_idx]
                ef = ball_shot_frames[i - 1]
                start = max(0, sf - CLIP_PADDING_FRAMES)
                end = min(total_frames - 1, ef + CLIP_PADDING_FRAMES)
                highlights.append({
                    "start_frame": start,
                    "end_frame": end,
                    "timestamp_seconds": round(sf / fps, 2),
                    "label": f"Long Padel Rally — {rally_length} shots (inc. wall bounces)",
                })
                rally_start_idx = i

    # Deduplicate overlapping highlights by merging close ones
    highlights = _merge_overlapping(highlights)
    return highlights

def _merge_overlapping(highlights: list[dict]) -> list[dict]:
    """Merge highlights whose frame ranges overlap or are within 30 frames."""
    if not highlights:
        return []
    highlights = sorted(highlights, key=lambda h: h["start_frame"])
    merged = [highlights[0]]
    for h in highlights[1:]:
        prev = merged[-1]
        if h["start_frame"] <= prev["end_frame"] + 30:
            prev["end_frame"] = max(prev["end_frame"], h["end_frame"])
            
            # Combine labels uniquely
            labels = set(prev["label"].split(" / ")) | set(h["label"].split(" / "))
            prev["label"] = " / ".join(labels)
        else:
            merged.append(h)
    return merged
