import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from copy import deepcopy
import constants
from utils import measure_distance, convert_pixel_distance_to_meters

logger = logging.getLogger(__name__)


def generate_statistics(
    player_mini_court_detections: list,
    ball_mini_court_detections: list,
    ball_shot_frames: list,
    mini_court,
    fps: float = 24.0,
) -> dict:
    """
    Calculate per-player shot speeds, movement speeds, distance covered for 4 players.
    Computes frame-by-frame movement stats independently of shot detection.
    """
    court_pixel_width = mini_court.get_width_of_mini_court()

    per_player_distance = {i: 0.0 for i in range(1, 5)}
    per_player_speeds = {i: [] for i in range(1, 5)}

    for f in range(1, len(player_mini_court_detections)):
        prev_frame = player_mini_court_detections[f - 1]
        curr_frame = player_mini_court_detections[f]
        for pid in range(1, 5):
            if pid in prev_frame and pid in curr_frame:
                d_px = measure_distance(prev_frame[pid], curr_frame[pid])
                d_m = convert_pixel_distance_to_meters(
                    d_px, constants.COURT_WIDTH, court_pixel_width,
                )
                per_player_distance[pid] += d_m
                speed_kmh = d_m * fps * 3.6
                per_player_speeds[pid].append(speed_kmh)

    for pid in range(1, 5):
        logger.info(
            "Player %d: distance=%.1fm, avg_speed=%.1f km/h, frames=%d",
            pid, per_player_distance[pid],
            sum(per_player_speeds[pid]) / max(len(per_player_speeds[pid]), 1),
            len(per_player_speeds[pid]),
        )

    base_stats = {'frame_num': 0}
    for i in range(1, 5):
        base_stats[f'player_{i}_number_of_shots'] = 0
        base_stats[f'player_{i}_total_shot_speed'] = 0
        base_stats[f'player_{i}_last_shot_speed'] = 0
        base_stats[f'player_{i}_total_player_speed'] = 0
        base_stats[f'player_{i}_last_player_speed'] = 0

    player_stats_data = [base_stats]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_seconds = (end_frame - start_frame) / fps

        if ball_shot_time_seconds == 0:
            continue

        if start_frame >= len(ball_mini_court_detections) or \
           end_frame >= len(ball_mini_court_detections):
            continue

        start_ball = ball_mini_court_detections[start_frame]
        end_ball = ball_mini_court_detections[end_frame]
        if 1 not in start_ball or 1 not in end_ball:
            continue

        distance_pixels = measure_distance(start_ball[1], end_ball[1])
        distance_meters = convert_pixel_distance_to_meters(
            distance_pixels, constants.COURT_WIDTH, court_pixel_width,
        )
        ball_speed_kmh = distance_meters / ball_shot_time_seconds * 3.6

        player_positions = player_mini_court_detections[start_frame]
        if not player_positions:
            continue

        player_shot_ball = min(
            player_positions.keys(),
            key=lambda pid: measure_distance(
                player_positions[pid], start_ball[1],
            ),
        )

        current = deepcopy(player_stats_data[-1])
        current['frame_num'] = start_frame
        current[f'player_{player_shot_ball}_number_of_shots'] += 1
        current[f'player_{player_shot_ball}_total_shot_speed'] += ball_speed_kmh
        current[f'player_{player_shot_ball}_last_shot_speed'] = ball_speed_kmh

        for pid in range(1, 5):
            if pid in player_mini_court_detections[start_frame] and \
               pid in player_mini_court_detections[end_frame]:
                p_dist_px = measure_distance(
                    player_mini_court_detections[start_frame][pid],
                    player_mini_court_detections[end_frame][pid],
                )
                p_dist_m = convert_pixel_distance_to_meters(
                    p_dist_px, constants.COURT_WIDTH, court_pixel_width,
                )
                p_speed = p_dist_m / ball_shot_time_seconds * 3.6
                current[f'player_{pid}_total_player_speed'] += p_speed
                current[f'player_{pid}_last_player_speed'] = p_speed

        player_stats_data.append(current)

    df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(player_mini_court_detections)))})
    df = pd.merge(frames_df, df, on='frame_num', how='left').ffill()
    df = df.fillna(0)

    for i in range(1, 5):
        num_shots = df[f'player_{i}_number_of_shots'].replace(0, float('nan'))
        df[f'player_{i}_average_shot_speed'] = df[f'player_{i}_total_shot_speed'] / num_shots
        df[f'player_{i}_average_player_speed'] = (
            df[f'player_{i}_total_player_speed'] / max(1, len(ball_shot_frames))
        )

    last = df.iloc[-1] if not df.empty else pd.Series(dtype=float)

    def safe(val):
        try:
            return round(float(val), 2) if not pd.isna(val) else 0.0
        except Exception:
            return 0.0

    summary = {"frame_stats_df": df}
    for i in range(1, 5):
        speeds = per_player_speeds[i]
        avg_speed = sum(speeds) / max(len(speeds), 1)

        summary[f"player_{i}"] = {
            "total_shots": int(last.get(f'player_{i}_number_of_shots', 0)),
            "avg_shot_speed": safe(last.get(f'player_{i}_average_shot_speed')),
            "max_shot_speed": safe(df[f'player_{i}_last_shot_speed'].max() if not df.empty else 0),
            "avg_player_speed": round(avg_speed, 2),
            "distance_covered": round(per_player_distance[i], 2),
        }

    logger.info("Stats summary: %s",
                {k: v for k, v in summary.items() if k != "frame_stats_df"})

    return summary
