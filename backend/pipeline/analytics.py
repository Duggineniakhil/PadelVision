import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from copy import deepcopy
import constants
from utils import measure_distance, convert_pixel_distance_to_meters

def generate_statistics(
    player_mini_court_detections: list,
    ball_mini_court_detections: list,
    ball_shot_frames: list,
    mini_court,
    fps: float = 24.0,
) -> dict:
    """
    Calculate per-player shot speeds, movement speeds, distance covered for 4 players.
    """
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

        # Ball distance & speed
        distance_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_meters = convert_pixel_distance_to_meters(
            distance_pixels,
            constants.COURT_WIDTH,
            mini_court.get_width_of_mini_court(),
        )
        ball_speed_kmh = distance_meters / ball_shot_time_seconds * 3.6

        # Which player hit the shot
        player_positions = player_mini_court_detections[start_frame]
        if not player_positions:
            continue
            
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda pid: measure_distance(
                player_positions[pid],
                ball_mini_court_detections[start_frame][1],
            ),
        )

        current = deepcopy(player_stats_data[-1])
        current['frame_num'] = start_frame
        current[f'player_{player_shot_ball}_number_of_shots'] += 1
        current[f'player_{player_shot_ball}_total_shot_speed'] += ball_speed_kmh
        current[f'player_{player_shot_ball}_last_shot_speed'] = ball_speed_kmh
        
        # Calculate movement for ALL players during this segment
        for pid in range(1, 5):
            if pid in player_mini_court_detections[start_frame] and pid in player_mini_court_detections[end_frame]:
                p_distance_pixels = measure_distance(
                    player_mini_court_detections[start_frame][pid],
                    player_mini_court_detections[end_frame][pid],
                )
                p_distance_meters = convert_pixel_distance_to_meters(
                    p_distance_pixels,
                    constants.COURT_WIDTH,
                    mini_court.get_width_of_mini_court(),
                )
                p_speed_kmh = p_distance_meters / ball_shot_time_seconds * 3.6
                current[f'player_{pid}_total_player_speed'] += p_speed_kmh
                current[f'player_{pid}_last_player_speed'] = p_speed_kmh

        player_stats_data.append(current)

    # Build full per-frame dataframe
    df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(player_mini_court_detections)))})
    df = pd.merge(frames_df, df, on='frame_num', how='left').ffill()
    
    # Fill any remaining NaNs with 0
    df = df.fillna(0)

    for i in range(1, 5):
        num_shots = df[f'player_{i}_number_of_shots'].replace(0, float('nan'))
        df[f'player_{i}_average_shot_speed'] = df[f'player_{i}_total_shot_speed'] / num_shots
        # Average movement speed (total speed accumulated over N segments / N segments)
        # We roughly approximate by using the number of shots as the segment count proxy, or better:
        df[f'player_{i}_average_player_speed'] = df[f'player_{i}_total_player_speed'] / max(1, len(ball_shot_frames))

    last = df.iloc[-1] if not df.empty else pd.Series(dtype=float)

    def safe(val):
        try:
            return round(float(val), 2) if not pd.isna(val) else 0.0
        except Exception:
            return 0.0

    summary = {"frame_stats_df": df}
    for i in range(1, 5):
        summary[f"player_{i}"] = {
            "total_shots": int(last.get(f'player_{i}_number_of_shots', 0)),
            "avg_shot_speed": safe(last.get(f'player_{i}_average_shot_speed')),
            "max_shot_speed": safe(df[f'player_{i}_last_shot_speed'].max() if not df.empty else 0),
            "avg_player_speed": safe(last.get(f'player_{i}_average_player_speed')),
            "distance_covered": safe(last.get(f'player_{i}_total_player_speed', 0)),
        }

    return summary
