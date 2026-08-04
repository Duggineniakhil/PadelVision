import numpy as np
import cv2
import math


def _safe(val):
    """Return 0.0 if val is NaN/Inf, otherwise float(val)."""
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def draw_player_stats(output_video_frames, player_stats):
    for index, row in player_stats.iterrows():
        if index >= len(output_video_frames):
            break

        frame = output_video_frames[index]
        frame_h, frame_w = frame.shape[0], frame.shape[1]

        scale = min(frame_w / 1280.0, frame_h / 720.0)
        scale = max(0.65, min(1.2, scale))

        team_a_shot_speed = _safe((row['player_1_last_shot_speed'] + row['player_2_last_shot_speed']) / 2)
        team_b_shot_speed = _safe((row['player_3_last_shot_speed'] + row['player_4_last_shot_speed']) / 2)
        team_a_speed = _safe((row['player_1_last_player_speed'] + row['player_2_last_player_speed']) / 2)
        team_b_speed = _safe((row['player_3_last_player_speed'] + row['player_4_last_player_speed']) / 2)
        avg_team_a_shot_speed = _safe((row['player_1_average_shot_speed'] + row['player_2_average_shot_speed']) / 2)
        avg_team_b_shot_speed = _safe((row['player_3_average_shot_speed'] + row['player_4_average_shot_speed']) / 2)
        avg_team_a_speed = _safe((row['player_1_average_player_speed'] + row['player_2_average_player_speed']) / 2)
        avg_team_b_speed = _safe((row['player_3_average_player_speed'] + row['player_4_average_player_speed']) / 2)

        # Compact size in bottom-left corner
        width = int(max(235, min(330, frame_w * 0.29)))
        height = int(max(105, min(145, frame_h * 0.22)))
        margin = int(max(10, 15 * scale))

        start_x = margin
        end_y = frame_h - margin
        start_y = end_y - height
        end_x = start_x + width

        # Background overlay with dark translucent box & white border
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (15, 20, 30), -1)
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1, cv2.LINE_AA)

        # Typography metrics
        f_scale = max(0.33, 0.40 * scale)
        f_thick = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Row positions
        row_step = height / 5.4
        y_header = int(start_y + row_step * 0.95)
        y_r1 = int(start_y + row_step * 2.05)
        y_r2 = int(start_y + row_step * 3.15)
        y_r3 = int(start_y + row_step * 4.25)
        y_r4 = int(start_y + row_step * 5.35)

        # Column positions
        col_label_x = start_x + int(8 * scale)
        col_ta_x = start_x + int(width * 0.41)
        col_tb_x = start_x + int(width * 0.72)

        # Draw Header
        cv2.putText(frame, "Team A (P1+P2)", (col_ta_x - 10, y_header), font, f_scale * 0.9, (255, 220, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, "Team B (P3+P4)", (col_tb_x - 10, y_header), font, f_scale * 0.9, (180, 180, 255), 1, cv2.LINE_AA)

        # Subtle divider under header
        div_y = int(start_y + row_step * 1.3)
        cv2.line(frame, (start_x + 5, div_y), (end_x - 5, div_y), (80, 90, 110), 1, cv2.LINE_AA)

        rows = [
            ("Shot Speed", f"{team_a_shot_speed:.1f}", f"{team_b_shot_speed:.1f}", y_r1),
            ("Player Speed", f"{team_a_speed:.1f}", f"{team_b_speed:.1f}", y_r2),
            ("avg. S. Speed", f"{avg_team_a_shot_speed:.1f}", f"{avg_team_b_shot_speed:.1f}", y_r3),
            ("avg. P. Speed", f"{avg_team_a_speed:.1f}", f"{avg_team_b_speed:.1f}", y_r4),
        ]

        for label, val_a, val_b, y_pos in rows:
            cv2.putText(frame, label, (col_label_x, y_pos), font, f_scale * 0.9, (190, 195, 205), f_thick, cv2.LINE_AA)
            cv2.putText(frame, f"{val_a} km/h", (col_ta_x, y_pos), font, f_scale, (255, 255, 255), f_thick, cv2.LINE_AA)
            cv2.putText(frame, f"{val_b} km/h", (col_tb_x, y_pos), font, f_scale, (255, 255, 255), f_thick, cv2.LINE_AA)

        output_video_frames[index] = frame

    return output_video_frames