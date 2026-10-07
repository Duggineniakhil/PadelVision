"""Annotated preview video for checking tracking by eye (not part of the analytics)."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from padelvision.court.calibration import CourtCalibration
from padelvision.court.draw import TEAM_COLORS, MiniCourt, draw_court_overlay


def render_preview(
    video_path: str | Path,
    players: pd.DataFrame,
    cal: CourtCalibration,
    out_path: str | Path,
    start_s: float = 0.0,
    seconds: float = 30.0,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = round(start_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    by_frame = {f: g for f, g in players.groupby("frame")}
    if players.empty or first > players.frame.max() or first + seconds * fps < players.frame.min():
        print(
            f"warning: {start_s:.0f}-{start_s + seconds:.0f}s has no analysed frames "
            f"(analysed up to {players.frame.max() / fps if len(players) else 0:.0f}s)"
        )
    mini = MiniCourt(height_px=min(300, h // 2))
    last = None
    try:
        for frame in range(first, first + round(seconds * fps)):
            ok, img = cap.read()
            if not ok:
                break
            rows = by_frame.get(frame, last)  # with stride > 1, hold the last analysed frame
            last = rows
            img = draw_court_overlay(img, cal, color=(0, 200, 200), thickness=1)
            canvas = mini.canvas()
            if rows is not None:
                for r in rows.itertuples():
                    color = TEAM_COLORS[r.team]
                    p1, p2 = (round(r.x1), round(r.y1)), (round(r.x2), round(r.y2))
                    cv2.rectangle(img, p1, p2, color, 2)
                    label = f"P{r.player}" + ("" if r.valid else " (feet cut)")
                    cv2.putText(
                        img,
                        label,
                        (p1[0], p1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    if r.valid:
                        cv2.circle(canvas, mini.to_px((r.x_m, r.y_m)), 5, color, -1, cv2.LINE_AA)
            mh, mw = canvas.shape[:2]
            img[10 : 10 + mh, w - mw - 10 : w - 10] = canvas
            cv2.putText(
                img,
                f"{frame / fps:7.2f}s",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(img)
    finally:
        cap.release()
        writer.release()
    return Path(out_path)
