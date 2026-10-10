"""Annotated preview video for checking tracking by eye (not part of the analytics)."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from padelvision.court.calibration import CourtCalibration
from padelvision.court.draw import TEAM_COLORS, MiniCourt, draw_court_overlay
from padelvision.overlay import StatsHud


def render_preview(
    video_path: str | Path,
    players: pd.DataFrame,
    cal: CourtCalibration,
    out_path: str | Path,
    start_s: float = 0.0,
    seconds: float = 30.0,
    ball: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    rallies: list[dict] | None = None,
    hud: StatsHud | None = None,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = round(start_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    by_frame = {f: g for f, g in players.groupby("frame")}
    ball_at = {} if ball is None else {int(r.frame): r for r in ball.itertuples()}
    shown = pd.DataFrame() if events is None else events[events.kind.isin(EVENT_STYLE)]
    show_frames = round(EVENT_SHOW_S * fps)
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
            _draw_ball(img, ball_at, frame)
            if len(shown):
                recent = shown[shown.frame.between(frame - show_frames, frame)]
                _draw_events(img, canvas, mini, recent)
            _draw_rally(img, rallies or [], shown, frame)
            mh, mw = canvas.shape[:2]
            img[10 : 10 + mh, w - mw - 10 : w - 10] = canvas
            if hud is not None:
                hud.draw(img, frame)  # includes the time
            else:
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


BALL_TRAIL = 8  # frames


def _draw_ball(img, ball_at: dict, frame: int) -> None:
    """Ball in play: filled when detected, hollow when interpolated, with a short trail."""
    pts = [ball_at[f] for f in range(frame - BALL_TRAIL, frame + 1) if f in ball_at]
    for a, b in zip(pts, pts[1:], strict=False):
        if b.frame - a.frame == 1 and a.track == b.track:
            pa, pb = (round(a.u), round(a.v)), (round(b.u), round(b.v))
            cv2.line(img, pa, pb, (0, 220, 255), 1, cv2.LINE_AA)
    r = ball_at.get(frame)
    if r is not None:
        filled = -1 if r.state == "detected" else 1
        cv2.circle(img, (round(r.u), round(r.v)), 6, (0, 255, 255), filled, cv2.LINE_AA)
        cv2.circle(img, (round(r.u), round(r.v)), 9, (0, 0, 0), 1, cv2.LINE_AA)


EVENT_SHOW_S = 0.5  # an event stays marked this long
# BGR; ball handling and unclassified turns are not drawn
EVENT_STYLE = {"hit": (0, 0, 255), "bounce": (0, 220, 0)}


def _draw_events(img, canvas, mini: MiniCourt, recent: pd.DataFrame) -> None:
    for e in recent.itertuples():
        color = EVENT_STYLE[e.kind]
        p = (round(e.u), round(e.v))
        cv2.circle(img, p, 16, color, 2, cv2.LINE_AA)
        label = f"HIT P{int(e.player)}" if e.kind == "hit" else "BOUNCE"
        cv2.putText(img, label, (p[0] + 18, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    cv2.LINE_AA)  # fmt: skip
        if e.kind == "bounce" and e.x_m == e.x_m:  # court position only for bounces
            cv2.drawMarker(
                canvas, mini.to_px((e.x_m, e.y_m)), color, cv2.MARKER_TILTED_CROSS, 10, 2
            )


def _draw_rally(img, rallies: list[dict], events: pd.DataFrame, frame: int) -> None:
    rally = next((r for r in rallies if r["start_frame"] <= frame <= r["end_frame"]), None)
    if rally is None:
        return
    hits = 0
    if len(events):
        hits = int(
            ((events.kind == "hit") & events.frame.between(rally["start_frame"], frame)).sum()
        )
    text = f"RALLY {rally['id']}  hits {hits}"
    cv2.rectangle(img, (8, 8), (30 + 13 * len(text), 42), (0, 0, 0), -1)
    cv2.putText(img, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
