"""Highlights: cut the rallies out of the match video and join them, with a title card per
clip, hit/bounce markers, the mini court and the stats overlay. No model is involved: the
rallies come from stage 5 (rallies.json).

mode "top": the `top_n` best rallies by a simple score, shown in match order.
mode "all": every rally in order (a condensed match without the dead time).

Score = hits + duration_s / 4 + fastest measured shot (km/h) / 25. A heuristic: longer
exchanges with more shots rank higher; the shot speed term only breaks ties, since it is an
estimate and often missing. Clip boundaries are only as good as hit detection: a missed
hit can split a point or start the clip late.

Inside a clip the ball's measured image path is drawn as a trail. After each clip a summary
card (CARD_S) shows each player's heatmap during the rally and its shot map (hitter's feet ->
landing bounce, both on the ground; the flight itself is not measured, so not drawn).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from padelvision.court.draw import TEAM_COLORS, MiniCourt
from padelvision.overlay import StatsHud
from padelvision.render import EVENT_SHOW_S, EVENT_STYLE, _draw_ball, _draw_events
from padelvision.visuals import rally_card

PAD_S = 0.5  # extra video kept before and after each rally
TITLE_S = 1.5  # the title card stays this long at the start of a clip
CARD_S = 3.0  # the summary card after each clip


def select_clips(summary: dict, fps: float, n_frames: int, mode: str = "top",
                 top_n: int = 5) -> list[dict]:  # fmt: skip
    """rallies.json summary -> clips [{rally_id, start_frame, end_frame, ...}] in match order."""
    if mode not in ("top", "all"):
        raise ValueError(f"mode must be 'top' or 'all', got {mode!r}")
    per = {r["id"]: r for r in summary.get("stats", {}).get("per_rally", [])}
    pad = round(PAD_S * fps)
    clips = []
    for r in summary.get("rallies", []):
        fastest = per.get(r["id"], {}).get("max_shot_kmh")
        score = r["hits"] + r["duration_s"] / 4 + (fastest or 0.0) / 25
        clips.append({
            "rally_id": r["id"],
            "start_frame": max(0, r["start_frame"] - pad),
            "end_frame": min(n_frames - 1, r["end_frame"] + pad),
            "start_s": round(max(0, r["start_frame"] - pad) / fps, 2),
            "end_s": round(min(n_frames - 1, r["end_frame"] + pad) / fps, 2),
            "duration_s": r["duration_s"], "hits": r["hits"], "max_shot_kmh": fastest,
            "score": round(score, 2),
        })  # fmt: skip
    if mode == "top":
        clips = sorted(clips, key=lambda c: -c["score"])[:top_n]
    return sorted(clips, key=lambda c: c["start_frame"])


def title_lines(clip: dict, index: int, total: int) -> list[str]:
    lines = [f"Rally {clip['rally_id']}   ({index + 1}/{total})",
             f"{clip['hits']} hits  -  {clip['duration_s']:.1f} s"]  # fmt: skip
    if clip["max_shot_kmh"] is not None:
        lines.append(f"fastest shot ~{clip['max_shot_kmh']:.0f} km/h (est.)")
    return lines


def render_highlights(
    video: str | Path,
    out_path: str | Path,
    clips: list[dict],
    players: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    hud: StatsHud | None = None,
    ball: pd.DataFrame | None = None,
    cards: dict[int, np.ndarray] | None = None,
) -> Path:
    """Write the clips (in order) to one video, each followed by its summary card
    (`cards`: rally id -> BGR image) if given. Reads the match video once, front to back."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame = {} if players is None else {f: g for f, g in players.groupby("frame")}
    shown = pd.DataFrame() if events is None else events[events.kind.isin(EVENT_STYLE)]
    show_frames = round(EVENT_SHOW_S * fps)
    ball_at = {} if ball is None else {int(r.frame): r for r in ball.itertuples()}
    mini = MiniCourt(height_px=min(300, h // 2))
    todo = sorted(clips, key=lambda c: c["start_frame"])
    try:
        f, ci = 0, 0
        while ci < len(todo):
            ok, img = cap.read()
            if not ok:
                break
            clip = todo[ci]
            if f >= clip["start_frame"]:
                canvas = mini.canvas()
                rows = by_frame.get(f)
                if rows is not None:
                    for r in rows.itertuples():
                        if r.valid:
                            cv2.circle(canvas, mini.to_px((r.x_m, r.y_m)), 5, TEAM_COLORS[r.team],
                                       -1, cv2.LINE_AA)  # fmt: skip
                _draw_ball(img, ball_at, f)
                if len(shown):
                    _draw_events(img, canvas, mini, shown[shown.frame.between(f - show_frames, f)])
                mh, mw = canvas.shape[:2]
                img[10 : 10 + mh, w - mw - 10 : w - 10] = canvas
                if hud is not None:
                    hud.draw(img, f)
                if f - clip["start_frame"] < TITLE_S * fps:
                    _title(img, title_lines(clip, ci, len(todo)))
                writer.write(img)
                if f >= clip["end_frame"]:
                    card = (cards or {}).get(clip["rally_id"])
                    if card is not None:
                        if card.shape[:2] != (h, w):
                            card = cv2.resize(card, (w, h), interpolation=cv2.INTER_AREA)
                        for _ in range(round(CARD_S * fps)):
                            writer.write(card)
                    ci += 1
            f += 1
    finally:
        cap.release()
        writer.release()
    return Path(out_path)


def highlights(video: str | Path, run_dir: str | Path, out_path: str | Path | None = None,
               mode: str = "top", top_n: int = 5) -> dict:  # fmt: skip
    """Select clips from a run folder, write highlights.json and the video."""
    run = Path(run_dir)
    info = json.loads((run / "video.json").read_text())
    summary = json.loads((run / "rallies.json").read_text())
    clips = select_clips(summary, info["fps"], info["frame_count"], mode, top_n)
    name = "highlights" if mode == "top" else "condensed"
    out = Path(out_path) if out_path else run / f"{name}.mp4"
    card_s = CARD_S if (run / "tracks.parquet").exists() else 0.0
    total_s = round(sum(c["end_s"] - c["start_s"] + card_s for c in clips), 1)
    result = {"mode": mode, "top_n": top_n if mode == "top" else None, "video": out.name,
              "clips": clips, "total_s": total_s}  # fmt: skip
    (run / f"{name}.json").write_text(json.dumps(result, indent=2))
    if not clips:
        print("No rallies found: nothing to cut.")
        return result
    players_path, events_path = run / "players.parquet", run / "events.parquet"
    ball_path, tracks_path = run / "ball.parquet", run / "tracks.parquet"
    cards = None
    if tracks_path.exists():
        tracks = pd.read_parquet(tracks_path)
        shots = summary.get("stats", {}).get("shots", [])
        by_id = {r["id"]: r for r in summary["rallies"]}
        size = (int(info["width"]), int(info["height"]))
        cards = {c["rally_id"]: rally_card(by_id[c["rally_id"]], tracks, shots, info["fps"], size)
                 for c in clips}  # fmt: skip
    render_highlights(
        video, out, clips,
        players=pd.read_parquet(players_path) if players_path.exists() else None,
        events=pd.read_parquet(events_path) if events_path.exists() else None,
        hud=StatsHud.from_run(run),
        ball=pd.read_parquet(ball_path) if ball_path.exists() else None,
        cards=cards,
    )  # fmt: skip
    print(f"{name}: {len(clips)} clips, {result['total_s']} s -> {out}")
    return result


def _title(img: np.ndarray, lines: list[str]) -> None:
    h, w = img.shape[:2]
    k = h / 720
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, (1.1 if i == 0 else 0.75) * k, 2)[0]
             for i, t in enumerate(lines)]  # fmt: skip
    box_w = max(s[0] for s in sizes) + round(40 * k)
    box_h = sum(s[1] for s in sizes) + round((20 + 18 * len(lines)) * k)
    x0, y0 = (w - box_w) // 2, round(h * 0.12)
    roi = img[y0 : y0 + box_h, x0 : x0 + box_w]
    img[y0 : y0 + box_h, x0 : x0 + box_w] = (roi * 0.3).astype(img.dtype)
    y = y0 + round(14 * k)
    for i, (t, s) in enumerate(zip(lines, sizes, strict=True)):
        y += s[1] + round(14 * k)
        cv2.putText(img, t, ((w - s[0]) // 2, y), font, (1.1 if i == 0 else 0.75) * k,
                    (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
