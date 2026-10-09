"""Stage 5: ball events (hits, floor bounces, wall rebounds) and rallies (pure Python).

Inputs: ball.parquet (stage 4: ball in play, image px, one row per frame, tracklet ids),
players.parquet (stage 3: player boxes), and the court calibration.

Events are sharp turns of the ball's image path, found
- inside a tracklet: the direction of motion before and after a frame differs by more
  than MIN_TURN_DEG (a smooth lob apex turns gradually and is not an event), and
- at the junction of two consecutive tracklets: the tracker breaks a tracklet when the ball
  changes direction abruptly, so a short gap with nearby end/start points is a turn too.

Each turn is classified:
- "hit":    the ball is in a player's reach zone (box widened for arm + racket) AND its size
            fits that player's depth: a 6.5 cm ball next to a ~1.75 m person is ~1/27 of the
            box height. With a low camera, a far-court ball often appears inside a near
            player's (large) box; it is 2-3x too small to be at their racket.
- "bounce": the ball was moving down the image and then up (floor bounce)
- "wall":   any other sharp turn (glass rebounds, net cord, a hit we can't attribute)

Only bounces get a court position: the ball is on the ground there, so the homography is
valid. Rallies are runs of ball-in-play frames separated by pauses > RALLY_GAP_S.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import COURT_LENGTH_M, COURT_WIDTH_M

WINDOW = 4  # frames on each side used to measure direction in/out of a point
MIN_STEP_PX = 1.5  # px/frame: slower than this, direction is noise
MIN_TURN_DEG = 40.0
NMS_FRAMES = 6  # keep the sharpest turn within this many frames
JUNCTION_MAX_GAP = 8  # frames between tracklets that still count as one turn
JUNCTION_MAX_DIST_PX = 120.0
REACH_X = 0.6  # reach zone: box widened by this fraction of its width on each side
REACH_TOP = 0.35  # and raised by this fraction of its height (overhead racket)
BOUNCE_COURT_MARGIN_M = 0.5
BALL_TO_PLAYER = 0.065 / 1.75  # expected ball diameter / player box height at contact
REACH_DEPTH_FACTOR = 2.0  # allowed ratio error (crouching, blur, box noise)
RALLY_GAP_S = 2.0
MIN_RALLY_S = 1.5

EVENT_COLUMNS = ["frame", "t", "kind", "u", "v", "size", "x_m", "y_m", "in_court", "player",
                 "team", "track", "turn_deg", "speed_in", "speed_out", "source"]  # fmt: skip


def detect_events(
    ball: pd.DataFrame, players: pd.DataFrame, cal: CourtCalibration, fps: float
) -> pd.DataFrame:
    turns = _turns_within(ball) + _turns_at_junctions(ball)
    turns = _suppress(turns)
    boxes = {f: g for f, g in players.groupby("frame")} if len(players) else {}
    rows = []
    for tr in turns:
        player, team = _reaching_player(boxes.get(tr["frame"]), tr["u"], tr["v"], tr["size"])
        if player is not None:
            kind = "hit"
        elif tr["dv_in"] > 0 and tr["dv_out"] < 0:
            kind = "bounce"
        else:
            kind = "wall"
        x = y = np.nan
        in_court = None
        if kind == "bounce":
            x, y = cal.to_court(np.array([[tr["u"], tr["v"]]]))[0]
            in_court = bool(
                abs(x) <= COURT_WIDTH_M / 2 + BOUNCE_COURT_MARGIN_M
                and abs(y) <= COURT_LENGTH_M / 2 + BOUNCE_COURT_MARGIN_M
            )
        rows.append((
            tr["frame"], tr["frame"] / fps, kind, tr["u"], tr["v"], tr["size"], x, y, in_court,
            player, team, tr["track"], tr["turn"], tr["speed_in"], tr["speed_out"], tr["source"],
        ))  # fmt: skip
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("frame").reset_index(drop=True)


def detect_rallies(ball: pd.DataFrame, events: pd.DataFrame, fps: float) -> list[dict]:
    """Runs of ball activity separated by pauses longer than RALLY_GAP_S."""
    if ball.empty:
        return []
    frames = ball.frame.to_numpy()
    breaks = np.flatnonzero(np.diff(frames) > RALLY_GAP_S * fps) + 1
    rallies = []
    for seg in np.split(frames, breaks):
        start, end = int(seg[0]), int(seg[-1])
        if (end - start) / fps < MIN_RALLY_S:
            continue
        ev = events[(events.frame >= start) & (events.frame <= end)]
        hits = ev[ev.kind == "hit"]
        rallies.append({
            "id": len(rallies) + 1,
            "start_frame": start, "end_frame": end,
            "start_s": round(start / fps, 2), "end_s": round(end / fps, 2),
            "duration_s": round((end - start) / fps, 2),
            "ball_coverage": round(len(seg) / (end - start + 1), 2),
            "hits": int(len(hits)),
            "bounces": int((ev.kind == "bounce").sum()),
            "walls": int((ev.kind == "wall").sum()),
            "hits_by_player": {str(int(k)): int(v) for k, v in hits.player.value_counts().items()},
        })  # fmt: skip
    return rallies


def _turns_within(ball: pd.DataFrame) -> list[dict]:
    out = []
    for track, g in ball.groupby("track"):
        g = g.sort_values("frame")
        f = g.frame.to_numpy()
        p = g[["u", "v"]].to_numpy()
        sizes = g["size"].to_numpy() if "size" in g else np.full(len(g), np.nan)
        for i in range(WINDOW, len(g) - WINDOW):
            # only where the window is contiguous in time (no gaps inside the tracklet)
            if f[i + WINDOW] - f[i - WINDOW] != 2 * WINDOW:
                continue
            vin = (p[i] - p[i - WINDOW]) / WINDOW
            vout = (p[i + WINDOW] - p[i]) / WINDOW
            t = _turn(vin, vout)
            if t is not None:
                size = _nanmedian(sizes[i - WINDOW : i + WINDOW + 1])
                out.append(_turn_row(int(f[i]), p[i], vin, vout, t, int(track), "within", size))
    return out


def _turns_at_junctions(ball: pd.DataFrame) -> list[dict]:
    out = []
    ends = []
    for track, g in ball.groupby("track"):
        g = g.sort_values("frame")
        if len(g) < 3:
            continue
        sizes = g["size"].to_numpy() if "size" in g else np.full(len(g), np.nan)
        f, p = g.frame.to_numpy(), g[["u", "v"]].to_numpy()
        ends.append((int(f[0]), int(f[-1]), int(track), p, f, sizes))
    ends.sort()
    pairs = zip(ends, ends[1:], strict=False)
    for (_, a_end, a_id, a_p, a_f, a_s), (b_start, _, _, b_p, b_f, b_s) in pairs:
        gap = b_start - a_end
        if not 0 < gap <= JUNCTION_MAX_GAP:
            continue
        if np.linalg.norm(b_p[0] - a_p[-1]) > JUNCTION_MAX_DIST_PX:
            continue
        k_in, k_out = min(WINDOW, len(a_p) - 1), min(WINDOW, len(b_p) - 1)
        vin = (a_p[-1] - a_p[-1 - k_in]) / (a_f[-1] - a_f[-1 - k_in])
        vout = (b_p[k_out] - b_p[0]) / (b_f[k_out] - b_f[0])
        t = _turn(vin, vout)
        if t is not None:
            mid = (a_p[-1] + b_p[0]) / 2
            size = _nanmedian(np.r_[a_s[-WINDOW:], b_s[:WINDOW]])
            out.append(_turn_row((a_end + b_start) // 2, mid, vin, vout, t, a_id, "junction", size))
    return out


def _turn(vin: np.ndarray, vout: np.ndarray) -> float | None:
    sin, sout = np.linalg.norm(vin), np.linalg.norm(vout)
    if sin < MIN_STEP_PX or sout < MIN_STEP_PX:
        return None
    cos = float(np.clip(vin @ vout / (sin * sout), -1, 1))
    deg = float(np.degrees(np.arccos(cos)))
    return deg if deg >= MIN_TURN_DEG else None


def _nanmedian(x) -> float:
    x = np.asarray(x, float)
    return float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")


def _turn_row(frame, p, vin, vout, turn, track, source, size=float("nan")) -> dict:
    return {
        "frame": frame, "u": float(p[0]), "v": float(p[1]), "turn": round(turn, 1),
        "speed_in": round(float(np.linalg.norm(vin)), 2),
        "speed_out": round(float(np.linalg.norm(vout)), 2),
        "dv_in": float(vin[1]), "dv_out": float(vout[1]), "track": track, "source": source,
        "size": size,
    }  # fmt: skip


def _suppress(turns: list[dict]) -> list[dict]:
    """Keep the sharpest turn within NMS_FRAMES of each other."""
    kept: list[dict] = []
    for t in sorted(turns, key=lambda t: -t["turn"]):
        if all(abs(t["frame"] - k["frame"]) > NMS_FRAMES for k in kept):
            kept.append(t)
    return sorted(kept, key=lambda t: t["frame"])


def _reaching_player(boxes: pd.DataFrame | None, u: float, v: float, size: float = float("nan")):
    """(player, team) whose reach zone contains the ball and whose depth matches the ball's
    size (when known), nearest box centre first."""
    if boxes is None or boxes.empty:
        return None, None
    w = boxes.x2 - boxes.x1
    h = boxes.y2 - boxes.y1
    inside = (
        (u >= boxes.x1 - REACH_X * w) & (u <= boxes.x2 + REACH_X * w)
        & (v >= boxes.y1 - REACH_TOP * h) & (v <= boxes.y2)
    )  # fmt: skip
    if np.isfinite(size) and size > 0:
        ratio = size / h.clip(lower=1)
        inside &= np.abs(np.log(ratio / BALL_TO_PLAYER)) <= np.log(REACH_DEPTH_FACTOR)
    if not inside.any():
        return None, None
    cand = boxes[inside]
    d = np.hypot((cand.x1 + cand.x2) / 2 - u, (cand.y1 + cand.y2) / 2 - v)
    best = cand.loc[d.idxmin()]
    return int(best.player), str(best.team)
