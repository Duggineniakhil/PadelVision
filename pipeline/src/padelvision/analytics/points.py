"""Stage 5b: serves, points and an estimate of the games, from events + players (no video).

Serve: in padel the server bounces the ball behind their service line and hits it underhand,
diagonally, into the service box across the net. In the data that is a hit
- from behind the service line (feet |y| >= SERVICE_LINE - BEHIND_MARGIN_M; the margin covers
  standing on the line and the foot position error),
- after a pause in play (no hit in the PAUSE_S before; between points the ball is collected),
- "confirmed" when its first bounce is seen in the diagonal service box (the other half,
  between the net and the service line, the other side of the centre line), "likely" when the
  landing wasn't seen, and flagged `fault_suspected` when it was seen outside the box.
Hand-checked on Test_video review frames: the serves at 32.7 s, 137.3 s and 196.5 s are real.

Points: every serve starts a point. A serve by the same server within FAULT_S of their
previous one, with no hit by anyone else in between, is a second serve: the same point.
A point ends where the next one starts.

Games (an estimate): one person serves a whole game, so a new game starts when the server
changes, and always at a changeover. Serves the ball detector missed hide points and can hide
whole games, so the result is checked against padel's rules and the checks are reported:
changeovers come after games 1, 3, 5, ... (two games between changeovers) and the serve
alternates between the pairs from game to game. Who won a point is not attempted: it needs the
ball followed to the end of every rally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from padelvision.court.geometry import SERVICE_LINE_FROM_NET_M as SERVICE

BEHIND_MARGIN_M = 0.5
MAX_DEPTH_M = 11.0  # behind the back wall is not a serve position
PAUSE_S = 4.0
LANDING_S = 2.5  # the serve's first bounce comes within this
BOX_MARGIN_M = 0.5  # landing position error allowed around the service box
CENTRE_MARGIN_M = 0.5
RETURN_S = 4.0  # the receiving pair hits within this when the serve was returned
FAULT_S = 25.0
FEET_WINDOW_S = 0.5  # the server's feet: median of their positions this close to the hit


def _feet(players: pd.DataFrame, player: int, frame: int, fps: float) -> tuple[float, float] | None:
    near_in_time = players.frame.between(frame - FEET_WINDOW_S * fps, frame + FEET_WINDOW_S * fps)
    rows = players[(players.player == player) & players.valid & near_in_time]
    if "changeover" in rows:
        rows = rows[~rows.changeover]
    if rows.empty:
        return None
    return float(rows.x_m.median()), float(rows.y_m.median())


def find_serves(events: pd.DataFrame, players: pd.DataFrame, fps: float) -> list[dict]:
    """Hits that look like serves, in time order (see the module docstring)."""
    hits = events[events.kind == "hit"].sort_values("frame")
    bounces = events[(events.kind == "bounce") & events.x_m.notna()].sort_values("frame")
    out = []
    prev_hit = None
    for h in hits.itertuples():
        pause = prev_hit is None or (h.frame - prev_hit) / fps >= PAUSE_S
        prev_hit = h.frame
        if not pause or h.player != h.player:  # NaN player: hitter unknown
            continue
        feet = _feet(players, int(h.player), int(h.frame), fps)
        if feet is None:
            continue
        x, y = feet
        if not SERVICE - BEHIND_MARGIN_M <= abs(y) <= MAX_DEPTH_M:
            continue
        near = y < 0
        own_left = x < 0 if near else x > 0
        land = bounces[bounces.frame.between(h.frame + 1, h.frame + LANDING_S * fps)].head(1)
        in_box = None
        landing = None
        if len(land):
            lx, ly = float(land.x_m.iloc[0]), float(land.y_m.iloc[0])
            landing = [round(lx, 2), round(ly, 2)]
            other_half = (ly > 0) == near
            in_depth = abs(ly) <= SERVICE + BOX_MARGIN_M
            diagonal = lx * x < 0 or abs(lx) <= CENTRE_MARGIN_M
            in_box = bool(other_half and in_depth and diagonal)
        returned = bool(((hits.frame > h.frame) & (hits.frame <= h.frame + RETURN_S * fps)
                         & (hits.team != h.team)).any())  # fmt: skip
        out.append({
            "frame": int(h.frame), "t": round(h.frame / fps, 2), "server": int(h.player),
            "team": "near" if near else "far", "side": "left" if own_left else "right",
            "feet": [round(x, 2), round(y, 2)], "landing": landing, "in_box": in_box,
            "returned": returned,
            "confidence": "confirmed" if in_box else "likely",
            "fault_suspected": in_box is False,
        })  # fmt: skip
    return out


def build_points(
    serves: list[dict], events: pd.DataFrame, fps: float, end_frame: int
) -> list[dict]:
    """Serves -> points (a fault and its second serve are one point)."""
    hits = events[events.kind == "hit"]
    points: list[dict] = []
    for s in serves:
        last = points[-1] if points else None
        if last is not None and last["server"] == s["server"]:
            prev = last["serves"][-1]
            others = hits[hits.frame.between(prev["frame"] + 1, s["frame"] - 1)
                          & (hits.player != s["server"])]  # fmt: skip
            if (s["frame"] - prev["frame"]) / fps <= FAULT_S and others.empty:
                last["serves"].append(s)  # the first one was a fault: second serve
                continue
        points.append({"server": s["server"], "team": s["team"], "side": s["side"], "serves": [s]})
    for i, p in enumerate(points):
        start = p["serves"][0]["frame"]
        end = points[i + 1]["serves"][0]["frame"] - 1 if i + 1 < len(points) else end_frame
        in_point = hits[hits.frame.between(start, end)]
        p.update({
            "id": i + 1, "start_frame": start, "end_frame": int(end),
            "start_s": round(start / fps, 2), "end_s": round(end / fps, 2),
            "second_serve": len(p["serves"]) > 1, "hits": int(len(in_point)),
        })  # fmt: skip
        p["serves"] = [s["frame"] for s in p["serves"]]
    return points


def estimate_games(points: list[dict], changeovers: list[dict], fps: float) -> dict:
    """Group points into games by serving pair and changeovers, and check padel's rules.

    By pair, not by person: which partner is which is less certain than which pair is on
    which side, and a partner mix-up must not split a game."""
    cuts = sorted(c["t"] for c in changeovers)
    games: list[dict] = []
    for p in points:
        period = int(np.searchsorted(cuts, p["start_s"]))
        pair = "A" if p["server"] <= 2 else "B"
        g = games[-1] if games else None
        if g is None or g["pair"] != pair or g["period"] != period:
            games.append({"pair": pair, "period": period, "points": [], "servers": []})
        games[-1]["points"].append(p["id"])
        games[-1]["servers"].append(p["server"])
    checks = []
    for k in range(1, len(cuts)):  # whole stretches between two changeovers hold two games
        n = sum(g["period"] == k for g in games)
        checks.append({"between_s": [cuts[k - 1], cuts[k]], "games_found": n, "expected": 2,
                       "ok": n == 2})  # fmt: skip
    alternating = all(a["pair"] != b["pair"] for a, b in zip(games, games[1:], strict=False))
    for i, g in enumerate(games):
        g["id"] = i + 1
        g["server"] = max(set(g["servers"]), key=g["servers"].count)
        del g["period"], g["servers"]
    consistent = bool(checks) and alternating and all(c["ok"] for c in checks)
    return {
        "games": games,
        "checks": checks,
        "serve_alternates_between_pairs": alternating,
        "consistent": consistent,
        "note": "estimate: missed serves hide points and can hide whole games"
        + ("" if consistent else "; padel's rules checks fail, so the game count is unreliable"),
    }


def points_summary(events: pd.DataFrame, players: pd.DataFrame, fps: float, end_frame: int,
                   changeovers: list[dict] | None = None) -> dict:  # fmt: skip
    serves = find_serves(events, players, fps)
    points = build_points(serves, events, fps, end_frame)
    return {
        "serves": serves,
        "points": points,
        "games_estimate": estimate_games(points, changeovers or [], fps),
        "counts": {
            "serves": len(serves),
            "confirmed": sum(s["confidence"] == "confirmed" for s in serves),
            "faults_suspected": sum(s["fault_suspected"] for s in serves),
            "points": len(points),
            "second_serves": sum(p["second_serve"] for p in points),
        },
    }
