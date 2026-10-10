"""Rally statistics from stage 5 (events.parquet + rallies) and the players' feet (stage 3).

Only what the events support, and only inside rallies (ball handling and warm-up excluded):
- rally count and lengths, hits per rally / player / team
- placement: floor bounces inside the court, each credited to the team whose hit came
  before it, by depth zone on the receiving side (net / transition / back)
- shot speed ESTIMATE: ground distance from the hitter's feet to the next in-court bounce on
  the other side of the net, divided by the time between them. The hit point is in the air, so
  the feet stand in for it (the homography is only valid on the ground), and the ball flies
  further than the ground line: it is a lower bound. Reported only with MIN_SAMPLES shots.
Anything without enough data is null with a reason instead of a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from padelvision.court.geometry import zone_of

MIN_SAMPLES = 3
SHOT_MAX_S = 2.5  # a bounce later than this after the hit is not that shot's landing
FEET_MAX_DT_S = 0.2  # the hitter's feet must be seen this close to the hit
MAX_SHOT_KMH = 250.0  # faster = a pairing or position error, not a shot
ZONES = ("net", "transition", "back")


def rally_stats(
    events: pd.DataFrame, rallies: list[dict], players: pd.DataFrame, fps: float
) -> dict:
    if not rallies:
        return {"rallies": 0, "insufficient_data": "no rallies detected"}
    in_rally = np.zeros(len(events), bool)
    for r in rallies:
        in_rally |= events.frame.between(r["start_frame"], r["end_frame"]).to_numpy()
    ev = events[in_rally].sort_values("frame")
    hits = ev[ev.kind == "hit"]
    durations = [r["duration_s"] for r in rallies]
    per_rally = [r["hits"] for r in rallies]
    longest = max(rallies, key=lambda r: r["duration_s"])

    shots = _shots(ev, rallies, players, fps)
    return {
        "rallies": len(rallies),
        "rally_time_s": round(sum(durations), 1),
        "duration_s": {
            "mean": round(float(np.mean(durations)), 1),
            "median": round(float(np.median(durations)), 1),
            "max": round(max(durations), 1),
        },  # fmt: skip
        "longest_rally_id": longest["id"],
        "hits_per_rally": {"mean": round(float(np.mean(per_rally)), 1), "max": max(per_rally)},
        "hits_by_player": {str(int(k)): int(v) for k, v in hits.player.value_counts().items()},
        "hits_by_team": {str(k): int(v) for k, v in hits.team.value_counts().items()},
        "placement": _placement(shots),
        "shot_speed_estimate": _speeds(shots),
        "per_rally": _per_rally(shots, rallies),
        # every rally hit, for overlays: frame, player, team, landing bounce, speed estimate
        "shots": shots,
    }


def _shots(ev: pd.DataFrame, rallies: list[dict], players: pd.DataFrame, fps: float) -> list:
    """Every hit inside a rally with its landing bounce (if any) and the hitter's feet."""
    feet = players[players.valid] if "valid" in players else players
    feet_by_player = {p: g.set_index("frame") for p, g in feet.groupby("player")}
    rally_of = {}
    for r in rallies:
        for f in ev.frame[ev.frame.between(r["start_frame"], r["end_frame"])]:
            rally_of[int(f)] = r["id"]
    rows = list(ev.itertuples())
    shots = []
    for i, h in enumerate(rows):
        if h.kind != "hit":
            continue
        landing = None
        for e in rows[i + 1 :]:
            if e.kind == "hit" or e.frame - h.frame > SHOT_MAX_S * fps:
                break
            if rally_of.get(int(e.frame)) != rally_of.get(int(h.frame)):
                break
            # the other side of the net from the hitter (near team = y < 0)
            if e.kind == "bounce" and e.in_court in (True,) and (e.y_m > 0) == (h.team == "near"):
                landing = e
                break
        shot = {"frame": int(h.frame), "player": int(h.player), "team": h.team,
                "bounce": None, "speed_kmh": None}  # fmt: skip
        if landing is not None:
            shot["bounce"] = {"frame": int(landing.frame), "x_m": round(float(landing.x_m), 2),
                              "y_m": round(float(landing.y_m), 2),
                              "zone": zone_of(landing.y_m)}  # fmt: skip
            xy = _feet_at(feet_by_player.get(h.player), int(h.frame), FEET_MAX_DT_S * fps)
            if xy is not None:
                dist = float(np.hypot(landing.x_m - xy[0], landing.y_m - xy[1]))
                kmh = dist / ((landing.frame - h.frame) / fps) * 3.6
                if kmh <= MAX_SHOT_KMH:
                    shot["speed_kmh"] = round(kmh, 1)
        shots.append(shot)
    return shots


def _feet_at(rows: pd.DataFrame | None, frame: int, max_df: float):
    if rows is None or rows.empty:
        return None
    idx = rows.index.to_numpy()
    j = int(np.argmin(np.abs(idx - frame)))
    if abs(idx[j] - frame) > max_df:
        return None
    r = rows.iloc[j]
    return float(r.x_m), float(r.y_m)


def _placement(shots: list) -> dict:
    out = {}
    for team in ("near", "far"):
        mine = [s for s in shots if s["team"] == team]
        landed = [s["bounce"] for s in mine if s["bounce"]]
        out[team] = {
            "shots": len(mine),
            "landed": len(landed),
            "zones": {z: sum(b["zone"] == z for b in landed) for z in ZONES},
            "bounces": landed,
        }
    return out


def _speeds(shots: list) -> dict:
    v = [s["speed_kmh"] for s in shots if s["speed_kmh"] is not None]
    note = ("ground distance hitter's feet -> landing bounce / time; a lower bound, "
            "approximate (about 0.3 m near / 1 m far position error)")  # fmt: skip
    if len(v) < MIN_SAMPLES:
        return {"samples": len(v), "insufficient_data": f"fewer than {MIN_SAMPLES} measured shots",
                "note": note}  # fmt: skip
    by_player: dict[str, list] = {}
    for s in shots:
        if s["speed_kmh"] is not None:
            by_player.setdefault(str(s["player"]), []).append(s["speed_kmh"])
    return {
        "samples": len(v),
        "median_kmh": round(float(np.median(v)), 1),
        "max_kmh": round(float(np.max(v)), 1),
        "by_player_median_kmh": {
            p: round(float(np.median(x)), 1) for p, x in sorted(by_player.items())
        },
        "by_player_max_kmh": {p: round(float(max(x)), 1) for p, x in sorted(by_player.items())},
        "by_player_samples": {p: len(x) for p, x in sorted(by_player.items())},
        "note": note,
    }


def _per_rally(shots: list, rallies: list[dict]) -> list[dict]:
    """Per rally: hits, measured shots and the fastest shot estimate (null if none measured)."""
    out = []
    for r in rallies:
        mine = [s for s in shots if r["start_frame"] <= s["frame"] <= r["end_frame"]]
        v = [s["speed_kmh"] for s in mine if s["speed_kmh"] is not None]
        out.append({"id": r["id"], "hits": len(mine), "measured_shots": len(v),
                    "max_shot_kmh": max(v) if v else None})  # fmt: skip
    return out
