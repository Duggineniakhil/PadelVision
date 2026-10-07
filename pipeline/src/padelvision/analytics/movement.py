"""Stage 6 (players): distance, speed, sprints, zones, team formation.

Only measured data is used. Gaps longer than MAX_GAP_S are never filled; a player
tracked for less than MIN_TRACKED_FRACTION of the analysed time gets null metrics
with a reason instead of numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from padelvision.court.geometry import BACK_ZONE_M, NET_ZONE_M
from padelvision.players.identify import SLOT_TEAM

MAX_GAP_S = 0.5  # interpolate across shorter gaps only
MEDIAN_WINDOW_S = 0.2  # removes single-frame jumps
SMOOTH_WINDOW_S = 0.5  # removes foot-point jitter (larger on the far side)
MAX_SPEED_MPS = 8.0  # faster steps are tracking errors, not running
SPRINT_SPEED_MPS = 4.0
SPRINT_MIN_S = 0.5
MIN_TRACKED_FRACTION = 0.3


def smooth_tracks(players: pd.DataFrame, fps: float, stride: int = 1) -> pd.DataFrame:
    """Valid player rows -> smoothed positions on the analysed frame grid.

    Returns columns: frame, t, player, x, y, segment. Segments are split at long gaps.
    """
    dt = stride / fps
    out = []
    valid = players[players.valid].sort_values("frame").drop_duplicates(["player", "frame"])
    for player, p in valid.groupby("player"):
        frames = p.frame.to_numpy()
        seg_ids = np.concatenate([[0], np.cumsum(np.diff(frames) / fps > MAX_GAP_S)])
        for seg, s in p.groupby(seg_ids):
            grid = np.arange(s.frame.iloc[0], s.frame.iloc[-1] + 1, stride)
            xy = s.set_index("frame")[["x_m", "y_m"]].reindex(grid).interpolate()
            xy = _smooth(xy, dt)
            out.append(pd.DataFrame({
                "frame": grid, "t": grid / fps, "player": player,
                "x": xy.x_m.to_numpy(), "y": xy.y_m.to_numpy(), "segment": seg,
            }))  # fmt: skip
    if not out:
        return pd.DataFrame(columns=["frame", "t", "player", "x", "y", "segment"])
    return pd.concat(out, ignore_index=True)


def _smooth(xy: pd.DataFrame, dt: float) -> pd.DataFrame:
    med = max(1, round(MEDIAN_WINDOW_S / dt) | 1)
    avg = max(1, round(SMOOTH_WINDOW_S / dt))
    xy = xy.rolling(med, center=True, min_periods=1).median()
    return xy.rolling(avg, center=True, min_periods=1).mean()


def movement_stats(
    tracks: pd.DataFrame, players: pd.DataFrame, fps: float, stride: int, analysed_samples: int
) -> dict:
    dt = stride / fps
    result: dict = {"players": {}, "teams": {}}

    for player, team in SLOT_TEAM.items():
        raw = players[(players.player == player) & players.valid]
        tracked = raw.frame.nunique() / analysed_samples if analysed_samples else 0.0
        entry: dict = {"team": team, "tracked_fraction": round(tracked, 3)}
        tr = tracks[tracks.player == player]
        if tracked < MIN_TRACKED_FRACTION or tr.empty:
            entry["insufficient_data"] = (
                f"tracked {tracked:.0%} of the time (< {MIN_TRACKED_FRACTION:.0%})"
            )
        else:
            entry.update(_player_metrics(tr, dt))
        result["players"][str(player)] = entry

    for team in ("near", "far"):
        result["teams"][team] = _team_metrics(tracks, team, analysed_samples)
    return result


def _player_metrics(tr: pd.DataFrame, dt: float) -> dict:
    distance, tracked_s, sprints, outliers, peak = 0.0, 0.0, 0, 0, 0.0
    for _, s in tr.groupby("segment"):
        tracked_s += len(s) * dt
        if len(s) < 2:
            continue
        speed = np.hypot(np.diff(s.x), np.diff(s.y)) / dt
        ok = speed <= MAX_SPEED_MPS
        outliers += int((~ok).sum())
        distance += float(speed[ok].sum() * dt)
        if ok.any():
            peak = max(peak, float(speed[ok].max()))
        sprints += _count_runs(ok & (speed >= SPRINT_SPEED_MPS), round(SPRINT_MIN_S / dt))

    d = tr.y.abs()
    zones = {
        "net": float((d <= NET_ZONE_M).mean()),
        "transition": float(((d > NET_ZONE_M) & (d <= BACK_ZONE_M)).mean()),
        "back": float((d > BACK_ZONE_M).mean()),
    }
    return {
        "tracked_s": round(tracked_s, 1),
        "distance_m": round(distance, 1),
        "avg_speed_mps": round(distance / tracked_s, 2) if tracked_s else None,
        "peak_speed_mps": round(peak, 2),
        "sprints": sprints,
        "zones": {k: round(v, 3) for k, v in zones.items()},
        "outlier_steps": outliers,
    }


def _count_runs(mask: np.ndarray, min_len: int) -> int:
    runs, n = 0, 0
    for m in mask:
        n = n + 1 if m else 0
        if n == max(min_len, 1):
            runs += 1
    return runs


def _team_metrics(tracks: pd.DataFrame, team: str, analysed_samples: int) -> dict:
    a, b = [p for p, t in SLOT_TEAM.items() if t == team]
    pa = tracks[tracks.player == a].set_index("frame")[["x", "y"]]
    pb = tracks[tracks.player == b].set_index("frame")[["x", "y"]]
    both = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner")
    frac = len(both) / analysed_samples if analysed_samples else 0.0
    if frac < MIN_TRACKED_FRACTION:
        return {"insufficient_data": f"both partners tracked {frac:.0%} of the time"}

    da, db = both.y_a.abs(), both.y_b.abs()
    net_a, net_b = da <= NET_ZONE_M, db <= NET_ZONE_M
    back_a, back_b = da > BACK_ZONE_M, db > BACK_ZONE_M
    formation = {
        "both_net": float((net_a & net_b).mean()),
        "both_back": float((back_a & back_b).mean()),
        "split": float(((net_a & back_b) | (back_a & net_b)).mean()),
    }
    formation["other"] = 1.0 - sum(formation.values())
    spacing = np.hypot(both.x_a - both.x_b, both.y_a - both.y_b)
    return {
        "both_tracked_fraction": round(frac, 3),
        "formation": {k: round(v, 3) for k, v in formation.items()},
        "partner_spacing_m": round(float(spacing.mean()), 2),
    }
