"""Rally statistics from stage 5 (events.parquet + rallies) and the players' feet (stage 3).

Only what the events support, and only inside rallies (ball handling and warm-up excluded):
- rally count and lengths, hits per rally / player / team
- placement: floor bounces inside the court, each credited to the team whose hit came
  before it, by depth zone on the receiving side (net / transition / back)
- shot speed ESTIMATE: ground distance from the hitter's feet to the next in-court bounce on
  the other side of the net, divided by the time between them. It is the AVERAGE ground speed
  of the flight: the ball is faster off the racket (it slows in the air) and flies further than
  the ground line, so this is a lower bound for the speed off the racket. The hit point is in
  the air, so the feet stand in for it (the homography is only valid on the ground).
  Only trustworthy shots get a number: the landing must be a bounce seen with the ball tracked
  through it (not a join between two track pieces), not at the image edge, the flight must
  last >= MIN_FLIGHT_S, and the error range must be within +-MAX_REL_ERR. Every speed comes
  with that range (position error from the calibration per zone plus the feet-to-contact
  offset, timing error per event; bounce times refined to a fraction of a frame from the
  ball's fall and rise). Other shots get speed_note instead. Summaries need MIN_SAMPLES.
Anything without enough data is null with a reason instead of a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import zone_of

MIN_SAMPLES = 3
SHOT_MAX_S = 2.5  # a bounce later than this after the hit is not that shot's landing
FEET_MAX_DT_S = 0.2  # the hitter's feet must be seen this close to the hit
MAX_SHOT_KMH = 250.0  # faster = a pairing or position error, not a shot
MIN_FLIGHT_S = 0.5  # shorter flights: the timing error is too large a share
EDGE_PX = 20  # a landing this close to the image border may be the ball leaving the view
MAX_REL_ERR = 0.25  # keep a speed only if its range is within +-25%
FEET_PX_ERR = 4.0  # foot point (box bottom) jitter
BOUNCE_PX_ERR = 2.0  # ball centre at the bounce
CONTACT_OFFSET_M = 0.7  # the racket meets the ball up to ~0.7 m from the feet (ground projection)
HIT_T_ERR = {"within": 1.0, "junction": 3.0}  # frames; a junction turn is in a tracking gap
POS_ERR_NO_CAL_M = 1.0  # position error when no calibration is given (tests, old runs)
ZONES = ("net", "transition", "back")


def rally_stats(
    events: pd.DataFrame,
    rallies: list[dict],
    players: pd.DataFrame,
    fps: float,
    cal: CourtCalibration | None = None,
    ball: pd.DataFrame | None = None,
) -> dict:
    """`cal` gives per-zone position errors and the image size (edge check); `ball`
    (ball.parquet) refines bounce times. Without them, wider fixed errors are used."""
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

    shots = _shots(ev, rallies, players, fps, cal, ball)
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
        # every rally hit, for overlays: frame, player, team, hitter's feet (ground, m),
        # landing bounce, speed estimate
        "shots": shots,
    }


def _shots(ev, rallies, players, fps, cal=None, ball=None) -> list:
    """Every hit inside a rally with its landing bounce (if any), the hitter's feet and, when
    trustworthy, a speed estimate with its range."""
    ball_det = None
    if ball is not None and len(ball):
        det = ball[ball.state == "detected"] if "state" in ball else ball
        ball_det = det.set_index("frame")[["u", "v"]].sort_index()
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
        xy = _feet_at(feet_by_player.get(h.player), int(h.frame), FEET_MAX_DT_S * fps)
        shot = {"frame": int(h.frame), "player": int(h.player), "team": h.team,
                "hitter_xy": None if xy is None else [round(xy[0], 2), round(xy[1], 2)],
                "bounce": None, "speed_kmh": None, "speed_range_kmh": None,
                "speed_note": "landing not seen"}  # fmt: skip
        if landing is not None:
            shot["bounce"] = {"frame": int(landing.frame), "x_m": round(float(landing.x_m), 2),
                              "y_m": round(float(landing.y_m), 2),
                              "zone": zone_of(landing.y_m)}  # fmt: skip
            if xy is None:
                shot["speed_note"] = "hitter's feet not seen"
            else:
                kmh, rng, note = _shot_speed(h, landing, xy, fps, cal, ball_det)
                shot.update(speed_kmh=kmh, speed_range_kmh=rng, speed_note=note)
        shots.append(shot)
    return shots


def _shot_speed(h, landing, xy, fps, cal, ball_det):
    """(speed km/h, [low, high], None) for a trustworthy shot, else (None, None, reason)."""
    if getattr(landing, "source", "within") != "within":
        return None, None, "bounce at a join between two ball tracks"
    if cal is not None:
        w, hgt = cal.image_size
        u, v = float(landing.u), float(landing.v)
        if min(u, v, w - 1 - u, hgt - 1 - v) < EDGE_PX:
            return None, None, "landing at the image edge"
    t_bounce, bounce_err = _refined_bounce_frame(ball_det, int(landing.frame))
    dt = (t_bounce - h.frame) / fps
    if dt < MIN_FLIGHT_S:
        return None, None, f"flight under {MIN_FLIGHT_S} s"
    dist = float(np.hypot(landing.x_m - xy[0], landing.y_m - xy[1]))
    if cal is not None:
        rms = cal.accuracy()["rms_px"] or 0.0
        feet_m = (cal.metres_per_px(xy) or 0.0) * np.hypot(FEET_PX_ERR, rms)
        bounce_xy = (float(landing.x_m), float(landing.y_m))
        bounce_m = (cal.metres_per_px(bounce_xy) or 0.0) * np.hypot(BOUNCE_PX_ERR, rms)
    else:
        feet_m = bounce_m = POS_ERR_NO_CAL_M
    d_err = float(np.hypot(np.hypot(feet_m, CONTACT_OFFSET_M), bounce_m))
    t_err = (HIT_T_ERR.get(getattr(h, "source", "within"), 3.0) + bounce_err) / fps
    kmh = dist / dt * 3.6
    lo = max(dist - d_err, 0.0) / (dt + t_err) * 3.6
    hi = (dist + d_err) / max(dt - t_err, 1e-6) * 3.6
    if kmh > MAX_SHOT_KMH:
        return None, None, "implausibly fast: a pairing or position error"
    if (hi - lo) / 2 > MAX_REL_ERR * kmh:
        return None, None, f"too uncertain ({lo:.0f}-{hi:.0f} km/h)"
    return round(kmh, 1), [round(lo), round(hi)], None


def _refined_bounce_frame(ball_det, frame: int) -> tuple[float, float]:
    """Bounce time to a fraction of a frame: where straight-line fits of the ball's image
    height before (falling) and after (rising) meet. Returns (frame, error in frames)."""
    if ball_det is None:
        return float(frame), 1.0
    before = ball_det.loc[frame - 5 : frame - 1]
    after = ball_det.loc[frame + 1 : frame + 5]
    if len(before) < 3 or len(after) < 3:
        return float(frame), 1.0
    fall = np.polyfit(before.index.to_numpy(float), before.v.to_numpy(float), 1)
    rise = np.polyfit(after.index.to_numpy(float), after.v.to_numpy(float), 1)
    if fall[0] <= 0 or rise[0] >= 0:  # not falling then rising (image v grows downwards)
        return float(frame), 1.0
    t = (rise[1] - fall[1]) / (fall[0] - rise[0])
    return (float(t), 0.5) if abs(t - frame) <= 2 else (float(frame), 1.0)


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
    note = ("average ground speed, hitter's feet -> landing bounce / time: a lower bound for "
            "the speed off the racket; only shots with a trustworthy landing and an error range "
            f"within +-{MAX_REL_ERR:.0%} (see each shot's speed_range_kmh)")  # fmt: skip
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
