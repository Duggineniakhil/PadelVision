"""Stage 5c: coarse shot types for every rally shot (no model, from data we already have).

Types, checked in this order (the first that applies wins):
- serve:       the hit is a serve (stage 5b, `points.py`).
- overhead:    the ball is above the hitter's head at contact: its image position is above
               the top of the player's box (smash, bandeja and vibora together; telling those
               apart needs the swing).
- lob:         a long, high flight to the back of the other side: landing seen behind the
               service line (|y| >= SERVICE_LINE) after >= LOB_MIN_FLIGHT_S.
- volley:      taken before it bounced: the opponents' previous shot has no landing and the
               ball was tracked almost the whole way (rallies.py "probably_volley").
- groundstroke: after a bounce on the hitter's side (the opponents' previous shot landed).
- unknown:     none of the above can be told (ball lost before the hit, first shot of a rally
               without a serve, hitter's box not seen).

Side (forehand / backhand) for every shot but serves and overheads, NEAR players only: they
face away from the camera, so image right of the body centre is their right. Players are assumed
right-handed (the pipeline can't know); a left-hander's sides come out swapped. Far players
are too small to tell reliably, so their side is null.
Every type carries `basis`: the measurement it came from, so it can be checked.
"""

from __future__ import annotations

import pandas as pd

from padelvision.court.geometry import SERVICE_LINE_FROM_NET_M as SERVICE

OVERHEAD_TOP = 0.0  # ball above box top + this many box heights (0 = above the head)
LOB_MIN_FLIGHT_S = 1.6
SIDE_MIN = 0.15  # |ball - box centre| >= this many box widths to call a side
BOX_MAX_DT = 2  # frames: the hitter's box must be seen this close to the hit
TYPES = ("serve", "overhead", "lob", "volley", "groundstroke", "unknown")


def classify_shots(shots: list[dict], events: pd.DataFrame, players: pd.DataFrame, fps: float,
                   serve_frames: set[int] | None = None) -> list[dict]:  # fmt: skip
    """rally_stats shots -> the same shots with `type`, `side` and `basis` added (in place)."""
    serve_frames = serve_frames or set()
    hits = events[events.kind == "hit"].set_index("frame")
    boxes = {p: g.set_index("frame") for p, g in players.groupby("player")}
    by_frame = {s["frame"]: s for s in shots}
    order = sorted(by_frame)
    for i, f in enumerate(order):
        s = by_frame[f]
        prev = by_frame.get(order[i - 1]) if i > 0 else None
        if prev is not None and (prev["team"] == s["team"] or f - prev["frame"] > 4 * fps):
            prev = None  # not the opponents' shot just before this one
        box = _box(boxes.get(s["player"]), f)
        ball_uv = (float(hits.u[f]), float(hits.v[f])) if f in hits.index else None
        s["type"], s["basis"] = _type(s, prev, box, ball_uv, fps, f in serve_frames)
        s["side"] = _side(s, box, ball_uv) if s["type"] not in ("serve", "overhead") else None
    return shots


def _box(rows: pd.DataFrame | None, frame: int):
    if rows is None or rows.empty:
        return None
    i = rows.index.get_indexer([frame], method="nearest")[0]
    if i < 0 or abs(rows.index[i] - frame) > BOX_MAX_DT:
        return None
    r = rows.iloc[i]
    return float(r.x1), float(r.y1), float(r.x2), float(r.y2)


def _type(s, prev, box, ball_uv, fps, is_serve) -> tuple[str, str]:
    if is_serve:
        return "serve", "serve (stage 5b)"
    if box is not None and ball_uv is not None:
        x1, y1, x2, y2 = box
        if ball_uv[1] < y1 + OVERHEAD_TOP * (y2 - y1):
            return "overhead", "ball above the hitter's head at contact"
    b = s.get("bounce")
    if (
        b is not None
        and abs(b["y_m"]) >= SERVICE
        and (b["frame"] - s["frame"]) / fps >= LOB_MIN_FLIGHT_S
    ):
        return (
            "lob",
            f"{(b['frame'] - s['frame']) / fps:.1f} s flight, landed {abs(b['y_m']):.1f} m deep",
        )
    if prev is not None:
        if prev.get("no_landing") == "probably_volley":
            return "volley", "no bounce since the opponents' shot (ball tracked)"
        if prev.get("bounce") is not None:
            return "groundstroke", "after a bounce on the hitter's side"
    return "unknown", "ball not followed since the opponents' shot"


def _side(s, box, ball_uv) -> str | None:
    if s["team"] != "near" or box is None or ball_uv is None:
        return None
    x1, _, x2, _ = box
    off = (ball_uv[0] - (x1 + x2) / 2) / max(x2 - x1, 1.0)
    if abs(off) < SIDE_MIN:
        return None
    return "forehand" if off > 0 else "backhand"  # right-handed, seen from behind


def summary(shots: list[dict]) -> dict:
    """Counts per type, per player (and forehand/backhand for near players)."""
    out: dict = {"counts": {t: 0 for t in TYPES}, "by_player": {}}
    for s in shots:
        t = s.get("type", "unknown")
        out["counts"][t] += 1
        p = out["by_player"].setdefault(
            str(s["player"]), {t: 0 for t in TYPES} | {"forehand": 0, "backhand": 0}
        )
        p[t] += 1
        if s.get("side"):
            p[s["side"]] += 1
    known = len(shots) - out["counts"]["unknown"]
    out["known_share"] = round(known / len(shots), 2) if shots else None
    out["note"] = ("coarse types from the ball and the hitter's box; forehand/backhand for near "
                   "players only, assuming right-handed players")  # fmt: skip
    return out
