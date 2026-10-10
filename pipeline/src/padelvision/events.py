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
            fits that player's depth. With a low camera, a far-court ball often appears inside a
            near player's (large) box; it is about half the size a ball at their racket would be.
            Expected size = a + b * player box height (the detector's boxes have a floor of ~7 px
            from blur/padding, so a pure ratio doesn't work). When several players are in reach,
            the best depth match gets the hit.
            A turn at a player within MAX_HIT_GAP_S of the other team's last hit is a return,
            so always a hit (far shots barely move in the image: depth is compressed).
            Turns by the same player within DUPLICATE_S are one contact (the ball's turn as
            it arrives and as it leaves): the one with the fastest outgoing ball is kept.
- "handling": any other "hit" after which the ball never leaves the player's surroundings
            (bouncing the ball between points, before a serve, catching it). A real shot sends
            the ball more than LEAVE_REACH box heights away within HANDLING_WINDOW_S. If the
            ball is mostly unseen in that window, it stays a hit.
- "bounce": the ball was moving down the image and then up (floor bounce)
- "wall":   any other sharp turn (glass rebounds, net cord, a hit we can't attribute)

Only bounces get a court position: the ball is on the ground there, so the homography is
valid. Ball activity is split into segments at pauses in the ball track > RALLY_GAP_S, and
again where no hit is seen for > MAX_HIT_GAP_S (players bouncing the ball between points
keep it in view). Segments with hits are trimmed to RALLY_LEAD_S before the first hit and
RALLY_TAIL_S after the last. A segment is a rally only if it shows an exchange: hits by both
teams, or a hit followed by a floor bounce on the other side of the net. Other segments
(ball handling, warm-up feeding, no hit seen) are reported separately, never as rallies.
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
# Expected ball box size (px) at a player of box height h: a + b * h. Fitted on 24 visually
# confirmed hits in Test_video with ball-detector v1 (ratio actual/expected 0.78-1.24); far balls
# seen inside a near player's box come out at ~0.53. Re-fit for another camera or detector.
BALL_SIZE_AT_PLAYER = (7.28, 0.0625)
SIZE_MATCH_TOL = 1.43  # accept actual/expected within [1/1.43, 1.43] = [0.70, 1.43]
# Ball handling vs a shot, tuned on Test_video: after the dribbles in rallies 4/7 the ball stays
# within ~0.35 box heights of the player; shots carry it 0.5-10 box heights away.
HANDLING_WINDOW_S = 0.6
LEAVE_REACH = 0.5  # box heights beyond the player's box (sides, top, bottom)
MIN_SEEN = 1 / 3  # fraction of the window with ball + player box needed to call it handling
# Visual review of Test_video events: duplicate turns of one contact were 0.2-0.3 s apart.
DUPLICATE_S = 0.35
RALLY_GAP_S = 2.0
MIN_RALLY_S = 1.5
# Hits in Test_video exchanges are 0.2-2.6 s apart; 4 s still allows one missed hit.
MAX_HIT_GAP_S = 4.0
RALLY_LEAD_S = 1.0  # kept before the first hit (serve bounce)
RALLY_TAIL_S = 2.0  # kept after the last hit (bounces, glass, the ball going dead)

EVENT_COLUMNS = ["frame", "t", "kind", "u", "v", "size", "x_m", "y_m", "in_court", "player",
                 "team", "track", "turn_deg", "speed_in", "speed_out", "source"]  # fmt: skip


def detect_events(
    ball: pd.DataFrame, players: pd.DataFrame, cal: CourtCalibration, fps: float
) -> pd.DataFrame:
    turns = _turns_within(ball) + _turns_at_junctions(ball)
    turns = _suppress(turns)
    boxes = {f: g for f, g in players.groupby("frame")} if len(players) else {}
    ball_at = dict(zip(ball.frame, zip(ball.u, ball.v, strict=True), strict=True))
    window = max(1, round(HANDLING_WINDOW_S * fps))
    rows = []
    last: tuple[int, str] | None = None  # (frame, team) of the last hit/handling
    for tr in turns:
        player, team = _reaching_player(boxes.get(tr["frame"]), tr["u"], tr["v"], tr["size"])
        if player is not None:
            returned = (
                last is not None
                and last[1] != team
                and tr["frame"] - last[0] <= MAX_HIT_GAP_S * fps
            )
            away = returned or _sends_ball_away(ball_at, boxes, player, tr["frame"], window)
            kind = "hit" if away else "handling"
            last = (tr["frame"], team)
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
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("frame")
    return _merge_duplicate_contacts(events.reset_index(drop=True), fps)


def _merge_duplicate_contacts(events: pd.DataFrame, fps: float) -> pd.DataFrame:
    """One row per contact: same-player hit/handling turns within DUPLICATE_S are merged into
    the one with the fastest outgoing ball (a hit if any of them was)."""
    contact = events.kind.isin(["hit", "handling"])
    drop = []
    group: list[int] = []
    for i in [*events.index[contact], None]:
        if group and (
            i is None
            or events.player[i] != events.player[group[-1]]
            or events.frame[i] - events.frame[group[-1]] > DUPLICATE_S * fps
        ):
            keep = max(group, key=lambda j: events.speed_out[j])
            if (events.kind[group] == "hit").any():
                events.loc[keep, "kind"] = "hit"
            drop += [j for j in group if j != keep]
            group = []
        if i is not None:
            group.append(i)
    return events.drop(index=drop).reset_index(drop=True)


def activity_segments(ball: pd.DataFrame, events: pd.DataFrame, fps: float) -> list[dict]:
    """Runs of ball activity separated by pauses longer than RALLY_GAP_S; `exchange` marks
    the ones that are rallies (see the module docstring)."""
    if ball.empty:
        return []
    frames = ball.frame.to_numpy()
    breaks = np.flatnonzero(np.diff(frames) > RALLY_GAP_S * fps) + 1
    spans = []
    for seg in np.split(frames, breaks):
        start, end = int(seg[0]), int(seg[-1])
        if (end - start) / fps >= MIN_RALLY_S:
            spans += _split_at_hit_gaps(events, start, end, fps)
    segments = []
    for start, end in spans:
        seg = frames[(frames >= start) & (frames <= end)]
        ev = events[(events.frame >= start) & (events.frame <= end)]
        hits = ev[ev.kind == "hit"]
        segments.append({
            "id": len(segments) + 1,
            "start_frame": start, "end_frame": end,
            "start_s": round(start / fps, 2), "end_s": round(end / fps, 2),
            "duration_s": round((end - start) / fps, 2),
            "ball_coverage": round(len(seg) / (end - start + 1), 2),
            "exchange": _has_exchange(ev),
            "hits": int(len(hits)),
            "handling": int((ev.kind == "handling").sum()),
            "bounces": int((ev.kind == "bounce").sum()),
            "walls": int((ev.kind == "wall").sum()),
            "hits_by_player": {str(int(k)): int(v) for k, v in hits.player.value_counts().items()},
        })  # fmt: skip
    return segments


def detect_rallies(ball: pd.DataFrame, events: pd.DataFrame, fps: float) -> list[dict]:
    """Activity segments that show an exchange (ids are the segment ids)."""
    return [s for s in activity_segments(ball, events, fps) if s["exchange"]]


def _split_at_hit_gaps(events: pd.DataFrame, start: int, end: int, fps: float) -> list[tuple]:
    """(start, end) frame spans: one per group of hits <= MAX_HIT_GAP_S apart, trimmed to the
    lead/tail around its hits. Without hits, the whole span is kept (as non-rally activity)."""
    hit_frames = np.sort(
        events.frame[(events.kind == "hit") & events.frame.between(start, end)].to_numpy()
    )
    if not len(hit_frames):
        return [(start, end)]
    cuts = np.flatnonzero(np.diff(hit_frames) > MAX_HIT_GAP_S * fps) + 1
    groups = np.split(hit_frames, cuts)
    spans = []
    for i, g in enumerate(groups):
        s = max(start, int(g[0]) - round(RALLY_LEAD_S * fps))
        e = min(end, int(g[-1]) + round(RALLY_TAIL_S * fps))
        if spans:
            s = max(s, spans[-1][1] + 1)
        if i + 1 < len(groups):
            e = min(e, int(groups[i + 1][0]) - 1)
        spans.append((s, e))
    return spans


def _has_exchange(ev: pd.DataFrame) -> bool:
    """Hits by both teams, or a hit followed by an in-court floor bounce on the other side of
    the net (near team = y < 0)."""
    hits = ev[ev.kind == "hit"]
    if hits.team.nunique() >= 2:
        return True
    if "in_court" not in ev:
        return False
    bounces = ev[(ev.kind == "bounce") & ev.in_court.eq(True)]  # not balls on the next court
    for h in hits.itertuples():
        later = bounces[bounces.frame > h.frame]
        if len(later) and ((later.y_m.iloc[0] > 0) == (h.team == "near")):
            return True
    return False


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


def _sends_ball_away(ball_at: dict, boxes: dict, player: int, frame: int, window: int) -> bool:
    """Does the ball get more than LEAVE_REACH box heights away from `player` within `window`
    frames after `frame`? True when there's too little data to tell (keep it a hit)."""
    out = []
    for f in range(frame + 1, frame + window + 1):
        if f not in ball_at or f not in boxes:
            continue
        b = boxes[f][boxes[f].player == player]
        if b.empty:
            continue
        x1, y1, x2, y2 = b[["x1", "y1", "x2", "y2"]].iloc[0]
        h = max(y2 - y1, 1.0)
        u, v = ball_at[f]
        dx = max(x1 - u, u - x2, 0.0)
        dy = max(y1 - v, v - y2, 0.0)
        out.append(max(dx, dy) / h)
    if len(out) < MIN_SEEN * window:
        return True
    return max(out) >= LEAVE_REACH


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
        a, b = BALL_SIZE_AT_PLAYER
        mismatch = np.abs(np.log(size / (a + b * h)))
        inside &= mismatch <= np.log(SIZE_MATCH_TOL)
        if not inside.any():
            return None, None
        best = boxes.loc[mismatch[inside].idxmin()]  # best depth match
    else:
        if not inside.any():
            return None, None
        cand = boxes[inside]
        d = np.hypot((cand.x1 + cand.x2) / 2 - u, (cand.y1 + cand.y2) / 2 - v)
        best = cand.loc[d.idxmin()]  # no size: nearest box
    return int(best.player), str(best.team)
