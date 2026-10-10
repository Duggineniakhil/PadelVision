import numpy as np
import pandas as pd
from conftest import project

from padelvision.events import activity_segments, detect_events, detect_rallies

FPS = 30.0


def _ball(points, track=0, size=8.0):
    """points: list of (frame, u, v)."""
    return pd.DataFrame(
        [(f, f / FPS, u, v, 0.5, size, "detected", track) for f, u, v in points],
        columns=["frame", "t", "u", "v", "conf", "size", "state", "track"],
    )


def _players(rows):
    """rows: (frame, player, team, x1, y1, x2, y2)."""
    return pd.DataFrame(rows, columns=["frame", "player", "team", "x1", "y1", "x2", "y2"])


NO_PLAYERS = _players([])


def _shot(u0, v0):
    """Ball arrives at (u0, v0) at frame 6 and is sent away up the image (toward the far court)."""
    k = np.arange(31) - 6
    return [
        (i, u0 + 8 * d, v0 + 6 * d) if d < 0 else (i, u0 - 4 * d, v0 - 14 * d)
        for i, d in enumerate(k)
    ]


def test_floor_bounce_gets_court_position(cal):
    u0, v0 = project([(1.5, -4.0)])[0]
    pts = [(i, u0 + 3 * (i - 6), v0 - 5 * abs(i - 6)) for i in range(13)]
    ev = detect_events(_ball(pts), NO_PLAYERS, cal, FPS)
    assert list(ev.kind) == ["bounce"]
    b = ev.iloc[0]
    assert b.frame == 6 and b.in_court
    assert np.hypot(b.x_m - 1.5, b.y_m + 4.0) < 0.2


def test_smooth_lob_apex_is_not_an_event(cal):
    pts = [(i, 300 + 6 * i, 500 - 12 * i + 0.25 * i * i) for i in range(60)]  # parabola
    assert detect_events(_ball(pts), NO_PLAYERS, cal, FPS).empty


def test_hit_needs_ball_size_matching_player_depth(cal):
    # Ball comes in and is sent away at (500, 450), inside a near player's large box.
    pts = _shot(500, 450)
    near_player = _players([(f, 2, "near", 460, 400, 560, 700) for f in range(31)])  # 300 px tall
    at_racket = detect_events(_ball(pts, size=25.0), near_player, cal, FPS)  # expected ~26 px
    assert list(at_racket.kind) == ["hit"] and at_racket.player.iloc[0] == 2
    behind = detect_events(_ball(pts, size=10.0), near_player, cal, FPS)  # far ball behind him
    assert "hit" not in set(behind.kind)
    unknown = detect_events(_ball(pts, size=np.nan), near_player, cal, FPS)  # no size: 2-D only
    assert list(unknown.kind) == ["hit"]


def test_turn_at_tracklet_junction(cal):
    a = [(i, 300 + 10 * i, 300) for i in range(10)]  # moving right
    b = [(i, 390 - 10 * (i - 12), 300 + 3 * (i - 12)) for i in range(12, 22)]  # back left
    ball = pd.concat([_ball(a, track=1), _ball(b, track=2)], ignore_index=True)
    ev = detect_events(ball, NO_PLAYERS, cal, FPS)
    assert len(ev) == 1 and ev.source.iloc[0] == "junction"
    assert ev.frame.iloc[0] == 10


def _events(rows):
    """rows: (frame, kind, player, team, y_m); a bounce with y_m is in the court."""
    ev = pd.DataFrame(rows, columns=["frame", "kind", "player", "team", "y_m"])
    ev["in_court"] = np.where(ev.y_m.notna(), True, None)
    return ev


def test_activity_splits_at_long_pauses_and_rallies_need_an_exchange(cal):
    rows = list(range(0, 120)) + list(range(200, 290)) + list(range(400, 420))  # 4 s, 3 s, 0.7 s
    ball = _ball([(f, 100 + f % 50, 300) for f in rows])
    ev = _events([
        (30, "hit", 2, "near", np.nan), (60, "hit", 3, "far", np.nan),
        (230, "hit", 1, "near", np.nan), (250, "handling", 1, "near", np.nan),
    ])  # fmt: skip
    segments = activity_segments(ball, ev, FPS)
    assert [s["start_frame"] for s in segments] == [0, 200]  # the 0.7 s burst is too short
    assert segments[0]["hits"] == 2 and segments[0]["hits_by_player"] == {"2": 1, "3": 1}
    assert segments[1]["handling"] == 1
    rallies = detect_rallies(ball, ev, FPS)
    assert [r["id"] for r in rallies] == [1]  # one near-side hit alone is not a rally


def test_hit_then_bounce_on_the_other_side_is_an_exchange(cal):
    ball = _ball([(f, 100 + f % 50, 300) for f in range(120)])
    over = _events([(30, "hit", 1, "near", np.nan), (50, "bounce", None, None, 4.0)])
    same_side = _events([(30, "hit", 1, "near", np.nan), (50, "bounce", None, None, -4.0)])
    next_court = over.assign(in_court=[None, False])  # a ball bouncing on the next court
    assert len(detect_rallies(ball, over, FPS)) == 1
    assert detect_rallies(ball, same_side, FPS) == []
    assert detect_rallies(ball, next_court, FPS) == []


def test_activity_splits_where_no_hit_is_seen_and_trims_to_the_hits(cal):
    # 20 s of ball in view: a player bounces the ball, then a real exchange from 15 s to 17 s.
    ball = _ball([(f, 100 + f % 50, 300) for f in range(600)])
    ev = _events([
        (60, "hit", 2, "near", np.nan), (120, "hit", 2, "near", np.nan),  # alone, 2 s apart
        (450, "hit", 2, "near", np.nan), (490, "hit", 4, "far", np.nan),
        (510, "hit", 1, "near", np.nan),
    ])  # fmt: skip
    segments = activity_segments(ball, ev, FPS)
    spans = [(s["start_frame"], s["end_frame"], s["exchange"]) for s in segments]
    assert spans == [(30, 180, False), (420, 570, True)]  # 1 s lead, 2 s tail


def test_ball_handling_is_not_a_hit(cal):
    # The player bounces the ball on the floor and racket beside him: it never goes far.
    pts = [(i, 520 + 0.5 * i, 600 - 150 * abs(((i / 16) % 1) * 2 - 1)) for i in range(64)]
    player = _players([(f, 2, "near", 460, 400, 560, 700) for f in range(64)])  # 300 px tall
    ev = detect_events(_ball(pts, size=25.0), player, cal, FPS)
    assert "handling" in set(ev.kind) and "hit" not in set(ev.kind)


def test_best_depth_match_gets_the_hit(cal):
    # The ball turns where a near player's big box and a far player's small box overlap in 2-D.
    pts = _shot(500, 420)
    boxes = [(f, 2, "near", 440, 380, 560, 700) for f in range(31)]  # 320 px tall
    boxes += [(f, 4, "far", 480, 390, 520, 450) for f in range(31)]  # 60 px tall
    ev = detect_events(_ball(pts, size=10.5), _players(boxes), cal, FPS)  # far-ball size
    assert list(ev.kind) == ["hit"] and ev.player.iloc[0] == 4
