import numpy as np
import pandas as pd
from conftest import project

from padelvision.events import detect_events, detect_rallies

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
    # Ball comes in and is sent back at (500, 450), inside a near player's large box.
    pts = [(i, 500 + 8 * (i - 6) * (1 if i < 6 else -1), 450 + 6 * (i - 6)) for i in range(13)]
    near_player = _players([(f, 2, "near", 460, 400, 560, 700) for f in range(13)])  # 300 px tall
    at_racket = detect_events(_ball(pts, size=11.0), near_player, cal, FPS)  # 11/300 ~ 1/27
    assert list(at_racket.kind) == ["hit"] and at_racket.player.iloc[0] == 2
    behind = detect_events(_ball(pts, size=4.0), near_player, cal, FPS)  # far ball behind him
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


def test_rallies_split_at_long_pauses(cal):
    rows = list(range(0, 120)) + list(range(200, 290)) + list(range(400, 420))  # 4 s, 3 s, 0.7 s
    ball = _ball([(f, 100 + f % 50, 300) for f in rows])
    ev = pd.DataFrame(
        [(30, "hit", 2), (60, "hit", 3), (230, "hit", 1)], columns=["frame", "kind", "player"]
    )
    rallies = detect_rallies(ball, ev, FPS)
    assert [r["start_frame"] for r in rallies] == [0, 200]  # the 0.7 s burst is too short
    assert rallies[0]["hits"] == 2 and rallies[0]["hits_by_player"] == {"2": 1, "3": 1}
