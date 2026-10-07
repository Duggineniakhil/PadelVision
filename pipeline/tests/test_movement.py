import numpy as np
import pandas as pd
import pytest

from padelvision.analytics import movement_stats, smooth_tracks

FPS = 30.0


def _players(paths: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """paths: player -> (frames, x, y)."""
    rows = []
    for p, (frames, xs, ys) in paths.items():
        team = "near" if p <= 2 else "far"
        for f, x, y in zip(frames, xs, ys, strict=True):
            rows.append((f, f / FPS, p, team, x, y, True))
    return pd.DataFrame(rows, columns=["frame", "t", "player", "team", "x_m", "y_m", "valid"])


def _stats(players, n_frames, stride=1):
    tracks = smooth_tracks(players, FPS, stride)
    return movement_stats(tracks, players, FPS, stride, n_frames // stride)


def test_constant_speed_distance_and_speed():
    f = np.arange(300)  # 10 s
    x = -4 + 0.8 * f / FPS  # 0.8 m/s
    players = _players({1: (f, x, np.full(300, -8.0))})
    p1 = _stats(players, 300)["players"]["1"]
    assert p1["distance_m"] == pytest.approx(8.0, rel=0.03)
    assert p1["avg_speed_mps"] == pytest.approx(0.8, rel=0.03)
    assert p1["peak_speed_mps"] == pytest.approx(0.8, rel=0.05)
    assert p1["sprints"] == 0
    assert p1["zones"]["back"] == 1.0


def test_sprint_detected():
    f = np.arange(300)
    t = f / FPS
    v = np.where((t > 4) & (t < 5.5), 5.0, 0.5)  # 1.5 s burst at 5 m/s
    x = np.cumsum(v) / FPS - 4
    players = _players({1: (f, x, np.full(300, -2.0))})
    p1 = _stats(players, 300)["players"]["1"]
    assert p1["sprints"] == 1
    assert p1["peak_speed_mps"] > 4.0
    assert p1["zones"]["net"] == 1.0


def test_long_gaps_are_not_filled():
    f = np.r_[np.arange(0, 90), np.arange(180, 300)]  # 3 s missing
    y = np.where(f < 150, -8.0, -2.0)  # player "teleports" 6 m across the gap
    players = _players({1: (f, np.zeros(len(f)), y)})
    p1 = _stats(players, 300)["players"]["1"]
    assert p1["distance_m"] < 0.5
    assert p1["tracked_s"] == pytest.approx(7.0, abs=0.1)


def test_jitter_does_not_inflate_distance():
    rng = np.random.default_rng(0)
    f = np.arange(300)
    players = _players({4: (f, rng.normal(0, 0.15, 300), 8 + rng.normal(0, 0.3, 300))})
    p4 = _stats(players, 300)["players"]["4"]
    assert p4["distance_m"] < 15  # raw jitter path would be ~100 m


def test_insufficient_data_gives_no_numbers():
    f = np.arange(20)  # 0.67 s out of 10 s
    players = _players({2: (f, np.zeros(20), np.full(20, -5.0))})
    stats = _stats(players, 300)
    assert "insufficient_data" in stats["players"]["2"]
    assert "distance_m" not in stats["players"]["2"]
    assert "insufficient_data" in stats["players"]["3"]
    assert "insufficient_data" in stats["teams"]["near"]


def test_team_formation():
    f = np.arange(300)
    players = _players({
        1: (f, np.full(300, -2.5), np.full(300, -2.0)),
        2: (f, np.full(300, 2.5), np.where(f < 150, -2.0, -9.0)),
    })  # fmt: skip
    near = _stats(players, 300)["teams"]["near"]
    assert near["formation"]["both_net"] == pytest.approx(0.5, abs=0.03)
    assert near["formation"]["split"] == pytest.approx(0.5, abs=0.03)
    assert near["partner_spacing_m"] > 5.0
