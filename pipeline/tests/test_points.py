import pandas as pd

from padelvision.analytics.points import build_points, estimate_games, find_serves, points_summary

FPS = 10.0


def _players(rows):
    """rows: (frame, player, x_m, y_m) -> players table (persons, valid)."""
    out = []
    for f, pl, x, y in rows:
        out.append((f, f / FPS, pl, "near" if y < 0 else "far", x, y, True, False))
    return pd.DataFrame(
        out, columns=["frame", "t", "player", "team", "x_m", "y_m", "valid", "changeover"]
    )


def _events(rows):
    """rows: (frame, kind, player, team, x_m, y_m)."""
    return pd.DataFrame(rows, columns=["frame", "kind", "player", "team", "x_m", "y_m"])


def _standing(seconds=120):
    """Player 1 behind the near service line on the right, 3 at the far net, 2 and 4 around."""
    rows = []
    for f in range(round(seconds * FPS)):
        rows += [(f, 1, 2.0, -8.0), (f, 2, -2.0, -3.0), (f, 3, -2.0, 7.5), (f, 4, 2.0, 3.0)]
    return _players(rows)


def test_serve_from_behind_the_line_after_a_pause_landing_in_the_diagonal_box():
    players = _standing()
    ev = _events([
        (100, "hit", 1, "near", None, None),  # serve: right side, after a pause
        (112, "bounce", None, None, -2.0, 4.0),  # diagonal box across the net
        (125, "hit", 3, "far", None, None),  # return
        (140, "hit", 2, "near", None, None),  # rally shot from the net: not a serve
        (400, "hit", 1, "near", None, None),  # next point, landing unseen
    ])  # fmt: skip
    serves = find_serves(ev, players, FPS)
    assert [s["frame"] for s in serves] == [100, 400]
    first = serves[0]
    assert first["server"] == 1 and first["team"] == "near" and first["side"] == "right"
    assert first["in_box"] and first["confidence"] == "confirmed" and first["returned"]
    assert serves[1]["confidence"] == "likely" and serves[1]["in_box"] is None


def test_landing_outside_the_box_is_a_suspected_fault_and_the_second_serve_same_point():
    players = _standing()
    ev = _events([
        (100, "hit", 1, "near", None, None),
        (110, "bounce", None, None, 2.0, 4.0),  # same side of the centre line: not diagonal
        (250, "hit", 1, "near", None, None),  # second serve 15 s later, nothing played between
        (262, "bounce", None, None, -2.0, 5.0),
        (275, "hit", 3, "far", None, None),
        (700, "hit", 1, "near", None, None),  # a new point
    ])  # fmt: skip
    serves = find_serves(ev, players, FPS)
    assert serves[0]["fault_suspected"] and not serves[1]["fault_suspected"]
    points = build_points(serves, ev, FPS, end_frame=1199)
    assert len(points) == 2
    assert points[0]["second_serve"] and points[0]["serves"] == [100, 250]
    assert points[0]["end_frame"] == 699 and points[1]["end_frame"] == 1199


def test_games_by_serving_pair_checked_against_changeovers():
    def pt(i, server, t):
        return {"id": i, "server": server, "start_s": t}

    # Game 1 (A), changeover, games 2 (B) and 3 (A), changeover, game 4 (B): consistent.
    points = [pt(1, 1, 10), pt(2, 2, 30), pt(3, 3, 70), pt(4, 4, 90), pt(5, 1, 130), pt(6, 3, 200)]
    est = estimate_games(points, [{"t": 50.0}, {"t": 170.0}], FPS)
    assert [g["pair"] for g in est["games"]] == ["A", "B", "A", "B"]
    assert est["games"][0]["points"] == [1, 2] and est["consistent"]

    # A game hidden by missed serves breaks the checks, and the result says so.
    est = estimate_games(points[:3] + points[5:], [{"t": 50.0}, {"t": 170.0}], FPS)
    assert not est["consistent"] and "unreliable" in est["note"]


def test_summary_counts():
    players = _standing()
    ev = _events([(100, "hit", 1, "near", None, None), (112, "bounce", None, None, -2.0, 4.0)])
    s = points_summary(ev, players, FPS, end_frame=1199)
    assert s["counts"] == {"serves": 1, "confirmed": 1, "faults_suspected": 0, "points": 1,
                           "second_serves": 0}  # fmt: skip
