import numpy as np
import pandas as pd

from padelvision.ball.track import MAX_INTERP_GAP, link, track_ball

FPS = 30.0


def _flight(f0, n, u0, v0, du, dv, g=0.4, conf=0.4, drop=()):
    return [(f0 + i, u0 + du * i, v0 + dv * i + 0.5 * g * i * i, conf)
            for i in range(n) if i not in drop]  # fmt: skip


def _dets(rows):
    return pd.DataFrame(rows, columns=["frame", "u", "v", "conf"])


def test_ball_in_play_ignores_spare_balls_and_noise():
    rng = np.random.default_rng(0)
    rows = _flight(0, 60, 100, 400, 9, -8, drop={20, 21, 22})  # 3-frame dropout
    rows += [
        (f, 640.0 + rng.normal(0, 0.3), 600.0 + rng.normal(0, 0.3), 0.6) for f in range(60)
    ]  # spare ball
    rows += [(f, rng.uniform(0, 1280), rng.uniform(0, 720), 0.15) for f in range(0, 60, 3)]  # noise
    rows += [(f, 50.0, 50.0, 0.05) for f in range(60)]  # below DET_CONF: ignored
    ball, stats = track_ball(_dets(rows), FPS)

    assert stats["tracklets"]["static"] >= 1
    assert set(ball.state) == {"detected", "interpolated"}
    assert (ball.state == "interpolated").sum() == 3
    truth = {f: (100 + 9 * (f - 0), 400 - 8 * f + 0.2 * f * f) for f in range(60)}
    on_flight = [
        np.hypot(r.u - truth[r.frame][0], r.v - truth[r.frame][1]) < 3 for r in ball.itertuples()
    ]
    assert sum(on_flight) >= 58 and len(ball) <= 61
    assert not ((ball.u - 640).abs() < 3).all()


def test_long_gaps_are_not_filled_and_tracklets_break():
    rows = _flight(0, 20, 100, 300, 10, 0, g=0) + _flight(40, 20, 300, 300, 10, 0, g=0)
    ball, _ = track_ball(_dets(rows), FPS)
    assert not ball.frame.between(20, 39).any()
    assert ball.track.nunique() == 2


def test_short_gap_limit():
    gap = MAX_INTERP_GAP + 2  # longer than MAX_MISSES too: two tracklets, nothing filled between
    rows = _flight(0, 10, 100, 300, 6, 0, g=0) + _flight(
        10 + gap, 10, 100 + 6 * (10 + gap), 300, 6, 0, g=0
    )
    ball, _ = track_ball(_dets(rows), FPS)
    assert (ball.state == "interpolated").sum() == 0


def test_strongest_overlapping_tracklet_wins_and_roi_filter():
    weak = _flight(0, 30, 900, 200, -5, 2, g=0, conf=0.15)
    strong = _flight(0, 30, 100, 500, 8, -6, g=0, conf=0.7)
    ball, stats = track_ball(_dets(weak + strong), FPS)
    assert stats["tracklets"]["moving"] == 2
    assert (ball.u < 400).all()  # every frame taken by the strong tracklet

    roi = np.zeros((720, 1280), bool)
    roi[:, 600:] = True  # only the right half is "court"
    ball, stats = track_ball(_dets(weak + strong), FPS, roi)
    assert stats["tracklets"]["off_court"] == 1
    assert (ball.u > 600).all()


def test_link_assigns_every_detection_once():
    rows = _flight(0, 15, 100, 300, 7, 1, g=0) + _flight(0, 15, 800, 300, -7, 1, g=0)
    tracklets = link(_dets(rows))
    assert sum(len(t.frames) for t in tracklets) == 30
    assert len(tracklets) == 2


def test_ball_in_play_does_not_jump_to_a_ball_it_could_not_reach():
    # The ball in play breaks into two tracklets (a 7-frame gap at a hit); meanwhile a ball on
    # another court moves on the far side of the image the whole time.
    a = _flight(0, 30, 100, 300, 8, 0, g=0, conf=0.7)
    a2 = _flight(37, 24, 396, 300, 8, 0, g=0, conf=0.7)
    other = _flight(0, 61, 1100, 300, -3, 0, g=0, conf=0.3)
    ball, stats = track_ball(_dets(a + a2 + other), FPS)
    assert stats["tracklets"]["moving"] == 3
    assert (ball.u < 700).all()  # never switches to the other ball
    assert not ball.frame.between(30, 36).any()  # the gap stays empty
