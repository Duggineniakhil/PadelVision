import cv2
import numpy as np
import pandas as pd

from padelvision.ball.candidates import frame_candidates, video_candidates
from padelvision.ball.link import link_tracks, pseudo_labels

W, H = 320, 240
BALL_BGR = (40, 230, 230)  # yellow


def _scene(t: int) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """Blue court, a static white line, a moving 'player' rectangle and a moving ball."""
    img = np.full((H, W, 3), (120, 60, 40), np.uint8)
    cv2.line(img, (0, 200), (W, 200), (230, 230, 230), 2)
    px = 40 + 3 * t
    player = np.array([[px, 80, px + 30, 180]])
    cv2.rectangle(img, (px, 80), (px + 30, 180), (200, 200, 200), -1)
    ball = (60 + 9 * t, 40 + 4 * t + 0.3 * t * t)  # parabola-ish flight
    cv2.circle(img, (round(ball[0]), round(ball[1])), 3, BALL_BGR, -1)
    return img, ball, player


def test_frame_candidates_find_ball_not_player_or_lines():
    (a, _, _), (b, ball, player), (c, _, _) = (_scene(t) for t in (4, 5, 6))
    cands = frame_candidates(a, b, c, person_boxes=player)
    assert cands, "ball not found"
    best = cands[0]
    assert np.hypot(best.u - ball[0], best.v - ball[1]) < 2.5
    assert best.color > 0.5


def test_blobs_inside_player_boxes_are_ignored():
    frames = []
    for t in (0, 1, 2):
        img = np.full((H, W, 3), (120, 60, 40), np.uint8)
        cv2.circle(img, (100 + 6 * t, 120), 3, BALL_BGR, -1)  # moving "ball" at the racket
        frames.append(img)
    box = np.array([[80, 60, 140, 200]])
    assert frame_candidates(*frames)
    assert not frame_candidates(*frames, person_boxes=box)


def test_video_candidates_and_pseudo_labels_follow_the_ball():
    frames = [(t, _scene(t)[0]) for t in range(20)]
    boxes = {t: _scene(t)[2] for t in range(20)}
    cands = video_candidates(frames, person_boxes=boxes)
    labels = pseudo_labels(cands)
    assert len(labels) >= 15
    for r in labels.itertuples():
        bu, bv = _scene(r.frame)[1]
        assert np.hypot(r.u - bu, r.v - bv) < 2.5


def test_linking_rejects_noise_and_static_flicker():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(60):
        rows.append((f, 100 + 8 * f, 300 - 6 * f + 0.2 * f * f, 0.9))  # ball
        rows.append((f, 640 + rng.normal(0, 0.5), 200 + rng.normal(0, 0.5), 0.5))  # flicker
        for _ in range(3):  # random noise
            rows.append((f, rng.uniform(0, 1280), rng.uniform(0, 720), 0.3))
    cands = pd.DataFrame(rows, columns=["frame", "u", "v", "score"])
    linked = link_tracks(cands)
    on = linked[linked.track >= 0]
    ball = on[(on.u - (100 + 8 * on.frame)).abs() < 1]
    assert len(ball) >= 55
    assert len(on) - len(ball) <= 3  # almost no false pseudo-labels


def test_linking_survives_short_occlusion():
    rows = [(f, 50 + 10 * f, 400.0, 1.0) for f in range(30) if f not in (12, 13)]
    labels = pseudo_labels(pd.DataFrame(rows, columns=["frame", "u", "v", "score"]))
    assert labels.track.nunique() == 1
    assert len(labels) == 28


def test_bootstrap_review_pack_and_labeler(tmp_path, cal, monkeypatch):
    import csv
    import json

    from conftest import SIZE
    from test_run import FakeTracker

    from padelvision import run as run_module
    from padelvision.ball.bootstrap import bootstrap, make_review_pack
    from padelvision.ball.label_tool import BallLabeler
    from padelvision.models import person_tracker

    # Video: blank court frames with a ball flying across the middle of the image.
    video = tmp_path / "match.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, SIZE)
    path = {}
    for f in range(90):
        img = np.full((SIZE[1], SIZE[0], 3), (120, 60, 40), np.uint8)
        if 10 <= f < 80:
            path[f] = (300 + 8 * (f - 10), 420 - 3 * (f - 10) + 0.05 * (f - 10) ** 2)
            cv2.circle(img, tuple(round(c) for c in path[f]), 4, BALL_BGR, -1)
        w.write(img)
    w.release()

    monkeypatch.setattr(person_tracker, "PersonTracker", FakeTracker)
    monkeypatch.setattr(run_module, "weights_path", lambda name: "fake.pt")
    court = tmp_path / "court.json"
    cal.save(court)
    run_dir = tmp_path / "run"
    run_module.analyze(video, court, run_dir, max_frames=90)

    pseudo = bootstrap(video, run_dir)
    assert len(pseudo) >= 50
    for r in pseudo.itertuples():
        assert np.hypot(r.u - path[r.frame][0], r.v - path[r.frame][1]) < 3

    pack = make_review_pack(video, run_dir, tmp_path / "review", n_pseudo=5, n_random=5)
    meta = json.loads((pack / "pack.json").read_text())
    assert len(meta["items"]) == 10
    for it in meta["items"]:
        assert (pack / "frames" / f"{it['frame']:06d}.jpg").exists()
        assert (pack / "motion" / f"{it['frame']:06d}.png").exists()

    tool = BallLabeler(pack)
    first = tool.item
    tool.on_key(ord("y") if first["proposal"] else ord("n"))
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 200)
    tool.on_key(ord("n"))
    tool.on_key(ord("m"))
    assert tool.render().shape[2] == 3

    resumed = BallLabeler(pack)  # labels persist; resumes at the first unlabelled frame
    assert len(resumed.labels) == 3
    assert resumed.i == 3
    rows = list(csv.DictReader(open(pack / "labels.csv")))
    assert [r["status"] for r in rows][1:] == ["ball", "none"]


def test_dull_or_weak_tracks_are_not_balls():
    rows = []
    for f in range(40):
        rows.append((f, 100 + 8 * f, 300.0, 0.9, 0.7, 60.0))  # yellow, sharp: ball
        rows.append((f, 100 + 8 * f, 500.0, 0.9, 0.0, 60.0))  # grey (a leg, a racket)
        rows.append((f, 100 + 8 * f, 650.0, 0.9, 0.7, 15.0))  # weak motion (far background)
    cands = pd.DataFrame(rows, columns=["frame", "u", "v", "score", "color", "motion"])
    labels = pseudo_labels(cands)
    assert set(labels.v.round()) == {300.0}
