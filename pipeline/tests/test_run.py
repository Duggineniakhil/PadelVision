"""End-to-end `analyze` + `render` with a fake tracker and a blank synthetic video."""

import json

import cv2
import numpy as np
from conftest import SIZE, synthetic_frames

from padelvision import run as run_module
from padelvision.models import person_tracker
from padelvision.render import render_preview


class FakeTracker:
    def __init__(self, weights, device=None, imgsz=1280):
        self.device = "fake"

    def track(self, video, stride=1, max_frames=None):
        for fd in synthetic_frames(30.0, seconds=10, stride=stride):
            if max_frames is not None and fd.frame >= max_frames:
                break
            yield fd


def _blank_video(path, frames=300):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, SIZE)
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    for _ in range(frames):
        w.write(img)
    w.release()


def test_analyze_and_render(tmp_path, cal, monkeypatch):
    monkeypatch.setattr(person_tracker, "PersonTracker", FakeTracker)
    monkeypatch.setattr(run_module, "weights_path", lambda name: "fake.pt")
    video = tmp_path / "match.avi"
    _blank_video(video)
    court = tmp_path / "court.json"
    cal.save(court)
    out = tmp_path / "run"

    stats = run_module.analyze(video, court, out, stride=2)

    for name in ["video.json", "court.json", "court_overlay.png", "detections.parquet",
                 "players.parquet", "tracks.parquet", "stats.json", "heatmaps.png"]:  # fmt: skip
        assert (out / name).exists(), name
    assert json.loads((out / "stats.json").read_text()) == json.loads(json.dumps(stats))
    for p in "1234":
        assert stats["players"][p]["tracked_fraction"] > 0.95
        assert stats["players"][p]["distance_m"] > 0
    assert stats["analysed"]["stride"] == 2

    # Second run re-uses cached detections.
    monkeypatch.setattr(person_tracker, "PersonTracker", None)
    run_module.analyze(video, court, out, stride=2)

    import pandas as pd

    preview = render_preview(video, pd.read_parquet(out / "players.parquet"), cal,
                             tmp_path / "preview.avi", seconds=1)  # fmt: skip
    cap = cv2.VideoCapture(str(preview))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    cap.release()


def test_restats_from_cached_detections(tmp_path, cal, monkeypatch):
    monkeypatch.setattr(person_tracker, "PersonTracker", FakeTracker)
    monkeypatch.setattr(run_module, "weights_path", lambda name: "fake.pt")
    video = tmp_path / "match.avi"
    _blank_video(video)
    court = tmp_path / "court.json"
    cal.save(court)
    out = tmp_path / "run"
    first = run_module.analyze(video, court, out, stride=2, max_frames=200)

    video.unlink()  # restats must not need the video
    again = run_module.restats(out)
    assert again["players"] == first["players"]
    assert again["analysed"]["frames"] == first["analysed"]["frames"] == 100


def test_detection_cache_invalidated_when_settings_change(tmp_path, cal, monkeypatch):
    calls = []

    class CountingTracker(FakeTracker):
        def track(self, video, stride=1, max_frames=None):
            calls.append(max_frames)
            yield from super().track(video, stride, max_frames)

    monkeypatch.setattr(person_tracker, "PersonTracker", CountingTracker)
    monkeypatch.setattr(run_module, "weights_path", lambda name: "fake.pt")
    video = tmp_path / "match.avi"
    _blank_video(video)
    court = tmp_path / "court.json"
    cal.save(court)
    out = tmp_path / "run"

    run_module.analyze(video, court, out, max_frames=60)
    run_module.analyze(video, court, out, max_frames=60)  # same settings: cached
    stats = run_module.analyze(video, court, out)  # full video: must re-detect
    assert calls == [60, None]
    assert stats["players"]["1"]["tracked_fraction"] > 0.9


def test_ball_track_stage_retrack_and_render(tmp_path, cal, monkeypatch):
    import json

    import pandas as pd

    from padelvision.models import ball_detectors

    class FakeBall:
        def __init__(self, weights, classes=None, imgsz=1280, conf=0.05, device=None):
            self.device = "fake"

        def detect_video(self, video, stride=1, max_frames=None):
            rows = [(f, 200 + 8 * f, 300 - 3 * f + 0.1 * f * f, 0.4) for f in range(40) if f != 15]
            rows += [(f, 900.0, 600.0, 0.5) for f in range(40)]  # a spare ball lying still
            return pd.DataFrame(rows, columns=["frame", "u", "v", "conf"])

    monkeypatch.setattr(ball_detectors, "YoloBall", FakeBall)
    monkeypatch.setattr(run_module, "weights_path", lambda name: "fake.pt")
    video = tmp_path / "match.avi"
    _blank_video(video, frames=40)
    out = tmp_path / "run"
    out.mkdir()
    cal.save(out / "court.json")

    stats = run_module.ball_track(video, out)
    ball = pd.read_parquet(out / "ball.parquet")
    assert stats["roi"] and stats["tracklets"]["static"] == 1
    assert len(ball) == 40 and (ball.state == "interpolated").sum() == 1
    assert (ball.u != 900.0).all()

    monkeypatch.setattr(ball_detectors, "YoloBall", None)  # retrack must not need the model
    video.unlink()
    again = run_module.ball_retrack(out)
    assert again == json.loads((out / "ball_stats.json").read_text())
    assert again["frames_with_ball"] == 40

    video2 = tmp_path / "again.avi"
    _blank_video(video2, frames=40)
    players = pd.DataFrame(
        columns=["frame", "player", "team", "x1", "y1", "x2", "y2", "x_m", "y_m", "valid"]
    )
    preview = render_preview(video2, players, cal, tmp_path / "p.avi", seconds=1, ball=ball)
    assert preview.exists()
