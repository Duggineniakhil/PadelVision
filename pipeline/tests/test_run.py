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
