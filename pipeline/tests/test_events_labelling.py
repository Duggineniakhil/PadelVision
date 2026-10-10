import json

import cv2
import numpy as np
import pandas as pd

from padelvision.events_eval import load_labels, score_events
from padelvision.events_label_tool import EventLabeler
from padelvision.events_review import choose_windows, make_event_pack

FPS = 30.0


def _seg(start, end, hits=1):
    return {"start_frame": start, "end_frame": end, "hits": hits}


def test_windows_cover_rallies_first_then_random_then_other():
    summary = {
        "rallies": [_seg(300, 450), _seg(460, 600)],  # merge after padding
        "other_activity": [_seg(1000, 1100), _seg(2000, 2100, hits=0)],  # no hits: skipped
    }
    wins = choose_windows(summary, n_frames=3000, fps=FPS, n_random=2)
    sources = [w["source"] for w in wins]
    assert sources.count("rally") == 1 and sources.count("other") == 1
    assert sources.count("random") == 2
    rally = next(w for w in wins if w["source"] == "rally")
    assert (rally["start"], rally["end"]) == (270, 630)  # padded by 1 s and merged
    spans = sorted((w["start"], w["end"]) for w in wins)
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:], strict=False))  # no overlaps

    capped = choose_windows(summary, 3000, FPS, n_random=2, max_total_s=15.0)
    assert [w["source"] for w in capped] == ["rally"]  # 12 s rally fits, nothing else does


def test_event_pack_extracts_window_frames(tmp_path):
    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for i in range(120):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()
    run = tmp_path / "run"
    run.mkdir()
    (run / "video.json").write_text(json.dumps({"fps": FPS, "frame_count": 120}))
    (run / "rallies.json").write_text(json.dumps({"rallies": [_seg(40, 60)]}))
    out = make_event_pack(video, run, tmp_path / "pack", n_random=0)
    pack = json.loads((out / "pack.json").read_text())
    assert [(w["start"], w["end"]) for w in pack["windows"]] == [(10, 90)]
    assert len(list((out / "frames").glob("*.jpg"))) == 81
    assert (out / "run" / "rallies.json").exists()


def _pack(tmp_path):
    pack = tmp_path / "pack"
    (pack / "run").mkdir(parents=True)
    windows = [{"id": 1, "start": 0, "end": 29, "source": "rally"},
               {"id": 2, "start": 100, "end": 129, "source": "random"}]  # fmt: skip
    (pack / "pack.json").write_text(json.dumps({"fps": FPS, "windows": windows}))
    pd.DataFrame(
        [(f, p, x, 100, x + 50, 300) for f in range(130) for p, x in ((1, 100), (3, 600))],
        columns=["frame", "player", "x1", "y1", "x2", "y2"],
    ).to_parquet(pack / "run" / "players.parquet")
    pd.DataFrame({"frame": range(130), "u": 400.0, "v": 200.0}).to_parquet(
        pack / "run" / "ball.parquet"
    )
    return pack


def test_labeler_marks_saves_and_resumes(tmp_path):
    pack = _pack(tmp_path)
    tool = EventLabeler(pack)
    tool.step(5)
    tool.click = (640.0, 150.0)  # inside player 3's box
    tool.mark("hit")
    assert tool.labels[5]["player"] == 3
    tool.set_player(1)
    tool.step(10)
    tool.mark("bounce")  # no click: the tracked ball position
    assert (tool.labels[15]["u"], tool.labels[15]["v"]) == (400.0, 200.0)
    tool.step(100)
    assert tool.frame == 29  # stays inside the window
    tool.finish_window()
    assert tool.win["id"] == 2 and tool.frame == 100

    again = EventLabeler(pack)  # resumes at the first unfinished window
    assert again.wi == 1 and again.done == {1}
    assert again.labels[5]["player"] == "1" and again.labels[15]["kind"] == "bounce"
    labels, windows = load_labels(pack / "labels.csv")
    assert list(labels.frame) == [5, 15] and windows == [{"id": 1, "start": 0, "end": 29}]


def test_score_events_matches_within_tolerance_inside_windows():
    labels = pd.DataFrame(
        [(10, "hit", 400, 200, 2), (40, "hit", 500, 200, 3), (25, "bounce", 450, 400, None)],
        columns=["frame", "kind", "u", "v", "player"],
    )
    pred = pd.DataFrame(
        [(12, "hit", 400, 200, 2.0),  # matches (2 frames off), right player
         (70, "hit", 300, 300, 1.0),  # nothing labelled there: false positive
         (26, "bounce", 455, 400, np.nan),  # matches
         (45, "handling", 500, 200, 3.0),  # not a hit: the labelled hit at 40 is missed
         (500, "hit", 0, 0, 1.0)],  # outside the labelled window: ignored
        columns=["frame", "kind", "u", "v", "player"],
    )  # fmt: skip
    res = score_events(pred, labels, [{"start": 0, "end": 99}], FPS)
    hit, bounce = res["hit"], res["bounce"]
    assert (hit["tp"], hit["fp"], hit["fn"]) == (1, 1, 1)
    assert hit["precision"] == 0.5 and hit["recall"] == 0.5 and hit["player_accuracy"] == 1.0
    assert (bounce["tp"], bounce["fp"], bounce["fn"]) == (1, 0, 0) and bounce["f1"] == 1.0
    assert bounce["median_px_error"] == 5.0
    assert res["labelled_s"] == round(100 / FPS, 1)
