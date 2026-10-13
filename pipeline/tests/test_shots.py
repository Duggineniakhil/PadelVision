import pandas as pd

from padelvision.analytics.shots import classify_shots, summary

FPS = 10.0


def _boxes(rows):
    """rows: (frame, player, team, x1, y1, x2, y2)."""
    return pd.DataFrame(rows, columns=["frame", "player", "team", "x1", "y1", "x2", "y2"])


def _hits(rows):
    """rows: (frame, player, team, u, v) -> events with kind hit."""
    return pd.DataFrame([(f, "hit", p, t, u, v) for f, p, t, u, v in rows],
                        columns=["frame", "kind", "player", "team", "u", "v"])  # fmt: skip


def _shot(frame, player, team, bounce=None, no_landing=None):
    return {
        "frame": frame,
        "player": player,
        "team": team,
        "bounce": bounce,
        "no_landing": no_landing,
    }


def test_types_and_sides():
    # near player 1 at x 600-680, y 400-600 (centre u 640); far player 3 at 600-640, 200-280
    boxes = _boxes([(f, 1, "near", 600, 400, 680, 600) for f in range(0, 200)]
                   + [(f, 3, "far", 600, 200, 640, 280) for f in range(0, 200)])  # fmt: skip
    deep = {"frame": 0, "x_m": 1.0, "y_m": 8.5, "zone": "back"}
    shots = [
        _shot(10, 1, "near", bounce={**deep, "frame": 18}),  # serve (from stage 5b)
        _shot(30, 3, "far", bounce={**deep, "frame": 36, "y_m": -5.0}),  # after a bounce: ground
        _shot(45, 1, "near", no_landing="probably_volley"),  # groundstroke, ball on the right
        _shot(60, 3, "far"),  # volleyed (the near shot before had no landing)
        _shot(80, 1, "near", bounce={**deep, "frame": 100}),  # 2 s, 8.5 m deep: lob
        _shot(110, 3, "far"),  # ball above the far player's head: overhead
        _shot(130, 1, "near"),  # nothing known about the shot before: unknown
    ]
    hits = _hits([(10, 1, "near", 640, 500), (30, 3, "far", 620, 240), (45, 1, "near", 680, 520),
                  (60, 3, "far", 620, 240), (80, 1, "near", 630, 500), (110, 3, "far", 620, 190),
                  (130, 1, "near", 600, 520)])  # fmt: skip
    classify_shots(shots, hits, boxes, FPS, serve_frames={10})
    assert [s["type"] for s in shots] == [
        "serve", "groundstroke", "groundstroke", "volley", "lob", "overhead", "unknown",
    ]  # fmt: skip
    assert shots[2]["side"] == "forehand"  # ball right of a near player's body
    assert shots[6]["side"] == "backhand"  # left of it (an unknown type still gets a side)
    assert shots[4]["side"] is None  # within SIDE_MIN of the centre: can't tell
    assert shots[1]["side"] is None and shots[3]["side"] is None  # far players: no side
    assert all(s["basis"] for s in shots)

    sm = summary(shots)
    assert sm["counts"]["groundstroke"] == 2 and sm["counts"]["unknown"] == 1
    assert sm["by_player"]["1"]["forehand"] == 1 and sm["by_player"]["1"]["backhand"] == 1
    assert sm["known_share"] == round(6 / 7, 2)


def test_shot_pack_label_and_score(tmp_path):
    import json

    import cv2
    import numpy as np

    from padelvision.shots_review import ShotLabeler, make_shot_pack, score_shots

    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for i in range(60):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()
    run = tmp_path / "run"
    run.mkdir()
    (run / "video.json").write_text(json.dumps({"fps": FPS, "frame_count": 60}))
    shots = [
        {"frame": 15, "player": 1, "team": "near", "type": "volley", "side": "forehand"},
        {"frame": 40, "player": 3, "team": "far", "type": "groundstroke", "side": None},
    ]
    (run / "rallies.json").write_text(json.dumps({"stats": {"shots": shots}}))
    _boxes([(f, p, "near", 10, 10, 20, 30) for f in range(60) for p in (1, 3)]).assign(
        x_m=0.0, y_m=-5.0
    ).to_parquet(run / "players.parquet")
    pack = make_shot_pack(video, run, tmp_path / "pack")
    assert len(list((pack / "frames").glob("*.jpg"))) == 2 * (2 * round(0.6 * FPS) + 1)

    tool = ShotLabeler(pack)
    for k in "vf":  # shot 1: volley, forehand (as predicted)
        tool.key(ord(k))
    tool.key(ord("n"))
    tool.key(ord("o"))  # shot 2: an overhead, predicted groundstroke
    assert tool.render().shape[2] == 3
    r = score_shots(pack)
    assert r["labelled"] == 2 and r["type_accuracy"] == 0.5 and r["side_accuracy"] == 1.0
    assert r["confusion_truth_vs_predicted"] == {
        "overhead -> groundstroke": 1,
        "volley -> volley": 1,
    }
    assert ShotLabeler(pack).labels[40]["type"] == "overhead"  # saved and reloaded
