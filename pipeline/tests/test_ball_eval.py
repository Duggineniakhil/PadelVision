import numpy as np
import pandas as pd

from padelvision.ball.evaluate import classify, curve, review_crops, score

LABELS = pd.DataFrame(
    [
        (10, "ball", 100.0, 100.0, "random"),
        (10, "ball", 400.0, 300.0, "pooled"),  # a spare ball on the floor
        (20, "ball", 300.0, 200.0, "random"),
        (20, "not_ball", 600.0, 100.0, "pooled"),  # a light reflection
        (30, "unsure", 50.0, 50.0, "pooled"),
    ],
    columns=["frame", "status", "u", "v", "source"],
)


def _preds(rows):
    return pd.DataFrame(rows, columns=["frame", "u", "v", "conf"])


def test_classify_outcomes():
    p = classify(
        _preds([
            (10, 103, 104, 0.9),  # ball
            (10, 99, 98, 0.8),  # same ball again
            (10, 401, 302, 0.7),  # the second ball
            (20, 601, 99, 0.95),  # reflection
            (30, 52, 49, 0.5),  # unsure spot
            (40, 1, 1, 0.6),  # never reviewed
        ]),
        LABELS,
    )  # fmt: skip
    assert list(p.outcome) == ["tp", "duplicate", "tp", "fp", "unsure", "unverified"]


def test_score_and_frames_restriction():
    preds = _preds(
        [(10, 101, 100, 0.9), (20, 600, 100, 0.95), (20, 300, 201, 0.3), (50, 5, 5, 0.6)]
    )
    s = score(preds, LABELS)
    assert (s["balls"], s["tp"], s["fp"], s["unverified"]) == (3, 2, 1, 1)
    assert s["recall"] == round(2 / 3, 3)
    assert s["precision"] == round(2 / 3, 3)
    assert score(preds, LABELS, conf=0.5)["tp"] == 1
    only10 = score(preds, LABELS, frames=[10])  # model not run on frame 20: its ball doesn't count
    assert (only10["balls"], only10["tp"], only10["fp"]) == (2, 1, 0)
    c = curve(preds, LABELS)
    assert c.recall.iloc[0] >= c.recall.iloc[-1]


def test_review_crops_only_unverified(tmp_path):
    frames = {f: np.zeros((240, 320, 3), np.uint8) for f in (10, 20, 50)}
    preds = _preds([(10, 101, 100, 0.9), (20, 600, 100, 0.95), (50, 5, 5, 0.6)])
    out = review_crops(preds, LABELS, frames, tmp_path, "toy")
    idx = pd.read_csv(out / "index.csv")
    assert list(idx.frame) == [50]
    assert all((out / f).exists() for f in idx.file)
