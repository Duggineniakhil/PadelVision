import numpy as np
import pandas as pd

from padelvision.ball.evaluate import match, recall_curve, review_crops, score

LABELS = pd.DataFrame(
    [
        (10, "ball", 100.0, 100.0, "random"),
        (20, "ball", 300.0, 200.0, "random"),
        (30, "ball", 50.0, 60.0, "pseudo"),
    ],  # fmt: skip
    columns=["frame", "status", "u", "v", "source"],
)


def _preds(rows):
    return pd.DataFrame(rows, columns=["frame", "u", "v", "conf"])


def test_match_outcomes():
    m = match(_preds([(10, 103, 104, 0.9), (20, 330, 200, 0.8), (40, 1, 1, 0.7)]), LABELS)
    assert list(m.outcome) == ["hit", "miss_location", "unverified"]
    assert m.dist.iloc[0] == 5.0


def test_score_uses_top_detection_per_frame():
    preds = _preds([
        (10, 400, 400, 0.4), (10, 101, 100, 0.9),  # top one is right
        (20, 302, 201, 0.3), (20, 600, 600, 0.95),  # top one is wrong
        (50, 5, 5, 0.6),
    ])  # fmt: skip
    s = score(preds, LABELS)
    assert (s["hits"], s["wrong_location"], s["missed"], s["unverified_detections"]) == (1, 1, 1, 1)
    assert s["recall"] == round(1 / 3, 3)
    assert score(preds, LABELS, conf=0.95)["hits"] == 0
    curve = recall_curve(preds, LABELS)
    assert curve.recall.iloc[0] >= curve.recall.iloc[-1]


def test_review_crops(tmp_path):
    frames = {f: np.zeros((240, 320, 3), np.uint8) for f in (10, 20, 50)}
    preds = _preds([(10, 101, 100, 0.9), (20, 30, 20, 0.95), (50, 5, 5, 0.6)])
    out = review_crops(preds, LABELS, frames, tmp_path, "toy")
    idx = pd.read_csv(out / "index.csv")
    assert list(idx.outcome) == ["miss_location", "unverified"]  # hits need no review
    assert all((out / f).exists() for f in idx.file)
