"""Score ball detectors against pooled labels.

Labels (one row per reviewed location): frame, status, u, v, source, where status is
- "ball":     a visually confirmed ball (a frame can contain several: spare balls on the floor,
              a ball in a player's hand, the ball in play)
- "not_ball": a confirmed non-ball spot that some detector fired on (lights, signs, shoes...)
- "unsure":   reviewed but undecidable; detections there are ignored

Each detection is classified: "tp" (on an unmatched ball), "duplicate" (on an already matched
ball; ignored), "fp" (on a not_ball spot), "unsure" (ignored) or "unverified" (nobody has
reviewed that spot yet). Unverified detections are neither right nor wrong until someone
reviews a crop of them (`review_crops`) and the verdict is added to the labels ("pooling"),
so every model is judged on the same reviewed locations.

Predictions: DataFrame(frame, u, v, conf), any number per frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TOLERANCE_PX = 10.0


def classify(preds: pd.DataFrame, labels: pd.DataFrame, tol: float = TOLERANCE_PX) -> pd.DataFrame:
    """Annotate each prediction with `outcome` (see module docstring)."""
    by_frame = {f: g for f, g in labels.groupby("frame")}
    outcome = pd.Series("unverified", index=preds.index, dtype=object)
    for f, g in preds.groupby("frame", sort=False):
        lab = by_frame.get(f)
        if lab is None:
            continue
        balls = lab[lab.status == "ball"][["u", "v"]].to_numpy()
        matched = np.zeros(len(balls), bool)
        not_ball = lab[lab.status == "not_ball"][["u", "v"]].to_numpy()
        unsure = lab[lab.status == "unsure"][["u", "v"]].to_numpy()
        for i, r in g.sort_values("conf", ascending=False).iterrows():
            p = np.array([r.u, r.v])
            if len(balls):
                d = np.linalg.norm(balls - p, axis=1)
                free = np.where(~matched & (d <= tol))[0]
                if len(free):
                    matched[free[np.argmin(d[free])]] = True
                    outcome[i] = "tp"
                    continue
                if (d <= tol).any():
                    outcome[i] = "duplicate"
                    continue
            if len(not_ball) and (np.linalg.norm(not_ball - p, axis=1) <= tol).any():
                outcome[i] = "fp"
            elif len(unsure) and (np.linalg.norm(unsure - p, axis=1) <= tol).any():
                outcome[i] = "unsure"
    return preds.assign(outcome=outcome)


def score(
    preds: pd.DataFrame,
    labels: pd.DataFrame,
    frames=None,
    conf: float = 0.0,
    tol: float = TOLERANCE_PX,
) -> dict:
    """Recall over labelled balls and precision over reviewed detections.

    `frames`: the frames the model was run on (labels elsewhere are ignored). Defaults to
    every labelled frame.
    """
    if frames is not None:
        frames = set(frames)
        labels = labels[labels.frame.isin(frames)]
        preds = preds[preds.frame.isin(frames)]
    p = classify(preds[preds.conf >= conf], labels, tol)
    n_balls = int((labels.status == "ball").sum())
    tp, fp = int((p.outcome == "tp").sum()), int((p.outcome == "fp").sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / n_balls if n_balls else None
    f1 = 2 * tp / (2 * tp + fp + (n_balls - tp)) if n_balls else None
    return {
        "balls": n_balls,
        "tp": tp,
        "fp": fp,
        "unverified": int((p.outcome == "unverified").sum()),
        "recall": None if recall is None else round(recall, 3),
        "precision": None if precision is None else round(precision, 3),
        "f1": None if f1 is None else round(f1, 3),
    }


def curve(
    preds: pd.DataFrame, labels: pd.DataFrame, frames=None, tol: float = TOLERANCE_PX
) -> pd.DataFrame:
    """Scores over confidence thresholds."""
    confs = sorted({0.0, *np.round(np.linspace(0.05, 0.95, 19), 2)})
    return pd.DataFrame([{"conf": c, **score(preds, labels, frames, c, tol)} for c in confs])


def review_crops(
    preds: pd.DataFrame,
    labels: pd.DataFrame,
    frames: dict[int, np.ndarray],
    out_dir,
    model: str,
    max_crops: int = 60,
    radius: int = 24,
) -> Path:
    """Save crops of the most confident *unverified* detections for ball / not-ball review."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = classify(preds, labels)
    todo = p[p.outcome == "unverified"].sort_values("conf", ascending=False).head(max_crops)
    rows = []
    for k, r in enumerate(todo.itertuples()):
        img = frames.get(int(r.frame))
        if img is None:
            continue
        u, v = int(round(r.u)), int(round(r.v))
        pad = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
        name = f"{model}_{k:03d}_f{int(r.frame)}.png"
        cv2.imwrite(str(out / name), pad[v : v + 2 * radius, u : u + 2 * radius])
        rows.append(
            {
                "file": name,
                "model": model,
                "frame": int(r.frame),
                "u": r.u,
                "v": r.v,
                "conf": r.conf,
            }
        )
    index = out / "index.csv"
    df = pd.DataFrame(rows)
    if index.exists():
        df = pd.concat([pd.read_csv(index), df], ignore_index=True)
    df.to_csv(index, index=False)
    return out
