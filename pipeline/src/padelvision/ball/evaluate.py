"""Score ball detectors against confirmed labels, with pooled review of unverified detections.

Labels list only visually confirmed balls (frame, u, v). Frames without a label are
*unverified*, not "no ball", so a detection there is neither right nor wrong until a person
reviews a crop of it (`review_crops`). Confirmed new balls are added to the labels
("pooling"), so a model that finds balls the classical bootstrap missed gets credit.

Predictions: DataFrame with columns frame, u, v, conf (any number per frame).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TOLERANCE_PX = 10.0


def match(preds: pd.DataFrame, labels: pd.DataFrame, tol: float = TOLERANCE_PX) -> pd.DataFrame:
    """Each prediction annotated with `dist` to the labelled ball in its frame (NaN if the
    frame has no label) and `outcome`: 'hit' | 'miss_location' | 'unverified'."""
    lab = labels[labels.status == "ball"].set_index("frame")[["u", "v"]]
    p = preds.copy()
    has = p.frame.isin(lab.index)
    p["dist"] = np.nan
    if has.any():
        lu = lab.loc[p.frame[has], "u"].to_numpy()
        lv = lab.loc[p.frame[has], "v"].to_numpy()
        p.loc[has, "dist"] = np.hypot(p.u[has].to_numpy() - lu, p.v[has].to_numpy() - lv)
    p["outcome"] = np.where(~has, "unverified", np.where(p.dist <= tol, "hit", "miss_location"))
    return p


def score(
    preds: pd.DataFrame, labels: pd.DataFrame, conf: float = 0.0, tol: float = TOLERANCE_PX
) -> dict:
    """Recall on labelled balls using each frame's top detection (above `conf`)."""
    lab = labels[labels.status == "ball"]
    top = preds[preds.conf >= conf].sort_values("conf", ascending=False).drop_duplicates("frame")
    m = match(top, labels, tol)
    hits = m[m.outcome == "hit"]
    n = len(lab)
    on_labelled = m[m.outcome != "unverified"]
    return {
        "labelled_balls": n,
        "recall": round(len(hits) / n, 3) if n else None,
        "hits": len(hits),
        "wrong_location": int((m.outcome == "miss_location").sum()),
        "missed": n - len(on_labelled),
        "median_error_px": round(float(hits.dist.median()), 1) if len(hits) else None,
        "unverified_detections": int((m.outcome == "unverified").sum()),
    }


def recall_curve(
    preds: pd.DataFrame, labels: pd.DataFrame, tol: float = TOLERANCE_PX
) -> pd.DataFrame:
    rows = []
    for c in sorted({0.0, *np.round(np.linspace(0.05, 0.95, 19), 2)}):
        s = score(preds, labels, c, tol)
        rows.append(
            {
                "conf": c,
                "recall": s["recall"],
                "detections_on_unverified": s["unverified_detections"],
                "wrong_location": s["wrong_location"],
            }
        )
    return pd.DataFrame(rows)


def review_crops(
    preds: pd.DataFrame,
    labels: pd.DataFrame,
    frames: dict[int, np.ndarray],
    out_dir,
    model: str,
    max_crops: int = 60,
    radius: int = 24,
) -> Path:
    """Save crops of the most confident detections that aren't confirmed hits (wrong location
    on a labelled frame, or on an unverified frame) for human ball / not-ball review."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    top = preds.sort_values("conf", ascending=False).drop_duplicates("frame")
    m = match(top, labels)
    todo = m[m.outcome != "hit"].head(max_crops)
    rows = []
    for k, r in enumerate(todo.itertuples()):
        img = frames.get(int(r.frame))
        if img is None:
            continue
        u, v = int(round(r.u)), int(round(r.v))
        pad = cv2.copyMakeBorder(img, radius, radius, radius, radius, cv2.BORDER_CONSTANT)
        crop = pad[v : v + 2 * radius, u : u + 2 * radius]
        name = f"{model}_{k:03d}_f{int(r.frame)}.png"
        cv2.imwrite(str(out / name), crop)
        rows.append({"file": name, "model": model, "frame": int(r.frame), "u": r.u, "v": r.v,
                     "conf": r.conf, "outcome": r.outcome})  # fmt: skip
    index = out / "index.csv"
    df = pd.DataFrame(rows)
    if index.exists():
        df = pd.concat([pd.read_csv(index), df], ignore_index=True)
    df.to_csv(index, index=False)
    return out
