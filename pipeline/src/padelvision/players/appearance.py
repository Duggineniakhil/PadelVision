"""Player appearance: the colours of the shirt and shorts (no model).

Positions alone can't tell whether the pairs changed ends when all four players meet at the
net and the tracker loses them, nor which partner is which when both swap sides for a game.
Clothing colours can. Every SAMPLE_S, each person box of stage 2 (detections.parquet, so a
re-run of stage 3 doesn't make this stale) gets the median CIELAB colour of two regions inside
the silhouette: the shirt (upper-middle of the box) and the shorts (below it).

Lab, not hue histograms: dark clothes have noisy hue and saturation, and far players are seen
through haze and a blue cast; lightness (L) separates dark from white kit on both ends.
Test_video: shirts L 23-40 (dark pair) vs 136-163 (white pair), near and far alike.

Writes appearance.parquet: frame, track_id, n_px, tL, ta, tb (shirt), lL, la, lb (shorts), in
OpenCV's 8-bit Lab (L 0-255, a/b centred on 128). Needs the video, so it runs in the notebooks
(Kaggle/Colab) next to detection; reading frames is cheap (no GPU).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SAMPLE_S = 0.5
MIN_BOX_H_PX = 30  # smaller boxes have too few pixels to say anything
# Regions as fractions of the box (x0, x1, y0, y1): shirt and shorts, inside the silhouette.
TORSO = (0.3, 0.7, 0.2, 0.45)
LEGS = (0.3, 0.7, 0.5, 0.65)
FEATURES = ["tL", "ta", "tb", "lL", "la", "lb"]
COLUMNS = ["frame", "track_id", "n_px", *FEATURES]
# A clearly different colour is about this far apart in Lab (8-bit units).
SCALE = 40.0


def box_descriptor(
    img_lab: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[np.ndarray, int]:
    """Median Lab of the shirt and shorts regions of one box (6 values) and the pixels used."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    out, n = [], 0
    for fx0, fx1, fy0, fy1 in (TORSO, LEGS):
        a, b = max(round(x1 + fx0 * w), 0), round(x1 + fx1 * w)
        c, d = max(round(y1 + fy0 * h), 0), round(y1 + fy1 * h)
        crop = img_lab[c:d, a:b].reshape(-1, 3)
        out += list(np.median(crop, axis=0)) if len(crop) else [np.nan] * 3
        n += len(crop)
    return np.array(out, dtype=np.float64), n


def extract(
    video: str | Path, dets: pd.DataFrame, fps: float, sample_s: float = SAMPLE_S
) -> pd.DataFrame:
    """Descriptors of the tracked person boxes on every `sample_s`-th second of the video."""
    step = max(1, round(sample_s * fps))
    dets = dets[(dets.track_id >= 0) & (dets.y2 - dets.y1 >= MIN_BOX_H_PX)]
    wanted = {f: g for f, g in dets[dets.frame % step == 0].groupby("frame")}
    rows = []
    cap = cv2.VideoCapture(str(video))
    try:
        last = max(wanted) if wanted else -1
        for f in range(last + 1):
            if f not in wanted:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            for r in wanted[f].itertuples():
                desc, n = box_descriptor(lab, (r.x1, r.y1, r.x2, r.y2))
                rows.append((f, int(r.track_id), n, *desc.astype(np.float32)))
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=COLUMNS)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Colour distance between two descriptors, ~1 = clearly different clothes."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / SCALE)
