"""Classical ball candidates: small moving blobs, used to bootstrap labels for a learned
ball detector (no model needed).

Three-frame differencing: a pixel belongs to a moving object in frame t if it differs from
both t-1 and t+1. Blobs inside (padded) person boxes are ignored (players move too), as is
everything outside the court region of interest. Remaining blobs that are small, compact
and ideally ball-coloured become candidates; most are still noise, so `link.py` keeps only
those forming ball-like trajectories.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import COURT_LENGTH_M, COURT_WIDTH_M

DIFF_THRESHOLD = 18  # grey levels
MIN_AREA_PX = 2
MAX_AREA_PX = 150
MAX_ASPECT = 6.0  # motion blur stretches the ball
PERSON_PAD_PX = 8
MAX_PER_FRAME = 8
# Ball colour (yellow-green; looks yellow under warm indoor light). OpenCV hue is 0-180.
HUE_RANGE = (18, 50)
MIN_SATURATION = 60
MIN_VALUE = 110

COLUMNS = ["frame", "u", "v", "area", "color", "motion", "score"]


@dataclass
class Candidate:
    u: float
    v: float
    area: int
    color: float  # fraction of blob pixels with ball colour
    motion: float  # mean frame difference inside the blob
    score: float


def court_roi(cal: CourtCalibration) -> np.ndarray:
    """Mask of where a ball in play can appear: the court and the air above it.

    Built from the visible court outline plus its upward projection to the top of the frame.
    """
    w, h = cal.image_size
    hw, hl = COURT_WIDTH_M / 2, COURT_LENGTH_M / 2
    t = np.linspace(-1, 1, 81)
    outline = np.r_[
        np.c_[np.full_like(t, -hw), t * hl],
        np.c_[np.full_like(t, hw), t * hl],
        np.c_[t * hw, np.full_like(t, -hl)],
        np.c_[t * hw, np.full_like(t, hl)],
    ]
    px = cal.to_image(outline)
    px = px[np.isfinite(px).all(axis=1)]
    px = np.clip(px, [0, 0], [w - 1, h - 1])
    pts = np.r_[px, np.c_[px[:, 0], np.zeros(len(px))]].astype(np.float32)
    mask = np.zeros((h, w), np.uint8)
    if len(pts) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts).astype(np.int32), 1)
    return mask.astype(bool)


def frame_candidates(
    prev: np.ndarray,
    cur: np.ndarray,
    nxt: np.ndarray,
    allowed: np.ndarray | None = None,
    person_boxes: np.ndarray | None = None,
) -> list[Candidate]:
    """Candidates in `cur` (BGR frames). `allowed`: bool mask; `person_boxes`: (N, 4) xyxy."""
    g = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0) for f in (prev, cur, nxt)]
    d1 = cv2.absdiff(g[1], g[0])
    d2 = cv2.absdiff(g[2], g[1])
    moving = (d1 > DIFF_THRESHOLD) & (d2 > DIFF_THRESHOLD)
    if allowed is not None:
        moving &= allowed
    if person_boxes is not None and len(person_boxes):
        h, w = moving.shape
        for x1, y1, x2, y2 in np.asarray(person_boxes):
            moving[
                max(0, int(y1) - PERSON_PAD_PX) : min(h, int(y2) + PERSON_PAD_PX),
                max(0, int(x1) - PERSON_PAD_PX) : min(w, int(x2) + PERSON_PAD_PX),
            ] = False

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(moving.astype(np.uint8), 8)
    if n <= 1:
        return []
    hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    colored = (
        (hsv[:, :, 0] >= HUE_RANGE[0])
        & (hsv[:, :, 0] <= HUE_RANGE[1])
        & (hsv[:, :, 1] >= MIN_SATURATION)
        & (hsv[:, :, 2] >= MIN_VALUE)
    )
    motion = np.minimum(d1, d2)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not MIN_AREA_PX <= area <= MAX_AREA_PX:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > MAX_ASPECT:
            continue
        blob = labels[y : y + bh, x : x + bw] == i
        color = float(colored[y : y + bh, x : x + bw][blob].mean())
        mot = float(motion[y : y + bh, x : x + bw][blob].mean())
        # Prefer ball-coloured, strongly moving, ball-sized blobs.
        size_fit = np.exp(-(((np.log(area) - np.log(20)) / 1.2) ** 2))
        score = (0.25 + color) * min(1.0, mot / 60) * size_fit
        out.append(Candidate(float(centroids[i][0]), float(centroids[i][1]), int(area), color,
                             mot, float(score)))  # fmt: skip
    out.sort(key=lambda c: -c.score)
    return out[:MAX_PER_FRAME]


def video_candidates(
    frames: Iterable[tuple[int, np.ndarray]],
    allowed: np.ndarray | None = None,
    person_boxes: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Candidates for a stream of consecutive (frame_index, BGR image) pairs."""
    rows = []
    window: list[tuple[int, np.ndarray]] = []
    for item in frames:
        window.append(item)
        if len(window) < 3:
            continue
        window = window[-3:]
        (_, prev), (f, cur), (_, nxt) = window
        boxes = person_boxes.get(f) if person_boxes else None
        for c in frame_candidates(prev, cur, nxt, allowed, boxes):
            rows.append((f, c.u, c.v, c.area, c.color, c.motion, c.score))
    return pd.DataFrame(rows, columns=COLUMNS)


def read_frames(video_path, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        f = 0
        while max_frames is None or f < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            yield f, img
            f += 1
    finally:
        cap.release()
