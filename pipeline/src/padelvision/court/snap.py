"""Snap a roughly clicked court line onto the image.

The user clicks a few points along a line (2 are enough; more follow the lens curve better).
Along that guide, every STEP_PX a short profile is read across the line and the line's exact
position is found in it:

- "paint": a painted court line, i.e. a thin strip brighter than the floor and not coloured.
- "edge":  the strongest colour change, e.g. where the floor meets a wall (the side lines) or a
           base strip (a baseline painted against the back wall).

Offsets that disagree with a smooth curve through the rest are dropped (players, balls, glare).
Returns the snapped points, or [] if fewer than MIN_POINTS were found.
"""

from __future__ import annotations

import cv2
import numpy as np

STEP_PX = 8.0  # one sample every 8 px along the guide
HALF_PX = 10.0  # search +-10 px across it
PAINT_CONTRAST = 18.0  # grey levels above the local floor
PAINT_MAX_SAT = 70.0  # max - min channel: painted lines are white-ish
EDGE_MIN_GRAD = 9.0  # colour change per pixel
MIN_POINTS = 3


def snap_line(image: np.ndarray, guide: list[tuple[float, float]], mode: str = "paint",
              half_px: float = HALF_PX) -> list[tuple[float, float]]:  # fmt: skip
    """image: BGR. guide: >= 2 clicked points along the line, in order."""
    if mode not in ("paint", "edge"):
        raise ValueError(f"mode must be 'paint' or 'edge', got {mode!r}")
    g = np.asarray(guide, dtype=np.float64).reshape(-1, 2)
    if len(g) < 2:
        raise ValueError("Need at least 2 guide points")
    pts, normals = _resample(g, STEP_PX)
    if len(pts) < MIN_POINTS:
        return []
    offsets = np.arange(-half_px, half_px + 1e-9, 0.5)
    xs = (pts[:, None, 0] + offsets[None, :] * normals[:, None, 0]).astype(np.float32)
    ys = (pts[:, None, 1] + offsets[None, :] * normals[:, None, 1]).astype(np.float32)
    h, w = image.shape[:2]
    inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    prof = cv2.remap(image, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    prof = prof.astype(np.float64)  # (N, M, 3)

    found = np.full(len(pts), np.nan)
    for i in range(len(pts)):
        if not inside[i].all():
            continue
        found[i] = _paint(prof[i], offsets) if mode == "paint" else _edge(prof[i], offsets)
    keep = _consistent(np.arange(len(pts)) * STEP_PX, found)
    if keep.sum() < MIN_POINTS:
        return []
    out = pts[keep] + found[keep, None] * normals[keep]
    return [(round(float(x), 2), round(float(y), 2)) for x, y in out]


def _resample(g: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` px along the polyline, with unit normals."""
    seg = np.linalg.norm(np.diff(g, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < step:
        return np.empty((0, 2)), np.empty((0, 2))
    t = np.arange(step / 2, s[-1] - step / 4, step)
    pts = np.c_[np.interp(t, s, g[:, 0]), np.interp(t, s, g[:, 1])]
    d = 2.0
    ahead = np.c_[np.interp(t + d, s, g[:, 0]), np.interp(t + d, s, g[:, 1])]
    behind = np.c_[np.interp(t - d, s, g[:, 0]), np.interp(t - d, s, g[:, 1])]
    tang = ahead - behind
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    return pts, np.c_[-tang[:, 1], tang[:, 0]]


def _paint(prof: np.ndarray, offsets: np.ndarray) -> float:
    grey = prof.mean(axis=1)
    sat = prof.max(axis=1) - prof.min(axis=1)
    p = grey - np.median(grey)
    p[sat > PAINT_MAX_SAT] = 0
    best = np.nan
    for k in _peaks(p, PAINT_CONTRAST, offsets):
        lo = hi = k  # the strip around the peak above half its height
        while lo > 0 and p[lo - 1] > p[k] / 2:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] > p[k] / 2:
            hi += 1
        if lo == 0 or hi == len(p) - 1:  # not a thin strip inside the window
            continue
        wgt = p[lo : hi + 1]
        best = float((wgt * offsets[lo : hi + 1]).sum() / wgt.sum())
        break
    return best


def _edge(prof: np.ndarray, offsets: np.ndarray) -> float:
    sm = cv2.GaussianBlur(prof.reshape(1, -1, 3).astype(np.float32), (5, 1), 0)[0].astype(float)
    grad = np.linalg.norm(np.gradient(sm, axis=0), axis=1) / 0.5  # per pixel
    grad[:2] = grad[-2:] = 0
    peaks = _peaks(grad, EDGE_MIN_GRAD, offsets)
    if not peaks:
        return np.nan
    k = peaks[0]
    a, b, c = grad[k - 1], grad[k], grad[k + 1]  # parabola through the peak
    den = a - 2 * b + c
    frac = 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return float(offsets[k] + 0.5 * frac)


def _peaks(v: np.ndarray, min_value: float, offsets: np.ndarray) -> list[int]:
    """Local maxima of `v` at least `min_value`, nearest to the guide (offset 0) first: where
    lines lie close together (far court), the one clicked wins over a stronger neighbour."""
    k = np.arange(1, len(v) - 1)
    k = k[(v[k] >= min_value) & (v[k] >= v[k - 1]) & (v[k] > v[k + 1])]
    return sorted(k.tolist(), key=lambda i: abs(offsets[i]))


def _consistent(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Mask of offsets that lie on a smooth curve (quadratic in arc length) through the rest."""
    keep = np.isfinite(d)
    for _ in range(3):
        if keep.sum() < MIN_POINTS:
            break
        deg = 2 if keep.sum() >= 6 else 1
        fit = np.polyval(np.polyfit(s[keep], d[keep], deg), s)
        res = np.abs(d - fit)
        mad = np.median(res[keep])
        new = np.isfinite(d) & (res <= max(1.5, 3.0 * 1.4826 * mad))
        if (new == keep).all():
            break
        keep = new
    return keep
