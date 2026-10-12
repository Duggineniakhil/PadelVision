"""Camera movement check: does the calibration still fit the video later on?

A camera knocked during a match makes every court position after that wrong, silently. Every
CHECK_EVERY_S a background is built from FRAMES frames around that moment (their median
removes the players), the court evidence is found in it (`auto.evidence`), and the court is
re-fitted locally from the calibration (`auto.search`, small range, lens kept). How far the
court lines moved in the image (median over the visible line points) is the camera movement.
Each checkpoint is compared with the first usable one (the camera at the start), both re-fitted
the same way, so the re-fit's own offset from the calibration cancels out. A checkpoint whose
re-fit still fits badly (crowd in front of the camera, something over the lens) is skipped.

Output (`camera.json`): one row per checkpoint, the moments the camera moved, and the
expected extra ground error after them. The analysis isn't re-run per segment: a moved camera
is reported so the video can be split and calibrated per part.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from padelvision.court.auto import ANCHORS, Evidence, evidence, scaled, score
from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import COURT_LINES

CHECK_EVERY_S = 60.0
WINDOW_S = 20.0  # the background around a checkpoint uses frames from this long
FRAMES = 15
SEARCH_PX = 36.0  # how far the camera may have moved (image px at 1280 wide)
MOVED_PX = 5.0
MAX_USABLE_SCORE = 2.5  # a background the court can't be fitted to (crowd, card, cover) is skipped


def line_points(cal: CourtCalibration) -> np.ndarray:
    pts = []
    for (x0, y0), (x1, y1) in COURT_LINES:
        t = np.linspace(0, 1, 40)[:, None]
        pts.append(np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
    return np.concatenate(pts)


def displacement_px(a: CourtCalibration, b: CourtCalibration) -> float:
    """Median image distance between two calibrations' court lines (visible part of `a`)."""
    pts = line_points(a)
    pa, pb = a.to_image(pts), b.to_image(pts)
    w, h = a.image_size
    ok = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
    ok &= (pa[:, 0] >= 0) & (pa[:, 0] < w) & (pa[:, 1] >= 0) & (pa[:, 1] < h)
    return float(np.median(np.hypot(*(pa[ok] - pb[ok]).T))) if ok.any() else float("nan")


def _moved(anchors: np.ndarray, dx: float, dy: float, rot_deg: float, scale: float) -> np.ndarray:
    """The anchors after the image moved: a shift, a rotation and a zoom about the image centre
    (a knocked camera turns a little; the image moves as a whole)."""
    centre = anchors.mean(axis=0)
    r = np.deg2rad(rot_deg)
    m = scale * np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return (anchors - centre) @ m.T + centre + [dx, dy]


def refit(
    ev: Evidence, cal: CourtCalibration
) -> tuple[CourtCalibration, CourtCalibration, float, float]:
    """How the image moved relative to `cal`: the best shift / rotation / zoom of the court
    (lens kept). Returns (court from the anchors before, after, score before, score after)."""
    from scipy.optimize import minimize

    size = cal.image_size
    lens, anc = scaled(cal, size)
    a0 = np.array([anc[n] for n in ANCHORS])
    before = score(ev, lens, a0, 3.0)
    step = size[0] / 1280 * 6
    k = int(SEARCH_PX // 6)
    shifts = [(dx, dy) for dx in np.arange(-k, k + 1) * step for dy in np.arange(-k, k + 1) * step]
    dx, dy = min(shifts, key=lambda d: score(ev, lens, a0 + d, 6.0))
    x = np.array([dx, dy, 0.0, 1.0])
    for cap in (6.0, 3.0):
        res = minimize(
            lambda v, c=cap: score(ev, lens, _moved(a0, *v), c),
            x,
            method="Nelder-Mead",
            options={
                "xatol": 0.05,
                "fatol": 1e-4,
                "maxiter": 600,
                "initial_simplex": [
                    x,
                    x + [3, 0, 0, 0],
                    x + [0, 3, 0, 0],
                    x + [0, 0, 0.5, 0],
                    x + [0, 0, 0, 0.01],
                ],
            },  # fmt: skip
        )
        x = res.x
    after = score(ev, lens, _moved(a0, *x), 3.0)

    def court(anchors: np.ndarray) -> CourtCalibration:
        return CourtCalibration.from_points(dict(zip(ANCHORS, map(tuple, anchors), strict=True)),
                                            size, lens=lens)  # fmt: skip

    return court(a0), court(_moved(a0, *x)), before, after


def check_backgrounds(backgrounds: list[tuple[float, np.ndarray]], cal: CourtCalibration) -> dict:
    """[(t, background image)] -> camera.json content (see the module docstring)."""
    rows, fits = [], []
    for t, img in backgrounds:
        if (img.shape[1], img.shape[0]) != tuple(cal.image_size):
            img = cv2.resize(img, tuple(cal.image_size), interpolation=cv2.INTER_AREA)
        base, fitted, before, after = refit(evidence(img), cal)
        usable = after <= MAX_USABLE_SCORE
        rows.append({"t": round(t, 1), "score_px": round(after, 2), "usable": bool(usable)})
        fits.append((base, fitted) if usable else None)
    ref = next((f for f in fits if f is not None), None)
    for row, f in zip(rows, fits, strict=True):
        row["moved_px"] = (
            None if f is None or ref is None else round(displacement_px(ref[1], f[1]), 1)
        )
        row["moved"] = row["moved_px"] is not None and row["moved_px"] >= MOVED_PX
    events, prev = [], None
    for row in rows:
        if not row["usable"]:
            continue
        if prev is not None and row["moved"] and not prev["moved"]:
            events.append({"between_s": [prev["t"], row["t"]], "moved_px": row["moved_px"]})
        prev = row
    worst = max((r["moved_px"] for r in rows if r["moved"]), default=0.0)
    m_px = {z: v["m_per_px"] for z, v in cal.accuracy()["zones"].items()}
    return {
        "checkpoints": rows,
        "moved": events,
        "camera_moved": bool(events),
        "start_fit_px": None if ref is None else round(displacement_px(ref[0], ref[1]), 1),
        "extra_error_m": {z: None if v is None else round(v * worst, 2) for z, v in m_px.items()}
        if events
        else None,
        "unusable_checkpoints": sum(not r["usable"] for r in rows),
        "note": "court re-fitted to a background at each checkpoint and compared with the "
        f"first usable one; moved = >= {MOVED_PX} px",
    }


def check_camera(video: str | Path, cal: CourtCalibration, every_s: float = CHECK_EVERY_S) -> dict:
    """Read backgrounds from the video every `every_s` and check the calibration on each."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    backgrounds = []
    try:
        for t in np.arange(min(WINDOW_S / 2, n / fps / 2), n / fps, every_s):
            centre = round(t * fps)
            span = round(WINDOW_S / 2 * fps)
            frames = []
            for f in np.linspace(max(0, centre - span), min(n - 1, centre + span), FRAMES).astype(
                int
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, img = cap.read()
                if ok:
                    frames.append(img)
            if len(frames) >= FRAMES // 2:
                backgrounds.append((float(t), np.median(np.stack(frames), axis=0).astype(np.uint8)))
    finally:
        cap.release()
    return check_backgrounds(backgrounds, cal)
