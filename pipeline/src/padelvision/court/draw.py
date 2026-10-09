"""Drawing court overlays on frames and a top-down mini court."""

from __future__ import annotations

import cv2
import numpy as np

from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import COURT_KEYPOINTS, COURT_LENGTH_M, COURT_LINES, COURT_WIDTH_M

TEAM_COLORS = {"near": (80, 200, 80), "far": (60, 140, 255)}  # BGR
ROUNDTRIP_TOL_M = 0.25


def draw_court_overlay(
    frame: np.ndarray, cal: CourtCalibration, color=(0, 255, 255), thickness: int = 2
) -> np.ndarray:
    """Project the court lines into the image (curved where the lens distorts)."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for (x0, y0), (x1, y1) in COURT_LINES:
        t = np.linspace(0, 1, 120)[:, None]
        court = np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]
        pts = cal.to_image(court)
        # Draw only parts that project sensibly: not behind the camera, inside the lens model's
        # valid range, and mapping back to the same court point (out-of-view parts of strongly
        # distorted lenses can land at bogus image positions).
        ok = np.isfinite(pts).all(axis=1) & (np.abs(pts - [w / 2, h / 2]) < [w, h]).all(axis=1)
        if ok.any():
            back = np.full_like(court, np.nan)
            back[ok] = cal.to_court(pts[ok])
            ok &= np.linalg.norm(back - court, axis=1) < ROUNDTRIP_TOL_M
        for run in np.split(np.arange(len(pts)), np.nonzero(np.diff(ok.astype(int)))[0] + 1):
            if ok[run[0]] and len(run) > 1:
                seg = pts[run].round().astype(np.int32)
                cv2.polylines(out, [seg], False, color, thickness, cv2.LINE_AA)
    for name, (px, py) in cal.image_points.items():
        cv2.circle(out, (round(px), round(py)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        proj = cal.to_image(np.array([COURT_KEYPOINTS[name]]))[0]
        if np.isfinite(proj).all():
            cv2.circle(out, (round(proj[0]), round(proj[1])), 9, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class MiniCourt:
    """Top-down court canvas. Near baseline (y = -10) is drawn at the bottom."""

    def __init__(self, height_px: int = 300, pad_px: int = 12):
        self.scale = (height_px - 2 * pad_px) / COURT_LENGTH_M
        self.pad = pad_px
        self.size = (round(COURT_WIDTH_M * self.scale) + 2 * pad_px, height_px)

    def to_px(self, xy_m) -> tuple[int, int]:
        x, y = xy_m
        u = self.pad + (x + COURT_WIDTH_M / 2) * self.scale
        v = self.pad + (COURT_LENGTH_M / 2 - y) * self.scale
        return round(u), round(v)

    def canvas(self) -> np.ndarray:
        w, h = self.size
        img = np.full((h, w, 3), (90, 50, 20), dtype=np.uint8)
        for p0, p1 in COURT_LINES:
            cv2.line(img, self.to_px(p0), self.to_px(p1), (235, 235, 235), 1, cv2.LINE_AA)
        return img
