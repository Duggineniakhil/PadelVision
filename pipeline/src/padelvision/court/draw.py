"""Drawing court overlays on frames and a top-down mini court."""

from __future__ import annotations

import cv2
import numpy as np

from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import COURT_KEYPOINTS, COURT_LENGTH_M, COURT_LINES, COURT_WIDTH_M

TEAM_COLORS = {"near": (80, 200, 80), "far": (60, 140, 255)}  # BGR


def draw_court_overlay(
    frame: np.ndarray, cal: CourtCalibration, color=(0, 255, 255), thickness: int = 2
) -> np.ndarray:
    """Project the court lines into the image (curved where the lens distorts)."""
    out = frame.copy()
    for (x0, y0), (x1, y1) in COURT_LINES:
        t = np.linspace(0, 1, 60)[:, None]
        pts = cal.to_image(np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
        cv2.polylines(out, [pts.round().astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    for name, (px, py) in cal.image_points.items():
        cv2.circle(out, (round(px), round(py)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        proj = cal.to_image(np.array([COURT_KEYPOINTS[name]]))[0]
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
