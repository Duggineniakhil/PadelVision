"""Mapping between image pixels and court metres via a ground-plane homography.

The homography is only valid for points ON THE GROUND (player feet, ball bounces).
Never use it for an airborne ball.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from padelvision.court.geometry import COURT_KEYPOINTS


@dataclass
class CourtCalibration:
    image_points: dict[str, tuple[float, float]]
    image_to_court: np.ndarray = field(repr=False)
    court_to_image: np.ndarray = field(repr=False)

    @classmethod
    def from_points(cls, image_points: dict[str, tuple[float, float]]) -> CourtCalibration:
        """Fit from named image points (pixel coords). Needs >= 4 known keypoints."""
        unknown = set(image_points) - set(COURT_KEYPOINTS)
        if unknown:
            raise ValueError(f"Unknown court keypoints: {sorted(unknown)}")
        if len(image_points) < 4:
            raise ValueError(f"Need at least 4 keypoints, got {len(image_points)}")

        names = sorted(image_points)
        src = np.array([image_points[n] for n in names], dtype=np.float64)
        dst = np.array([COURT_KEYPOINTS[n] for n in names], dtype=np.float64)
        h, _ = cv2.findHomography(src, dst, method=0)
        if h is None:
            raise ValueError("Could not fit a homography; are the points collinear?")
        return cls(
            image_points={n: (float(x), float(y)) for n, (x, y) in image_points.items()},
            image_to_court=h,
            court_to_image=np.linalg.inv(h),
        )

    def to_court(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) image pixels -> (N, 2) court metres."""
        return _apply(self.image_to_court, pixels)

    def to_image(self, metres: np.ndarray) -> np.ndarray:
        """(N, 2) court metres -> (N, 2) image pixels."""
        return _apply(self.court_to_image, metres)

    def reprojection_error_m(self) -> float:
        """Mean distance (metres) between the clicked points and their true court positions."""
        names = sorted(self.image_points)
        projected = self.to_court(np.array([self.image_points[n] for n in names]))
        truth = np.array([COURT_KEYPOINTS[n] for n in names])
        return float(np.linalg.norm(projected - truth, axis=1).mean())

    def save(self, path: str | Path) -> None:
        data = {
            "image_points": self.image_points,
            "image_to_court": self.image_to_court.tolist(),
            "reprojection_error_m": self.reprojection_error_m(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CourtCalibration:
        """Load a court.json. Only `image_points` is required; the homography is refit."""
        data = json.loads(Path(path).read_text())
        points = {n: tuple(p) for n, p in data["image_points"].items()}
        return cls.from_points(points)


def _apply(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)
