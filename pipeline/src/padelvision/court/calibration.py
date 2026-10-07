"""Mapping between image pixels and court metres.

image pixel --(lens undistort)--> undistorted pixel --(homography)--> court metres

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
from padelvision.court.lens import LensModel, fit_lens


@dataclass
class CourtCalibration:
    image_size: tuple[int, int]
    image_points: dict[str, tuple[float, float]]
    lens: LensModel
    image_to_court: np.ndarray = field(repr=False)
    court_to_image: np.ndarray = field(repr=False)
    lines: list[list[tuple[float, float]]] = field(default_factory=list, repr=False)
    line_rms_px: float | None = None

    @classmethod
    def from_points(
        cls,
        image_points: dict[str, tuple[float, float]],
        image_size: tuple[int, int],
        lines: list[list[tuple[float, float]]] | None = None,
        lens: LensModel | None = None,
    ) -> CourtCalibration:
        """Fit from named image points (raw pixel clicks). Needs >= 4 known keypoints.

        `lines`: point lists clicked along straight real-world lines, used to fit lens
        distortion. Without lines (and no explicit `lens`), no distortion is assumed.
        """
        unknown = set(image_points) - set(COURT_KEYPOINTS)
        if unknown:
            raise ValueError(f"Unknown court keypoints: {sorted(unknown)}")
        if len(image_points) < 4:
            raise ValueError(f"Need at least 4 keypoints, got {len(image_points)}")

        w, h = image_size
        lines = [[(float(x), float(y)) for x, y in line] for line in (lines or [])]
        line_rms = None
        if lens is None:
            if lines:
                lens, line_rms = fit_lens([np.array(line) for line in lines], w, h)
            else:
                lens = LensModel.identity(w, h)

        names = sorted(image_points)
        src = lens.undistort(np.array([image_points[n] for n in names], dtype=np.float64))
        dst = np.array([COURT_KEYPOINTS[n] for n in names], dtype=np.float64)
        hom, _ = cv2.findHomography(src, dst, method=0)
        if hom is None:
            raise ValueError("Could not fit a homography; are the points collinear?")
        return cls(
            image_size=(int(w), int(h)),
            image_points={n: (float(x), float(y)) for n, (x, y) in image_points.items()},
            lens=lens,
            image_to_court=hom,
            court_to_image=np.linalg.inv(hom),
            lines=lines,
            line_rms_px=line_rms,
        )

    def to_court(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) raw image pixels -> (N, 2) court metres (ground plane only)."""
        return _apply(self.image_to_court, self.lens.undistort(pixels))

    def to_image(self, metres: np.ndarray) -> np.ndarray:
        """(N, 2) court metres -> (N, 2) raw image pixels."""
        return self.lens.distort(_apply(self.court_to_image, metres))

    def point_errors_m(self) -> dict[str, float]:
        """Per keypoint: distance (metres) between the clicked point and its true position."""
        names = sorted(self.image_points)
        projected = self.to_court(np.array([self.image_points[n] for n in names]))
        truth = np.array([COURT_KEYPOINTS[n] for n in names])
        errors = np.linalg.norm(projected - truth, axis=1)
        return {n: float(e) for n, e in zip(names, errors, strict=True)}

    def reprojection_error_m(self) -> float:
        """Mean keypoint error in metres. Only meaningful with more than 4 keypoints."""
        return float(np.mean(list(self.point_errors_m().values())))

    def to_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "image_points": {n: list(p) for n, p in self.image_points.items()},
            "lines": [[list(p) for p in line] for line in self.lines],
            "lens": self.lens.to_dict(),
            "line_rms_px": self.line_rms_px,
            "image_to_court": self.image_to_court.tolist(),
            "reprojection_error_m": self.reprojection_error_m(),
            "point_errors_m": self.point_errors_m(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CourtCalibration:
        """Load a court.json. The homography is refit from the stored clicks."""
        data = json.loads(Path(path).read_text())
        lens = LensModel(**data["lens"]) if data.get("lens") else None
        cal = cls.from_points(
            {n: tuple(p) for n, p in data["image_points"].items()},
            image_size=tuple(data["image_size"]),
            lines=data.get("lines"),
            lens=lens,
        )
        cal.line_rms_px = data.get("line_rms_px")
        return cal


def _apply(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)
