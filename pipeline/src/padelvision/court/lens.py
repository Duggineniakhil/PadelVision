"""Radial lens distortion (wide-angle / GoPro / phone cameras).

Model (Brown, radial only), in normalised coordinates x = (u - cx) / f:
    x_distorted = x_undistorted * (1 + k1 * r^2 + k2 * r^4),  r = |x_undistorted|

The focal length is unknown and fixed at the image width; k1/k2 absorb the difference.
Parameters are fitted from straight lines in the scene: after undistortion, points
clicked along one real-world straight line must be collinear. No checkerboard needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LensModel:
    k1: float
    k2: float
    f: float
    cx: float
    cy: float

    @classmethod
    def identity(cls, width: int, height: int) -> LensModel:
        return cls(0.0, 0.0, float(width), width / 2, height / 2)

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) distorted image pixels -> (N, 2) undistorted pixels."""
        xd = self._normalise(pixels)
        rd = np.linalg.norm(xd, axis=1)
        ru = rd.copy()
        for _ in range(30):  # Newton on r_u * (1 + k1 r_u^2 + k2 r_u^4) = r_d
            g = ru * (1 + self.k1 * ru**2 + self.k2 * ru**4) - rd
            dg = 1 + 3 * self.k1 * ru**2 + 5 * self.k2 * ru**4
            ru = ru - g / np.where(np.abs(dg) < 1e-6, 1e-6, dg)
        scale = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 1e-12)
        return self._denormalise(xd * scale[:, None])

    def distort(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) undistorted pixels -> (N, 2) distorted image pixels."""
        xu = self._normalise(pixels)
        r2 = (xu**2).sum(axis=1)
        return self._denormalise(xu * (1 + self.k1 * r2 + self.k2 * r2**2)[:, None])

    def to_dict(self) -> dict:
        return asdict(self)

    def _normalise(self, pixels: np.ndarray) -> np.ndarray:
        p = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        return (p - [self.cx, self.cy]) / self.f

    def _denormalise(self, x: np.ndarray) -> np.ndarray:
        return x * self.f + [self.cx, self.cy]


def line_residuals(points: np.ndarray) -> np.ndarray:
    """Perpendicular distances of points from their best-fit line."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[1]


def fit_lens(lines: list[np.ndarray], width: int, height: int) -> tuple[LensModel, float]:
    """Fit k1, k2 so every clicked line becomes straight.

    Returns the model and the RMS straightness error in pixels after correction.
    Lines through the image centre carry no information; long lines near the edges do.
    """
    from scipy.optimize import least_squares

    lines = [np.asarray(line, dtype=np.float64) for line in lines if len(line) >= 3]
    if len(lines) < 2:
        raise ValueError("Need at least 2 lines with >= 3 points each to fit lens distortion")

    base = LensModel.identity(width, height)

    def residuals(k: np.ndarray) -> np.ndarray:
        lens = LensModel(k[0], k[1], base.f, base.cx, base.cy)
        return np.concatenate([line_residuals(lens.undistort(line)) for line in lines])

    sol = least_squares(residuals, x0=[0.0, 0.0], bounds=([-1.0, -1.0], [1.0, 1.0]))
    lens = LensModel(float(sol.x[0]), float(sol.x[1]), base.f, base.cx, base.cy)
    rms = float(np.sqrt(np.mean(residuals(sol.x) ** 2)))
    return lens, rms
