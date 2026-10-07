"""Radial lens distortion (wide-angle / GoPro / phone cameras).

Division model, in normalised coordinates x = (u - cx) / f, distortion centre at the image
centre and focal length fixed at the image width (k1/k2 absorb the difference):

    x_undistorted = x_distorted / (1 + k1 * r^2 + k2 * r^4),   r = |x_distorted|

Barrel distortion has k1 < 0. The division model handles the strong distortion of action
cameras far better than the polynomial (Brown) model; on the developer's GoPro-style
footage the polynomial model left 6.7 px of curvature in the service line, this one 1.2 px.

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
        r2 = (xd**2).sum(axis=1)
        return self._denormalise(xd / (1 + self.k1 * r2 + self.k2 * r2**2)[:, None])

    def distort(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) undistorted pixels -> (N, 2) distorted image pixels.

        Points far outside the camera's field of view may have no valid distorted position;
        those come back as NaN.
        """
        xu = self._normalise(pixels)
        ru = np.linalg.norm(xu, axis=1)
        rd = ru.copy()
        for _ in range(50):  # Newton on r_d / (1 + k1 r_d^2 + k2 r_d^4) = r_u
            d = 1 + self.k1 * rd**2 + self.k2 * rd**4
            dd = 2 * self.k1 * rd + 4 * self.k2 * rd**3
            g = rd / d - ru
            dg = (d - rd * dd) / d**2
            rd = rd - g / np.where(np.abs(dg) < 1e-9, 1e-9, dg)
        d = 1 + self.k1 * rd**2 + self.k2 * rd**4
        bad = (np.abs(rd / d - ru) > 1e-6) | (rd < 0) | (d <= 0)
        scale = np.divide(rd, ru, out=np.ones_like(ru), where=ru > 1e-12)
        out = self._denormalise(xu * scale[:, None])
        out[bad] = np.nan
        return out

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

    sol = least_squares(residuals, x0=[0.0, 0.0], bounds=([-3.0, -3.0], [3.0, 3.0]))
    lens = LensModel(float(sol.x[0]), float(sol.x[1]), base.f, base.cx, base.cy)
    rms = float(np.sqrt(np.mean(residuals(sol.x) ** 2)))
    return lens, rms
