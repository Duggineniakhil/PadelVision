"""Mapping between image pixels and court metres.

image pixel --(lens undistort)--> undistorted pixel --(homography)--> court metres

Calibration constraints can be named keypoints (a clicked point with known court position)
and/or named court lines (points clicked anywhere along e.g. the near service line). Lines
help a lot when corners are out of frame or hidden: each line counts like one point.

The homography is only valid for points ON THE GROUND (player feet, ball bounces).
Never use it for an airborne ball.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from padelvision.court.geometry import COURT_KEYPOINTS, NAMED_LINES
from padelvision.court.lens import LensModel, fit_lens

Pts = list[tuple[float, float]]


@dataclass
class CourtCalibration:
    image_size: tuple[int, int]
    image_points: dict[str, tuple[float, float]]
    lens: LensModel
    image_to_court: np.ndarray = field(repr=False)
    court_to_image: np.ndarray = field(repr=False)
    lines: list[Pts] = field(default_factory=list, repr=False)
    named_lines: dict[str, Pts] = field(default_factory=dict, repr=False)
    line_rms_px: float | None = None

    @classmethod
    def from_points(
        cls,
        image_points: dict[str, tuple[float, float]],
        image_size: tuple[int, int],
        lines: list[Pts] | None = None,
        lens: LensModel | None = None,
        named_lines: dict[str, Pts] | None = None,
    ) -> CourtCalibration:
        """Fit from raw pixel clicks.

        image_points: named keypoints (see COURT_KEYPOINTS).
        named_lines:  points along named court lines (see NAMED_LINES), >= 2 points each.
        lines:        points along any other real-world straight lines (lens fitting only).
        Needs keypoints + named lines >= 4, with no 3 keypoints collinear on their own.
        Lens distortion is fitted from all lines with >= 3 points unless `lens` is given.
        """
        named_lines = {k: _pts(v) for k, v in (named_lines or {}).items()}
        lines = [_pts(line) for line in (lines or [])]
        unknown = set(image_points) - set(COURT_KEYPOINTS)
        unknown |= {f"line:{n}" for n in set(named_lines) - set(NAMED_LINES)}
        if unknown:
            raise ValueError(f"Unknown court keypoints/lines: {sorted(unknown)}")
        if len(image_points) + len(named_lines) < 4:
            raise ValueError(
                f"Need at least 4 keypoints/lines, got {len(image_points) + len(named_lines)}"
            )

        w, h = image_size
        line_rms = None
        if lens is None:
            lens_lines = [np.array(p) for p in [*lines, *named_lines.values()] if len(p) >= 3]
            if len(lens_lines) >= 2:
                lens, line_rms = fit_lens(lens_lines, w, h)
            else:
                lens = LensModel.identity(w, h)

        court_to_image = _fit_homography(
            {n: lens.undistort(np.array([p]))[0] for n, p in image_points.items()},
            {n: lens.undistort(np.array(p)) for n, p in named_lines.items()},
        )
        if line_rms is not None:
            # Lens was fitted from line straightness alone; now refine lens and court
            # together so the keypoints can correct the lens too.
            lens, court_to_image, line_rms = _joint_refine(
                lens, court_to_image, image_points, named_lines, lines
            )
        return cls(
            image_size=(int(w), int(h)),
            image_points={n: (float(x), float(y)) for n, (x, y) in image_points.items()},
            lens=lens,
            image_to_court=np.linalg.inv(court_to_image),
            court_to_image=court_to_image,
            lines=lines,
            named_lines=named_lines,
            line_rms_px=line_rms,
        )

    def to_court(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) raw image pixels -> (N, 2) court metres (ground plane only)."""
        return _apply(self.image_to_court, self.lens.undistort(pixels))

    def to_image(self, metres: np.ndarray) -> np.ndarray:
        """(N, 2) court metres -> (N, 2) raw image pixels (NaN if not visible to the camera)."""
        pts = np.asarray(metres, dtype=np.float64).reshape(-1, 2)
        homog = np.c_[pts, np.ones(len(pts))] @ self.court_to_image.T
        w = homog[:, 2:]
        # Same sign as the image of the court centre = in front of the camera.
        front = np.sign(self.court_to_image[2, 2])
        px = np.where(w * front > 1e-9, homog[:, :2] / np.where(w == 0, 1e-12, w), np.nan)
        return self.lens.distort(px)

    def point_errors_m(self) -> dict[str, float]:
        """Per constraint, in metres: keypoint distance from its true position, or for a named
        line ("line:<name>") the mean distance of its clicked points from the true line."""
        errors = {}
        for name, p in self.image_points.items():
            proj = self.to_court(np.array([p]))[0]
            errors[name] = float(np.linalg.norm(proj - COURT_KEYPOINTS[name]))
        for name, pts in self.named_lines.items():
            a, b, c = NAMED_LINES[name]
            xy = self.to_court(np.array(pts))
            errors[f"line:{name}"] = float(np.mean(np.abs(a * xy[:, 0] + b * xy[:, 1] + c)))
        return errors

    def reprojection_error_m(self) -> float:
        """Mean constraint error in metres. Only meaningful with more than 4 constraints."""
        return float(np.mean(list(self.point_errors_m().values())))

    def to_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "image_points": {n: list(p) for n, p in self.image_points.items()},
            "named_lines": {n: [list(p) for p in pts] for n, pts in self.named_lines.items()},
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
            named_lines=data.get("named_lines"),
        )
        cal.line_rms_px = data.get("line_rms_px")
        return cal


def _pts(points) -> Pts:
    return [(float(x), float(y)) for x, y in points]


def _fit_homography(points: dict[str, np.ndarray], lines: dict[str, np.ndarray]) -> np.ndarray:
    """Court -> undistorted image homography from point and line correspondences (DLT).

    Point: image p ~ H c.  Line: court line L ~ H^T l (l = image line through the points).
    Coordinates are normalised (Hartley) for numerical stability.
    """
    img_pts = [p for p in points.values()] + [p for pts in lines.values() for p in pts]
    centre = np.mean(img_pts, axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(np.array(img_pts) - centre, axis=1))
    t_img = np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])
    t_court = np.diag([0.1, 0.1, 1.0])  # metres -> ~unit range

    rows = []
    for name, p in points.items():
        c = t_court @ [*COURT_KEYPOINTS[name], 1.0]
        rows += _cross_rows(t_img @ [*p, 1.0], _point_coeffs(c))
    for name, pts in lines.items():
        if len(pts) < 2:
            raise ValueError(f"Line '{name}' needs at least 2 points")
        l_img = np.linalg.inv(t_img).T @ _fit_line(pts)
        l_court = np.linalg.inv(t_court).T @ np.array(NAMED_LINES[name])
        rows += _cross_rows(l_court / np.linalg.norm(l_court[:2]), _line_coeffs(l_img))

    _, s, vt = np.linalg.svd(np.array(rows))
    if s[-2] < 1e-9 * s[0]:
        raise ValueError("Calibration is degenerate; add constraints (are 3 points collinear?)")
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_img) @ h_norm @ t_court
    h = h / h[2, 2]
    if np.linalg.cond(h) > 1e12:
        raise ValueError("Calibration is degenerate; add constraints")
    return _refine(h, points, lines)


def _refine(h0: np.ndarray, points: dict[str, np.ndarray], lines: dict[str, np.ndarray]):
    """Minimise pixel distances (points to projected keypoints, line points to projected
    court lines), starting from the algebraic DLT solution."""
    from scipy.optimize import least_squares

    p_obs = np.array(list(points.values())).reshape(-1, 2)
    p_court = np.array([[*COURT_KEYPOINTS[n], 1.0] for n in points]).reshape(-1, 3)
    line_data = [(np.array(NAMED_LINES[n]), np.asarray(pts)) for n, pts in lines.items()]

    def residuals(x):
        h = np.append(x, 1.0).reshape(3, 3)
        res = []
        if len(p_obs):
            proj = p_court @ h.T
            res.append((proj[:, :2] / proj[:, 2:] - p_obs).ravel())
        h_inv_t = np.linalg.inv(h).T
        for court_line, pts in line_data:
            l_img = h_inv_t @ court_line
            # Each line weighs like one point (2 residuals), however many points were clicked.
            weight = np.sqrt(2 / len(pts))
            res.append(weight * (pts @ l_img[:2] + l_img[2]) / np.linalg.norm(l_img[:2]))
        return np.concatenate(res)

    n_res = 2 * len(p_obs) + sum(len(p) for _, p in line_data)
    if n_res <= 8:  # exactly determined: nothing to refine
        return h0
    sol = least_squares(residuals, h0.ravel()[:8], method="lm")
    return np.append(sol.x, 1.0).reshape(3, 3)


def _joint_refine(lens: LensModel, h0: np.ndarray, image_points, named_lines, lines):
    """Jointly refine k1, k2 and the homography against every constraint (in undistorted
    pixels). Each line weighs like one point regardless of how many points it has."""
    from scipy.optimize import least_squares

    from padelvision.court.lens import line_residuals

    p_obs = np.array([image_points[n] for n in image_points]).reshape(-1, 2)
    p_court = np.array([[*COURT_KEYPOINTS[n], 1.0] for n in image_points]).reshape(-1, 3)
    named = [(np.array(NAMED_LINES[n]), np.array(p)) for n, p in named_lines.items()]
    free = [np.array(line) for line in lines if len(line) >= 3]

    def build(x):
        return (
            LensModel(x[0], x[1], lens.f, lens.cx, lens.cy),
            np.append(x[2:], 1.0).reshape(3, 3),
        )

    def residuals(x):
        lz, h = build(x)
        res = []
        if len(p_obs):
            proj = p_court @ h.T
            res.append((proj[:, :2] / proj[:, 2:] - lz.undistort(p_obs)).ravel())
        h_inv_t = np.linalg.inv(h).T
        for court_line, pts in named:
            l_img = h_inv_t @ court_line
            u = lz.undistort(pts)
            res.append(
                np.sqrt(2 / len(pts)) * (u @ l_img[:2] + l_img[2]) / np.linalg.norm(l_img[:2])
            )
        for pts in free:
            res.append(np.sqrt(2 / len(pts)) * line_residuals(lz.undistort(pts)))
        return np.concatenate(res)

    x0 = np.r_[lens.k1, lens.k2, (h0 / h0[2, 2]).ravel()[:8]]
    if len(residuals(x0)) <= len(x0):
        return lens, h0, None
    sol = least_squares(residuals, x0, method="lm")
    new_lens, h = build(sol.x)
    all_lines = [p for _, p in named if len(p) >= 3] + free
    straight = np.concatenate([line_residuals(new_lens.undistort(p)) for p in all_lines])
    return new_lens, h, float(np.sqrt(np.mean(straight**2)))


def _point_coeffs(c: np.ndarray) -> list[np.ndarray]:
    """Coefficient vectors over h (row-major) for each component of H @ c."""
    out = []
    for i in range(3):
        v = np.zeros(9)
        v[3 * i : 3 * i + 3] = c
        out.append(v)
    return out


def _line_coeffs(l_img: np.ndarray) -> list[np.ndarray]:
    """Coefficient vectors over h for each component of H^T @ l."""
    out = []
    for j in range(3):
        v = np.zeros(9)
        v[[j, 3 + j, 6 + j]] = l_img
        out.append(v)
    return out


def _cross_rows(a: np.ndarray, m: list[np.ndarray]) -> list[np.ndarray]:
    """Rows of a x (M h) = 0."""
    return [
        a[1] * m[2] - a[2] * m[1],
        a[2] * m[0] - a[0] * m[2],
        a[0] * m[1] - a[1] * m[0],
    ]


def _fit_line(pts: np.ndarray) -> np.ndarray:
    """Homogeneous image line (unit normal) best fitting the points."""
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre)
    n = vt[1]
    return np.array([n[0], n[1], -n @ centre])


def _apply(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)
