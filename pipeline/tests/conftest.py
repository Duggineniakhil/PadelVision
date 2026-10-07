"""Synthetic camera used across tests: a known homography plus optional lens distortion."""

import numpy as np
import pytest

from padelvision.court import COURT_KEYPOINTS, CourtCalibration, LensModel
from padelvision.court.geometry import COURT_LINES

SIZE = (1280, 720)
# Court metres -> undistorted pixels for a camera behind the near baseline.
TRUE_H = np.array(
    [
        [60.0, 22.4, 640.0],
        [0.0, -12.0, 330.0],
        [0.0, 0.035, 1.0],
    ]
)
TRUE_LENS = LensModel(k1=-0.6, k2=0.15, f=1280.0, cx=640.0, cy=360.0)


def project(court_pts, lens: LensModel | None = None) -> np.ndarray:
    pts = np.asarray(court_pts, dtype=np.float64).reshape(-1, 2)
    homog = np.c_[pts, np.ones(len(pts))] @ TRUE_H.T
    px = homog[:, :2] / homog[:, 2:]
    return lens.distort(px) if lens else px


def clicks(names, lens: LensModel | None = None) -> dict[str, tuple[float, float]]:
    court = np.array([COURT_KEYPOINTS[n] for n in names])
    return {n: tuple(p) for n, p in zip(names, project(court, lens), strict=True)}


def line_clicks(lens: LensModel, n: int = 8) -> list[list[tuple[float, float]]]:
    lines = []
    for (x0, y0), (x1, y1) in COURT_LINES:
        t = np.linspace(0.05, 0.95, n)[:, None]
        pts = project(np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], lens)
        lines.append([tuple(p) for p in pts])
    return lines


def true_positions(t: float) -> dict[int, tuple[float, float]]:
    """Ground truth court positions (metres) of 4 moving players at time t."""
    return {
        1: (-2.5 + 1.5 * np.sin(t), -7.0 + 0.5 * t),  # near left, moving forward
        2: (2.5 + np.sin(0.7 * t), -8.0),
        3: (-2.0 + 0.3 * t, 3.0),  # far side, walks right, crosses partner's x
        4: (2.0 - 0.3 * t, 6.0),
    }


def synthetic_frames(fps: float = 30.0, seconds: float = 10.0, stride: int = 1):
    """FrameDetections for 4 players + a spectator + a duplicate box + a tracker id switch."""
    from padelvision.models.person_tracker import FrameDetections

    for frame in range(0, round(fps * seconds), stride):
        t = frame / fps
        people = []  # (court_xy, conf, track_id)
        for p, xy in true_positions(t).items():
            tid = 10 + p
            if p == 1 and t >= 5.0:
                tid = 99  # tracker lost player 1 and gave a new id
            people.append((xy, 0.9, tid))
        people.append(((7.5, 0.0), 0.95, 50))  # spectator outside the glass
        dup = true_positions(t)[2]
        people.append(((dup[0] + 0.1, dup[1]), 0.5, -1))  # duplicate box on player 2
        feet = project([xy for xy, _, _ in people])
        boxes = np.c_[feet[:, 0] - 15, feet[:, 1] - 80, feet[:, 0] + 15, feet[:, 1]]
        yield FrameDetections(
            frame=frame,
            boxes=boxes,
            conf=np.array([c for _, c, _ in people]),
            track_ids=np.array([tid for _, _, tid in people]),
        )


@pytest.fixture
def cal() -> CourtCalibration:
    """Exact calibration of the synthetic camera (no lens distortion)."""
    return CourtCalibration.from_points(clicks(COURT_KEYPOINTS), SIZE)
