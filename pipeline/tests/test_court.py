import numpy as np
import pytest

from padelvision.court import COURT_KEYPOINTS, CourtCalibration
from padelvision.court.geometry import CORNER_NAMES, in_court

# A plausible behind-the-baseline camera: court metres -> 1280x720 pixels.
TRUE_H = np.array(
    [
        [60.0, 22.4, 640.0],
        [0.0, -12.0, 330.0],
        [0.0, 0.035, 1.0],
    ]
)


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homog = np.c_[pts, np.ones(len(pts))] @ h.T
    return homog[:, :2] / homog[:, 2:]


def _clicks(names) -> dict[str, tuple[float, float]]:
    court = np.array([COURT_KEYPOINTS[n] for n in names])
    return {n: tuple(p) for n, p in zip(names, _project(TRUE_H, court), strict=True)}


def test_corners_recover_homography():
    cal = CourtCalibration.from_points(_clicks(CORNER_NAMES))
    assert cal.reprojection_error_m() < 1e-6

    court_pts = np.array([[0.0, 0.0], [2.5, -6.95], [-4.0, 8.0]])
    pixels = _project(TRUE_H, court_pts)
    np.testing.assert_allclose(cal.to_court(pixels), court_pts, atol=1e-6)
    np.testing.assert_allclose(cal.to_image(court_pts), pixels, atol=1e-4)


def test_noisy_clicks_give_small_error():
    rng = np.random.default_rng(0)
    clicks = _clicks(list(COURT_KEYPOINTS))
    noisy = {n: (x + rng.normal(0, 1.5), y + rng.normal(0, 1.5)) for n, (x, y) in clicks.items()}
    cal = CourtCalibration.from_points(noisy)
    assert cal.reprojection_error_m() < 0.15


def test_save_load_roundtrip(tmp_path):
    cal = CourtCalibration.from_points(_clicks(CORNER_NAMES))
    path = tmp_path / "court.json"
    cal.save(path)
    loaded = CourtCalibration.load(path)
    np.testing.assert_allclose(loaded.image_to_court, cal.image_to_court)


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 4"):
        CourtCalibration.from_points(_clicks(CORNER_NAMES[:3]))
    with pytest.raises(ValueError, match="Unknown"):
        CourtCalibration.from_points({"middle_of_nowhere": (0, 0), **_clicks(CORNER_NAMES)})


def test_in_court():
    assert in_court(0, 0)
    assert in_court(5, -10)
    assert not in_court(5.5, 0)
    assert in_court(5.5, 0, margin_m=1.0)
