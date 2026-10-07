import numpy as np
import pytest
from conftest import SIZE, TRUE_LENS, clicks, line_clicks, project

from padelvision.court import COURT_KEYPOINTS, CourtCalibration, LensModel, fit_lens
from padelvision.court.geometry import CORNER_NAMES, in_court, zone_of

SAMPLE_M = np.array([[0.0, 0.0], [2.5, -6.95], [-4.0, 8.0], [4.5, -9.5]])


def test_corners_recover_homography():
    cal = CourtCalibration.from_points(clicks(CORNER_NAMES), SIZE)
    assert cal.reprojection_error_m() < 1e-6
    np.testing.assert_allclose(cal.to_court(project(SAMPLE_M)), SAMPLE_M, atol=1e-6)
    np.testing.assert_allclose(cal.to_image(SAMPLE_M), project(SAMPLE_M), atol=1e-4)


def test_any_four_visible_points_work():
    # Low camera: near corners out of frame; use service-line ends and net posts instead.
    names = ["near_service_left", "near_service_right", "net_left", "net_right", "far_left"]
    cal = CourtCalibration.from_points(clicks(names), SIZE)
    np.testing.assert_allclose(cal.to_court(project(SAMPLE_M)), SAMPLE_M, atol=1e-6)


def test_noisy_clicks_give_small_error():
    rng = np.random.default_rng(0)
    noisy = {n: (x + rng.normal(0, 1.5), y + rng.normal(0, 1.5))
             for n, (x, y) in clicks(COURT_KEYPOINTS).items()}  # fmt: skip
    cal = CourtCalibration.from_points(noisy, SIZE)
    assert cal.reprojection_error_m() < 0.15


def test_lens_distort_undistort_roundtrip():
    px = np.random.default_rng(1).uniform([0, 0], SIZE, size=(200, 2))
    np.testing.assert_allclose(TRUE_LENS.undistort(TRUE_LENS.distort(px)), px, atol=1e-6)
    np.testing.assert_allclose(TRUE_LENS.distort(TRUE_LENS.undistort(px)), px, atol=1e-6)


def test_fit_lens_straightens_lines():
    lines = [np.array(line) for line in line_clicks(TRUE_LENS)]
    lens, rms = fit_lens(lines, *SIZE)
    assert rms < 0.05
    assert lens.k1 == pytest.approx(TRUE_LENS.k1, abs=0.02)


def test_calibration_with_distortion_is_accurate():
    distorted = clicks(COURT_KEYPOINTS, TRUE_LENS)
    without = CourtCalibration.from_points(distorted, SIZE)
    with_lens = CourtCalibration.from_points(distorted, SIZE, lines=line_clicks(TRUE_LENS))

    truth_px = project(SAMPLE_M, TRUE_LENS)
    err_without = np.linalg.norm(without.to_court(truth_px) - SAMPLE_M, axis=1).max()
    err_with = np.linalg.norm(with_lens.to_court(truth_px) - SAMPLE_M, axis=1).max()
    assert err_without > 0.1  # ignoring the lens is visibly wrong
    assert err_with < 0.02
    np.testing.assert_allclose(with_lens.to_image(SAMPLE_M), truth_px, atol=0.5)


def test_save_load_roundtrip(tmp_path):
    cal = CourtCalibration.from_points(
        clicks(COURT_KEYPOINTS, TRUE_LENS), SIZE, lines=line_clicks(TRUE_LENS)
    )
    path = tmp_path / "court.json"
    cal.save(path)
    loaded = CourtCalibration.load(path)
    assert loaded.lens == cal.lens
    assert loaded.image_size == SIZE
    np.testing.assert_allclose(loaded.image_to_court, cal.image_to_court)


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 4"):
        CourtCalibration.from_points(clicks(CORNER_NAMES[:3]), SIZE)
    with pytest.raises(ValueError, match="Unknown"):
        CourtCalibration.from_points({"middle_of_nowhere": (0, 0), **clicks(CORNER_NAMES)}, SIZE)
    with pytest.raises(ValueError, match="2 lines"):
        fit_lens([np.zeros((3, 2))], *SIZE)


def test_identity_lens_is_noop():
    lens = LensModel.identity(*SIZE)
    px = np.array([[0.0, 0.0], [1280.0, 720.0], [300.0, 500.0]])
    np.testing.assert_allclose(lens.undistort(px), px)


def test_geometry_helpers():
    assert in_court(0, 0)
    assert in_court(5, -10)
    assert not in_court(5.5, 0)
    assert in_court(5.5, 0, margin_m=1.0)
    np.testing.assert_array_equal(in_court(np.array([0, 6]), np.array([0, 0])), [True, False])
    assert [zone_of(y) for y in (1, -5, 9)] == ["net", "transition", "back"]
    assert set(COURT_KEYPOINTS) >= set(CORNER_NAMES)
