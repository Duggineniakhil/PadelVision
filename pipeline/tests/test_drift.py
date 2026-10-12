import cv2
import numpy as np
from conftest import SIZE
from test_auto_calibration import _court_image, _true

from padelvision.court.drift import check_backgrounds


def _bumped(img: np.ndarray, dx: float, dy: float, rot: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((SIZE[0] / 2, SIZE[1] / 2), rot, 1.0)
    m[:, 2] += [dx, dy]
    return cv2.warpAffine(img, m, SIZE, borderMode=cv2.BORDER_REPLICATE)


def test_steady_camera_is_not_reported():
    img = _court_image()
    r = check_backgrounds([(0.0, img), (60.0, img), (120.0, img)], _true())
    assert not r["camera_moved"] and r["extra_error_m"] is None
    assert all(c["usable"] and c["moved_px"] < 1.0 for c in r["checkpoints"])


def test_knocked_camera_is_found_with_its_size_and_time():
    img = _court_image()
    bgs = [
        (0.0, img),
        (60.0, img),
        (120.0, _bumped(img, 12, 8, 0.8)),
        (180.0, _bumped(img, 12, 8, 0.8)),
    ]
    r = check_backgrounds(bgs, _true())
    assert r["camera_moved"]
    assert r["moved"] == [{"between_s": [60.0, 120.0], "moved_px": r["checkpoints"][2]["moved_px"]}]
    assert 10 <= r["checkpoints"][2]["moved_px"] <= 20
    assert r["extra_error_m"]["near_half"] > 0


def test_unusable_background_is_skipped():
    img = _court_image()
    covered = np.full_like(img, 40)  # something in front of the lens
    r = check_backgrounds([(0.0, img), (60.0, covered), (120.0, img)], _true())
    assert [c["usable"] for c in r["checkpoints"]] == [True, False, True]
    assert not r["camera_moved"] and r["unusable_checkpoints"] == 1
