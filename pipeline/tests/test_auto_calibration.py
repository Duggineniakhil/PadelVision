import cv2
import numpy as np
from conftest import SIZE, project

from padelvision.court import COURT_KEYPOINTS, CourtCalibration
from padelvision.court.auto import ANCHORS, evidence, refine, scaled, search
from padelvision.court.geometry import COURT_LINES


def _court_image() -> np.ndarray:
    """The synthetic camera's view: grey walls, a blue floor, white lines, a dark see-through
    net mesh standing on the net line with a white band on top."""
    w, h = SIZE
    img = np.full((h, w, 3), (70, 70, 70), np.uint8)

    def poly(pts):
        return project(pts).round().astype(np.int32)

    t = np.linspace(0, 1, 50)[:, None]
    floor = np.r_[
        np.c_[-5 + 10 * t, np.full(50, -10.0)],
        np.c_[np.full(50, 5.0), -10 + 20 * t],
        np.c_[5 - 10 * t, np.full(50, 10.0)],
        np.c_[np.full(50, -5.0), 10 - 20 * t],
    ]
    cv2.fillPoly(img, [poly(floor)], (150, 85, 35))
    for (x0, y0), (x1, y1) in COURT_LINES[4:7]:  # painted: service lines and centre line
        cv2.polylines(img, [poly(np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])], False,
                      (235, 235, 235), 2, cv2.LINE_AA)  # fmt: skip
    base = poly(np.c_[-5 + 10 * t, np.zeros(50)])
    top = base - [0, 28]  # the net stands ~28 px tall here
    mesh = np.zeros(img.shape[:2], np.uint8)  # a see-through mesh: the floor shows through it
    cv2.fillPoly(mesh, [np.r_[base, top[::-1]]], 1)
    grid = np.zeros_like(mesh)
    grid[::3, :] = grid[:, ::3] = 1
    img[(mesh & grid).astype(bool)] = (25, 25, 25)
    cv2.polylines(img, [top], False, (240, 240, 240), 3, cv2.LINE_AA)
    return img


def _true() -> CourtCalibration:
    names = ("near_service_left", "near_service_right", "far_service_left", "far_service_right",
             "far_left", "far_right", "net_left", "net_right")  # fmt: skip
    court = np.array([COURT_KEYPOINTS[n] for n in names])
    return CourtCalibration.from_points(
        dict(zip(names, map(tuple, project(court)), strict=True)), SIZE
    )


def test_finds_the_court_from_a_shifted_template():
    img = _court_image()
    true = _true()
    lens, anchors = scaled(true, SIZE)
    start = np.array([anchors[n] for n in ANCHORS]) + [[30, 18], [22, 25], [18, -12], [25, -8]]
    ev = evidence(img)
    a, _ = search(ev, lens, start, starts=3, grid=1)  # a small grid: the start is close
    rough = CourtCalibration.from_points(
        dict(zip(ANCHORS, map(tuple, a), strict=True)), SIZE, lens=lens
    )
    cal, lines = refine(img, rough, ev)
    assert {"near_service_line", "far_service_line", "center_line", "net_base"} <= set(lines)
    # the found court maps ground points within a few cm of the truth in both halves
    pts = np.array([(x, y) for x in (-3.0, 0.0, 3.0) for y in (-6.0, -2.0, 2.0, 6.0)])
    err = np.hypot(*(cal.to_court(true.to_image(pts)) - pts).T)
    assert err.max() < 0.15
