import cv2
import numpy as np
import pytest

from padelvision.court.snap import snap_line

FLOOR = (150, 90, 40)  # BGR blue floor


def _line_y(x):
    return 200 + 0.0002 * (x - 320) ** 2  # slightly curved, like a distorted court line


def _image():
    img = np.full((400, 640, 3), FLOOR, np.uint8)
    xs = np.arange(0, 640)
    pts = np.c_[xs, _line_y(xs)].round().astype(np.int32)
    cv2.polylines(img, [pts], False, (235, 235, 235), 3, cv2.LINE_AA)
    img[:120] = (60, 60, 60)  # a grey wall above y = 120
    return img


def test_paint_snaps_onto_the_line_from_a_rough_guide():
    img = _image()
    guide = [(40, _line_y(40) + 6), (320, _line_y(320) - 4), (600, _line_y(600) - 5)]
    pts = snap_line(img, guide, "paint")
    assert len(pts) > 50
    err = [abs(y - _line_y(x)) for x, y in pts]
    assert max(err) < 0.8


def test_paint_drops_points_on_an_occluder():
    img = _image()
    cv2.rectangle(img, (300, 150), (340, 260), (20, 30, 200), -1)  # a player in red
    cv2.rectangle(img, (400, 185), (404, 215), (255, 255, 255), -1)  # a bright post next to it
    pts = snap_line(img, [(40, _line_y(40)), (600, _line_y(600))], "paint")
    assert all(abs(y - _line_y(x)) < 1.0 for x, y in pts)
    assert not any(300 <= x <= 340 for x, _ in pts)


def test_edge_finds_floor_meeting_wall():
    img = _image()
    pts = snap_line(img, [(20, 126), (620, 115)], "edge")
    assert len(pts) > 50
    assert all(abs(y - 119.5) < 0.8 for _, y in pts)


def test_nothing_to_snap_returns_empty():
    img = np.full((200, 200, 3), FLOOR, np.uint8)
    assert snap_line(img, [(10, 100), (190, 100)], "paint") == []
    with pytest.raises(ValueError):
        snap_line(img, [(10, 100)], "paint")
