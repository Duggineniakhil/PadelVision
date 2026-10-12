import cv2
import numpy as np
import pandas as pd

from padelvision.players.appearance import FEATURES, distance, extract

FPS = 10.0


def _video(path, frames: int = 20):
    """Two 'players': a white shirt over dark shorts, and a red shirt over white shorts."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 240))
    for _ in range(frames):
        img = np.full((240, 320, 3), (150, 90, 40), np.uint8)  # blue court
        img[40:100, 40:100] = (235, 235, 235)  # player 1: shirt rows 40-100 of box 20-220
        img[100:160, 40:100] = (25, 25, 25)
        img[40:100, 200:260] = (30, 30, 200)  # player 2: red shirt (BGR)
        img[100:160, 200:260] = (235, 235, 235)
        writer.write(img)
    writer.release()


def test_extract_reads_shirt_and_shorts_colours(tmp_path):
    path = tmp_path / "tiny.avi"
    _video(path)
    rows = [
        (f, f / FPS, tid, 0.9, x1, 0, x1 + 80, 240)
        for f in range(20)
        for tid, x1 in ((1, 30), (2, 190))
    ]
    dets = pd.DataFrame(rows, columns=["frame", "t", "track_id", "conf", "x1", "y1", "x2", "y2"])
    app = extract(path, dets, FPS, sample_s=0.5)
    assert sorted(app.frame.unique()) == [0, 5, 10, 15]
    one = app[app.track_id == 1][FEATURES].median().to_numpy()
    two = app[app.track_id == 2][FEATURES].median().to_numpy()
    assert one[0] > 200 and one[3] < 60  # white shirt (L), dark shorts
    assert two[1] > 160 and two[3] > 200  # red shirt (a > 128), white shorts
    assert distance(one, two) > 3
    assert distance(one, one) == 0
