import cv2
import numpy as np
import pytest

from padelvision.video import probe


def test_probe_reads_real_fps(tmp_path):
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(15):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()

    info = probe(path)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(30.0)
    assert info.frame_count == 15
    assert info.duration_s == pytest.approx(0.5)


def test_probe_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Cannot open"):
        probe(tmp_path / "nope.mp4")
