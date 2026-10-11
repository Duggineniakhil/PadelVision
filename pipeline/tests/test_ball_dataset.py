import cv2
import numpy as np
import pandas as pd
import yaml

from padelvision.ball.dataset import (
    count,
    import_yolo_dataset,
    is_test_frame,
    is_train_frame,
    write_data_yaml,
    write_video_frames,
)


def test_block_split_alternates():
    fps = 30.0
    frames = np.array([0, 599, 600, 1199, 1200])
    assert list(is_test_frame(frames, fps, 20)) == [False, False, True, True, False]
    # train frames keep a 2 s margin from the test blocks on both sides
    assert list(is_train_frame(np.array([30, 60, 300, 540, 570, 900, 1260]), fps, 20, 2.0)) == [
        False,
        True,
        True,
        True,
        False,
        False,
        True,
    ]


def test_write_video_frames(tmp_path):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    for _ in range(10):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    boxes = pd.DataFrame([(2, 160, 120, 16), (2, 10, 20, 10), (7, 32, 24, 12)],
                         columns=["frame", "u", "v", "size"])  # fmt: skip
    assert write_video_frames(video, boxes, tmp_path / "ds", "train") == 2
    lines = (tmp_path / "ds/labels/train/vid_000002.txt").read_text().split("\n")
    assert lines[0] == "0 0.500000 0.500000 0.050000 0.066667"
    assert len([x for x in lines if x]) == 2
    assert (tmp_path / "ds/images/train/vid_000007.jpg").exists()


def test_import_keeps_only_ball_classes(tmp_path):
    src = tmp_path / "rf"
    (src / "train/images").mkdir(parents=True)
    (src / "train/labels").mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"names": ["ball", "player", "racket"]}))
    for i, txt in enumerate(
        ["0 .5 .5 .02 .02\n1 .3 .3 .2 .4\n", "1 .3 .3 .2 .4\n", "2 .1 .1 .1 .1\n0 .7 .7 .02 .02\n"]
    ):
        cv2.imwrite(str(src / f"train/images/img{i}.jpg"), np.zeros((64, 64, 3), np.uint8))
        (src / f"train/labels/img{i}.txt").write_text(txt)
    stats = import_yolo_dataset(src, tmp_path / "ds", "rf", val_fraction=0.0)
    assert stats["kept_classes"] == [0]
    assert (stats["images"], stats["balls"]) == (2, 2)  # img1 has no ball: skipped
    assert (tmp_path / "ds/labels/train/rf_img2.txt").read_text() == "0 .7 .7 .02 .02\n"
    data = write_data_yaml(tmp_path / "ds", [stats])
    assert yaml.safe_load(data.read_text())["names"] == {0: "ball"}
    assert count(tmp_path / "ds") == {"train": 2, "val": 0}


def test_import_without_ball_class_imports_nothing(tmp_path):
    src = tmp_path / "rf"
    (src / "train/labels").mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"names": {0: "player"}}))
    assert import_yolo_dataset(src, tmp_path / "ds", "rf")["images"] == 0


def test_ball_size_by_row_and_merge():
    import numpy as np
    import pandas as pd

    from padelvision.ball.dataset import DEFAULT_BOX_PX, ball_size_by_row, merge_ball_labels

    v = np.linspace(200, 700, 100)
    ball = pd.DataFrame({"v": v, "size": 5 + 0.03 * v, "state": "detected"})
    size = ball_size_by_row(ball)
    assert abs(size(400) - 17.0) < 0.1 and abs(size(10_000) - ball["size"].quantile(0.95)) < 1e-9
    assert ball_size_by_row(ball.head(5))(300) == DEFAULT_BOX_PX

    a = pd.DataFrame({"frame": [1, 1, 2], "u": [100, 300, 50], "v": [100, 300, 50], "size": 9.0})
    b = pd.DataFrame({"frame": [1, 3], "u": [104, 10], "v": [103, 10], "size": 12.0})
    m = merge_ball_labels([a, b])
    assert len(m) == 4 and m[m.frame == 1]["size"].tolist() == [9.0, 9.0]  # (104, 103) = (100, 100)
