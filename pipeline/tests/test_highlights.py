import cv2
import numpy as np
import pandas as pd
from test_run import _blank_video

from padelvision.highlights import render_highlights, select_clips, title_lines
from padelvision.overlay import StatsHud

FPS = 30.0


def _summary():
    rallies = [
        {"id": 1, "start_frame": 30, "end_frame": 90, "duration_s": 2.0, "hits": 2},
        {"id": 3, "start_frame": 150, "end_frame": 330, "duration_s": 6.0, "hits": 7},
        {"id": 5, "start_frame": 400, "end_frame": 460, "duration_s": 2.0, "hits": 3},
    ]
    per = [{"id": 1, "max_shot_kmh": None}, {"id": 3, "max_shot_kmh": 50.0},
           {"id": 5, "max_shot_kmh": 75.0}]  # fmt: skip
    shots = [{"frame": 160, "player": 1, "team": "near", "speed_kmh": 40.0},
             {"frame": 200, "player": 3, "team": "far", "speed_kmh": None}]  # fmt: skip
    return {"rallies": rallies, "stats": {"per_rally": per, "shots": shots}}


def test_top_clips_by_score_in_match_order_and_all_mode():
    s = _summary()
    top = select_clips(s, FPS, n_frames=470, mode="top", top_n=2)
    assert [c["rally_id"] for c in top] == [3, 5]  # 7 hits; then 3 hits + 75 km/h beats 2 hits
    assert top[0]["start_frame"] == 135 and top[1]["end_frame"] == 469  # 0.5 s pad, video end
    assert [c["rally_id"] for c in select_clips(s, FPS, 470, mode="all")] == [1, 3, 5]
    assert "fastest shot" not in " ".join(title_lines(select_clips(s, FPS, 470, "all")[0], 0, 3))


def _tracks():
    rows = [(f, 1, 0.0, -8.0 + 0.1 * f / 3, 0.0, 1.0, 0) for f in range(0, 300)]  # 1 m/s
    rows += [(f, 3, 0.0, 8.0, 0.0, 0.0, 0) for f in range(0, 20)]
    df = pd.DataFrame(rows, columns=["frame", "player", "x", "y", "vx", "vy", "segment"])
    df["t"] = df.frame / FPS
    return df


def test_hud_shows_measured_values_only():
    stats = {"players": {"1": {}, "2": {"insufficient_data": "tracked 5%"}, "3": {}, "4": {}}}
    hud = StatsHud(FPS, _tracks(), stats, _summary())
    assert abs(hud.distance_m(1, 150) - 5.0) < 1e-6  # 150 frames at 1 m/s
    assert abs(hud.speed_kmh(1, 150) - 3.6) < 1e-6
    assert hud.speed_kmh(3, 150) is None  # not seen for a long time: no "current" speed
    assert hud.player_line(2, 150) == "P2  n/a" and hud.player_line(4, 150) == "P4  n/a"
    assert hud.hits_so_far(1, 170) == 1 and hud.hits_so_far(1, 150) == 0
    assert "RALLY 3  hits 2  last shot ~40 km/h est." in hud.rally_line(250)
    assert hud.rally_line(120).endswith("between points")
    img = np.zeros((720, 1280, 3), np.uint8)
    hud.draw(img, 250)
    assert img[680:].any()


def test_render_highlights_writes_only_the_clips(tmp_path):
    video = tmp_path / "match.avi"
    _blank_video(video, frames=120)
    clips = [{"rally_id": 1, "start_frame": 10, "end_frame": 29, "duration_s": 0.6, "hits": 2,
              "max_shot_kmh": None},
             {"rally_id": 2, "start_frame": 60, "end_frame": 89, "duration_s": 1.0, "hits": 4,
              "max_shot_kmh": 55.0}]  # fmt: skip
    out = render_highlights(video, tmp_path / "h.avi", clips,
                            hud=StatsHud(FPS, _tracks(), None, _summary()))  # fmt: skip
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == 20 + 30
