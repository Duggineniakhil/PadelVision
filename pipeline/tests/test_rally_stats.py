import numpy as np
import pandas as pd
from test_run import _blank_video

from padelvision.analytics import rally_stats
from padelvision.render import render_preview
from padelvision.visuals import save_placement_map

FPS = 30.0
COLS = ["frame", "kind", "u", "v", "x_m", "y_m", "in_court", "player", "team"]


def _rally(rid, start, end, hits):
    return {"id": rid, "start_frame": start, "end_frame": end,
            "duration_s": (end - start) / FPS, "hits": hits}  # fmt: skip


def _feet(rows):
    """rows: (frame, player, x_m, y_m)."""
    df = pd.DataFrame(rows, columns=["frame", "player", "x_m", "y_m"])
    df["valid"] = True
    return df


def _example():
    ev = pd.DataFrame(
        [
            (30, "hit", 500, 600, np.nan, np.nan, None, 1, "near"),  # near player at y = -8
            (60, "bounce", 600, 300, 0.0, 7.0, True, np.nan, None),  # lands 15 m away, 1 s later
            (75, "hit", 600, 280, np.nan, np.nan, None, 3, "far"),
            (90, "bounce", 650, 320, 1.0, 2.0, True, np.nan, None),  # same side as the far hitter
            (120, "hit", 500, 600, np.nan, np.nan, None, 1, "near"),  # no landing seen
            (400, "hit", 500, 600, np.nan, np.nan, None, 2, "near"),  # outside every rally
            (430, "bounce", 600, 300, 0.0, 5.0, True, np.nan, None),
        ],
        columns=COLS,
    )
    feet = _feet([(30, 1, 0.0, -8.0), (75, 3, 1.0, 8.0), (120, 1, 0.0, -8.0), (400, 2, 0, -8.0)])
    return ev, [_rally(1, 0, 150, 3)], feet


def test_rally_stats_counts_placement_and_speed_estimate():
    ev, rallies, feet = _example()
    s = rally_stats(ev, rallies, feet, FPS)
    assert s["rallies"] == 1 and s["hits_by_player"] == {"1": 2, "3": 1}
    assert s["hits_by_team"] == {"near": 2, "far": 1}
    near = s["placement"]["near"]
    assert near["shots"] == 2 and near["landed"] == 1  # the bounce at 430 is outside the rally
    assert near["bounces"][0]["zone"] == "back" and near["zones"]["back"] == 1
    assert s["placement"]["far"]["landed"] == 0  # its bounce was on its own side
    speed = s["shot_speed_estimate"]
    assert speed["samples"] == 1 and "insufficient_data" in speed  # one shot is not enough


def test_shot_speed_estimate_with_enough_shots():
    rows, feet = [], []
    for k in range(3):
        f = 100 * k
        rows += [(f, "hit", 500, 600, np.nan, np.nan, None, 1, "near"),
                 (f + 30, "bounce", 600, 300, 0.0, 7.0, True, np.nan, None)]  # fmt: skip
        feet.append((f + 2, 1, 0.0, -8.0))  # feet seen 2 frames after the hit
    s = rally_stats(pd.DataFrame(rows, columns=COLS), [_rally(1, 0, 300, 3)], _feet(feet), FPS)
    speed = s["shot_speed_estimate"]
    assert speed["samples"] == 3 and speed["median_kmh"] == 54.0  # 15 m in 1 s


def test_no_rallies_gives_no_numbers(tmp_path):
    ev, _, feet = _example()
    s = rally_stats(ev, [], feet, FPS)
    assert s == {"rallies": 0, "insufficient_data": "no rallies detected"}
    save_placement_map(s, tmp_path / "placement.png")  # still draws ("not enough data")
    assert (tmp_path / "placement.png").exists()


def test_placement_map_and_preview_with_events(tmp_path, cal):
    ev, rallies, feet = _example()
    save_placement_map(rally_stats(ev, rallies, feet, FPS), tmp_path / "placement.png")
    assert (tmp_path / "placement.png").stat().st_size > 10_000

    video = tmp_path / "v.avi"
    _blank_video(video, frames=70)
    players = pd.DataFrame(
        columns=["frame", "player", "team", "x1", "y1", "x2", "y2", "x_m", "y_m", "valid"]
    )
    out = render_preview(video, players, cal, tmp_path / "p.avi", seconds=2, events=ev,
                         rallies=rallies)  # fmt: skip
    assert out.exists()
