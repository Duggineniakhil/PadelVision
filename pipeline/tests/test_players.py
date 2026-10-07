import numpy as np
import pandas as pd
from conftest import SIZE, synthetic_frames, true_positions

from padelvision.players import assign_players, detections_to_frame
from padelvision.players.identify import ground_positions

FPS = 30.0


def _true_player(row) -> int:
    truth = true_positions(row.t)
    return min(truth, key=lambda p: np.hypot(truth[p][0] - row.x_m, truth[p][1] - row.y_m))


def test_identities_are_stable_and_spectators_dropped(cal):
    dets = detections_to_frame(synthetic_frames(FPS), FPS)
    players = assign_players(dets, cal)

    assert players.groupby("frame").size().max() == 4
    assert not (players.track_id == 50).any()  # spectator outside the court
    assert (players.x_m.abs() <= 5.5).all()

    # Each output id always follows the same real person, despite the id switch at t=5s
    # and the far players crossing.
    players["truth"] = players.apply(_true_player, axis=1)
    mapping = players.groupby("player").truth.nunique()
    assert (mapping == 1).all(), mapping
    assert sorted(players.groupby("player").truth.first()) == [1, 2, 3, 4]
    assert set(players[players.team == "near"].player) == {1, 2}


def test_positions_match_ground_truth(cal):
    dets = detections_to_frame(synthetic_frames(FPS, seconds=1), FPS)
    players = assign_players(dets, cal)
    for row in players.itertuples():
        tx, ty = true_positions(row.t)[_true_player(row)]
        assert np.hypot(tx - row.x_m, ty - row.y_m) < 1e-3


def test_feet_cut_off_marked_invalid(cal):
    dets = pd.DataFrame(
        [(0, 0.0, 1, 0.9, 600.0, SIZE[1] - 120, 640.0, SIZE[1] - 1)],
        columns=["frame", "t", "track_id", "conf", "x1", "y1", "x2", "y2"],
    )
    g = ground_positions(dets, cal)
    assert g.foot_clipped.iloc[0]
    out = assign_players(dets, cal)
    assert len(out) == 0 or not out.valid.any()


def test_stride_and_empty_input(cal):
    dets = detections_to_frame(synthetic_frames(FPS, seconds=2, stride=3), FPS)
    assert sorted(dets.frame.unique())[:3] == [0, 3, 6]
    empty = assign_players(dets.iloc[0:0], cal)
    assert empty.empty
