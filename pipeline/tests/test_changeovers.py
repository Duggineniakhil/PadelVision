import numpy as np
import pandas as pd

from padelvision.players import assign_persons, assign_players

FPS = 10.0
WALK = (60.0, 70.0)  # the pairs change ends in this window

# person -> (x, y) before and after the changeover, and where they pass the net (x).
# Persons keep their side habit: 1 and 3 play on their own left, 2 and 4 on their right
# (own left = image left on the near side, image right on the far side).
PATHS = {
    1: ((-2.5, -6.0), (2.5, 6.0), -4.0),
    2: ((2.5, -7.0), (-2.5, 7.0), -2.0),
    3: ((2.5, 6.0), (-2.5, -6.0), 2.0),
    4: ((-2.5, 7.0), (2.5, -7.0), 4.0),
}


def _pos(person: int, t: float, swap: bool, paths=None) -> tuple[float, float]:
    (x0, y0), (x1, y1), xn = (paths or PATHS)[person]
    wobble = 0.6 * np.sin(0.5 * t + person)
    if not swap or t < WALK[0]:
        return x0 + wobble, y0
    if t >= WALK[1]:
        return x1 + wobble, y1
    a = (t - WALK[0]) / (WALK[1] - WALK[0])  # to the net, then on to the new place
    if a < 0.5:
        return x0 + (xn - x0) * 2 * a, y0 * (1 - 2 * a)
    return xn + (x1 - xn) * (2 * a - 1), y1 * (2 * a - 1)


def _match(
    seconds: float = 140.0, swap: bool = True, gather: bool = False, paths=None, all_new_ids=False
) -> pd.DataFrame:
    from conftest import project

    rows = []
    for f in range(round(seconds * FPS)):
        t = f / FPS
        for person in PATHS:
            x, y = _pos(person, t, swap, paths)
            if gather and WALK[0] <= t < WALK[1]:  # everyone meets at the net
                x, y = (-3 + 2 * person, 0.5 * (-1) ** person)
                if all_new_ids and t >= WALK[1] - 4:  # ... and the tracker loses them there
                    continue
            # 2 and 4 lose their tracker id at the net; 1 and 3 keep it (unless all_new_ids)
            kept = person in (1, 3) and not all_new_ids
            switch = WALK[1] - 4 if all_new_ids else (WALK[0] + WALK[1]) / 2
            tid = person if kept or t < switch else 20 + person
            rows.append((f, tid, x, y))
    feet = project([(x, y) for _, _, x, y in rows])
    data = [(f, f / FPS, tid, 0.9, u - 15, v - 80, u + 15, v)
            for (f, tid, _, _), (u, v) in zip(rows, feet, strict=True)]  # fmt: skip
    return pd.DataFrame(data, columns=["frame", "t", "track_id", "conf", "x1", "y1", "x2", "y2"])


def test_changeover_found_and_persons_followed_across_it(cal):
    dets = _match()
    players, info = assign_persons(assign_players(dets, cal), dets, cal, FPS)
    assert len(info["changeovers"]) == 1
    co = info["changeovers"][0]
    assert WALK[0] <= co["t"] <= WALK[1] and co["evidence"] == "strong"
    assert info["periods"][1]["near_pair"] == "B"

    _assert_persons_play_where_they_should(players, swapped=True)
    after = players[players.t > WALK[1] + 2]
    # slot = court position, pair = A for persons 1-2; team stays the side the row is on
    assert set(after[after.team == "near"].player) == {3, 4}
    assert set(after[after.team == "near"].slot) == {1, 2}
    assert (players.pair == np.where(players.player <= 2, "A", "B")).all()
    # the walk between ends is flagged and left out of the smoothed tracks
    assert players[players.t.between(*WALK)].changeover.all()
    assert not players[(players.t < WALK[0] - 10) | (players.t > WALK[1] + 10)].changeover.any()
    from padelvision.analytics import smooth_tracks

    tracks = smooth_tracks(players, FPS)
    assert not tracks.t.between(WALK[0] + 0.5, WALK[1] - 0.5).any()


def test_without_changeover_person_is_the_slot(cal):
    dets = _match(swap=False)
    players, info = assign_persons(assign_players(dets, cal), dets, cal, FPS)
    assert info["changeovers"] == [] and len(info["periods"]) == 1
    assert (players.player == players.slot).all()


def test_gathering_at_the_net_without_crossing_is_unresolved_not_a_changeover(cal):
    dets = _match(swap=False, gather=True)
    players, info = assign_persons(assign_players(dets, cal), dets, cal, FPS)
    assert info["changeovers"] == []
    assert len(info["unresolved_gatherings"]) == 1
    _assert_persons_play_where_they_should(players, swapped=False)


def _assert_persons_play_where_they_should(players: pd.DataFrame, swapped: bool) -> None:
    """After the walk, each person is found where that person plays (their side habit)."""
    after = players[players.t > WALK[1] + 2]
    for person in PATHS:
        mine = after[after.player == person]
        x, y = PATHS[person][1 if swapped else 0]
        assert len(mine) > 0.9 * (140 - WALK[1] - 2) * FPS
        assert np.sign(mine.y_m.median()) == np.sign(y)
        assert np.sign(mine.x_m.median()) == np.sign(x)


# person -> shirt and shorts colour (Lab): pair A dark shirts, pair B white; partners' shorts differ
KIT = {1: (30, 128, 128, 30, 128, 128), 2: (30, 128, 128, 200, 128, 128),
       3: (160, 128, 150, 40, 128, 128), 4: (160, 128, 150, 120, 128, 128)}  # fmt: skip


def _appearance(dets: pd.DataFrame) -> pd.DataFrame:
    """Every 5th frame, each track's box gets its person's colours (+ noise)."""
    from padelvision.players.appearance import COLUMNS

    rng = np.random.default_rng(3)
    rows = []
    for r in dets[dets.frame % 5 == 0].itertuples():
        person = r.track_id % 20
        rows.append((r.frame, r.track_id, 300, *(np.array(KIT[person]) + rng.normal(0, 6, 6))))
    return pd.DataFrame(rows, columns=COLUMNS)


def test_colours_show_a_changeover_positions_cannot_see(cal):
    # Everyone meets at the net, the tracker loses them all and they reappear with new ids:
    # no crossing to see.
    dets = _match(swap=True, gather=True, all_new_ids=True)
    players, info = assign_persons(assign_players(dets, cal), dets, cal, FPS)
    assert info["changeovers"] == []  # positions alone: no swap assumed ...
    players, info = assign_persons(assign_players(dets, cal), dets, cal, FPS, _appearance(dets))
    assert len(info["changeovers"]) == 1  # ... the colours show it
    assert info["changeovers"][0]["colours_agree"] and info["unresolved_gatherings"] == []
    _assert_persons_play_where_they_should(players, swapped=True)


def test_colours_follow_partners_who_also_swap_sides(cal):
    # After the changeover pair A's partners also swap sides, which fools the side habit.
    paths = dict(PATHS)
    paths[1] = ((-2.5, -6.0), (-2.5, 7.0), -4.0)
    paths[2] = ((2.5, -7.0), (2.5, 6.0), -2.0)
    dets = _match(paths=paths)
    players, info = assign_persons(assign_players(dets, cal), dets, cal, FPS, _appearance(dets))
    after = players[players.t > WALK[1] + 2]
    for person in (1, 2):
        mine = after[after.player == person]
        assert np.sign(mine.x_m.median()) == np.sign(paths[person][1][0])
    assert info["periods"][1]["partners_by"] == ["colours", "colours"]
