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


def _pos(person: int, t: float, swap: bool) -> tuple[float, float]:
    (x0, y0), (x1, y1), xn = PATHS[person]
    wobble = 0.6 * np.sin(0.5 * t + person)
    if not swap or t < WALK[0]:
        return x0 + wobble, y0
    if t >= WALK[1]:
        return x1 + wobble, y1
    a = (t - WALK[0]) / (WALK[1] - WALK[0])  # to the net, then on to the new place
    if a < 0.5:
        return x0 + (xn - x0) * 2 * a, y0 * (1 - 2 * a)
    return xn + (x1 - xn) * (2 * a - 1), y1 * (2 * a - 1)


def _match(seconds: float = 140.0, swap: bool = True, gather: bool = False) -> pd.DataFrame:
    from conftest import project

    rows = []
    for f in range(round(seconds * FPS)):
        t = f / FPS
        for person in PATHS:
            x, y = _pos(person, t, swap)
            if gather and WALK[0] <= t < WALK[1]:  # everyone chats at the net, then goes back
                x, y = (-3 + 2 * person, 0.5 * (-1) ** person)
            # 2 and 4 lose their tracker id at the net; 1 and 3 keep it
            tid = person if person in (1, 3) or t < (WALK[0] + WALK[1]) / 2 else 20 + person
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

    after = players[players.t > WALK[1] + 2]
    for person in PATHS:
        mine = after[after.player == person]
        x1, y1 = PATHS[person][1]
        assert len(mine) > 0.9 * (140 - WALK[1] - 2) * FPS
        assert np.sign(mine.y_m.median()) == np.sign(y1)
        assert np.sign(mine.x_m.median()) == np.sign(x1)
    # slot = court position, pair = A for persons 1-2; team stays the side the row is on
    assert set(after[after.team == "near"].player) == {3, 4}
    assert set(after[after.team == "near"].slot) == {1, 2}
    assert (players.pair == np.where(players.player <= 2, "A", "B")).all()


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
    assert (players.player == players.slot).all()
