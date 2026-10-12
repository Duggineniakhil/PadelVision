"""Stage 3b: changeovers and persons.

Players 1-4 from stage 3 are side slots (1-2 near, 3-4 far). In a match the pairs change
ends (padel: after games 1, 3, 5, ...), so a slot holds different people over the match.
This stage finds the changeovers and maps each slot, in each period between them, to a
person: persons 1-2 are pair A (the pair on the near side at the start of the video),
persons 3-4 pair B. Without changeovers, person == player.

Positions only (no pixels):
- Changeovers: at a changeover the players walk across the net and stay. Tracker ids that
  pass the net are found in the raw detections (a walk around the net post is outside the
  court, so the detections are used, not the on-court slots); a track that ends at the net
  and one that starts there right after are linked into one chain. A changeover is a time
  window in which chains moved for good to the other side, in both directions (or several
  in one direction when the others were lost). A player fetching a ball over the net comes
  back, so doesn't count.
- Which pair member is which after a changeover: padel players keep their side (forehand
  right / backhand left), so in each period the court-position slot that played more on its
  own left half is the pair's left player. Seen from the camera, a near player's left is
  image-left and a far player's left is image-right. Each person's left share is learned
  from the periods before. (Tracker ids carried across the net were tried as evidence too, but
  they merge people at the net and players aren't in position right after a changeover.)
Every period's mapping carries a confidence (minutes of agreement, the margin over the
swapped mapping) and the evidence is saved in stats.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from padelvision.court.calibration import CourtCalibration
from padelvision.players.identify import SLOT_TEAM, ground_positions

AREA_X_M = 7.0  # the walk around the net post can be outside the side walls
AREA_Y_M = 11.0
BIN_S = 0.5  # positions are reduced to a median per bin
SIDE_M = 2.5  # clearly on one side: |y| beyond this
NET_ZONE_M = 3.0  # a track that ends / starts this close to the net may be linked
LINK_GAP_S = 3.0  # end -> start within this long ...
LINK_DIST_M = 2.5  # ... and this close
HOLD_BEFORE_S = 1.0  # a crossing chain was this long on its old side
STAY_S = 4.0  # ... and stays this long on its new side (a ball fetch comes back)
WINDOW_S = 75.0  # crossings this close together are one changeover
MIN_GAP_S = 90.0  # two changeovers are at least a game apart
GATHER_M = 2.0  # a net gathering: >= GATHER_MIN people within this of the net ...
GATHER_MIN = 3
GATHER_S = 5.0  # ... for this long
GATHER_NEAR_S = 20.0  # a lone crossing this close to a gathering makes a likely changeover
STATIC_M = 0.3  # tracks that never move (a net post, a bag) are not people


@dataclass
class Crossing:
    chain: int
    t: float  # when it passed the net
    direction: str  # "to_far" | "to_near"
    tracks: list[int] = field(default_factory=list)


def chains(dets: pd.DataFrame, cal: CourtCalibration) -> pd.DataFrame:
    """Raw detections -> (bin time, chain, track_id, x_m, y_m), tracks linked through the net."""
    g = ground_positions(dets[dets.track_id >= 0], cal)
    g = g[(g.x_m.abs() <= AREA_X_M) & (g.y_m.abs() <= AREA_Y_M) & ~g.bystander]
    if g.empty:
        return pd.DataFrame(columns=["t", "chain", "track_id", "x_m", "y_m"])
    g = g.assign(t=(g.t / BIN_S).round() * BIN_S)
    b = g.groupby(["track_id", "t"], as_index=False)[["x_m", "y_m"]].median()
    life = b.groupby("track_id").agg(t0=("t", "min"), t1=("t", "max"))
    first = b.loc[b.groupby("track_id").t.idxmin()].set_index("track_id")
    last = b.loc[b.groupby("track_id").t.idxmax()].set_index("track_id")

    chain_of = {tid: tid for tid in life.index}
    taken: set[int] = set()

    def root(t: int) -> int:
        while chain_of[t] != t:
            t = chain_of[t]
        return t

    # Link a track ending at the net to the nearest track starting there right after.
    ends = last[last.y_m.abs() <= NET_ZONE_M].sort_values("t")
    starts = first[first.y_m.abs() <= NET_ZONE_M]
    for tid, e in ends.iterrows():
        cand = starts[(starts.t > e.t - BIN_S) & (starts.t <= e.t + LINK_GAP_S)]
        cand = cand[~cand.index.isin(taken) & (cand.index != tid)]
        if cand.empty:
            continue
        d = np.hypot(cand.x_m - e.x_m, cand.y_m - e.y_m)
        nxt = int(d.idxmin())
        if d.min() <= LINK_DIST_M and root(nxt) != root(int(tid)):  # no loops
            taken.add(nxt)
            chain_of[nxt] = int(tid)

    b["chain"] = b.track_id.map(lambda t: root(int(t)))
    return b[["t", "chain", "track_id", "x_m", "y_m"]].sort_values(["chain", "t"])


def crossings(ch: pd.DataFrame) -> list[Crossing]:
    """Chains that moved for good to the other side of the net."""
    out = []
    for cid, c in ch.groupby("chain"):
        side = np.where(c.y_m > SIDE_M, 1, np.where(c.y_m < -SIDE_M, -1, 0))
        t = c.t.to_numpy()
        clear = side != 0
        if clear.sum() < 2:
            continue
        ts, ss = t[clear], side[clear]
        # Runs of consecutive same-side samples.
        cuts = np.nonzero(np.diff(ss))[0] + 1
        edges = zip(np.r_[0, cuts], np.r_[cuts, len(ss)], strict=True)
        runs = [(ss[a], ts[a], ts[b - 1]) for a, b in edges]
        for (_, a0, a1), (s1, b0, b1) in zip(runs, runs[1:], strict=False):
            if a1 - a0 < HOLD_BEFORE_S - 1e-9:
                continue
            # Stays: the rest of the chain after b0 is mostly on the new side, for STAY_S.
            rest = ss[ts >= b0]
            rest_t = ts[ts >= b0]
            if (rest_t[-1] - b0) < STAY_S or (rest == s1).mean() < 0.8:
                continue
            mid = c[(c.t > a1) & (c.t < b0)]
            t_net = float(mid.t[mid.y_m.abs().idxmin()]) if len(mid) else (a1 + b0) / 2
            tracks = sorted(set(c.track_id[(c.t >= a0) & (c.t <= b1)].astype(int)))
            out.append(Crossing(int(cid), t_net, "to_far" if s1 > 0 else "to_near", tracks))
            break  # one move per chain: anything after it is a later changeover's business
    return sorted(out, key=lambda x: x.t)


def gatherings(ch: pd.DataFrame) -> list[tuple[float, float]]:
    """(start, end) of the times >= GATHER_MIN people stood near the net for >= GATHER_S:
    changeovers, but also set breaks and chats, so on their own they decide nothing."""
    near = ch[ch.y_m.abs() < GATHER_M]
    moving = ch.groupby("track_id").agg(sx=("x_m", "std"), sy=("y_m", "std"), n=("t", "size"))
    still = (moving.sx.fillna(0) < STATIC_M) & (moving.sy.fillna(0) < STATIC_M)
    static = moving[(moving.n >= 4) & still]
    near = near[~near.track_id.isin(static.index)]
    if near.empty:
        return []
    count = near.groupby("t").track_id.nunique()
    busy = count[count >= GATHER_MIN].index.to_numpy()
    out: list[tuple[float, float]] = []
    for t in busy:
        if out and t - out[-1][1] <= 2 * BIN_S:
            out[-1] = (out[-1][0], float(t))
        else:
            out.append((float(t), float(t)))
    return [(a, b + BIN_S) for a, b in out if b + BIN_S - a >= GATHER_S]


def find_changeovers(
    cross: list[Crossing], gathers: list[tuple[float, float]] | None = None
) -> list[dict]:
    """Crossings grouped in time -> changeovers [{t, to_far, to_near, chains, evidence}].

    "strong": chains moved both ways, or several one way. "likely": a single chain moved, at a
    net gathering. A gathering without any crossing is reported by `unresolved`, not here."""
    gathers = gathers or []
    groups: list[list[Crossing]] = []
    for c in cross:
        if groups and c.t - groups[-1][0].t <= WINDOW_S:
            groups[-1].append(c)
        else:
            groups.append([c])
    out = []
    for g in groups:
        to_far = {c.chain for c in g if c.direction == "to_far"}
        to_near = {c.chain for c in g if c.direction == "to_near"}
        t = float(np.median([c.t for c in g]))
        if (to_far and to_near) or len(to_far) >= 2 or len(to_near) >= 2:
            evidence = "strong"
        elif any(a - GATHER_NEAR_S <= t <= b + GATHER_NEAR_S for a, b in gathers):
            evidence = "likely"
        else:
            continue
        if out and t - out[-1]["t"] < MIN_GAP_S:
            continue
        out.append({"t": round(t, 1), "to_far": len(to_far), "to_near": len(to_near),
                    "chains": sorted(to_far | to_near), "evidence": evidence})  # fmt: skip
    return out


def unresolved(gathers: list[tuple[float, float]], cos: list[dict]) -> list[dict]:
    """Net gatherings with no changeover found: the pairs may or may not have changed ends."""
    return [{"start_s": round(a, 1), "end_s": round(b, 1)} for a, b in gathers
            if not any(a - GATHER_NEAR_S <= c["t"] <= b + GATHER_NEAR_S for c in cos)]  # fmt: skip


def _own_left(x: pd.Series, team: pd.Series) -> pd.Series:
    """True when the player is on their own left half (facing the net)."""
    return np.where(team == "near", x < 0, x > 0)


def assign_persons(players: pd.DataFrame, dets: pd.DataFrame, cal: CourtCalibration,
                   fps: float) -> tuple[pd.DataFrame, dict]:  # fmt: skip
    """Stage 3 rows -> the same rows with `player` = person (1-4), `slot` = the stage 3 side
    slot and `pair` ("A"/"B"); `team` stays the side the row is on. And the evidence."""
    ch = chains(dets, cal)
    cross = crossings(ch)
    gathers = gatherings(ch)
    cos = find_changeovers(cross, gathers)
    out = players.copy()
    t_end = float(players.t.max()) if len(players) else 0.0
    bounds = [0.0] + [c["t"] for c in cos] + [t_end + 1.0]
    out["period"] = np.searchsorted(bounds[1:-1], out.t.to_numpy(), side="right")

    # Side preference per slot per period: share of time on the slot's own left half.
    valid = out[out.valid]
    left = (valid.assign(left=_own_left(valid.x_m, valid.team))
            .groupby(["period", "player"]).left.agg(["mean", "size"]))  # fmt: skip

    periods = [{"period": 0, "start_s": 0.0, "near_pair": "A", "slots": {p: p for p in SLOT_TEAM},
                "confidence": None}]  # fmt: skip
    history: dict[int, list[tuple[float, float]]] = {p: [] for p in SLOT_TEAM}  # (share, minutes)

    def learn(k: int, m: dict[int, int]) -> None:
        for slot, person in m.items():
            if (k, slot) in left.index:
                r = left.loc[(k, slot)]
                history[person].append((float(r["mean"]), float(r["size"]) / fps / 60))

    def agreement(k: int, slot: int, person: int) -> float:
        """Minutes this slot played like the person's side habit, minus minutes it didn't."""
        if (k, slot) not in left.index or not history[person]:
            return 0.0
        share, minutes = left.loc[(k, slot)]["mean"], left.loc[(k, slot)]["size"] / fps / 60
        habit = np.average([h for h, _ in history[person]],
                           weights=[max(w, 1e-6) for _, w in history[person]])  # fmt: skip
        return float(minutes * (1 - 2 * abs(share - habit)))

    learn(0, periods[0]["slots"])
    for k in range(1, len(bounds) - 1):
        prev = periods[-1]["slots"]
        m: dict[int, int] = {}
        margins = []
        for here, there in (((1, 2), (3, 4)), ((3, 4), (1, 2))):
            # The persons now on this side were on the other side before.
            a, b = sorted(prev[s] for s in there)
            keep = agreement(k, here[0], a) + agreement(k, here[1], b)
            swap = agreement(k, here[0], b) + agreement(k, here[1], a)
            m.update(zip(here, (a, b) if keep >= swap else (b, a), strict=True))
            margins.append(abs(keep - swap))
        learn(k, m)
        periods.append({"period": k, "start_s": bounds[k], "near_pair": "A" if k % 2 == 0 else "B",
                        "slots": m, "confidence": round(min(margins), 2)})  # fmt: skip
    mapping = {p["period"]: p["slots"] for p in periods}

    out["slot"] = out.player
    out["player"] = [mapping[k][p] for k, p in zip(out.period, out.slot, strict=True)]
    out["pair"] = np.where(out.player <= 2, "A", "B")
    info = {
        "changeovers": cos,
        "periods": [{**p, "slots": {str(s): q for s, q in p["slots"].items()}} for p in periods],
        "crossings": len(cross),
        "unresolved_gatherings": unresolved(gathers, cos),
        "note": "player = person: 1-2 pair A (near at the start), 3-4 pair B; slots per period map "
        "court position slots (1-2 near, 3-4 far) to persons",
    }
    return out.drop(columns="period"), info
