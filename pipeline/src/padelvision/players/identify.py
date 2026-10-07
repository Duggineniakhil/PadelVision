"""Stage 3: on-court filtering and 4 stable player identities.

- Foot point = bottom-centre of the box, projected to court metres.
- Only people whose feet are on the court (+ margin) are kept: spectators, coaches
  and people behind the glass are dropped.
- Team = side of the net. Players 1-2 are on the near side, 3-4 on the far side.
  Teams swap ends during a match; until appearance-based re-identification exists,
  ids mean "near/far side slot", not a specific person across changeovers.
- Within a side, a slot keeps following its tracker id. When a new tracker id appears,
  it takes the free slot whose last position is closest, but only if the person could
  physically have got there (no hopping between bystanders on opposite sides of the court).
- Bystanders right outside the side walls (e.g. next to the net posts) can land just inside
  the margin. A tracker id that stays within WALL_BAND_M of a side wall for its whole life
  of at least BYSTANDER_MIN_S is treated as a bystander: players don't hug the side wall.
  Short-lived newcomers at the wall can't take a player slot either.
- Feet cut off by the bottom image edge give a wrong ground position: those rows keep
  their identity but are marked `valid = False` and are excluded from metrics.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from padelvision.court.calibration import CourtCalibration
from padelvision.court.geometry import COURT_LENGTH_M, COURT_WIDTH_M

SLOT_TEAM = {1: "near", 2: "near", 3: "far", 4: "far"}
# Side walls are solid, so nobody plays outside them: keep the sideways margin tight
# (bystanders stand right next to the net posts). Depth is less accurate, so allow more.
COURT_MARGIN_X_M = 0.25
COURT_MARGIN_Y_M = 0.5
DUPLICATE_DIST_M = 0.4  # two boxes whose feet are this close are the same person
NEW_SLOT_COST_M = 2.0  # cost of filling a never-used (or long-lost) slot
UNASSIGNED_COST_M = 3.0  # cost of leaving a free slot empty this frame
MAX_SPEED_MPS = 8.0
JUMP_TOLERANCE_M = 1.5  # position noise allowance, mostly far-side depth error
REACQUIRE_S = 2.0  # after this long unseen, a slot may be re-filled anywhere on its side
MAX_CANDIDATES = 4
WALL_BAND_M = 0.5
BYSTANDER_MIN_S = 3.0
EDGE_PX = 2

OUT_COLUMNS = [
    "frame", "t", "player", "team", "track_id", "conf",
    "x1", "y1", "x2", "y2", "x_m", "y_m", "valid",
]  # fmt: skip


@dataclass
class _Slot:
    player: int
    track_id: int | None = None
    last_xy: np.ndarray | None = None
    last_t: float | None = None


def ground_positions(dets: pd.DataFrame, cal: CourtCalibration) -> pd.DataFrame:
    """Adds x_m, y_m (court metres of the feet), foot_clipped and on_court columns."""
    out = dets.copy()
    feet = np.c_[(out.x1 + out.x2) / 2, out.y2]
    xy = cal.to_court(feet) if len(out) else np.empty((0, 2))
    out["x_m"], out["y_m"] = xy[:, 0], xy[:, 1]
    out["foot_clipped"] = out.y2 >= cal.image_size[1] - EDGE_PX
    out["on_court"] = (out.x_m.abs() <= COURT_WIDTH_M / 2 + COURT_MARGIN_X_M) & (
        out.y_m.abs() <= COURT_LENGTH_M / 2 + COURT_MARGIN_Y_M
    )
    out["bystander"] = out.track_id.isin(_wall_huggers(out[out.on_court]))
    return out


def _wall_huggers(dets: pd.DataFrame) -> set[int]:
    tracked = dets[dets.track_id >= 0]
    life = tracked.groupby("track_id").agg(
        t0=("t", "min"), t1=("t", "max"), inner=("x_m", lambda x: x.abs().min())
    )
    hug = (life.t1 - life.t0 >= BYSTANDER_MIN_S) & (life.inner >= COURT_WIDTH_M / 2 - WALL_BAND_M)
    return set(life.index[hug])


def assign_players(dets: pd.DataFrame, cal: CourtCalibration) -> pd.DataFrame:
    """Detections table (stage 2) -> one row per (frame, player) with court positions."""
    g = ground_positions(dets, cal)
    g = g[g.on_court & ~g.bystander].sort_values(["frame", "conf"], ascending=[True, False])

    slots = {p: _Slot(p) for p in SLOT_TEAM}
    track_to_slot: dict[int, int] = {}
    rows = []

    for frame, fdets in g.groupby("frame", sort=True):
        for team in ("near", "far"):
            side = fdets[(fdets.y_m < 0) == (team == "near")]
            cands = _dedupe(side)
            side_slots = [p for p, s in SLOT_TEAM.items() if s == team]

            mapped, unmapped = [], []
            for c in cands:
                slot = track_to_slot.get(c["track_id"]) if c["track_id"] >= 0 else None
                if slot is not None and slot not in side_slots:
                    # Tracker id moved to the other side of the net: release it.
                    slots[slot].track_id = None
                    del track_to_slot[c["track_id"]]
                    slot = None
                (mapped if slot is not None else unmapped).append((slot, c))
            mapped = mapped[:2]

            free = [p for p in side_slots if p not in {s for s, _ in mapped}]
            # A newcomer right at a side wall is most likely a bystander outside it; it can
            # only take a slot once it's further in. Tracked players may still go to the wall.
            inner = COURT_WIDTH_M / 2 - WALL_BAND_M
            new = [c for _, c in unmapped if abs(c["x_m"]) < inner][:MAX_CANDIDATES]
            assignment = list(mapped) + _match(new, [slots[p] for p in free])

            for player, c in assignment:
                slot = slots[player]
                tid = c["track_id"]
                if tid >= 0 and slot.track_id != tid:
                    if slot.track_id is not None:
                        track_to_slot.pop(slot.track_id, None)
                    slot.track_id = tid
                    track_to_slot[tid] = player
                slot.last_xy = np.array([c["x_m"], c["y_m"]])
                slot.last_t = c["t"]
                rows.append((
                    frame, c["t"], player, team, tid, c["conf"],
                    c["x1"], c["y1"], c["x2"], c["y2"], c["x_m"], c["y_m"],
                    not c["foot_clipped"],
                ))  # fmt: skip

    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def _dedupe(side: pd.DataFrame) -> list[dict]:
    kept: list[dict] = []
    for c in side.to_dict("records"):  # sorted by conf, highest first
        if all(np.hypot(c["x_m"] - k["x_m"], c["y_m"] - k["y_m"]) > DUPLICATE_DIST_M for k in kept):
            kept.append(c)
    return kept


def _match(cands: list[dict], free: list[_Slot]) -> list[tuple[int, dict]]:
    """Assign new detections to free slots (<= 2 slots, brute force).

    Minimises total distance; a slot may stay empty, and a recently seen slot can't take
    a detection further away than the person could have moved.
    """
    if not cands or not free:
        return []

    def cost(slot: _Slot, c: dict) -> float:
        if slot.last_xy is None or slot.last_t is None:
            return NEW_SLOT_COST_M
        dt = c["t"] - slot.last_t
        if dt > REACQUIRE_S:
            return NEW_SLOT_COST_M
        dist = float(np.hypot(*(slot.last_xy - [c["x_m"], c["y_m"]])))
        return dist if dist <= MAX_SPEED_MPS * dt + JUMP_TOLERANCE_M else np.inf

    options = [None, *range(len(cands))]
    best, best_cost = [], np.inf
    for choice in itertools.product(options, repeat=len(free)):
        used = [i for i in choice if i is not None]
        if len(used) != len(set(used)):
            continue
        total = sum(
            UNASSIGNED_COST_M if i is None else cost(s, cands[i])
            for s, i in zip(free, choice, strict=True)
        )
        if total < best_cost:
            best, best_cost = list(zip(free, choice, strict=True)), total
    return [(s.player, cands[i]) for s, i in best if i is not None]
