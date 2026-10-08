"""Link per-frame ball candidates into trajectories; keep only ball-like ones as pseudo-labels.

A ball in flight moves fast and smoothly: over a few frames its image path is close to
constant-velocity. Noise (flickering lights, reflections, distant movement) rarely forms
such tracks. Tracks are built greedily with a constant-velocity prediction; a track may
skip MAX_MISSES frames (occlusion, blur).

Pseudo-labels are only the frames where a candidate was actually detected; nothing is
interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIRST_STEP_MAX_PX = 70  # how far the ball can move between two frames (fast smash, 30 fps)
GATE_PX = 18  # allowed deviation from the constant-velocity prediction
GATE_PER_SPEED = 0.35  # extra deviation per px/frame of speed (blur, curvature)
MAX_MISSES = 2
MIN_TRACK_LEN = 6  # detections
MIN_SPEED_PX = 2.5  # median px/frame: the ball in play moves; static flicker doesn't
MAX_JERK_PX = 9.0  # median |second difference|, after accounting for missed frames
# Track-level appearance (medians over the track), tuned on 132 hand-checked pseudo-labels from
# the developer's footage: precision went from ~40% to ~90% while keeping ~84% of real balls.
# Rejects players' legs/rackets/shoes and people on neighbouring courts: dull and slow-changing.
MIN_TRACK_COLOR = 0.25  # fraction of ball-coloured pixels
MIN_TRACK_MOTION = 30.0  # frame-difference strength

PSEUDO_COLUMNS = ["frame", "u", "v", "track", "score"]


@dataclass
class _Track:
    id: int
    frames: list[int] = field(default_factory=list)
    pts: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    colors: list[float] = field(default_factory=list)
    motions: list[float] = field(default_factory=list)

    def predict(self, frame: int) -> np.ndarray:
        if len(self.pts) < 2:
            return self.pts[-1]
        vel = (self.pts[-1] - self.pts[-2]) / (self.frames[-1] - self.frames[-2])
        return self.pts[-1] + vel * (frame - self.frames[-1])

    def gate(self) -> float:
        if len(self.pts) < 2:
            return FIRST_STEP_MAX_PX
        speed = np.linalg.norm(self.pts[-1] - self.pts[-2]) / (self.frames[-1] - self.frames[-2])
        return GATE_PX + GATE_PER_SPEED * speed


def link_tracks(cands: pd.DataFrame) -> pd.DataFrame:
    """All candidates -> candidates with a `track` id (-1 = not on any track)."""
    cands = cands.sort_values(["frame", "score"], ascending=[True, False]).reset_index(drop=True)
    track_of = np.full(len(cands), -1)
    active: list[_Track] = []
    finished: list[_Track] = []
    next_id = 0
    idx_of: dict[int, list[int]] = {}

    for frame, group in cands.groupby("frame", sort=True):
        active, done = (
            [t for t in active if frame - t.frames[-1] <= MAX_MISSES + 1],
            [t for t in active if frame - t.frames[-1] > MAX_MISSES + 1],
        )
        finished += done
        pts = group[["u", "v"]].to_numpy()
        colors = group.color.to_numpy() if "color" in group else np.ones(len(group))
        motions = group.motion.to_numpy() if "motion" in group else np.full(len(group), 255.0)
        free = list(range(len(group)))
        # Longer tracks choose first: they have reliable velocity estimates.
        for t in sorted(active, key=lambda t: -len(t.pts)):
            if not free:
                break
            pred = t.predict(frame)
            d = np.linalg.norm(pts[free] - pred, axis=1)
            j = int(np.argmin(d))
            if d[j] <= t.gate():
                k = free.pop(j)
                row = group.index[k]
                t.frames.append(frame)
                t.pts.append(pts[k])
                t.scores.append(float(group.score.iloc[k]))
                t.colors.append(float(colors[k]))
                t.motions.append(float(motions[k]))
                idx_of.setdefault(t.id, []).append(row)
        for k in free:  # every unmatched candidate may start a new track
            t = _Track(next_id, [frame], [pts[k]], [float(group.score.iloc[k])],
                       [float(colors[k])], [float(motions[k])])  # fmt: skip
            idx_of[t.id] = [group.index[k]]
            next_id += 1
            active.append(t)

    for t in finished + active:
        if _ball_like(t):
            track_of[idx_of[t.id]] = t.id
    return cands.assign(track=track_of)


def pseudo_labels(cands: pd.DataFrame) -> pd.DataFrame:
    """Frames with one confident ball position (from a ball-like track)."""
    linked = link_tracks(cands)
    on = linked[linked.track >= 0]
    # If two tracks claim the same frame, keep the higher-scoring candidate.
    on = on.sort_values("score", ascending=False).drop_duplicates("frame").sort_values("frame")
    return on[PSEUDO_COLUMNS].reset_index(drop=True)


def _ball_like(t: _Track) -> bool:
    if len(t.pts) < MIN_TRACK_LEN:
        return False
    if np.median(t.colors) < MIN_TRACK_COLOR or np.median(t.motions) < MIN_TRACK_MOTION:
        return False
    f = np.asarray(t.frames, dtype=float)
    p = np.asarray(t.pts)
    vel = np.diff(p, axis=0) / np.diff(f)[:, None]
    if np.median(np.linalg.norm(vel, axis=1)) < MIN_SPEED_PX:
        return False
    jerk = np.linalg.norm(np.diff(vel, axis=0), axis=1)
    return bool(np.median(jerk) <= MAX_JERK_PX)
