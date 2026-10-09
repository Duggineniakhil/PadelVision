"""Stage 4: ball in play from per-frame ball detections (pure Python, no model).

1. Link detections (conf >= DET_CONF) into tracklets with a constant-velocity prediction.
   Tracklets naturally break at hits and bounces (sudden direction changes).
2. Drop tracklets that are too short, static (spare balls on the floor, fixed false spots),
   or mostly outside the court region (balls on neighbouring courts).
3. Where moving tracklets overlap in time, the strongest (sum of confidences) is the ball
   in play.
4. Gaps of <= MAX_INTERP_GAP frames inside a tracklet are filled linearly and marked
   state="interpolated"; longer gaps stay empty. Nothing is invented between tracklets.

Positions are image pixels. An airborne ball has no valid ground-plane position, so never
project these through the court homography (bounce points are handled in phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DET_CONF = 0.10  # model card: best F1 threshold for ball-detector v1
FIRST_STEP_MAX_PX = 80.0  # fastest plausible move between two frames (smash at 30 fps)
GATE_PX = 25.0  # allowed deviation from the constant-velocity prediction
GATE_PER_SPEED = 0.5  # extra deviation per px/frame of speed
MAX_MISSES = 4  # frames a tracklet may go undetected and still continue
MIN_DETECTIONS = 4
STATIC_SPEED_PX = 1.5  # median px/frame below this = a resting ball or a fixed false spot
MIN_IN_ROI = 0.5  # fraction of a tracklet's points that must lie in the court region
MAX_INTERP_GAP = 5  # frames

BALL_COLUMNS = ["frame", "t", "u", "v", "conf", "state", "track"]


@dataclass
class Tracklet:
    id: int
    frames: list[int] = field(default_factory=list)
    pts: list[np.ndarray] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)

    def predict(self, frame: int) -> np.ndarray:
        if len(self.pts) < 2:
            return self.pts[-1]
        vel = (self.pts[-1] - self.pts[-2]) / (self.frames[-1] - self.frames[-2])
        return self.pts[-1] + vel * (frame - self.frames[-1])

    def gate(self) -> float:
        if len(self.pts) < 2:
            return FIRST_STEP_MAX_PX
        step = np.linalg.norm(self.pts[-1] - self.pts[-2]) / (self.frames[-1] - self.frames[-2])
        return GATE_PX + GATE_PER_SPEED * step

    @property
    def speed(self) -> float:
        """Median px/frame between consecutive detections."""
        if len(self.pts) < 2:
            return 0.0
        p, f = np.asarray(self.pts), np.asarray(self.frames, float)
        return float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(f)))

    @property
    def score(self) -> float:
        return float(np.sum(self.confs))


def link(dets: pd.DataFrame) -> list[Tracklet]:
    """Detections (frame, u, v, conf) -> tracklets (every detection ends up in one)."""
    dets = dets.sort_values(["frame", "conf"], ascending=[True, False])
    active: list[Tracklet] = []
    done: list[Tracklet] = []
    next_id = 0
    for frame, g in dets.groupby("frame", sort=True):
        done += [t for t in active if frame - t.frames[-1] > MAX_MISSES + 1]
        active = [t for t in active if frame - t.frames[-1] <= MAX_MISSES + 1]
        pts = g[["u", "v"]].to_numpy()
        confs = g.conf.to_numpy()
        free = list(range(len(g)))
        for t in sorted(active, key=lambda t: -t.score):  # strong tracklets choose first
            if not free:
                break
            d = np.linalg.norm(pts[free] - t.predict(frame), axis=1)
            j = int(np.argmin(d))
            if d[j] <= t.gate():
                k = free.pop(j)
                t.frames.append(int(frame))
                t.pts.append(pts[k])
                t.confs.append(float(confs[k]))
        for k in free:
            active.append(Tracklet(next_id, [int(frame)], [pts[k]], [float(confs[k])]))
            next_id += 1
    return done + active


def classify_tracklets(tracklets: list[Tracklet], roi: np.ndarray | None = None) -> dict[int, str]:
    """Tracklet id -> 'moving' | 'short' | 'static' | 'off_court'."""
    kinds = {}
    for t in tracklets:
        if len(t.frames) < MIN_DETECTIONS:
            kinds[t.id] = "short"
        elif t.speed < STATIC_SPEED_PX:
            kinds[t.id] = "static"
        elif roi is not None and _in_roi_fraction(t, roi) < MIN_IN_ROI:
            kinds[t.id] = "off_court"
        else:
            kinds[t.id] = "moving"
    return kinds


def track_ball(
    dets: pd.DataFrame, fps: float, roi: np.ndarray | None = None, n_frames: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Per-frame ball in play + summary stats. `dets`: frame, u, v, conf (all detections);
    `n_frames`: number of frames analysed (for coverage; default: last detection frame + 1)."""
    used = dets[dets.conf >= DET_CONF]
    tracklets = link(used)
    kinds = classify_tracklets(tracklets, roi)
    moving = sorted((t for t in tracklets if kinds[t.id] == "moving"), key=lambda t: -t.score)

    claimed: dict[int, tuple] = {}
    for t in moving:
        for frame, row in _expand(t).items():
            claimed.setdefault(frame, row)
    rows = [(f, f / fps, *claimed[f]) for f in sorted(claimed)]
    ball = pd.DataFrame(rows, columns=BALL_COLUMNS)

    counts = pd.Series(kinds).value_counts().to_dict()
    if n_frames is None:
        n_frames = int(dets.frame.max()) + 1 if len(dets) else 0
    stats = {
        "detections": int(len(dets)),
        "detections_used": int(len(used)),
        "det_conf": DET_CONF,
        "tracklets": {k: int(counts.get(k, 0)) for k in ("moving", "static", "off_court", "short")},
        "frames_with_ball": int(len(ball)),
        "frames_detected": int((ball.state == "detected").sum()),
        "frames_interpolated": int((ball.state == "interpolated").sum()),
        "coverage": round(len(ball) / n_frames, 3) if n_frames else 0.0,
    }
    return ball, stats


def _expand(t: Tracklet) -> dict[int, tuple]:
    """Tracklet -> {frame: (u, v, conf, state, track)}, filling short gaps."""
    out = {}
    for i, f in enumerate(t.frames):
        u, v = t.pts[i]
        out[f] = (float(u), float(v), t.confs[i], "detected", t.id)
        if i + 1 < len(t.frames):
            gap = t.frames[i + 1] - f
            if 1 < gap <= MAX_INTERP_GAP + 1:
                for k in range(1, gap):
                    w = k / gap
                    p = t.pts[i] * (1 - w) + t.pts[i + 1] * w
                    out[f + k] = (float(p[0]), float(p[1]), np.nan, "interpolated", t.id)
    return out


def _in_roi_fraction(t: Tracklet, roi: np.ndarray) -> float:
    h, w = roi.shape
    p = np.asarray(t.pts).round().astype(int)
    inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    ok = np.zeros(len(p), bool)
    ok[inside] = roi[p[inside, 1], p[inside, 0]]
    return float(ok.mean())
