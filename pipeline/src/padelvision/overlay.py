"""Stats overlay (HUD) for videos: a band along the bottom of the frame with each player's
running stats and the current rally, drawn from the run's result files.

Only measured values are shown. A player tracked too little for stats (stats.json says
insufficient_data) shows "n/a"; a player not seen at this moment shows "-" for speed. Ball
speed is never shown for a ball in flight, only the hit-to-landing ESTIMATE of the last
measured shot, marked "est.".
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from padelvision.analytics.movement import MAX_SPEED_MPS, running_distance
from padelvision.players.identify import SLOT_TEAM

BAND_H = 58  # px at 720p; scaled with the frame height
SPEED_MAX_GAP = 2  # frames: a speed older than this isn't "now"
TEAM_BGR = {"near": (80, 200, 80), "far": (60, 140, 255)}


class StatsHud:
    def __init__(
        self,
        fps: float,
        tracks: pd.DataFrame | None = None,
        stats: dict | None = None,
        rallies: dict | None = None,
        stride: int = 1,
    ):
        """`stats`: stats.json; `rallies`: rallies.json (with "stats" from stage 5/6)."""
        self.fps = fps
        players = (stats or {}).get("players", {})
        self.no_data = {int(p) for p, v in players.items() if "insufficient_data" in v}
        self.dist: dict[int, pd.Series] = {}
        self.speed: dict[int, pd.Series] = {}
        if tracks is not None and len(tracks):
            self.dist = running_distance(tracks, fps, stride)
            for p, tr in tracks.groupby("player"):
                v = np.hypot(tr.vx, tr.vy).to_numpy(dtype=float, copy=True)
                v[v > MAX_SPEED_MPS] = np.nan
                self.speed[int(p)] = pd.Series(v * 3.6, index=tr.frame.to_numpy()).sort_index()
        rallies = rallies or {}
        self.rallies = rallies.get("rallies", [])
        shots = rallies.get("stats", {}).get("shots", [])
        cols = ["frame", "player", "team", "speed_kmh", "speed_range_kmh"]
        self.hits = pd.DataFrame(shots, columns=cols)
        self.hits = self.hits.sort_values("frame") if len(self.hits) else self.hits

    @classmethod
    def from_run(cls, run_dir: str | Path) -> StatsHud | None:
        """Build from a run folder (tracks.parquet, stats.json, rallies.json), or None if
        it has no player stats."""
        run = Path(run_dir)
        if not (run / "tracks.parquet").exists() or not (run / "stats.json").exists():
            return None
        fps = json.loads((run / "video.json").read_text())["fps"]
        stats = json.loads((run / "stats.json").read_text())
        rallies_path = run / "rallies.json"
        rallies = json.loads(rallies_path.read_text()) if rallies_path.exists() else None
        stride = stats.get("analysed", {}).get("stride", 1)
        return cls(fps, pd.read_parquet(run / "tracks.parquet"), stats, rallies, stride)

    # --- values at a frame ------------------------------------------------------------------
    def distance_m(self, player: int, frame: int) -> float | None:
        s = self.dist.get(player)
        if s is None or player in self.no_data:
            return None
        i = s.index.searchsorted(frame, side="right")
        return float(s.iloc[i - 1]) if i else 0.0

    def speed_kmh(self, player: int, frame: int) -> float | None:
        s = self.speed.get(player)
        if s is None or player in self.no_data:
            return None
        i = s.index.searchsorted(frame, side="right")
        if not i or frame - s.index[i - 1] > SPEED_MAX_GAP or not np.isfinite(s.iloc[i - 1]):
            return None
        return float(s.iloc[i - 1])

    def hits_so_far(self, player: int, frame: int) -> int:
        if not len(self.hits):
            return 0
        return int(((self.hits.player == player) & (self.hits.frame <= frame)).sum())

    def rally_at(self, frame: int) -> dict | None:
        return next((r for r in self.rallies if r["start_frame"] <= frame <= r["end_frame"]), None)

    def rally_line(self, frame: int) -> str:
        r = self.rally_at(frame)
        clock = f"{frame / self.fps:6.1f}s   "
        if r is None:
            return clock + "between points"
        h = self.hits[(self.hits.frame >= r["start_frame"]) & (self.hits.frame <= frame)]
        text = clock + f"RALLY {r['id']}  hits {len(h)}"
        measured = h[h.speed_kmh.notna()]
        if len(measured):
            last = measured.iloc[-1]
            rng = last.speed_range_kmh
            span = f" ({rng[0]}-{rng[1]})" if isinstance(rng, list | tuple) else ""
            text += f"  last shot ~{last.speed_kmh:.0f}{span} km/h est."
        return text

    def player_line(self, player: int, frame: int) -> str:
        if player in self.no_data or player not in self.dist:
            return f"P{player}  n/a"
        d, v = self.distance_m(player, frame), self.speed_kmh(player, frame)
        speed = "-" if v is None else f"{v:.0f} km/h"
        n = self.hits_so_far(player, frame)
        return f"P{player}  {d:.0f} m  {speed}  {n} hit{'' if n == 1 else 's'}"

    # --- drawing ------------------------------------------------------------------------------
    def draw(self, img: np.ndarray, frame: int) -> None:
        h, w = img.shape[:2]
        k = h / 720
        band = round(BAND_H * k)
        y0 = h - band
        img[y0:] = (img[y0:] * 0.35).astype(img.dtype)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, max(1, round(1.4 * k))
        cv2.putText(img, self.rally_line(frame), (round(12 * k), y0 + round(22 * k)), font,
                    0.6 * k, (255, 255, 255), thick, cv2.LINE_AA)  # fmt: skip
        col_w = w / 4
        for i, (player, team) in enumerate(SLOT_TEAM.items()):
            x = round(12 * k + i * col_w)
            cv2.putText(img, self.player_line(player, frame), (x, y0 + round(46 * k)), font,
                        scale, TEAM_BGR[team], thick, cv2.LINE_AA)  # fmt: skip
