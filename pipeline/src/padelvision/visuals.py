"""Stage 7: heatmap images from smoothed player positions; shot placement map from rally stats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from padelvision.court.geometry import COURT_LENGTH_M, COURT_LINES, COURT_WIDTH_M
from padelvision.players.identify import SLOT_TEAM

BIN_M = 0.25
MARGIN_M = 1.0


def save_heatmaps(tracks: pd.DataFrame, stats: dict, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    hw, hl = COURT_WIDTH_M / 2 + MARGIN_M, COURT_LENGTH_M / 2 + MARGIN_M
    xbins = np.arange(-hw, hw + BIN_M, BIN_M)
    ybins = np.arange(-hl, hl + BIN_M, BIN_M)

    fig, axes = plt.subplots(1, 4, figsize=(12, 6.5), facecolor="#0f172a")
    for ax, player in zip(axes, SLOT_TEAM, strict=True):
        ax.set_facecolor("#1e3a5f")
        info = stats["players"][str(player)]
        pts = tracks[tracks.player == player]
        if "insufficient_data" in info or pts.empty:
            ax.text(0, 0, "not enough\ndata", color="white", ha="center", va="center")
        else:
            hist, _, _ = np.histogram2d(pts.x, pts.y, bins=[xbins, ybins])
            hist = gaussian_filter(hist, sigma=2)
            hist = np.ma.masked_less(hist / hist.max(), 0.02)
            ax.imshow(
                hist.T,
                origin="lower",
                extent=(-hw, hw, -hl, hl),
                cmap="inferno",
                alpha=0.9,
                aspect="equal",
            )
        for (x0, y0), (x1, y1) in COURT_LINES:
            ax.plot([x0, x1], [y0, y1], color="white", lw=1)
        ax.set_xlim(-hw, hw)
        ax.set_ylim(-hl, hl)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Player {player} ({SLOT_TEAM[player]})", color="white")
        ax.text(0, -hl + 0.3, "camera side", color="#94a3b8", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


PLACEMENT_COLORS = {"near": "#50c850", "far": "#ff8c3c"}  # match the preview's team colours


def save_placement_map(stats: dict, path: str | Path) -> None:
    """Where each team's shots landed (in-rally floor bounces on the other side of the net).

    `stats`: the output of `padelvision.analytics.rally_stats`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hw, hl = COURT_WIDTH_M / 2 + MARGIN_M, COURT_LENGTH_M / 2 + MARGIN_M
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 6.5), facecolor="#0f172a")
    placement = stats.get("placement", {})
    for ax, team in zip(axes, ("near", "far"), strict=True):
        ax.set_facecolor("#1e3a5f")
        for (x0, y0), (x1, y1) in COURT_LINES:
            ax.plot([x0, x1], [y0, y1], color="white", lw=1)
        info = placement.get(team, {})
        bounces = info.get("bounces", [])
        if bounces:
            ax.scatter([b["x_m"] for b in bounces], [b["y_m"] for b in bounces], s=40,
                       color=PLACEMENT_COLORS[team], edgecolors="white", linewidths=0.5,
                       zorder=3)  # fmt: skip
            z = info["zones"]
            text = f"net {z['net']}  transition {z['transition']}  back {z['back']}"
        else:
            text = "not enough data"
            ax.text(0, hl / 2 if team == "near" else -hl / 2, "no landings seen",
                    color="white", ha="center", va="center")  # fmt: skip
        ax.set_xlim(-hw, hw)
        ax.set_ylim(-hl, hl)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{team.capitalize()} team's shots\n{info.get('landed', 0)} of {info.get('shots', 0)} "
            "landings seen",
            color="white", fontsize=10,
        )  # fmt: skip
        ax.text(0, -hl - 0.6, text, color="#cbd5e1", ha="center", va="top", fontsize=8)
        ax.text(0, -hl + 0.3, "camera side", color="#94a3b8", ha="center", fontsize=8)
    fig.text(0.5, 0.01, "Court positions are approximate (about 0.3 m near / 1 m far)",
             color="#94a3b8", ha="center", fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


CARD_MIN_TRACKED_S = 1.0  # less tracking than this in a rally: "not enough data"


def _speed_label(n: int, shot: dict) -> str:
    rng = shot.get("speed_range_kmh")
    span = f" ({rng[0]}-{rng[1]})" if rng else ""
    return f"#{n} ~{shot['speed_kmh']:.0f}{span} km/h"


def rally_card(
    rally: dict, tracks: pd.DataFrame, shots: list[dict], fps: float, size: tuple[int, int]
) -> np.ndarray:
    """Summary image (BGR, `size` = (w, h)) for one rally: each player's heatmap during it and
    its shot map. Shot lines run from the hitter's feet to the landing bounce, both measured
    on the ground; the flight in between is not measured, so it is not drawn as a curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    w, h = size
    hw, hl = COURT_WIDTH_M / 2 + MARGIN_M, COURT_LENGTH_M / 2 + MARGIN_M
    xbins = np.arange(-hw, hw + BIN_M, BIN_M)
    ybins = np.arange(-hl, hl + BIN_M, BIN_M)
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#0f172a")
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.7], left=0.02, right=0.98,
                            top=0.84, bottom=0.08, wspace=0.08)  # fmt: skip
    fig.text(0.5, 0.93, f"Rally {rally['id']}  -  {rally['hits']} hits  -  "
             f"{rally['duration_s']:.1f} s", color="white", ha="center", fontsize=20)  # fmt: skip
    in_rally = tracks[tracks.frame.between(rally["start_frame"], rally["end_frame"])]

    def court(ax):
        ax.set_facecolor("#1e3a5f")
        for (x0, y0), (x1, y1) in COURT_LINES:
            ax.plot([x0, x1], [y0, y1], color="white", lw=0.8)
        ax.set_xlim(-hw, hw)
        ax.set_ylim(-hl, hl)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for i, (player, team) in enumerate(SLOT_TEAM.items()):
        ax = fig.add_subplot(grid[0, i])
        court(ax)
        pts = in_rally[in_rally.player == player]
        if len(pts) / fps < CARD_MIN_TRACKED_S:
            ax.text(0, 0, "not enough\ndata", color="white", ha="center", va="center", fontsize=9)
        else:
            hist, _, _ = np.histogram2d(pts.x, pts.y, bins=[xbins, ybins])
            hist = gaussian_filter(hist, sigma=2)
            hist = np.ma.masked_less(hist / hist.max(), 0.04)
            ax.imshow(hist.T, origin="lower", extent=(-hw, hw, -hl, hl), cmap="inferno",
                      alpha=0.9, aspect="equal")  # fmt: skip
        ax.set_title(f"P{player} ({team})", color="white", fontsize=11)

    ax = fig.add_subplot(grid[0, 4])
    court(ax)
    mine = [s for s in shots if rally["start_frame"] <= s["frame"] <= rally["end_frame"]]
    landed = 0
    for n, s in enumerate(mine, start=1):
        color = PLACEMENT_COLORS[s["team"]]
        start, b = s.get("hitter_xy"), s.get("bounce")
        if start is not None and b is not None:
            landed += 1
            ax.annotate("", xy=(b["x_m"], b["y_m"]), xytext=start,
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 2})  # fmt: skip
            ax.scatter([b["x_m"]], [b["y_m"]], s=30, color=color, edgecolors="white", zorder=3)
            if s.get("speed_kmh") is not None:
                # Next to its own landing point (towards the middle), in the shot's colour and
                # with its number, so crossing arrows can't be confused.
                right = b["x_m"] < 0
                ax.text(b["x_m"] + (0.5 if right else -0.5), b["y_m"],
                        _speed_label(n, s), color=color, fontsize=8,
                        fontweight="bold", ha="left" if right else "right", va="center",
                        bbox={"facecolor": "#0f172a", "edgecolor": "none", "alpha": 0.75,
                              "pad": 1.5}, zorder=6)  # fmt: skip
        if start is not None:
            ax.scatter([start[0]], [start[1]], s=110, color=color, edgecolors="white", zorder=4)
            ax.text(start[0], start[1], str(n), color="black", fontsize=8, ha="center",
                    va="center", zorder=5)  # fmt: skip
    ax.set_title(f"Shots: {landed} of {len(mine)} landings seen", color="white", fontsize=11)
    fig.text(0.98, 0.025, "Numbers: hitter's position at each shot, in order. Arrows: to where it "
             "landed. km/h: estimate. Flight not measured.", color="#94a3b8", ha="right",
             fontsize=9)  # fmt: skip
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    img = np.ascontiguousarray(rgba[:, :, 2::-1])  # RGBA -> BGR
    if img.shape[:2] != (h, w):
        import cv2

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img
