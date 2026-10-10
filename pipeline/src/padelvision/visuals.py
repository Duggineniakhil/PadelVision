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
