"""Stage 7: heatmap images from smoothed player positions."""

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
