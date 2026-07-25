import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import constants

logger = logging.getLogger(__name__)

def _draw_court_outline(ax):
    """Draw a simple padel court schematic on the given axes."""
    ax.set_facecolor("#154c79") # Padel blue
    
    court_w = constants.COURT_WIDTH
    court_h = constants.COURT_LENGTH

    # Outer boundary
    court = patches.Rectangle((0, 0), court_w, court_h,
                               linewidth=2, edgecolor="white", facecolor="#154c79")
    ax.add_patch(court)

    # Net
    mid_y = court_h / 2
    ax.plot([0, court_w], [mid_y, mid_y], "white", lw=3) # Thicker net

    # Service boxes
    service_line_dist = constants.SERVICE_LINE_DIST
    ax.plot([0, court_w], [mid_y - service_line_dist, mid_y - service_line_dist], "white", lw=1)
    ax.plot([0, court_w], [mid_y + service_line_dist, mid_y + service_line_dist], "white", lw=1)

    # Center service line
    ax.plot([court_w / 2, court_w / 2], [mid_y - service_line_dist, mid_y + service_line_dist], "white", lw=1)

    # Baselines
    ax.plot([0, court_w], [0, 0], "white", lw=2)
    ax.plot([0, court_w], [court_h, court_h], "white", lw=2)


def generate_heatmap(
    player_mini_court_positions: list[dict],
    mini_court,
    output_dir: str,
) -> dict[str, str]:
    """
    Generate heatmap PNGs for players 1 to 4.

    Args:
        player_mini_court_positions: list of per-frame dicts { player_id: (x, y) }
        mini_court: MiniCourt instance (used for coordinate normalization)
        output_dir: directory to save the PNG files

    Returns:
        { "player_1": "/path/to/heatmap_p1.png", ... }
    """
    os.makedirs(output_dir, exist_ok=True)

    court_w = mini_court.get_width_of_mini_court()
    court_start_x = mini_court.court_start_x
    court_start_y = mini_court.court_start_y
    court_end_y = mini_court.court_end_y
    court_h_pixels = court_end_y - court_start_y

    paths = {}
    actual_court_w = constants.COURT_WIDTH
    actual_court_h = constants.COURT_LENGTH

    for player_id in [1, 2, 3, 4]:
        xs, ys = [], []
        for frame in player_mini_court_positions:
            if player_id in frame:
                px, py = frame[player_id]
                # Normalize to [0, 1] within the mini-court bounds
                norm_x = (px - court_start_x) / max(court_w, 1)
                norm_y = (py - court_start_y) / max(court_h_pixels, 1)
                xs.append(np.clip(norm_x, 0, 1) * actual_court_w)
                ys.append(np.clip(norm_y, 0, 1) * actual_court_h)

        fig, ax = plt.subplots(figsize=(5, 10))
        _draw_court_outline(ax)

        logger.info("Player %d: %d position samples", player_id, len(xs))
        if xs:
            grid_size = 100
            heatmap_data, xedges, yedges = np.histogram2d(
                xs, ys,
                bins=grid_size,
                range=[[0, actual_court_w], [0, actual_court_h]],
            )
            heatmap_data = gaussian_filter(heatmap_data, sigma=3)
            heatmap_data = heatmap_data / (heatmap_data.max() + 1e-8)

            ax.imshow(
                heatmap_data.T,
                extent=[0, actual_court_w, 0, actual_court_h],
                origin="lower",
                cmap="YlOrRd",
                alpha=0.65,
                aspect="auto",
                vmin=0,
                vmax=1,
            )

        ax.set_xlim(0, actual_court_w)
        ax.set_ylim(0, actual_court_h)
        ax.axis("off")
        ax.set_title(f"Player {player_id} — Court Coverage", color="white", fontsize=12, pad=8)
        fig.patch.set_facecolor("#1a1a2e")

        out_path = os.path.join(output_dir, f"heatmap_p{player_id}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths[f"player_{player_id}"] = out_path

    return paths
