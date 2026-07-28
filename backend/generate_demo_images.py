import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from matplotlib.collections import LineCollection

sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))
import constants
from pipeline.heatmap import _draw_court_outline as draw_heatmap_court
from pipeline.shot_map import _draw_court_outline as draw_shotmap_court

def generate_fake_heatmaps(output_dir):
    actual_court_w = constants.COURT_WIDTH
    actual_court_h = constants.COURT_LENGTH

    for player_id in [1, 2, 3, 4]:
        if player_id == 1:
            center_x, center_y = actual_court_w * 0.25, actual_court_h * 0.15
        elif player_id == 2:
            center_x, center_y = actual_court_w * 0.75, actual_court_h * 0.15
        elif player_id == 3:
            center_x, center_y = actual_court_w * 0.25, actual_court_h * 0.85
        else:
            center_x, center_y = actual_court_w * 0.75, actual_court_h * 0.85
        
        xs = np.random.normal(center_x, actual_court_w * 0.12, 1000)
        ys = np.random.normal(center_y, actual_court_h * 0.12, 1000)
        xs = np.clip(xs, 0, actual_court_w)
        ys = np.clip(ys, 0, actual_court_h)

        fig, ax = plt.subplots(figsize=(5, 10))
        draw_heatmap_court(ax)

        grid_size = 100
        heatmap_data, _, _ = np.histogram2d(
            xs, ys, bins=grid_size, range=[[0, actual_court_w], [0, actual_court_h]]
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
        print(f"Generated {out_path}")

def generate_fake_shotmap(output_dir):
    actual_court_w = constants.COURT_WIDTH
    actual_court_h = constants.COURT_LENGTH

    positions = []
    # Serve from near right
    x, y = actual_court_w * 0.75, actual_court_h * 0.1
    positions.append((x, y))
    
    for _ in range(12):
        if y < actual_court_h / 2:
            x = np.random.uniform(0.1 * actual_court_w, 0.9 * actual_court_w)
            y = np.random.uniform(actual_court_h * 0.6, actual_court_h * 0.95)
        else:
            x = np.random.uniform(0.1 * actual_court_w, 0.9 * actual_court_w)
            y = np.random.uniform(actual_court_h * 0.05, actual_court_h * 0.4)
        
        last_x, last_y = positions[-1]
        num_interp = 8
        for i in range(1, num_interp + 1):
            interp_x = last_x + (x - last_x) * (i / num_interp)
            interp_y = last_y + (y - last_y) * (i / num_interp)
            positions.append((interp_x, interp_y))

    fig, ax = plt.subplots(figsize=(5, 10))
    draw_shotmap_court(ax, draw_zones=True)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    speeds = [0.0]
    for i in range(1, len(positions)):
        d = ((positions[i][0] - positions[i-1][0])**2 + (positions[i][1] - positions[i-1][1])**2)**0.5
        speeds.append(d)
    
    speeds = np.array(speeds)
    max_speed = speeds.max() or 1.0
    norm_speeds = speeds / max_speed

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=1.5, alpha=0.7)
    lc.set_array(norm_speeds[1:])
    lc.set_clim(0, 1)
    ax.add_collection(lc)

    sc = ax.scatter(xs, ys, c=norm_speeds, cmap="plasma", s=10, alpha=0.7, zorder=5, edgecolors="white", linewidths=0.3)
    
    # Optional colorbar, if desired
    # plt.colorbar(sc, ax=ax, label="Relative speed", fraction=0.03, pad=0.04)

    ax.set_xlim(0, actual_court_w)
    ax.set_ylim(0, actual_court_h)
    ax.axis("off")
    ax.set_title("Ball Trajectory Map", color="white", fontsize=12, pad=8)
    fig.patch.set_facecolor("#1a1a2e")

    out_path = os.path.join(output_dir, "shot_map.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "demo_fake_data")
    os.makedirs(output_dir, exist_ok=True)
    generate_fake_heatmaps(output_dir)
    generate_fake_shotmap(output_dir)
    print(f"\nAll fake data successfully generated in: {output_dir}")
