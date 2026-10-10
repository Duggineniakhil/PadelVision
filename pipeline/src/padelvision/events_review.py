"""Hit/bounce labelling pack: time windows to label exhaustively, with every frame extracted.

    padelvision event-review-pack <video> --run runs/match1 --out event_review   (Kaggle)
    padelvision label-events data/event_review                                  (local GUI)

Windows: every rally and every other-activity segment with a hit (from rallies.json, padded
by PAD_S), plus `n_random` random windows elsewhere so that play the detector missed entirely
gets labelled too. Every hit and floor bounce inside a finished window is marked; a frame
without a mark means "no event", which makes recall measurable. The total length is capped at
`max_total_s` (rallies first, then random windows, then other activity).

The pack holds frames/<frame>.jpg, a copy of the run's stage outputs (run/) for overlays and
scoring, and pack.json with the windows.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from padelvision.ball.candidates import read_frames

PAD_S = 1.0
N_RANDOM = 3
RANDOM_S = 6.0
MAX_TOTAL_S = 120.0
JPEG_QUALITY = 88
RUN_FILES = ["video.json", "court.json", "players.parquet", "ball.parquet", "events.parquet",
             "rallies.json"]  # fmt: skip
PRIORITY = {"rally": 0, "random": 1, "other": 2}


def choose_windows(
    summary: dict,
    n_frames: int,
    fps: float,
    n_random: int = N_RANDOM,
    max_total_s: float = MAX_TOTAL_S,
    seed: int = 0,
) -> list[dict]:
    """rallies.json summary -> [{id, start, end, source}] (inclusive frames, sorted by time)."""
    pad = round(PAD_S * fps)
    found = [(s, "rally") for s in summary.get("rallies", [])]
    found += [(s, "other") for s in summary.get("other_activity", []) if s.get("hits", 0) > 0]
    wins = _merge([
        [max(0, s["start_frame"] - pad), min(n_frames - 1, s["end_frame"] + pad), src]
        for s, src in found
    ])  # fmt: skip

    length = round(RANDOM_S * fps)
    rng = np.random.default_rng(seed)
    randoms: list[list] = []
    for _ in range(200 * max(n_random, 1)):
        if len(randoms) >= n_random or n_frames <= length:
            break
        s = int(rng.integers(0, n_frames - length))
        cand = [s, s + length - 1, "random"]
        if not any(_overlap(cand, w, pad) for w in wins + randoms):
            randoms.append(cand)

    chosen: list[list] = []
    total = 0.0
    for w in sorted(wins + randoms, key=lambda w: (PRIORITY[w[2]], w[0])):
        dur = (w[1] - w[0] + 1) / fps
        if total + dur <= max_total_s:
            chosen.append(w)
            total += dur
    chosen.sort()
    return [
        {"id": i + 1, "start": s, "end": e, "source": src} for i, (s, e, src) in enumerate(chosen)
    ]


def make_event_pack(
    video,
    run_dir,
    out_dir,
    n_random: int = N_RANDOM,
    max_total_s: float = MAX_TOTAL_S,
    seed: int = 0,
) -> Path:
    run, out = Path(run_dir), Path(out_dir)
    info = json.loads((run / "video.json").read_text())
    fps, n_frames = float(info["fps"]), int(info["frame_count"])
    summary = json.loads((run / "rallies.json").read_text())
    windows = choose_windows(summary, n_frames, fps, n_random, max_total_s, seed)
    wanted = {f for w in windows for f in range(w["start"], w["end"] + 1)}

    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "run").mkdir(exist_ok=True)
    for name in RUN_FILES:
        if (run / name).exists():
            shutil.copy(run / name, out / "run" / name)
    if wanted:
        for f, img in read_frames(video, max(wanted) + 1):
            if f in wanted:
                cv2.imwrite(str(out / "frames" / f"{f:06d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])  # fmt: skip
    pack = {"video": str(video), "fps": fps, "windows": windows}
    (out / "pack.json").write_text(json.dumps(pack, indent=1))
    secs = len(wanted) / fps
    kinds = {k: sum(w["source"] == k for w in windows) for k in PRIORITY}
    print(f"event review pack: {len(windows)} windows {kinds}, {secs:.0f} s, {len(wanted)} frames")
    return out


def _merge(wins: list[list]) -> list[list]:
    out: list[list] = []
    for s, e, src in sorted(wins):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2] = min(out[-1][2], src, key=PRIORITY.__getitem__)
        else:
            out.append([s, e, src])
    return out


def _overlap(a, b, gap: int) -> bool:
    return a[0] <= b[1] + gap and b[0] <= a[1] + gap
