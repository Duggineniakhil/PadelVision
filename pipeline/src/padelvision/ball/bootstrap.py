"""Phase 2 data bootstrap: classical candidates -> pseudo-labels -> human review pack.

Run on Kaggle/Colab (reads the video):
    padelvision ball-bootstrap <video> --run runs/match1
    padelvision ball-review-pack <video> --run runs/match1 --out ball_review

Writes into the run folder:
    ball_candidates.parquet   every candidate blob (frame, u, v, area, color, motion, score)
    ball_pseudo.parquet       one ball position per frame where a ball-like track was found
The review pack (frames + motion maps + pack.json) is downloaded and labelled locally with
`padelvision label-ball`; those labels measure pseudo-label precision and form the eval set.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from padelvision.ball.candidates import court_roi, read_frames, video_candidates
from padelvision.ball.link import pseudo_labels
from padelvision.court.calibration import CourtCalibration


def bootstrap(video, run_dir, max_frames: int | None = None) -> pd.DataFrame:
    run = Path(run_dir)
    cal = CourtCalibration.load(run / "court.json")
    dets = pd.read_parquet(run / "detections.parquet")
    boxes = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in dets.groupby("frame")}
    roi = court_roi(cal)

    start = time.perf_counter()

    def logged(frames):
        for f, img in frames:
            if f and f % 1000 == 0:
                print(f"  candidates: frame {f} ({f / (time.perf_counter() - start):.0f} fps)")
            yield f, img

    cands = video_candidates(logged(read_frames(video, max_frames)), roi, boxes)
    cands.to_parquet(run / "ball_candidates.parquet")
    pseudo = pseudo_labels(cands)
    pseudo.to_parquet(run / "ball_pseudo.parquet")
    n_frames = max_frames or int(json.loads((run / "video.json").read_text())["frame_count"])
    print(
        f"ball: {len(cands)} candidates, {len(pseudo)} pseudo-labelled frames "
        f"({len(pseudo) / max(1, n_frames):.0%} of frames), {pseudo.track.nunique()} tracks"
    )
    return pseudo


def make_review_pack(
    video, run_dir, out_dir, n_pseudo: int = 150, n_random: int = 150, seed: int = 0
) -> Path:
    """Sample frames for human review: some pseudo-labelled (checks precision) and some
    random frames during play (checks recall; these become the eval set)."""
    run, out = Path(run_dir), Path(out_dir)
    pseudo = pd.read_parquet(run / "ball_pseudo.parquet")
    cands = pd.read_parquet(run / "ball_candidates.parquet")
    players = pd.read_parquet(run / "players.parquet")
    n_total = int(json.loads((run / "video.json").read_text())["frame_count"])
    fps = float(json.loads((run / "video.json").read_text())["fps"])

    inner = lambda f: f[(f > 0) & (f < n_total - 1)]  # noqa: E731  (motion needs neighbours)
    pf = inner(pseudo.frame.to_numpy())
    pick_pseudo = (
        pf[np.linspace(0, len(pf) - 1, min(n_pseudo, len(pf))).round().astype(int)]
        if len(pf)
        else pf
    )
    in_play = players.groupby("frame").size()
    play = inner(in_play.index[in_play >= 2].to_numpy())
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(play, pick_pseudo)
    pick_random = np.sort(rng.choice(pool, min(n_random, len(pool)), replace=False))

    proposal = pseudo.set_index("frame")[["u", "v"]]
    best = cands.sort_values("score", ascending=False).drop_duplicates("frame").set_index("frame")
    items = []
    for f, source in [*((f, "pseudo") for f in pick_pseudo), *((f, "random") for f in pick_random)]:
        f = int(f)
        if f in proposal.index:
            prop = proposal.loc[f].tolist()
        elif f in best.index:
            prop = best.loc[f, ["u", "v"]].tolist()
        else:
            prop = None
        items.append({"frame": f, "t": f / fps, "source": source, "proposal": prop})
    items.sort(key=lambda it: it["frame"])

    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "motion").mkdir(parents=True, exist_ok=True)
    wanted = {it["frame"] for it in items}
    need = wanted | {f - 1 for f in wanted} | {f + 1 for f in wanted}
    buf: dict[int, np.ndarray] = {}
    for f, img in read_frames(video, max(need) + 1):
        if f in need:
            buf[f] = img
        if f - 1 in wanted:
            _save_item(out, f - 1, buf[f - 2], buf[f - 1], buf[f])
        for old in [k for k in buf if k < f - 2]:
            del buf[old]
    (out / "pack.json").write_text(json.dumps({"video": str(video), "items": items}, indent=1))
    print(
        f"review pack: {len(items)} frames ({len(pick_pseudo)} pseudo, {len(pick_random)} random)"
    )
    return out


def _save_item(out: Path, f: int, prev, cur, nxt) -> None:
    cv2.imwrite(str(out / "frames" / f"{f:06d}.jpg"), cur, [cv2.IMWRITE_JPEG_QUALITY, 92])
    g = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (prev, cur, nxt)]
    motion = np.minimum(cv2.absdiff(g[1], g[0]), cv2.absdiff(g[2], g[1]))
    cv2.imwrite(
        str(out / "motion" / f"{f:06d}.png"),
        np.clip(motion.astype(int) * 4, 0, 255).astype(np.uint8),
    )
