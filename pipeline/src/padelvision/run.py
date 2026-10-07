"""`padelvision analyze`: runs the player stages and writes everything to one run folder.

runs/<name>/
  video.json            stage 0  video metadata
  court.json            stage 1  calibration used (copy)
  court_overlay.png     stage 1  court lines projected on a frame (check alignment)
  detections.parquet    stage 2  raw person detections + tracker ids (expensive; cached)
  players.parquet       stage 3  4 player identities with court positions
  tracks.parquet        stage 6  smoothed positions
  stats.json            stage 6  movement metrics
  heatmaps.png          stage 7
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import pandas as pd

from padelvision.analytics import movement_stats, smooth_tracks
from padelvision.court.calibration import CourtCalibration
from padelvision.court.draw import draw_court_overlay
from padelvision.models import weights_path
from padelvision.players import assign_players, run_detection
from padelvision.video import probe
from padelvision.visuals import save_heatmaps


def analyze(
    video: str | Path,
    court_json: str | Path,
    out_dir: str | Path,
    model: str = "person-detector",
    stride: int = 1,
    max_frames: int | None = None,
    imgsz: int = 1280,
    device: str | int | None = None,
    force: bool = False,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = probe(video)
    (out / "video.json").write_text(json.dumps(info.to_dict(), indent=2))
    print(f"[0] video: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration_s:.0f} s")

    cal = CourtCalibration.load(court_json)
    if tuple(cal.image_size) != (info.width, info.height):
        raise ValueError(
            f"court.json was made on a {cal.image_size} image but the video is "
            f"{info.width}x{info.height}; calibrate on a frame from this video"
        )
    cal.save(out / "court.json")
    _save_overlay(video, info.frame_count // 2, cal, out / "court_overlay.png")
    print(f"[1] court: mean keypoint error {cal.reprojection_error_m() * 100:.0f} cm")

    det_path = out / "detections.parquet"
    if det_path.exists() and not force:
        dets = pd.read_parquet(det_path)
        print(f"[2] detections: cached ({len(dets)} boxes); use --force to recompute")
    else:
        from padelvision.models.person_tracker import PersonTracker

        tracker = PersonTracker(weights_path(model), device=device, imgsz=imgsz)
        print(f"[2] detections: {model} on device={tracker.device}, stride={stride}")
        dets = run_detection(tracker, video, info.fps, info.frame_count, stride, max_frames)
        dets.to_parquet(det_path)
        print(f"    {len(dets)} boxes")

    limit = min(info.frame_count, max_frames) if max_frames else info.frame_count
    return _player_stages(dets, cal, info.fps, stride, limit, out)


def restats(run_dir: str | Path, court_json: str | Path | None = None) -> dict:
    """Re-run stages 3-7 from a run folder's cached detections. No video or GPU needed,
    so identity/metric logic can be tuned locally on detections downloaded from Kaggle.

    Needs video.json, detections.parquet and stats.json (for stride / analysed length)
    in `run_dir`; uses `court_json` if given, else the run's court.json.
    """
    out = Path(run_dir)
    info = json.loads((out / "video.json").read_text())
    analysed = json.loads((out / "stats.json").read_text())["analysed"]
    cal = CourtCalibration.load(court_json or out / "court.json")
    if court_json:
        cal.save(out / "court.json")
    dets = pd.read_parquet(out / "detections.parquet")
    limit = round(analysed["duration_s"] * info["fps"])
    return _player_stages(dets, cal, info["fps"], analysed["stride"], limit, out)


def _player_stages(dets, cal, fps: float, stride: int, limit: int, out: Path) -> dict:
    players = assign_players(dets, cal)
    players.to_parquet(out / "players.parquet")
    print(f"[3] players: {len(players)} rows, {int((~players.valid).sum())} with feet cut off")

    analysed = math.ceil(limit / stride)
    tracks = smooth_tracks(players, fps, stride)
    tracks.to_parquet(out / "tracks.parquet")
    stats = movement_stats(tracks, players, fps, stride, analysed)
    stats["analysed"] = {"frames": analysed, "stride": stride, "duration_s": limit / fps}
    stats["court_error_m"] = cal.reprojection_error_m()
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print("[6] stats.json written")

    save_heatmaps(tracks, stats, out / "heatmaps.png")
    print("[7] heatmaps.png written")
    return stats


def _save_overlay(video, frame_idx: int, cal: CourtCalibration, path: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(path), draw_court_overlay(frame, cal))
