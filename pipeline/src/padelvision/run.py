"""`padelvision analyze`: runs the player stages and writes everything to one run folder.

runs/<name>/
  video.json            stage 0  video metadata
  court.json            stage 1  calibration used (copy)
  court_overlay.png     stage 1  court lines projected on a frame (check alignment)
  detections.parquet    stage 2  raw person detections + tracker ids (expensive; cached)
  detections.json       stage 2  settings the cache was made with (re-run if they differ)
  players.parquet       stage 3  4 player identities with court positions
  tracks.parquet        stage 6  smoothed positions
  stats.json            stage 6  movement metrics
  heatmaps.png          stage 7
  ball_detections.parquet  stage 4  raw ball detections, every frame (expensive; cached)
  ball_detections.json     stage 4  settings the cache was made with
  ball.parquet             stage 4  ball in play per frame (image px; detected / interpolated)
  ball_stats.json          stage 4  tracking summary
  events.parquet           stage 5  hits / handling / floor bounces (with court position) / walls
  rallies.json             stage 5  rallies (segments with an exchange) + other ball activity
                                    + rally stats (counts, placement, shot speed estimate)
  placement.png            stage 7  where each team's shots landed
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import pandas as pd

from padelvision.analytics import movement_stats, rally_stats, smooth_tracks
from padelvision.ball.candidates import court_roi
from padelvision.ball.track import track_ball
from padelvision.court.calibration import CourtCalibration
from padelvision.court.draw import draw_court_overlay
from padelvision.models import weights_path
from padelvision.players import assign_players, run_detection
from padelvision.video import probe
from padelvision.visuals import save_heatmaps, save_placement_map


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

    det_path, det_meta = out / "detections.parquet", out / "detections.json"
    params = {"model": model, "stride": stride, "max_frames": max_frames, "imgsz": imgsz}
    cached = det_meta.exists() and json.loads(det_meta.read_text()) == params
    if det_path.exists() and cached and not force:
        dets = pd.read_parquet(det_path)
        print(f"[2] detections: cached ({len(dets)} boxes); use --force to recompute")
    else:
        from padelvision.models.person_tracker import PersonTracker

        tracker = PersonTracker(weights_path(model), device=device, imgsz=imgsz)
        print(f"[2] detections: {model} on device={tracker.device}, stride={stride}")
        dets = run_detection(tracker, video, info.fps, info.frame_count, stride, max_frames)
        dets.to_parquet(det_path)
        det_meta.write_text(json.dumps(params))
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


BALL_DETECT_CONF = 0.05  # keep weak detections in the cache; the tracker applies its own threshold


def ball_track(
    video: str | Path,
    out_dir: str | Path,
    model: str = "ball-detector",
    max_frames: int | None = None,
    imgsz: int = 1280,
    device: str | int | None = None,
    force: bool = False,
) -> dict:
    """Stage 4: detect the ball on every frame (cached), then track the ball in play."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    (out / "video.json").write_text(json.dumps(info.to_dict(), indent=2))

    det_path, det_meta = out / "ball_detections.parquet", out / "ball_detections.json"
    params = {
        "model": model,
        "max_frames": max_frames,
        "imgsz": imgsz,
        "conf": BALL_DETECT_CONF,
        "columns": "frame,u,v,conf,size",
    }  # older caches lack size: re-detect
    cached = det_meta.exists() and json.loads(det_meta.read_text()) == params
    if det_path.exists() and cached and not force:
        dets = pd.read_parquet(det_path)
        print(f"[4] ball detections: cached ({len(dets)}); use --force to recompute")
    else:
        from padelvision.models.ball_detectors import YoloBall

        det = YoloBall(
            weights_path(model), classes=[0], imgsz=imgsz, conf=BALL_DETECT_CONF, device=device
        )
        print(f"[4] ball detections: {model} on device={det.device}")
        dets = det.detect_video(video, max_frames=max_frames)
        dets.to_parquet(det_path)
        det_meta.write_text(json.dumps(params))
        print(f"    {len(dets)} detections")
    limit = min(info.frame_count, max_frames) if max_frames else info.frame_count
    return _ball_stages(dets, info.fps, limit, out)


def ball_retrack(run_dir: str | Path) -> dict:
    """Re-run ball tracking from cached ball detections (no video or GPU needed)."""
    out = Path(run_dir)
    info = json.loads((out / "video.json").read_text())
    meta = json.loads((out / "ball_detections.json").read_text())
    limit = min(info["frame_count"], meta["max_frames"] or info["frame_count"])
    dets = pd.read_parquet(out / "ball_detections.parquet")
    return _ball_stages(dets, info["fps"], limit, out)


def _ball_stages(dets: pd.DataFrame, fps: float, n_frames: int, out: Path) -> dict:
    court = out / "court.json"
    roi = court_roi(CourtCalibration.load(court)) if court.exists() else None
    ball, stats = track_ball(dets, fps, roi, n_frames)
    stats["roi"] = roi is not None
    ball.to_parquet(out / "ball.parquet")
    (out / "ball_stats.json").write_text(json.dumps(stats, indent=2))
    print(
        f"[4] ball in play on {stats['frames_with_ball']} of {n_frames} frames "
        f"({stats['coverage']:.0%}; {stats['frames_interpolated']} interpolated); "
        f"tracklets {stats['tracklets']}"
    )
    return stats


def events(run_dir: str | Path) -> dict:
    """Stage 5: events + rallies from ball.parquet and players.parquet (no video or GPU)."""
    from padelvision.events import activity_segments, detect_events

    out = Path(run_dir)
    fps = json.loads((out / "video.json").read_text())["fps"]
    ball = pd.read_parquet(out / "ball.parquet")
    players = pd.read_parquet(out / "players.parquet")
    cal = CourtCalibration.load(out / "court.json")
    ev = detect_events(ball, players, cal, fps)
    ev.to_parquet(out / "events.parquet")
    segments = activity_segments(ball, ev, fps)
    rallies = [s for s in segments if s["exchange"]]
    bounces = ev[ev.kind == "bounce"]
    summary = {
        "events": {k: int(v) for k, v in ev.kind.value_counts().items()},
        "ball_size_known": bool(ball["size"].notna().any()) if "size" in ball else False,
        "bounces_in_court": int(bounces.in_court.eq(True).sum()),
        "hits_by_player": {
            str(int(k)): int(v) for k, v in ev[ev.kind == "hit"].player.value_counts().items()
        },
        "rallies": rallies,
        # ball activity without an exchange (handling, warm-up, or no hit seen): not rallies
        "other_activity": [s for s in segments if not s["exchange"]],
        "stats": rally_stats(ev, rallies, players, fps),
    }
    (out / "rallies.json").write_text(json.dumps(summary, indent=2))
    save_placement_map(summary["stats"], out / "placement.png")
    speed = summary["stats"].get("shot_speed_estimate", {})
    print(
        f"[5] events {summary['events']}; {len(rallies)} rallies "
        f"({len(segments) - len(rallies)} other activity segments); "
        f"hits by player {summary['hits_by_player']}\n"
        f"[6] rally stats + placement.png; shot speed estimate: "
        f"{speed.get('median_kmh', speed.get('insufficient_data'))} "
        f"(median km/h, {speed.get('samples', 0)} shots)"
    )
    return summary
