"""Runs on local disk: one folder per run under the runs directory.

A run folder holds the pipeline's files (video.json, stats.json, rallies.json, *.parquet, ...)
plus meta.json written here: name, created, status, and the attached match video.
Imports come from the notebooks' zips (run_cache.zip, results.zip, events_review.zip); after an
import the light stages (identities/stats, ball tracking, events) are re-run with the current
code in a background thread. Models never run here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RUN_FILES = {  # everything a run may contain (anything else in an imported zip is ignored)
    "video.json", "court.json", "stats.json", "detections.parquet", "detections.json",
    "players.parquet", "tracks.parquet", "ball_detections.parquet", "ball_detections.json",
    "ball.parquet", "ball_stats.json", "events.parquet", "rallies.json", "heatmaps.png",
    "placement.png", "court_overlay.png", "highlights.mp4", "highlights.json", "condensed.mp4",
    "condensed.json",
}  # fmt: skip
MEDIA_FILES = {"highlights.mp4", "condensed.mp4", "heatmaps.png", "placement.png",
               "court_overlay.png"}  # fmt: skip
# Files that can be replaced in an existing run (rendered outputs; the analysis stays as it is).
UPDATABLE_FILES = MEDIA_FILES | {"highlights.json", "condensed.json"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
ID_RE = re.compile(r"^[a-z0-9-]{1,64}$")


class RunNotFound(Exception):
    pass


class ImportFailed(Exception):
    pass


def runs_dir() -> Path:
    d = Path(os.environ.get("PADELVISION_RUNS", "runs")).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_path(run_id: str) -> Path:
    if not ID_RE.match(run_id):
        raise RunNotFound(run_id)
    p = runs_dir() / run_id
    if not (p / "meta.json").exists():
        raise RunNotFound(run_id)
    return p


def read_meta(run: Path) -> dict:
    return json.loads((run / "meta.json").read_text())


def write_meta(run: Path, **changes) -> dict:
    meta = read_meta(run) if (run / "meta.json").exists() else {}
    meta.update(changes)
    tmp = run / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    for attempt in range(20):  # Windows refuses to replace a file another thread is reading
        try:
            tmp.replace(run / "meta.json")
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)
    return meta


def list_runs() -> list[Path]:
    runs = [p for p in runs_dir().iterdir() if (p / "meta.json").exists()]
    return sorted(runs, key=lambda p: read_meta(p).get("created", ""), reverse=True)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "match"
    return f"{s}-{uuid.uuid4().hex[:6]}"


def import_run(uploads: list[tuple[str, Path]], name: str | None = None) -> str:
    """Create a run from uploaded files: zips (any folder layout) and/or loose run files.
    `uploads`: (original filename, temp path). Returns the new run id; processing starts in
    the background."""
    staging = Path(tempfile.mkdtemp(prefix="pv-import-"))
    try:
        for filename, path in uploads:
            if filename.lower().endswith(".zip"):
                _extract_zip(path, staging / Path(filename).stem)
            elif Path(filename).name in RUN_FILES:
                shutil.copy(path, staging / Path(filename).name)
        found = _collect_run_files(staging)
        if "video.json" not in found:
            raise ImportFailed(
                "No video.json found. Import the run_cache.zip or results.zip from notebook 08 "
                "(or a run folder zipped as a whole)."
            )
        if not ({"stats.json", "detections.parquet"} & set(found)):
            raise ImportFailed("The import has no player data (stats.json or detections.parquet).")
        video = json.loads(found["video.json"].read_text())
        default = Path(video.get("path", "")).stem or "Match"
        run_id = _slug(name or default)
        run = runs_dir() / run_id
        run.mkdir()
        for fname, src in found.items():
            shutil.copy(src, run / fname)
        write_meta(
            run, id=run_id, name=(name or default).strip() or "Match",
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            status="processing", status_message="Preparing the results", video_path=None,
        )  # fmt: skip
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    start_processing(run_id)
    return run_id


def add_files(run_id: str, uploads: list[tuple[str, Path]]) -> list[str]:
    """Replace rendered outputs (highlight videos, their clip lists, images) of an existing run
    from notebook zips. Returns the names of the files updated."""
    run = run_path(run_id)
    staging = Path(tempfile.mkdtemp(prefix="pv-add-"))
    try:
        for filename, path in uploads:
            if filename.lower().endswith(".zip"):
                _extract_zip(path, staging / Path(filename).stem)
            elif Path(filename).name in UPDATABLE_FILES:
                shutil.copy(path, staging / Path(filename).name)
        found = _collect_run_files(staging)
        if "video.json" in found:  # same match? (zips from the notebooks carry it)
            theirs = json.loads(found["video.json"].read_text())
            ours = json.loads((run / "video.json").read_text())
            if theirs.get("frame_count") != ours.get("frame_count"):
                raise ImportFailed(
                    f"These files belong to another video ({theirs.get('frame_count')} frames; "
                    f"this match has {ours.get('frame_count')})."
                )
        updated = sorted(n for n in found if n in UPDATABLE_FILES)
        if not updated:
            raise ImportFailed(
                "No highlight videos or images found. Add highlights.zip from notebook 07 or "
                "results.zip from notebook 08."
            )
        for name in updated:
            shutil.copy(found[name], run / name)
        return updated
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(root)):  # zip slip
                raise ImportFailed(f"Unsafe path in zip: {info.filename}")
            if Path(info.filename).name in RUN_FILES and not info.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)


def _collect_run_files(staging: Path) -> dict[str, Path]:
    """Known run files found anywhere in the staging tree (the newest copy of each name)."""
    found: dict[str, Path] = {}
    for p in staging.rglob("*"):
        if p.is_file() and p.name in RUN_FILES:
            if p.name not in found or p.stat().st_mtime > found[p.name].stat().st_mtime:
                found[p.name] = p
    return found


# --- processing ---------------------------------------------------------------------------
_lock = threading.Lock()


def start_processing(run_id: str) -> None:
    threading.Thread(target=_process, args=(run_id,), daemon=True).start()


def _process(run_id: str) -> None:
    from padelvision import run as stages

    run = runs_dir() / run_id
    t0 = time.time()
    try:
        with _lock:  # one heavy job at a time
            if (run / "detections.parquet").exists() and (run / "stats.json").exists():
                write_meta(run, status_message="Recomputing player identities and movement")
                stages.restats(run)
            if all((run / f).exists() for f in ("ball_detections.parquet", "ball_detections.json")):
                write_meta(run, status_message="Tracking the ball")
                stages.ball_retrack(run)
            if all((run / f).exists() for f in ("ball.parquet", "players.parquet", "court.json")):
                write_meta(run, status_message="Detecting hits, bounces and rallies")
                stages.events(run)
        write_meta(run, status="ready", status_message=None,
                   processed_s=round(time.time() - t0, 1))  # fmt: skip
    except Exception as e:  # surfaced in the UI; the files imported so far stay usable
        write_meta(run, status="error", status_message=f"{type(e).__name__}: {e}")


def attach_video(run_id: str, path: Path, copied: bool = False) -> dict:
    """Attach the match video after checking it is the video this run was made from."""
    from padelvision.video import probe

    run = run_path(run_id)
    if not path.exists() or path.suffix.lower() not in VIDEO_EXT:
        raise ImportFailed(f"Not a video file: {path}")
    info = probe(path)
    expected = json.loads((run / "video.json").read_text())
    # Overlays are drawn in analysis coordinates and scaled to the displayed size, so any
    # resolution works (e.g. the original 4K file of a 1280-wide analysis) as long as it is the
    # same footage: same aspect ratio and frame count.
    same_shape = abs(info.width / info.height - expected["width"] / expected["height"]) < 0.01
    if not same_shape or abs(info.frame_count - expected["frame_count"]) > 2:
        if copied:
            path.unlink(missing_ok=True)
        raise ImportFailed(
            f"This video has {info.frame_count} frames at {info.width}x{info.height}, but the "
            f"analysis was made on {expected['frame_count']} frames at "
            f"{expected['width']}x{expected['height']}. Choose the video this match was analysed "
            "from."
        )
    return write_meta(run, video_path=str(path.resolve()), video_copied=copied)


def video_file(run_id: str) -> Path | None:
    p = read_meta(run_path(run_id)).get("video_path")
    return Path(p) if p and Path(p).exists() else None


def delete_run(run_id: str) -> None:
    run = run_path(run_id)
    shutil.rmtree(run)
