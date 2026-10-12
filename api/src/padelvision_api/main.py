"""PadelVision local API.

padelvision-api                      # serves on http://127.0.0.1:8000 (runs in ./runs)
PADELVISION_RUNS=/path/to/runs padelvision-api
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from padelvision_api import calibration, store, views
from padelvision_api import schemas as S

app = FastAPI(title="PadelVision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("PADELVISION_WEB_ORIGINS",
                                 "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["*"], allow_headers=["*"], expose_headers=["Content-Range", "Accept-Ranges"],
)  # fmt: skip


def _run(run_id: str) -> Path:
    try:
        return store.run_path(run_id)
    except store.RunNotFound as e:
        raise HTTPException(404, f"No run '{run_id}'") from e


def _save_uploads(files: list[UploadFile], tmp: Path) -> list[tuple[str, Path]]:
    saved = []
    for i, f in enumerate(files):
        dest = tmp / f"{i}_{Path(f.filename or 'upload').name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out, length=1 << 20)
        saved.append((f.filename or dest.name, dest))
    return saved


@app.get("/api/runs", response_model=list[S.RunListItem])
def list_runs() -> list[S.RunListItem]:
    return [views.list_item(p) for p in store.list_runs()]


@app.post("/api/runs", response_model=S.RunListItem, status_code=201)
def import_run(
    files: Annotated[list[UploadFile], File()], name: Annotated[str | None, Form()] = None
) -> S.RunListItem:
    """Import notebook output zips (run_cache.zip, results.zip, ...) as a new run."""
    tmp = Path(tempfile.mkdtemp(prefix="pv-upload-"))
    try:
        run_id = store.import_run(_save_uploads(files, tmp), name)
    except store.ImportFailed as e:
        raise HTTPException(422, str(e)) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return views.list_item(store.run_path(run_id))


@app.get("/api/runs/{run_id}", response_model=S.RunDetail)
def get_run(run_id: str) -> S.RunDetail:
    return views.detail(_run(run_id))


@app.patch("/api/runs/{run_id}", response_model=S.RunListItem)
def rename_run(run_id: str, body: S.RenameRun) -> S.RunListItem:
    run = _run(run_id)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "The name can't be empty")
    store.write_meta(run, name=name[:120])
    return views.list_item(run)


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str) -> None:
    _run(run_id)
    store.delete_run(run_id)


@app.post("/api/runs/{run_id}/files", response_model=S.FilesAdded)
def add_files(run_id: str, files: Annotated[list[UploadFile], File()]) -> S.FilesAdded:
    """Replace a run's highlight videos and images with newer ones (e.g. notebook 07's zip)."""
    _run(run_id)
    tmp = Path(tempfile.mkdtemp(prefix="pv-upload-"))
    try:
        updated = store.add_files(run_id, _save_uploads(files, tmp))
    except store.ImportFailed as e:
        raise HTTPException(422, str(e)) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return S.FilesAdded(updated=updated)


@app.post("/api/runs/{run_id}/reprocess", response_model=S.RunListItem)
def reprocess(run_id: str) -> S.RunListItem:
    run = _run(run_id)
    if store.read_meta(run)["status"] == "processing":
        raise HTTPException(409, "This run is already being processed")
    store.write_meta(run, status="processing", status_message="Preparing the results")
    store.start_processing(run_id)
    return views.list_item(run)


@app.put("/api/runs/{run_id}/video", response_model=S.RunDetail)
def attach_video_path(run_id: str, body: S.AttachVideo) -> S.RunDetail:
    """Attach the match video by its path on this computer (no copy)."""
    _run(run_id)
    try:
        store.attach_video(run_id, Path(body.path.strip().strip('"')))
    except store.ImportFailed as e:
        raise HTTPException(422, str(e)) from e
    return views.detail(_run(run_id))


@app.post("/api/runs/{run_id}/video", response_model=S.RunDetail)
def upload_video(run_id: str, file: Annotated[UploadFile, File()]) -> S.RunDetail:
    """Attach the match video by uploading it (copied into the run folder)."""
    run = _run(run_id)
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in store.VIDEO_EXT:
        raise HTTPException(422, f"Unsupported video type '{ext}'")
    dest = run / f"video{ext}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out, length=1 << 20)
    try:
        store.attach_video(run_id, dest, copied=True)
    except store.ImportFailed as e:
        raise HTTPException(422, str(e)) from e
    return views.detail(run)


@app.get("/api/runs/{run_id}/video")
def get_video(run_id: str) -> FileResponse:
    _run(run_id)
    path = store.video_file(run_id)
    if path is None:
        raise HTTPException(404, "No video attached to this run")
    return FileResponse(path)


@app.get("/api/runs/{run_id}/files/{name}")
def get_file(run_id: str, name: str) -> FileResponse:
    run = _run(run_id)
    if name not in store.MEDIA_FILES or not (run / name).exists():
        raise HTTPException(404, f"No file '{name}' in this run")
    return FileResponse(run / name)


@app.get("/api/runs/{run_id}/rallies", response_model=S.RalliesResponse)
def get_rallies(run_id: str) -> S.RalliesResponse:
    return views.rallies(_run(run_id))


@app.get("/api/runs/{run_id}/events", response_model=list[S.Event])
def get_events(run_id: str) -> list[S.Event]:
    return views.events(_run(run_id))


@app.get("/api/runs/{run_id}/tracks", response_model=S.TracksResponse)
def get_tracks(run_id: str, sample_fps: float = Query(5.0, gt=0, le=30)) -> S.TracksResponse:
    return views.tracks(_run(run_id), sample_fps)


@app.get("/api/runs/{run_id}/overlay", response_model=S.OverlayResponse)
def get_overlay(run_id: str, start: int = Query(..., ge=0), end: int = Query(..., ge=0)) \
        -> S.OverlayResponse:  # fmt: skip
    if end < start:
        raise HTTPException(422, "end must be >= start")
    return views.overlay(_run(run_id), start, end)


@app.get("/api/runs/{run_id}/highlights", response_model=S.HighlightsResponse)
def get_highlights(run_id: str) -> S.HighlightsResponse:
    return views.highlights(_run(run_id))


# --- court calibration -----------------------------------------------------------------------
def _cal_error(e: Exception) -> HTTPException:
    return HTTPException(422, str(e))


@app.post("/api/calibration/images", response_model=S.CalibrationImage, status_code=201)
def upload_calibration_image(files: Annotated[list[UploadFile], File()]) -> S.CalibrationImage:
    """An image to calibrate on: a frame / background.png, or notebook 08's calibration.zip."""
    tmp = Path(tempfile.mkdtemp(prefix="pv-upload-"))
    try:
        return calibration.save_image(_save_uploads(files, tmp))
    except (calibration.CalibrationError, zipfile.BadZipFile) as e:
        raise _cal_error(e) from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/api/calibration/images/{image_id}")
def get_calibration_image(image_id: str) -> FileResponse:
    try:
        return FileResponse(calibration.image_path(image_id))
    except calibration.CalibrationError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/calibration/snap", response_model=S.SnapResult)
def snap(body: S.SnapRequest) -> S.SnapResult:
    """Snap roughly clicked points onto the painted line (or edge) they lie along."""
    try:
        return S.SnapResult(points=calibration.snap(body))
    except (calibration.CalibrationError, ValueError) as e:
        raise _cal_error(e) from e


@app.post("/api/calibration/fit", response_model=S.CalibrationFit)
def fit(body: S.CalibrationInput) -> S.CalibrationFit:
    try:
        return calibration.fit(body)
    except calibration.CalibrationError as e:
        raise _cal_error(e) from e


@app.get("/api/runs/{run_id}/court", response_model=S.RunCourt)
def get_run_court(run_id: str) -> S.RunCourt:
    _run(run_id)
    return calibration.run_court(run_id)


@app.post("/api/runs/{run_id}/court/background", response_model=S.CalibrationImage)
def run_background(run_id: str, refresh: bool = False) -> S.CalibrationImage:
    """The court without players: median of frames from the attached video (takes a while)."""
    _run(run_id)
    try:
        return calibration.run_background(run_id, refresh)
    except (calibration.CalibrationError, ValueError) as e:
        raise _cal_error(e) from e


@app.put("/api/runs/{run_id}/court", response_model=S.RunListItem)
def save_run_court(run_id: str, body: S.CalibrationInput) -> S.RunListItem:
    """Save a new calibration and recompute the analysis that depends on it."""
    run = _run(run_id)
    try:
        calibration.save_run_court(run_id, body)
    except calibration.CalibrationError as e:
        raise _cal_error(e) from e
    return views.list_item(run)


@app.post("/api/runs/{run_id}/court/restore", response_model=S.RunListItem)
def restore_run_court(run_id: str) -> S.RunListItem:
    """Swap back to the calibration used before the last save."""
    run = _run(run_id)
    try:
        calibration.restore_previous(run_id)
    except calibration.CalibrationError as e:
        raise _cal_error(e) from e
    return views.list_item(run)


def serve() -> None:
    import uvicorn

    uvicorn.run("padelvision_api.main:app", host=os.environ.get("PADELVISION_HOST", "127.0.0.1"),
                port=int(os.environ.get("PADELVISION_PORT", "8000")))  # fmt: skip
