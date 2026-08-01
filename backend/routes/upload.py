"""
routes/upload.py

POST /api/upload
  - Accepts a video file (multipart/form-data)
  - Saves it to backend/uploads/{job_id}.mp4
  - Creates a SQLite job record
  - Starts background processing
  - Returns { job_id } immediately
"""

import os
import sys
import uuid
import logging
import traceback

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from jobs.store import create_job, update_job, complete_job, fail_job
from pipeline.processor import process_video
from app.schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


def _run_pipeline(job_id: str, input_path: str):
    """Background task: runs the full pipeline and updates the job store."""
    def _on_progress(stage: str, pct: int):
        update_job(job_id, status="processing", stage=stage, progress=pct)

    try:
        update_job(job_id, status="processing", stage="Starting", progress=1)

        result = process_video(
            input_path=input_path,
            job_id=job_id,
            on_progress=_on_progress,
        )

        # Build URL-friendly relative paths (served as static files)
        job_result = {
            "video_url": f"/outputs/{job_id}/video.mp4",
            "heatmap_p1_url": f"/outputs/{job_id}/heatmap_p1.png",
            "heatmap_p2_url": f"/outputs/{job_id}/heatmap_p2.png",
            "heatmap_p3_url": f"/outputs/{job_id}/heatmap_p3.png",
            "heatmap_p4_url": f"/outputs/{job_id}/heatmap_p4.png",
            "shot_map_url": f"/outputs/{job_id}/shot_map.png",
            "highlights_video_url": f"/outputs/{job_id}/highlights.mp4" if result.get("highlights_video_path") else None,
            "player_1": result["stats"]["player_1"],
            "player_2": result["stats"]["player_2"],
            "player_3": result["stats"]["player_3"],
            "player_4": result["stats"]["player_4"],
            "highlights": result["highlights"],
            "total_frames": result["stats"]["total_frames"],
            "fps": result["stats"]["fps"],
        }
        complete_job(job_id, job_result)
        logger.info("Job %s completed successfully.", job_id)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("Job %s failed: %s", job_id, error_msg)
        fail_job(job_id, error=str(exc))


@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # Validate extension
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    job_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOADS_DIR, f"{job_id}{ext}")

    # Stream file to disk while enforcing the documented upload limit.
    bytes_written = 0
    try:
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
                    )
                f.write(chunk)
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

    logger.info("Uploaded video saved to %s (job: %s)", save_path, job_id)

    # Register job in SQLite
    create_job(job_id)

    # Kick off background processing
    background_tasks.add_task(_run_pipeline, job_id, save_path)

    return UploadResponse(job_id=job_id, message="Upload received. Processing started.")
