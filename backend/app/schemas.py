"""
Pydantic schemas for the PadelVision API.
Compatible with Pydantic v2.
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional


class UploadResponse(BaseModel):
    job_id: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str          # queued | processing | done | error
    stage: str
    progress: int        # 0–100


class PlayerStats(BaseModel):
    total_shots: int
    avg_shot_speed: float
    max_shot_speed: float
    avg_player_speed: float
    distance_covered: float


class Highlight(BaseModel):
    start_frame: int
    end_frame: int
    timestamp_seconds: float
    label: str


class AnalysisResult(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: int
    player_1: Optional[PlayerStats] = None
    player_2: Optional[PlayerStats] = None
    player_3: Optional[PlayerStats] = None
    player_4: Optional[PlayerStats] = None
    highlights: Optional[list[Highlight]] = None
    video_url: Optional[str] = None
    heatmap_p1_url: Optional[str] = None
    heatmap_p2_url: Optional[str] = None
    heatmap_p3_url: Optional[str] = None
    heatmap_p4_url: Optional[str] = None
    shot_map_url: Optional[str] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models: dict[str, bool]
