"""Response models. The web app's TypeScript types are generated from these (OpenAPI)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

RunStatus = Literal["processing", "ready", "error"]
Team = Literal["near", "far"]


class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


class RunListItem(BaseModel):
    id: str
    name: str
    created: str
    status: RunStatus
    status_message: str | None = None
    duration_s: float | None = None
    rallies: int | None = None
    hits: int | None = None
    has_video: bool
    has_highlights: bool


class ZoneAccuracy(BaseModel):
    m_per_px: float | None = None
    expected_error_m: float | None = None


class CourtInfo(BaseModel):
    image_width: int
    image_height: int
    rms_px: float | None = None
    zones: dict[str, ZoneAccuracy]
    lines_image: list[list[tuple[float, float]]]  # court lines drawn on the video (px)


class PlayerStats(BaseModel):
    player: int
    team: Team
    tracked_fraction: float
    insufficient_data: str | None = None
    tracked_s: float | None = None
    distance_m: float | None = None
    avg_speed_kmh: float | None = None
    peak_speed_kmh: float | None = None
    sprints: int | None = None
    zones: dict[str, float] | None = None
    rally_hits: int = 0
    shot_speed_median_kmh: float | None = None
    shot_speed_max_kmh: float | None = None
    shot_speed_samples: int = 0


class TeamStats(BaseModel):
    team: Team
    insufficient_data: str | None = None
    both_tracked_fraction: float | None = None
    formation: dict[str, float] | None = None
    partner_spacing_m: float | None = None


class RallySummary(BaseModel):
    count: int
    total_s: float
    median_s: float | None = None
    max_s: float | None = None
    hits_total: int
    hits_per_rally: float | None = None
    other_activity: int


class SpeedSummary(BaseModel):
    samples: int
    median_kmh: float | None = None
    max_kmh: float | None = None
    insufficient_data: str | None = None
    note: str | None = None


class PointsSummary(BaseModel):
    serves: int
    confirmed_serves: int  # the serve's landing was seen in the diagonal service box
    faults_suspected: int
    points: int
    second_serves: int
    games: int | None = None  # only when the estimate passes padel's rules checks
    games_note: str


class QualityNote(BaseModel):
    level: Literal["info", "warning"]
    title: str
    detail: str


class Changeover(BaseModel):
    t: float  # seconds into the video
    evidence: Literal["strong", "likely"]


class RunDetail(BaseModel):
    id: str
    name: str
    created: str
    status: RunStatus
    status_message: str | None = None
    video: VideoInfo | None = None
    has_video: bool
    video_name: str | None = None
    media: list[str]
    court: CourtInfo | None = None
    ball_coverage: float | None = None
    players: list[PlayerStats]
    teams: list[TeamStats]
    rally_summary: RallySummary | None = None
    speed_summary: SpeedSummary | None = None
    points_summary: PointsSummary | None = None
    # the pairs changed ends at these times; players are persons across them, teams are pairs
    # (A = players 1-2, the pair on the near side at the start)
    changeovers: list[Changeover] = []
    unresolved_gatherings: int = 0
    quality: list[QualityNote]


class Bounce(BaseModel):
    frame: int
    t: float
    x_m: float
    y_m: float
    zone: str


class Shot(BaseModel):
    frame: int
    t: float
    rally_id: int
    index: int  # 1-based position in its rally
    player: int
    team: Team
    hitter_xy: tuple[float, float] | None = None
    bounce: Bounce | None = None
    speed_kmh: float | None = None
    speed_range_kmh: tuple[float, float] | None = None
    speed_note: str | None = None
    no_landing: Literal["probably_volley", "not_seen"] | None = None


class Rally(BaseModel):
    id: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    duration_s: float
    hits: int
    bounces: int
    hits_by_player: dict[str, int]
    max_shot_kmh: float | None = None
    measured_shots: int
    landed: int
    probable_volleys: int


class RalliesResponse(BaseModel):
    rallies: list[Rally]
    shots: list[Shot]


class Event(BaseModel):
    frame: int
    t: float
    kind: Literal["hit", "bounce", "handling", "turn"]
    u: float
    v: float
    player: int | None = None
    team: Team | None = None
    x_m: float | None = None
    y_m: float | None = None
    in_court: bool | None = None


class PlayerTrack(BaseModel):
    t: list[float]
    x: list[float]
    y: list[float]


class TracksResponse(BaseModel):
    sample_fps: float
    players: dict[str, PlayerTrack]


class OverlayResponse(BaseModel):
    """Columnar per-frame data for drawing on the video between two frames."""

    start_frame: int
    end_frame: int
    box_frame: list[int]
    box_player: list[int]
    box_x1: list[float]
    box_y1: list[float]
    box_x2: list[float]
    box_y2: list[float]
    ball_frame: list[int]
    ball_u: list[float]
    ball_v: list[float]
    ball_detected: list[bool]


class Clip(BaseModel):
    rally_id: int
    start_s: float
    end_s: float
    duration_s: float
    hits: int
    max_shot_kmh: float | None = None
    score: float


class HighlightsVideo(BaseModel):
    file: str
    version: int  # file modification time: changes when the video is replaced
    total_s: float
    clips: list[Clip]


class HighlightsResponse(BaseModel):
    highlights: HighlightsVideo | None = None
    condensed: HighlightsVideo | None = None


class FilesAdded(BaseModel):
    updated: list[str]


class AttachVideo(BaseModel):
    path: str


class RenameRun(BaseModel):
    name: str


# --- court calibration ------------------------------------------------------------------------
Point = tuple[float, float]


class Lens(BaseModel):
    k1: float
    k2: float
    f: float
    cx: float
    cy: float


class CalibrationInput(BaseModel):
    """Clicked constraints (raw image pixels). `lens` given = keep that lens, else it is fitted
    from the lines with >= 3 points (identity if fewer than 2 such lines)."""

    image_size: tuple[int, int]
    image_points: dict[str, Point] = {}
    named_lines: dict[str, list[Point]] = {}
    lines: list[list[Point]] = []  # other straight lines (lens only)
    lens: Lens | None = None


class CalibrationAccuracy(BaseModel):
    rms_px: float | None = None
    constraints_px: dict[str, float]
    zones: dict[str, ZoneAccuracy]


class CalibrationFit(BaseModel):
    accuracy: CalibrationAccuracy
    court_lines: list[list[Point]]  # projected court lines (px), split where not visible
    keypoints: dict[str, Point]  # projected court keypoints near the image
    lens: Lens
    line_rms_px: float | None = None
    court_json: dict[str, Any]  # what is saved as court.json


class CalibrationImage(BaseModel):
    id: str
    width: int
    height: int
    source: Literal["upload", "video"]


class AutoRequest(BaseModel):
    image_id: str


class AutoCalibration(BaseModel):
    """The court found automatically. `ok` = it passed the quality checks; otherwise `reasons`
    say why, and `calibration` (if any) is a starting point to correct by hand."""

    ok: bool
    reasons: list[str]
    template: str | None = None  # the saved calibration the search started from
    lines: dict[str, int]  # points snapped per court line
    calibration: CalibrationInput | None = None
    accuracy: CalibrationAccuracy | None = None


class SnapRequest(BaseModel):
    image_id: str
    guide: list[Point]
    mode: Literal["paint", "edge"] = "paint"


class SnapResult(BaseModel):
    points: list[Point]


class RunCourt(BaseModel):
    calibration: CalibrationInput | None = None
    accuracy: CalibrationAccuracy | None = None
    video_size: tuple[int, int]
    has_video: bool
    can_recompute: bool  # raw detections present, so a new calibration can be applied
    has_previous: bool
