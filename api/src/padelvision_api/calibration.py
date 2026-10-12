"""Court calibration in the web app: images to click on, line snapping, fitting, and saving a
run's court.json. The geometry itself lives in the pipeline (`padelvision.court`).

Images are kept in <runs>/_calibration/<id>.png: an uploaded frame / background.png (or the
calibration.zip notebook 08 writes when it has no calibration), or the median background of a
run's attached video, resized to the analysis resolution.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from padelvision_api import schemas as S
from padelvision_api import store

IMAGE_ID_RE = re.compile(r"^[a-z0-9-]{1,80}$")
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
# Preferred image inside a calibration.zip: the background has the players removed.
ZIP_PREFERENCE = ("background.png", "frame.png")
BACKGROUND_FRAMES = 25


class CalibrationError(Exception):
    pass


def images_dir() -> Path:
    d = store.runs_dir() / "_calibration"
    d.mkdir(exist_ok=True)
    return d


def image_path(image_id: str) -> Path:
    p = images_dir() / f"{image_id}.png"
    if not IMAGE_ID_RE.match(image_id) or not p.exists():
        raise CalibrationError(f"No calibration image '{image_id}'")
    return p


def save_image(uploads: list[tuple[str, Path]]) -> S.CalibrationImage:
    """Store the first usable image: a png/jpg, or the background/frame inside a zip."""
    import cv2

    data = None
    for filename, path in uploads:
        ext = Path(filename).suffix.lower()
        if ext in IMAGE_EXT:
            data = path.read_bytes()
        elif ext == ".zip":
            with zipfile.ZipFile(path) as z:
                names = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
                pick = next((names[n] for n in ZIP_PREFERENCE if n in names), None)
                pick = pick or next((n for b, n in names.items() if Path(b).suffix in IMAGE_EXT),
                                    None)  # fmt: skip
                if pick:
                    data = z.read(pick)
        if data:
            break
    if not data:
        raise CalibrationError(
            "Choose an image of the court (png/jpg), or the calibration.zip from notebook 08."
        )
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise CalibrationError("The image could not be read.")
    image_id = uuid.uuid4().hex[:12]
    cv2.imwrite(str(images_dir() / f"{image_id}.png"), img)
    return S.CalibrationImage(id=image_id, width=img.shape[1], height=img.shape[0],
                              source="upload")  # fmt: skip


def run_background(run_id: str, refresh: bool = False) -> S.CalibrationImage:
    """Median background of the run's attached video at the analysis resolution (cached)."""
    import cv2
    from padelvision.video import median_background

    run = store.run_path(run_id)
    video = store.video_file(run_id)
    if video is None:
        raise CalibrationError("Attach the match video first, or add an image of the court.")
    info = json.loads((run / "video.json").read_text())
    size = (int(info["width"]), int(info["height"]))
    image_id = f"run-{run_id}"
    dest = images_dir() / f"{image_id}.png"
    if refresh or not dest.exists():
        _, background = median_background(video, BACKGROUND_FRAMES, size)
        cv2.imwrite(str(dest), background)
    return S.CalibrationImage(id=image_id, width=size[0], height=size[1], source="video")


@lru_cache(maxsize=4)
def _image(path: str, mtime: float) -> np.ndarray:
    import cv2

    return cv2.imread(path)


def snap(body: S.SnapRequest) -> list[tuple[float, float]]:
    from padelvision.court.snap import snap_line

    p = image_path(body.image_id)
    return snap_line(_image(str(p), p.stat().st_mtime), body.guide, body.mode)


def _calibration(body: S.CalibrationInput):
    from padelvision.court import CourtCalibration, LensModel

    try:
        return CourtCalibration.from_points(
            {n: tuple(p) for n, p in body.image_points.items()},
            image_size=tuple(body.image_size),
            lines=[list(map(tuple, line)) for line in body.lines],
            lens=LensModel(**body.lens.model_dump()) if body.lens else None,
            named_lines={n: list(map(tuple, pts)) for n, pts in body.named_lines.items()},
        )
    except (ValueError, np.linalg.LinAlgError) as e:
        raise CalibrationError(str(e)) from e


def fit(body: S.CalibrationInput) -> S.CalibrationFit:
    from padelvision.court.draw import court_polylines
    from padelvision.court.geometry import COURT_KEYPOINTS

    cal = _calibration(body)
    acc = cal.accuracy()
    w, h = cal.image_size
    keypoints = {}
    for name, xy in COURT_KEYPOINTS.items():
        px = cal.to_image(np.array([xy]))[0]
        if np.isfinite(px).all() and -0.1 * w <= px[0] <= 1.1 * w and -0.1 * h <= px[1] <= 1.1 * h:
            keypoints[name] = (round(float(px[0]), 1), round(float(px[1]), 1))
    return S.CalibrationFit(
        accuracy=_accuracy(acc),
        court_lines=[
            [(round(float(x), 1), round(float(y), 1)) for x, y in seg]
            for seg in court_polylines(cal, samples=160)
        ],  # fmt: skip
        keypoints=keypoints,
        lens=S.Lens(**cal.lens.to_dict()),
        line_rms_px=None if cal.line_rms_px is None else round(cal.line_rms_px, 2),
        court_json=cal.to_dict(),
    )


def _accuracy(acc: dict) -> S.CalibrationAccuracy:
    return S.CalibrationAccuracy(
        rms_px=acc["rms_px"], constraints_px=acc["constraints_px"],
        zones={k: S.ZoneAccuracy(**v) for k, v in acc["zones"].items()},
    )  # fmt: skip


def run_court(run_id: str) -> S.RunCourt:
    run = store.run_path(run_id)
    info = json.loads((run / "video.json").read_text())
    calibration = accuracy = None
    if (run / "court.json").exists():
        data = json.loads((run / "court.json").read_text())
        calibration = S.CalibrationInput(
            image_size=tuple(data["image_size"]),
            image_points=data.get("image_points", {}),
            named_lines=data.get("named_lines", {}),
            lines=data.get("lines", []),
            lens=data.get("lens"),
        )
        acc = data.get("accuracy")
        if not acc:  # court.json files from before accuracy() was stored
            from padelvision.court import CourtCalibration

            acc = CourtCalibration.load(run / "court.json").accuracy()
        accuracy = _accuracy(acc)
    return S.RunCourt(
        calibration=calibration, accuracy=accuracy,
        video_size=(int(info["width"]), int(info["height"])),
        has_video=store.video_file(run_id) is not None,
        can_recompute=all((run / f).exists() for f in ("detections.parquet", "stats.json")),
        has_previous=(run / "court.prev.json").exists(),
    )  # fmt: skip


def save_run_court(run_id: str, body: S.CalibrationInput) -> None:
    """Replace the run's court.json (the old one is kept as court.prev.json) and recompute the
    analysis that depends on it: player court positions and stats, ball tracking, events."""
    run = store.run_path(run_id)
    meta = store.read_meta(run)
    if meta.get("status") == "processing":
        raise CalibrationError("This match is being processed; wait until it's ready.")
    if not all((run / f).exists() for f in ("detections.parquet", "stats.json")):
        raise CalibrationError(
            "This match was imported without the raw detections (run_cache.zip), so it can't be "
            "recomputed with a new calibration. Import run_cache.zip to enable this."
        )
    info = json.loads((run / "video.json").read_text())
    if tuple(body.image_size) != (int(info["width"]), int(info["height"])):
        raise CalibrationError(
            f"The calibration was made on a {body.image_size[0]}x{body.image_size[1]} image but "
            f"the analysis is {info['width']}x{info['height']}. Use an image at that resolution."
        )
    cal = _calibration(body)
    if (run / "court.json").exists():
        shutil.copy(run / "court.json", run / "court.prev.json")
    cal.save(run / "court.json")
    store.write_meta(run, status="processing", status_message="Applying the new calibration")
    store.start_processing(run_id)


def restore_previous(run_id: str) -> None:
    run = store.run_path(run_id)
    if store.read_meta(run).get("status") == "processing":
        raise CalibrationError("This match is being processed; wait until it's ready.")
    prev = run / "court.prev.json"
    if not prev.exists():
        raise CalibrationError("There is no previous calibration to restore.")
    current = run / "court.json"
    tmp = run / "court.swap.json"
    shutil.copy(current, tmp)
    shutil.copy(prev, current)
    tmp.replace(prev)
    store.write_meta(run, status="processing", status_message="Restoring the previous calibration")
    store.start_processing(run_id)
