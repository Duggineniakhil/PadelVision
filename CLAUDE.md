# PadelVision V2

AI analytics for padel matches: a user uploads a fixed-camera match video and gets player
movement, positioning, rally and shot analytics, first through a CLI and later through a local web app.
This is a personal/portfolio **project**, not a product. Keep things simple: no auth,
no cloud services, local disk storage.

This is a **from-scratch rebuild**. The old V1 code on `main` is reference only. Don't copy
it over wholesale. The full plan, phases and model strategy are in `docs/V2_PLAN.md`; read
it before starting larger work.

## Status
Phase 0 is done. Phase 1 (player analytics) code is written and unit-tested on synthetic data,
but it hasn't been validated on real footage yet. That validation is the next step:
`ml/notebooks/01_player_analytics.ipynb` on Kaggle. `web/` and `api/` don't exist yet. Don't invent
commands for tooling that hasn't been set up; add them here once they exist.

Pipeline modules: `court/` (geometry, lens distortion, calibration, click tool, drawing),
`models/` (manifest registry, `person_tracker.py`, the only ultralytics import), `players/`
(stage 2 detection, stage 3 on-court filtering + identities), `analytics/movement.py` (stage 6),
`visuals.py` (heatmaps), `render.py` (preview video), `run.py` (`analyze` orchestration).
The tuning constants (smoothing windows, speed caps, zones, margins) sit at the top of each module.

Calibration notes: the lens uses a **division model** (the polynomial model couldn't straighten
GoPro lines). Constraints can be keypoints and/or **named court lines** (`NAMED_LINES` in
`geometry.py`); the lens and homography are refined jointly. Freeing the distortion centre
overfits, so keep it fixed. Calibrations for specific videos live in `ml/calibrations/<video stem>.json`.
For the developer's low camera, depth near the net is about 6 px/m: expect roughly 0.3 m error in the near
half and about 1 m near the net and beyond.

## Commands (run from repo root; Windows venv paths shown)
```bash
.venv/Scripts/python -m pip install -e "pipeline[dev]"   # add [ml] for ultralytics
.venv/Scripts/python -m pytest pipeline                  # unit tests (no weights/GPU needed)
.venv/Scripts/ruff check pipeline scripts
.venv/Scripts/ruff format pipeline scripts
.venv/Scripts/python scripts/fetch_models.py [name ...]  # download + verify weights
.venv/Scripts/padelvision probe <video>                  # print video metadata
.venv/Scripts/padelvision calibrate data/court_frame.png -o data/court.json   # local GUI
# Kaggle/Colab only (needs GPU + [ml]):
padelvision analyze <video> --court court.json --out runs/match1 [--stride 2] [--max-frames 900]
padelvision render <video> --run runs/match1 --start 60 --seconds 30
```
The developer's laptop **cannot process real videos**. Anything that runs models on
video goes in a notebook under `ml/notebooks/` to be run on Kaggle/Colab. Locally, only run unit tests
on tiny synthetic fixtures.

## Repository layout (target)
```
web/        Next.js (App Router) + TypeScript + Tailwind frontend (phase 4)
api/        Thin local FastAPI app: upload, run the pipeline in the background, serve runs/. No CV code here.
pipeline/   `padelvision` Python package (src layout): the CV and analytics core.
            No web/DB dependencies; used by the CLI, the worker and notebooks.
ml/         Training notebooks (Kaggle/Colab), dataset configs, eval scripts
models/     manifest.yaml (tracked) + weights/ (gitignored)
scripts/    fetch_models.py and other dev utilities
docs/       Plans and design notes
infra/      Dockerfiles, docker-compose for local dev
```

## Hard rules
- **Never fabricate data.** No fake or demo heatmaps, stats or tracks when the real data is weak.
  If a metric can't be computed, return null/"insufficient data" and show that in the UI.
- **Never commit model weights or videos.** Models are declared in `models/manifest.yaml`
  (url, sha256, license, metrics) and downloaded by `scripts/fetch_models.py`. Code loads
  them only through `padelvision.models.registry`. A missing model raises a clear error
  that names the file and the fetch command.
- **The homography is only valid for points on the ground** (player feet = bottom-centre
  of the box, ball bounce points). Never project an airborne ball through it to get speed or position.
- **Read FPS and timestamps from the video.** Never hardcode a frame rate.
- **Court coordinates are in metres**, origin at the court centre, x across the court (−5..5), y along
  it (−10..10), net at y = 0. All court dimensions come from `padelvision/court/geometry.py`.
- **Pipeline stages are pure and cached:** each stage reads the previous stages' outputs and writes
  its own file to `runs/<job_id>/` (json/parquet). There are no hidden global state or side effects.
- **Every model sits behind an interface** in `padelvision/models/` so it can be swapped. The project uses
  Ultralytics YOLO11 (AGPL-3.0, fine for a non-commercial project). Don't import ultralytics outside that module.
- The pipeline must run on CPU (slowly) as well as GPU. Choose the device automatically and never assume CUDA.

## Conventions
- Python >= 3.10 (local is 3.13; Kaggle/Colab use their bundled version), type hints everywhere, dataclasses or pydantic for stage outputs, `ruff` + `pytest`.
- Tests use tiny fixtures (a few frames, synthetic tracks) and must not need weights or a GPU.
- TypeScript strict mode. API types are generated from FastAPI's OpenAPI schema rather than written by hand.
- Overlays are drawn in the browser on a canvas from result JSON. A rendered MP4 is only an optional export.

## Target footage
The developer's own videos come from a **low, wide-angle camera near one baseline** (lens distortion,
near corners out of frame, far court compressed, tiny ball). Design for that first; see "Supported
input" in `docs/V2_PLAN.md`. Never assume the 4 outer corners are visible.

## Compute constraints
The developer has no local GPU. Training and long test runs happen on **Kaggle** (preferred)
or **Google Colab**. Notebooks in `ml/notebooks/` must:
- install the pipeline with `pip install -e pipeline/` from a fresh clone
- fetch models with `scripts/fetch_models.py`
- save checkpoints to persistent storage, so a disconnected session can resume
- push final weights to the Hugging Face Hub and update `models/manifest.yaml`

Local CPU is used for development and short clips (30–60 s). Full matches run in Kaggle/Colab notebooks.
