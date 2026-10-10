# PadelVision V2

AI analytics for padel matches: a user uploads a fixed-camera match video and gets player
movement, positioning, rally and shot analytics, first through a CLI and later through a local web app.
This is a personal/portfolio **project**, not a product. Keep things simple: no auth,
no cloud services, local disk storage.

This is a **from-scratch rebuild**. The old V1 code on `main` is reference only. Don't copy
it over wholesale. The full plan, phases and model strategy are in `docs/V2_PLAN.md`; read
it before starting larger work.

## Status
Phases 0–1 are done. Player analytics has been validated on the developer's Test_video: bystander
filtering, identity gating and Kalman smoothing were tuned on real detections.
Phase 2 (ball) is in progress. The label bootstrap is built (`ball/candidates.py`, `ball/link.py`,
`ball/bootstrap.py`, `ball/label_tool.py`, notebook `02_ball_bootstrap.ipynb`). The pseudo-label track
filter was tuned on hand-checked crops (precision ~85%). Eval labels are in `ml/eval/labels/` (93 visually
confirmed balls; unlabelled frames are *unverified*, not "no ball"; new balls are added by pooled crop review).
Next is the pretrained-detector comparison (`03_ball_bakeoff.ipynb`, `ball/evaluate.py`,
`models/ball_detectors.py`: COCO sports ball, V1 padel YOLO, TrackNet, Roboflow Universe models via a
Kaggle secret `ROBOFLOW_API_KEY`). Bake-off result with pooled labels (207 balls, 401 non-ball spots):
Roboflow `padel-ball-detection-nazbq/4` F1 0.68 > COCO sports ball 0.60 > others. Hosted models are
too costly per frame, so we train our own: `04_train_ball.ipynb` + `ball/dataset.py` fine-tune YOLO11s on our
pseudo-labels (`ml/datasets/Test_video_ball_pseudo.csv`) plus Universe datasets downloaded inside Kaggle.
Train/test split: 20 s blocks (odd blocks = test, 2 s margin) to avoid near-duplicate leakage. The session has
several balls on court (spares, balls in hand), so the tracker must pick the ball in play (motion).
Result: the trained `ball-detector` (manifest; GitHub release `ball-yolo11s-v1`; card in
`docs/models/padel_ball_yolo11s.md`) scores F1 0.79 on held-out blocks (conf >= 0.10).
Stage 4 (`ball/track.py`, notebook 05) is done: on Test_video the ball in play is found on 55-57% of frames,
98% of reviewed tracked positions are real balls, and 75% of motion-found moving balls are tracked within 10 px.
When two balls move at once it used to flip between them (balls on the next court are seen through the side
fence, so a 2-D court mask can't drop them). The tracker now keeps the current ball and only switches to a ball
it could have reached (80 px/frame since last seen): impossible jumps went 69 -> 9, coverage 57% -> 53%.
Phase 3 (stage 5, `events.py`) in progress: sharp turns of the ball's image path (inside tracklets and at
tracklet junctions) classified as hit (ball in a player's reach zone AND ball size consistent with that
player's depth: a low camera puts far balls inside near players' boxes), handling (a "hit" after which the
ball stays near the player: bouncing it between points; Test_video has long stretches of this), floor bounce
(down then up the image; gets a court position) or wall. Activity is split at ball-track pauses > 2 s and at
> 4 s without a hit, then trimmed to 1 s before the first hit / 2 s after the last. A rally needs an exchange
(hits by both teams, or a hit then an in-court bounce on the other side); the rest goes to `other_activity`
in `rallies.json`. A turn right after the other team's hit is a return (always a hit: far shots barely move in
the image), and same-player turns within 0.35 s are one contact. Visual precision check of the predicted events
(`ml/eval/labels/Test_video_events_reviewed.csv`, precision only): hits inside rallies 17/17 real (+1 unclear);
the remaining false hits are ball tapping outside rallies; bounces 9/11 decided; walls ~0/6 (unreliable).
Recall still needs human `label-events` labels. Test_video: 4 rallies (8.0-12.9, 31.7-48.4, 67.2-74.0, 136.3-141.9 s).
Ball detections now carry `size` (min box side, px). `web/` and `api/` don't exist yet. Don't invent commands
for tooling that hasn't been set up; add them here once they exist.

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
padelvision render <video> --run runs/match1 --start 0 --seconds 30
padelvision ball-bootstrap <video> --run runs/match1          # classical candidates + pseudo-labels
padelvision ball-review-pack <video> --run runs/match1 --out ball_review
padelvision ball-track <video> --run runs/match1               # stage 4: ball in play (ml/notebooks/05)
padelvision event-review-pack <video> --run runs/match1 --out event_review   # frames to label hits/bounces in
# Local: label the downloaded review pack (Y = ring on ball, click = ball, N = none, M = motion view)
.venv/Scripts/padelvision label-ball data/ball_review
# Local again: redo stages 3-7 from a downloaded Kaggle run folder (no video/GPU needed)
.venv/Scripts/padelvision restats data/runs/match1 [--court other_court.json]
.venv/Scripts/padelvision ball-retrack data/runs/match1   # redo ball tracking from cached ball detections
.venv/Scripts/padelvision events data/runs/match1         # stage 5: hits/bounces/rallies (ml/notebooks/06)
# Local: label every hit/bounce in the pack's windows (A/D step, H hit, B bounce, V window done), then score
.venv/Scripts/padelvision label-events data/event_review
.venv/Scripts/padelvision events-eval data/runs/match1 --labels ml/eval/labels/Test_video_events.csv
```
To tune identity or stats logic, ask for the run folder (`detections.parquet`, `video.json`,
`stats.json`) from Kaggle and iterate locally with `restats`. Don't send the developer back to
Kaggle for every tweak.
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
