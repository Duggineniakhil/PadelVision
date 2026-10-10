# PadelVision V2 — Project Plan

Status (2026-10-11): phases 0-2 done; phase 3 (events, rallies, rally stats, placement map) built and
precision-checked on Test_video; hit/bounce recall still needs human labels. Next: phase 4 (local web app). Scope: a **personal / portfolio project**, not a commercial
product yet. Product concerns (accounts, cloud hosting, payments) are parked in §8.

## 1. Goal

Upload a padel match video from a fixed camera and get **trustworthy** analytics:
player movement, positioning, heatmaps, then rallies and shots. Use it through a CLI
first and then a simple local web app.

### Lessons from V1 → rules for V2

| V1 problem | V2 rule |
|---|---|
| Model weights missing, Google Drive links, "model not found" | Every model is listed in `models/manifest.yaml` (URL + SHA256) and fetched by one script. A missing model gives a clear error |
| Fake heatmaps/shot maps when tracking was weak | **Never fabricate data.** Show "not enough data" instead |
| Ball speed from projecting an airborne ball through a ground-plane homography | The homography is only valid for points **on the ground** (feet, bounces) |
| Hardcoded 24 FPS | Read FPS and timestamps from the video |
| One big script, hard to debug | Pipeline split into stages that each save their results to disk, so they can be cached and tested |
| No way to tell if a change helped | A small hand-labelled evaluation set + metrics |

### Supported input
A fixed camera (no pan or zoom), 720p or higher, 25–60 fps, doubles (4 players).

**Primary target: the developer's own footage.** A wide-angle (GoPro/phone) camera mounted low near one
baseline, at an indoor club, 1280×720 @ 30 fps. Consequences for the pipeline:
- **Lens distortion:** court lines are visibly curved. Phase 1 needs an undistortion step that fits
  radial distortion from the fact that court lines must be straight. No checkerboard or training needed.
- **The near corners are out of frame:** calibration uses whichever ≥ 4 keypoints are visible (service-line
  ends, net-post bases, far corners), not just the 4 outer corners.
- **The far court is compressed:** position accuracy for the far team is lower. Report per-side accuracy and
  use looser targets for the far side.
- **Players are small and the ball is about 5 px:** run detection at `imgsz=1280` and train the ball model at high resolution.

High-camera footage (from above the back glass) is a secondary, easier case.

## 2. Architecture (kept simple)

```
pipeline/  padelvision package ──► CLI:  padelvision analyze match.mp4 --court court.json
                │                         writes runs/<id>/ (json, parquet, png)
                ├──► Kaggle/Colab notebooks (full matches, training)
                └──► api/ (FastAPI, local) ◄──► web/ (Next.js, local)
```

- **pipeline/**: the core. All CV and analytics code lives here, with no web dependencies.
- **api/**: a thin FastAPI app. It accepts an upload, runs the pipeline in a background
  process, and serves results from `runs/`. Everything is stored on local disk; there's no database at first (SQLite if needed).
- **web/**: Next.js pages for upload, court calibration (click 4 corners), progress, and a dashboard
  where overlays are drawn on a canvas over the original video.
- No auth, no cloud storage, no queue service. Add them only if the project becomes a product.

### Pipeline stages (each writes a file into `runs/<id>/`)

| # | Stage | Output | Notes |
|---|---|---|---|
| 0 | probe | `video.json` | fps, resolution, duration (ffprobe) |
| 1 | court | `court.json` | clicked corners ↔ court metres, homography, reprojection error |
| 2 | players | `players.parquet` | person detection, tracking, kept only if feet are inside the court (+ margin) |
| 3 | identities | `players_id.parquet` | 4 stable IDs; team = side of the net |
| 4 | ball | `ball.parquet` | per-frame detection; short gaps filled, long gaps left empty |
| 5 | events | `events.json` | hits, bounces, rallies |
| 6 | analytics | `stats.json` | distance, speed, zones, formation, rally stats |
| 7 | visuals | `*.png` | heatmaps, placement map |
| 8 | render | `annotated.mp4` | optional export |

Court coordinates: metres, origin at the court centre, x across the court (−5..5), y along it (−10..10), net at y = 0.

## 3. Models

**Library: Ultralytics YOLO11.** Its AGPL licence is fine for a personal or open-source project, and it's
the easiest to train on Colab/Kaggle. Keep it behind a small interface (`padelvision/models/`)
so you could switch to an Apache-2.0 model like RF-DETR if this ever becomes a closed-source product.

| Need | Source | Train? | Phase |
|---|---|---|---|
| Person detection | `yolo11s.pt` / `yolo11m.pt` (COCO, downloads automatically) | No | 1 |
| Tracking | ByteTrack (built into Ultralytics) | No | 1 |
| Court calibration | The user clicks 4 corners (OpenCV window in phase 1, web UI later) | No | 1 |
| Ball | Fine-tune `yolo11n/s` on padel/tennis-ball images | **Yes**, about 1–3 h on Kaggle | 2 |
| Ball (if recall is poor) | A TrackNet-style model that uses 3 frames; fine-tune public tennis weights | Fine-tune | 4 |
| Automatic court keypoints | `yolo11n-pose` fine-tuned on about 300–500 labelled frames | Small | 4 |
| Player pose | `yolo11n-pose.pt` (COCO) | No | 4 |

### Ball training data
1. **Roboflow Universe**: search "padel ball", "padel", "tennis ball". Export in YOLO format and check quality and licence.
2. **Your own frames**: about 1,500–3,000 frames taken from your test videos.
3. **Labelling**: Roboflow (free tier), CVAT or Label Studio. Once a first model exists, let it pre-label new frames and just correct them.
4. **Include hard cases**: a blurred ball, the ball against the glass or near the lights, the ball lying on the floor, and frames with **no ball** at all.

### Training on Kaggle (preferred) / Colab
- Kaggle: free T4 ×2 / P100. There's a weekly GPU quota of roughly 30 h and sessions of about 12 h (check the current limits).
  Upload the dataset as a private Kaggle Dataset and run `ml/notebooks/train_ball.ipynb`.
- Colab: free T4. Save checkpoints to Drive every epoch and resume after a disconnect.
- Settings: `imgsz` 960–1280 (the ball is small), 50–100 epochs, small model.
- When training finishes: upload the weights to the **Hugging Face Hub** (free), then add the URL, sha256 and metrics to `models/manifest.yaml`.

### Loading models
`scripts/fetch_models.py` reads the manifest, downloads weights to `models/weights/` (gitignored) and checks the hashes.
Code loads models only through `padelvision.models.registry.load(name)`.

## 4. Where things run
- **Local CPU:** development and short clips (30–60 s). This is fine with small models.
- **Kaggle/Colab:** full matches and all training. Notebooks clone the repo, run `pip install -e pipeline/`, then run the CLI.
- **Optional demo:** later, a free Hugging Face Space (CPU) that processes short clips, for a portfolio link.

## 5. Analytics (only what we can actually measure)
- **Phase 1:** distance, average/peak speed, sprints, heatmaps, time in each zone (net/transition/back), team formation, partner spacing.
- **Phase 3:** rallies, hits per player, bounce locations, placement map, approximate ball speed between a hit and the next bounce (labelled as an estimate).
- **Later:** shot types (pose + trajectory), highlight clips.

## 6. Phases

**Phase 0 — Foundations**
- Repo skeleton, the `padelvision` package, ruff + pytest, the manifest + fetch script.
- Collect 3–5 test clips. Hand-label court corners, players on every 10th frame, and the ball on about 500 frames.
- *Done when:* `pytest` passes and running `fetch_models` on a fresh clone works.

**Phase 1 — Player analytics CLI (no training)**
- Stages 0–3, 6 and 7 for players. Calibration uses a small OpenCV click tool that saves `court.json`.
- *Done when:* the expected ground error in the near half is under 15 cm (at the net and beyond a low camera
  can't reach that: report per zone, see `CourtCalibration.accuracy()`), there are fewer than 2 ID switches per minute, and distances look right on the test clips.

**Phase 2 — Ball model (Kaggle)**
- **Bootstrap labels without a model** (`ml/notebooks/02_ball_bootstrap.ipynb`): three-frame
  differencing finds small moving blobs in the court area, outside player boxes. Trajectory
  linking keeps only smooth, fast ball flights, and those become pseudo-labels.
- **Human review** (local, `padelvision label-ball`): about 150 pseudo-labelled frames (to measure
  pseudo-label precision) plus about 150 random in-play frames (to measure recall). These labels are the eval set.
- **Train** YOLO11 on the pseudo-labels on Kaggle (eval frames excluded), evaluate on the reviewed set,
  then add stage 4 (ball tracking with smoothing).
- *Done when:* recall is at least 80% and precision at least 90% (within 10 px) on visible-ball eval frames.

**Phase 3 — Events**
- Hit, bounce and rally detection; placement map; rally stats.
- *Built:* `events.py` (hit / handling / bounce / wall, rallies need an exchange), `analytics/rallies.py`
  (rally stats, placement, shot speed estimate = hitter's feet -> landing bounce, a lower bound),
  `placement.png`, hit/bounce markers in the preview. Eval: `events-eval` + `label-events`.

**Phase 4 — Local web app**
- FastAPI + Next.js: upload, calibration, progress, dashboard with canvas overlays.

**Phase 5 — Extras (pick any)**
- Automatic court keypoints, shot types, highlights, TrackNet ball model, HF Space demo.

## 7. Evaluation
- `ml/eval/` scripts measure court error, ID switches, ball precision/recall and event F1.
- The manifest stores each model's metrics. A new model replaces the old one only if it doesn't make things worse.
- Unit tests use tiny fixtures and never need weights or a GPU.

## 8. Parked until it becomes a product
Auth and accounts, cloud storage, job queue + Postgres, serverless GPU inference (Modal/RunPod),
usage limits, payments, privacy and data-retention policy, and the AGPL licensing decision (switch to an
Apache-2.0 detector or buy an Ultralytics licence if closed-source).
