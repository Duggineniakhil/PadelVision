# PadelVision V2

AI analytics for padel matches: a user uploads a fixed-camera match video and gets player
movement, positioning, rally and shot analytics, first through a CLI and later through a local web app.
This is a personal/portfolio **project**, not a product. Keep things simple: no auth,
no cloud services, local disk storage.

This is a **from-scratch rebuild**. The old V1 code on `main` is reference only. Don't copy
it over wholesale. The full plan, phases and model strategy are in `docs/V2_PLAN.md`; read
it before starting larger work.

## Status
Phases 0-4 are built (phase 4: local web app). Everything is tuned and measured on one video, the
developer's Test_video (206 s, mostly warm-up and ball handling, 4 short rallies). Current numbers and gaps are
in `ml/eval/README.md`; open items: ball
detector below target, hit/bounce recall unmeasured (needs human `label-events` labels).
Second video GX010012 (19 min match, notebook 08, 2026-10-11): players tracked 85-87%, but the ball detector finds
the real ball on only ~33% of frames (91% of frames hold only fixed false spots: lights, spare balls, the camera
mount; masking them changed nothing), so only 34 rally fragments (165 s, 2.2 hits each) are found. The fix is
retraining the ball detector with this court; tracker rules can't recover undetected balls.

- Players (stages 2-3, 6-7): bystander filtering, identity gating and Kalman smoothing tuned on real detections.
  Visual ID check (94 s): 0 switches; all 12 tracker-id changes were the same person picked up again.
- Changeovers (stage 3b, `players/changeovers.py`): stage 3's ids are side slots; this finds candidate changeovers
  from positions (tracks linked through the net that cross for good, both ways = "strong", one crossing at a net
  gathering = "likely"; net gatherings without a crossing are candidates too) and decides each one, and which partner
  is which, by clothing colours when `appearance.parquet` exists (stage 2b, `players/appearance.py`: median Lab of
  shirt and shorts per person box every 0.5 s, read from the video in notebooks 07/08 with `padelvision appearance`;
  Lab because dark kit has noisy hue and far players have a blue cast), else by positions: changeovers swap,
  gatherings don't (reported unresolved), partners by side habit (the slot that played more on its own left = the
  pair's left player; tracker ids carried across the net were tried and merge people). players.parquet: `player` = person (1-2 pair A = near at the
  start, 3-4 pair B), `slot` = side slot, `team` = side the row is on, `pair`; so stats, hits, speeds, heatmaps,
  HUD are per person; event/rally geometry stays side-based. stats.json `persons` holds the evidence. GX010012:
  changeovers at 40 s (likely), 283, 543 (strong), 858 s (likely) = every 2 games; side habit consistent in all 5
  periods (left share ~0.85 vs ~0.2 for pair A); 7 unresolved gatherings. Not checked against the video by eye.
  Test_video: one changeover at 98 s (likely), confirmed by eye on review frames (black-shirt pair near before,
  white-shirt pair near after). With colours from the review frames: shirts L 23-40 vs 136-163, the 98 s changeover
  confirmed by colour (margin 8), its 3 net gatherings decided "no changeover" (margins 13-14), and pair B's partners
  matched by shorts colour (side habit had swapped them). GX010012 has no appearance.parquet yet (re-run nb 07).
- Movement (stage 6, `analytics/movement.py`): before smoothing, stretches where the raw position moves further in
  0.5 s than 8 m/s + 3 sigma (feet hidden behind the net band, the tracker moving onto another box) are dropped;
  tracks are split at tracker-id changes (on the far side a 2 m jump is within the position error); peak speed
  must be held 0.5 s. GX010012 peaks 19-26 -> 12-20 km/h, distances -1-3%.
  Appearance (shirt colours) would settle unresolved gatherings but needs a pixel pass on Kaggle.
- Ball detector: own YOLO11s (`ball-detector` in the manifest, GitHub release `ball-yolo11s-v1`, card
  `docs/models/padel_ball_yolo11s.md`), fine-tuned on Test_video pseudo-labels + Roboflow Universe data
  (`04_train_ball.ipynb`, `ball/dataset.py`); held-out 20 s blocks: recall 0.74, precision 0.85 (conf >= 0.10).
  Pooled eval labels in `ml/eval/labels/` (unlabelled spots are *unverified*, not wrong). Detections carry `size`.
- Ball in play (stage 4, `ball/track.py`): several balls are often in view (spares, balls in hand, the next
  court seen through the side fence, so a 2-D court mask can't drop them). The tracker keeps its ball and only
  switches to one it could have reached (80 px/frame since last seen). Found on 53% of frames.
- Events (stage 5, `events.py`): sharp turns of the ball's image path -> hit (ball in a player's reach zone AND
  ball size fits that player's depth; a turn right after the other team's hit is always a hit, a return; same-player
  turns within 0.35 s are one contact; far players may hit balls down to the detector's ~7 px floor), handling (ball stays near the player: bouncing it between points),
  floor bounce (gets a court position) or turn (unclassified; walls were unreliable; used in no statistic).
  Activity splits at ball pauses > 2 s and > 4 s without a hit, trimmed to the hits; a rally needs an exchange
  (both teams hit, or a hit then an in-court bounce on the other side). `rallies.json` holds rallies,
  `other_activity` and `stats` (`analytics/rallies.py`: counts, hits per player/team, placement ->
  `placement.png`, shot speed ESTIMATE = hitter's feet -> landing bounce / time: the average ground speed, a lower
  bound for the speed off the racket). A shot only gets a speed if its landing is a bounce tracked through (not a
  join between ball tracks), not at the image edge, the flight >= 0.5 s and the error range <= +-25%; each speed has
  `speed_range_kmh` (calibration per zone + 0.7 m feet-to-contact + timing), others a `speed_note`. Summaries need
  3 shots. Never validated against a radar. Shots without a landing get `no_landing`: "probably_volley" (opponents
  hit next, ball tracked >= 90% of the frames in between, no bounce) or "not_seen"; cards draw volleys hollow. Rally stats live there, not in stats.json (`restats` rewrites that).
  `render` draws hits, bounces and a rally banner. Test_video: 4 rallies, 18 rally hits (visual review: 17 real,
  1 unclear); 5 shots with a trustworthy speed, 33-56 km/h, each about +-20% (an 80 km/h one was a track join).
- Serves, points, games (stage 5b, `analytics/points.py`, rallies.json `points`): a serve = a hit from behind the
  service line (feet |y| >= 6.45 m) after >= 4 s without a hit; "confirmed" when its first bounce is seen in the
  diagonal service box, `fault_suspected` when seen outside. Every serve starts a point; the same server again
  within 25 s with nothing played between = a second serve (same point). Games = runs of the same serving PAIR
  (partners are less certain than sides), cut at changeovers; checked against padel's rules (2 games between
  changeovers, serving pair alternates) and only shown when the checks pass. Checked by eye: Test_video serves at
  32.7, 137.3, 196.5 s are real; 9.0 s probably not. GX010012: 37 serves (14 confirmed), 32 points; games not
  countable (far serves mostly missed by the ball detector; returns after a missed serve look like serves; the ball
  before a hit can't tell them apart yet because the tracker may follow a spare ball). Who won a point: not attempted.
- Videos: `render` (preview) and `highlights` (`highlights.py`: rallies ranked by hits + length + fastest
  measured shot, cut and joined, title card per clip, ball trail, and after each clip a 3 s summary card from
  `visuals.rally_card`: per-player heatmaps of that rally + shot map = hitter's feet -> landing bounce, the
  flight itself is not drawn; `--mode all` = condensed match; writes highlights.json /
  condensed.json) both draw the stats overlay (`overlay.py` StatsHud: per player running distance, current
  speed, hits; current rally and last measured shot estimate; "n/a" when stats.json has insufficient data).
  No model involved. Rally stats also give per-player max/median shot speed estimates and per-rally summaries.
- Eval tooling: `event-review-pack` (Kaggle) -> `label-events` (local GUI) -> `events-eval`.

- Web app (phase 4): `api/` (FastAPI, package `padelvision_api`, runs stored in ./runs or $PADELVISION_RUNS) imports
  notebook zips (run_cache.zip, results.zip), re-runs restats / ball-retrack / events in a background thread, and
  serves JSON views (`views.py`), media and the match video (HTTP Range). A video is attached by local path or
  upload and checked by frame count + aspect ratio (any resolution of the same footage). `web/` (Next.js 16 App
  Router, TS strict, Tailwind 4, SWR): Matches, then per match Overview / Video / Players / Rallies / Shots /
  Highlights / Court, Ctrl+K command palette. Court calibration in the browser (`components/calibration.tsx`): pick a named line,
  click >= 2 points along it, the API snaps them (`court/snap.py`: "paint" = bright thin strip, "edge" = strongest
  colour change, e.g. floor/wall; the peak nearest the clicks wins, outliers vs a smooth curve dropped) and refits
  live (`/api/calibration/fit`: accuracy + projected court lines). `/calibrate` exports court.json for a new video
  (image = notebook 08's calibration.zip); a match's Court tab uses the median background of its attached video
  (`video.median_background`, cached in <runs>/_calibration) and saving rewrites court.json (old one kept as
  court.prev.json, restorable) and re-runs restats / ball-retrack / events; needs detections.parquet (run_cache.zip). Video overlays are drawn on a canvas from /overlay windows, synced with
  requestVideoFrameCallback. Next 16 has Cache Components on: pages are a server `page.tsx` (`instant = false`)
  rendering a client `view.tsx`; URL hooks sit under `<Suspense>`. Read `web/node_modules/next/dist/docs/` before
  using Next APIs (see web/AGENTS.md). API types: `python scripts/export_openapi.py && (cd web && npm run gen:api)`.
  Design: neutral zinc UI, one accent (court blue), team colours near green / far orange, no gradients.

Don't invent commands for tooling that hasn't been set up; add them here once they exist.

Pipeline modules: `court/` (geometry, lens distortion, calibration, click tool, line snapping, automatic
calibration, drawing),
`models/` (manifest registry, `person_tracker.py`, the only ultralytics import), `players/`
(stage 2 detection, stage 3 on-court filtering + identities, stage 3b changeovers -> persons), `ball/` (bootstrap, labelling, evaluation,
dataset, stage 4 tracking), `events.py` (stage 5), `analytics/points.py` (stage 5b: serves, points, games),
`analytics/movement.py` + `analytics/rallies.py` (stage 6),
`visuals.py` (heatmaps, placement map), `render.py` (preview video), `events_review.py` /
`events_label_tool.py` / `events_eval.py` (hit/bounce labelling + scoring), `run.py` (stage orchestration).
The tuning constants (smoothing windows, speed caps, zones, margins) sit at the top of each module.

Automatic calibration (`court/auto.py`, `padelvision auto-calibrate <image|video>`, notebook 08 when no
calibration is given, web "Detect automatically"): evidence from a background image = painted lines (top-hat, low
saturation, on the floor only), the floor (Lab a/b distance to the colour at the bottom middle; outer outline only:
glare makes holes) and the net base (dark mesh above, floor below; the net's white top band looks like a painted
line, and without the net the far lines slipped by one and still fit at 1.3 px). Rough placement: start from each
`ml/calibrations/*.json` (same camera type), coarse grid of shift/scale/far-pair moves + Nelder-Mead on 4 anchor
points with the template lens, scored by truncated chamfer distance + floor inside/outside the court. Precise fit:
snap every named line along its projection (wall lines first to the floor outline, then 3 px to the image edge;
net base on its own ridge), refit with the template lens KEPT (refitting it bent the image edges and overfitted),
twice. Gate: >= 4 lines, lines on both sides of the net, the net base found, fit <= 2.5 px, near half <= 15 cm.
`net_base` (y = 0 on the floor) is a named line. Leave-one-out (each court from the other's calibration): Test_video
3.0 px median from the hand-made calibration (ground 8 / 7 / 31 cm near / net / far), GX010012 5.1 px (6 / 14 /
15 cm), ~1 min per template. Where they differ the hand-made one is not always right (Test_video's left wall was
from sparse clicks). Only two courts tested.

Calibration notes: the lens uses a **division model** (the polynomial model couldn't straighten
GoPro lines). Constraints can be keypoints and/or **named court lines** (`NAMED_LINES` in
`geometry.py`); the lens and homography are refined jointly. Freeing the distortion centre
overfits, so keep it fixed. Calibrations for specific videos live in `ml/calibrations/<video stem>.json`.
Judge a calibration with `cal.accuracy()` (printed at stage 1, stored in court.json and stats.json): constraint
fit in pixels plus expected ground error per zone (near half / net / far half). Don't use the old mean-metres
error: near the net 1 px is ~0.11 m and in the far half ~0.3 m, so a 4 px click there dominated it. The target
(< 15 cm) applies to the near half. Test_video (2026-10-11): fit 2.2 px rms; near half 0.04 m, net 0.25 m, far
half 0.67 m. Its far baseline, far side edges and far centre line were snapped from a median background of the
video; 6 old right-side points that sat on a floor stripe were dropped; the lens was kept fixed (refitting it
from short far lines overfits). With it, far players near the far baseline are no longer dropped as off-court.
GX010012 (19 min match, another club, GoPro 4K converted to 1280x720 by notebook 08): built 2026-10-11 from the
notebook's background image with snapped lines only (near/far service lines, centre line, both side walls, far
baseline), lens fitted freely (k1 -1.105, close to Test_video's -1.053: same GoPro mode). Fit 1.6 px; the net,
not a constraint, projects onto the net base within ~2-5 px. Near half 0.02 m, net 0.11 m, far half 0.31 m.

## Commands (run from repo root; Windows venv paths shown)
```bash
.venv/Scripts/python -m pip install -e "pipeline[dev]"   # add [ml] for ultralytics
.venv/Scripts/python -m pytest pipeline                  # unit tests (no weights/GPU needed)
.venv/Scripts/ruff check pipeline scripts
.venv/Scripts/ruff format pipeline scripts
.venv/Scripts/python scripts/fetch_models.py [name ...]  # download + verify weights
.venv/Scripts/padelvision probe <video>                  # print video metadata
.venv/Scripts/padelvision calibrate data/court_frame.png -o data/court.json   # local GUI
.venv/Scripts/padelvision auto-calibrate data/court_background.png -o court.json  # automatic (exit 1 = unreliable)
# Kaggle/Colab only (needs GPU + [ml]):
padelvision analyze <video> --court court.json --out runs/match1 [--stride 2] [--max-frames 900]
padelvision render <video> --run runs/match1 --start 0 --seconds 30   # + stats overlay (--no-hud to drop it)
padelvision highlights <video> --run runs/match1 [--mode top|all] [--top 5]   # best rallies / condensed match
#   ml/notebooks/07_highlights.ipynb: reuses a cached run (notebook 06's events_review.zip as a Kaggle dataset),
#   refreshes stages 3-6 on CPU, then builds the highlights; no GPU or re-detection needed.
#   ml/notebooks/08_full_match.ipynb: end to end for a long match from Google Drive (gdown -> /tmp, converted to
#   1280 wide @ <= 30 fps, calibration check/stop, stages 0-7, summary, highlights + condensed, results.zip).
#   ml/notebooks/09_retrain_ball.ipynb: fine-tune the ball detector on a new court from v1 (train blocks: v1-tracked
#   ball + motion pseudo-labels, + Test_video + Universe), v1 vs v2 on held-out test blocks, 200-frame eval pack.
padelvision appearance <video> --run runs/match1   # shirt/shorts colours (no model); then restats uses them
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
# Web app (local): API on :8000, web on :3000
.venv/Scripts/python -m pip install -e api
.venv/Scripts/padelvision-api
cd web && npm install && npm run dev        # lint: npm run lint; types: npx tsc --noEmit
.venv/Scripts/python -m pytest api          # API tests
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
