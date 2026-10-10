# Evaluation

How every model and heuristic is measured. Labels are small CSV/JSON files in `labels/` (in git);
videos stay on Kaggle/Drive (never in git). So far everything is measured on one video, the developer's
`Test_video.mp4` (206 s, 1280x720 @ 30 fps, low wide-angle camera near one baseline; mostly warm-up and
ball handling, with 4 short rallies). Numbers will move when a second video is added.

## Court calibration
Calibrations live in `ml/calibrations/<video stem>.json` (made with `padelvision calibrate`). The pipeline prints
the mean keypoint error at stage 1 (`[1] court: mean keypoint error ...`).

## Ball (phase 2): pooled labels
```
labels/Test_video_ball.csv          frame,status,u,v,source     status = ball | not_ball | unsure
labels/Test_video_ball_frames.json  {"video", "frames", "excluded", "note"}   the 300 reviewed frames
```
- Labels are pooled: every detection any model made on the reviewed frames was checked on a crop. A frame can
  hold several balls (spares, a ball in hand). `unsure` spots (tiny distant dots) are ignored when scoring.
- Detections on spots nobody reviewed are *unverified*, not wrong. Review their crops and add the verdicts
  (`ball/evaluate.py`: `classify`, `score`, `review_crops`) so every model is judged on the same spots.
- A detection is correct within 10 px. Test frames come from held-out 20 s blocks (see the model card).

## Hits and bounces (phase 3)
### Exhaustive labels (recall + precision): not made yet
Made with `padelvision label-events` on a pack from `event-review-pack` (notebook 06), then copied here:
```
labels/<video stem>_events.csv            frame,kind,u,v,player   kind = hit | bounce | wall (walls not scored)
labels/<video stem>_events_windows.json   [{"id", "start", "end"}]  windows labelled exhaustively
```
- Windows: every rally, other activity with a hit, and a few random windows (so missed play is covered too).
- Inside a finished window every hit and floor bounce is marked: no mark = no event. Only predictions inside
  finished windows are scored (`padelvision events-eval`, same kind within 0.2 s).
- A hit is a real shot (serves included). Bouncing the ball between points or tapping it to a partner is not.
- Hit `player` is the slot id 1-4 (1-2 near, 3-4 far), as in `players.parquet`.

### Reviewed predicted events (precision only)
`labels/Test_video_events_reviewed.csv`: frame, predicted, truth, player, note. Every hit / handling / bounce /
turn (then called "wall") the pipeline predicted inside the event pack's windows (run of 2026-10-10), checked
visually on full-resolution crop strips (-0.2 s .. +0.2 s) by Claude, not by a human. truth = hit | handling |
bounce | none | duplicate (a second turn of the same contact) | unclear. It measures precision and hit
attribution only: events the pipeline missed entirely are not in it, so it says nothing about recall.

## Player identities (phase 1): no labels yet
ID switches have not been measured. A labelled set would be player boxes with consistent ids 1-4
(1-2 near, 3-4 far) every 10th frame.

## Targets and current results (Test_video)
| Metric | Target | Current | Phase |
|---|---|---|---|
| Court mean keypoint error | < 15 cm | 32 cm (not met) | 1 |
| Player ID switches | < 2 per minute | not measured | 1 |
| Ball detector recall / precision (10 px, held-out blocks) | >= 80% / >= 90% | 74% / 85% (not met) | 2 |
| Ball in play found (stage 4) | | 53% of frames | 2 |
| Hits inside rallies: precision (visual review) | | 17 of 17 decided | 3 |
| Floor bounces: precision (visual review) | | 9 of 11 decided | 3 |
| Hit / bounce recall and F1 (within 0.2 s) | set after the first labels | not measured | 3 |
