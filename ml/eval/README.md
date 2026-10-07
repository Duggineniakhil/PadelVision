# Evaluation set

A small hand-labelled set used to measure every model and heuristic. Videos and labels
live on Kaggle/Drive (not in git, since videos are large). This file defines the format.

## Clips
3–5 clips of about 60 s each, from **different courts or cameras**, filmed from a fixed camera that shows the whole court.
Name them `clip01.mp4`, `clip02.mp4`, … and keep them in a Kaggle Dataset (e.g. `padelvision-eval`)
or a Drive folder (`padel/eval/`).

## Labels (one folder per clip)
```
clip01/
  court.json      # {"image_points": {"near_left": [x, y], "near_right": [...], ...}}
                  #   keypoint names: pipeline/src/padelvision/court/geometry.py
  players.csv     # frame,player_id,x1,y1,x2,y2   every 10th frame, player_id 1-4 kept consistent
  ball.csv        # frame,visible,x,y              ~500 frames; visible=0 when hidden/out of frame
```
- Players 1–2 are the near team and 3–4 the far team (consistent within a clip).
- Label the ball centre in pixels. Mark `visible=0` instead of guessing a hidden ball.
- Use CVAT or Label Studio and export to these CSVs (a converter script comes in phase 1).

## Metrics (scripts arrive with each phase)
| Metric | Target | Phase |
|---|---|---|
| Court reprojection error | < 15 cm | 1 |
| Player ID switches | < 2 per minute | 1 |
| Ball recall / precision (within 10 px) | ≥ 80% / ≥ 90% | 2 |
| Hit / bounce F1 | TBD | 3 |
