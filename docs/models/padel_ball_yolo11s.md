# Model card: `padel_ball_yolo11s.pt` (ball detector, v1)

One-class ("ball") YOLO11s detector for padel footage, fine-tuned from the COCO-pretrained
`yolo11s.pt`. It was trained with `ml/notebooks/04_train_ball.ipynb` on Kaggle (2× T4, 60 epochs,
imgsz 1280, batch 8). It's built for the developer's low, wide-angle baseline camera, where the ball is ~5–30 px.

- File: `padel_ball_yolo11s.pt` (19.3 MB), published as GitHub release `ball-yolo11s-v1`;
  `models/manifest.yaml` entry `ball-detector` (`python scripts/fetch_models.py ball-detector`)
- sha256: `112bfdeaecc51a4074d9d3ee823cd856c829f80d7931ddb661b4fb1ef4dd2e5d`
- Licence: AGPL-3.0 (Ultralytics YOLO11). Training data credits are below.
- Recommended confidence threshold: **0.10**. The model's confidences are low; the median detection is about 0.1.

## Training data

| Source | Images | Balls | Licence |
|---|---|---|---|
| `Test_video.mp4` (developer's footage), train blocks only: classical pseudo-labels + visually confirmed balls | 725 | 738 | own footage |
| Roboflow Universe [`padel-analysis-eulgl/padel-ball-detection-nazbq` v4](https://universe.roboflow.com/padel-analysis-eulgl/padel-ball-detection-nazbq/dataset/4), class `ball` | 573 | 573 | CC BY 4.0 |
| Roboflow Universe [`padel-ll7pp/padel-dataset` v1](https://universe.roboflow.com/padel-ll7pp/padel-dataset/dataset/1), class `ball` only (`InactiveBall`, `net`, `player` dropped) | 7,735 | 10,754 | CC BY 4.0 |

Ultralytics validation (90% of the mixed data trains, 10% validates; best epoch 52): mAP50 0.754, mAP50-95 0.337,
P 0.69, R 0.75.

## Evaluation on the developer's footage

Test frames come from **held-out 20 s blocks** of `Test_video.mp4`. They are never trained on, and training frames
keep a ≥ 2 s margin from them. Every detection by every model on these frames was reviewed visually
(pooled labels in `ml/eval/labels/Test_video_ball.csv`: `ball` / `not_ball` / `unsure`; several balls per frame
are possible). A detection counts as correct within 10 px.

| Model | Frames | Recall | Precision | F1 |
|---|---|---|---|---|
| **padel_ball_yolo11s (this)**, conf ≥ 0.10 | 140 (112 balls) | **0.74** | **0.85** | **0.79** |
| COCO yolo11s "sports ball", best conf 0.40 | 140 | 0.39 | 0.80 | 0.53 |
| **padel_ball_yolo11s (this)**, conf ≥ 0.10 | 68 (Roboflow subset) | **0.72** | **0.88** | **0.79** |
| Roboflow hosted `padel-ball-detection-nazbq/4`, best conf 0.25 | 68 | 0.54 | 0.83 | 0.65 |
| COCO yolo11s "sports ball" | 68 | 0.49 | 0.54 | 0.52 |

## Known limitations

- **Single test video.** The numbers measure in-domain performance on one court, camera and lighting. A new
  court needs its own labelled test frames before these numbers can be trusted there.
- **Labels are visual crop reviews.** About 94 locations were undecidable (tiny distant dots) and are excluded
  from scoring.
- **Spare balls.** `padel-dataset` labels resting balls as `InactiveBall`, which was dropped, so the model may
  partly ignore balls lying still. That suits tracking the ball in play, but lowers recall on static balls.
- **False positives** still occur on some static spots (light patches on the fence) at low confidence. The
  stage 4 ball tracker should require motion consistency.
