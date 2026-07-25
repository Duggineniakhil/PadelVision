# PadelVision: Technical Overview

PadelVision is an AI-powered Padel match video analysis platform. You upload a Padel match video, and the system automatically processes it through a computer vision pipeline to generate player tracking, ball tracking, court detection, match statistics, heatmaps, a shot map, and highlight clips. The results are displayed through a Next.js web dashboard and also embedded as overlays on the processed output video.

---

## What the Project Actually Does

### 1. Video Ingestion

A match video is either submitted through the web dashboard (drag-and-drop upload) or pointed to via the CLI script (`backend/main.py`). The video is loaded frame-by-frame using OpenCV (`cv2.VideoCapture`).

---

### 2. Player Detection & Tracking

- Uses **YOLOv8x** (`yolov8x.pt`) to detect every person visible in each frame.
- The `.track()` method from Ultralytics is used to assign persistent tracking IDs to players across frames (`persist=True`).
- After tracking, the four players most consistently inside or near the detected court region are selected and remapped to fixed IDs of **Player 1**, **Player 2**, **Player 3**, and **Player 4**. Any other detected people, such as staff or spectators, are filtered out.
- Bounding boxes are drawn on the output video frames labelling each player.

---

### 3. Padel Ball Detection & Interpolation

- Uses a **custom fine-tuned YOLO model** (`models/padel_ball_detector.pt`) specifically trained to detect a Padel ball.
- The ball is detected per-frame using `model.predict()` at a confidence threshold of 0.15 (lower threshold to catch fast-moving balls).
- Because the ball is often occluded or motion-blurred, detected positions are collected into a Pandas DataFrame and **interpolated** (filling gaps between missing detections) and **backward-filled** to create a complete ball position history for every frame.

---

### 4. Court Line & Keypoint Detection

- Uses a **fine-tuned YOLO model** (`models/padel_court_keypoints.pt`) that outputs court landmark keypoints (e.g., corners, service line points, net endpoints).
- The model runs on the first frame of the video only, since the camera is assumed to be static.
- Keypoints are returned as a dictionary of `{class_name: [(x, y), ...]}`.
- These keypoints define the boundaries of the Padel court (10m × 20m), the service boxes, and the net, and are used as the spatial reference for the homography transform.

---

### 5. 2D Mini-Court Projection

- A `MiniCourt` object is created and drawn as a small top-down Padel court diagram in the top-right corner of every output video frame (250×400 px canvas with a blue background).
- The real-world positions of Player 1–4 and the ball are converted from raw pixel coordinates into **2D mini-court coordinates** using a homography-based perspective transform (`CourtMapper`):
  - The 4 outer court corners from the YOLO keypoint detection are sorted into top-left, top-right, bottom-right, bottom-left order.
  - A `cv2.findHomography()` matrix maps video pixels to mini-court pixels.
  - For players, the foot position (bottom-center of the bounding box) is used as their ground-plane position.
  - For the ball, the center of the bounding box is used.
- The positions are drawn as coloured circles on the mini-court overlay in every frame (green for players, yellow for ball).

---

### 6. Ball Shot Detection

- The ball's Y-position (vertical) rolling mean is calculated across a sliding window of 5 frames.
- A **direction change** in the ball's vertical trajectory (delta_y sign flip) that persists for at least 25 frames is classified as a **ball hit** (a shot).
- This produces a list of frame numbers where shots occurred.

---

### 7. Match Statistics Calculation

- For each pair of consecutive shot frames (a shot segment), the pipeline calculates:
  - **Ball shot speed** in km/h using the Euclidean distance the ball travelled across the mini-court divided by the time elapsed (in seconds at 24 FPS).
  - **Opponent movement speed** in km/h (the player who did not hit the ball, measured over the same time window).
- Stats are accumulated per player across all shot segments for all 4 players:
  - Total shots
  - Total and average shot speed (km/h)
  - Maximum shot speed (km/h)
  - Average player movement speed (km/h)
  - Total distance covered (metres) — approximated from accumulated speed over time
- A per-frame stats DataFrame is generated to drive the **stats HUD overlay** drawn on each video frame.

---

### 8. Player Stats Overlay (HUD)

- A semi-transparent stats table is rendered directly onto the bottom-right of each video frame using OpenCV.
- Displayed in real time as the match progresses, showing Team A (Player 1 + Player 2) and Team B (Player 3 + Player 4) columns with:
  - Shot speed (last shot)
  - Player speed
  - Average shot speed
  - Average player speed

---

### 9. Heatmap Generation

- After all frames are processed, the mini-court positions of each player are accumulated.
- Positions are normalized to real-world Padel court coordinates (10m × 20m).
- A **2D histogram density grid** is built using `np.histogram2d` and smoothed with `scipy.ndimage.gaussian_filter` (sigma=3) separately for Player 1–4.
- The heatmaps are saved as PNG images on a dark-themed Padel court schematic showing court coverage intensity — where each player spent the most time during the match.

---

### 10. Shot Map Generation

- All ball positions at every frame are plotted as a trajectory on a Padel court schematic.
- The trajectory is drawn as coloured line segments using `matplotlib.collections.LineCollection`, with a `plasma` colormap representing relative ball speed.
- Scatter dots are overlaid at each position.
- A colour bar indicates relative speed (slower = cool colours, faster = warm colours).
- The shot map is saved as `shot_map.png`.

---

### 11. Highlight Detection

- Detected shot frames are analysed for two types of highlights:
  - **Fast Padel Smashes:** Shots where the ball speed exceeds 40% of the maximum shot speed in the match.
  - **Long Padel Rallies:** Sequences of 5+ consecutive shots (with gaps < 3 seconds between shots).
- Each highlight stores its start frame, end frame (with ±60 frame padding), timestamp in seconds, and a descriptive label.
- Overlapping highlights are merged to avoid duplicates.

---

### 12. Output Artifacts

After the pipeline completes, the following files are saved under `backend/outputs/{job_id}/`:

| File | Description |
|---|---|
| `video.avi` | Fully processed video with bounding boxes, mini-court overlay, and stats HUD |
| `heatmap_p1.png` | Court coverage heatmap for Player 1 |
| `heatmap_p2.png` | Court coverage heatmap for Player 2 |
| `heatmap_p3.png` | Court coverage heatmap for Player 3 |
| `heatmap_p4.png` | Court coverage heatmap for Player 4 |
| `shot_map.png` | Ball trajectory map coloured by relative speed |
| `analysis.json` | JSON file with full stats, highlight timestamps, and all result URLs |

---

### 13. Web Dashboard

The results are surfaced through a **Next.js 16** web frontend served at `http://localhost:3000`:

- **Upload page:** Drag-and-drop or click-to-browse video upload. Submits to `POST /api/upload`.
- **Processing status:** A circular progress indicator polls `GET /api/status/{job_id}` every 2.5 seconds and shows the current pipeline stage name and percentage, with a checklist of completed steps.
- **Analysis dashboard:** Once processing is complete, the dashboard shows a 3-column SaaS-style layout:
  - **Center:** The processed output video in a custom video player (with play/pause, mute, seek, and fullscreen controls)
  - **Left sidebar:** Tab navigation — Home (video + stats + heatmaps + shot map), Shot Explorer, Game Stats, Leaderboards
  - **Right panel:** Clickable highlight timeline — clicking a highlight seeks the video to that timestamp
  - **Stats panel:** Player 1 vs Player 2 head-to-head comparison with animated progress bars for: Total Shots, Avg Shot Speed, Max Shot Speed, Avg Movement, Distance Run
  - **Heatmaps:** Side-by-side court coverage for each player
  - **Shot map:** Full ball trajectory with speed colouring
  - **Leaderboards:** Placeholder tournament comparison metrics

---

## Technology Stack

| Layer | Technology |
|---|---|
| Player detection | YOLOv8x (Ultralytics) |
| Ball detection | Custom fine-tuned YOLO (`padel_ball_detector.pt`) |
| Court keypoint detection | Fine-tuned YOLO (`padel_court_keypoints.pt`) |
| Spatial transform | OpenCV homography (`CourtMapper`) |
| Image processing | OpenCV, NumPy |
| Data processing | Pandas, SciPy |
| Backend API | FastAPI (Python), SQLite (job store) |
| Background processing | FastAPI `BackgroundTasks` |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS v4, Lucide icons |

---

## Video Requirements

The pipeline is calibrated for **broadcast-style, high-angle Padel footage** (from behind the baseline or from a high overhead position). Net-level or side-on footage will not work correctly because:

- The YOLO court keypoint model was trained on high-angle frames.
- The homography transform for the 2D mini-court projection requires all 4 court corners to be clearly visible.
- Player occlusion at net level breaks both the tracking and coordinate transforms.

---

## Model Weights

The following model files are required but are **not included in this repository** (they are tracked in `.gitignore`). Download them before running:

| Model | Purpose | Download |
|---|---|---|
| `yolov8x.pt` | Player detection (person class) | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| `padel_ball_detector.pt` | Padel ball detection | [Google Drive](https://drive.google.com/file/d/1GNuhEyX-Akpb9UwbCrFBMS3EI-VQx0lx/view?usp=sharing) |
| `padel_court_keypoints.pt` | Court keypoint / landmark detection | [Google Drive](https://drive.google.com/file/d/1aXIFR855U27v2QAKy07q0HFOd7KCRlYu/view?usp=sharing) |
