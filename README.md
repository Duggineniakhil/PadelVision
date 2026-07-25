# PadelVision

An AI-powered Padel match video analysis tool that automatically processes uploaded match videos through a computer vision pipeline — detecting players and the ball, mapping court geometry, calculating match statistics, and generating heatmaps, shot maps, and highlight clips. Results are delivered through both an interactive web dashboard and a fully annotated output video.

---

## What It Does

Upload a Padel match video. The pipeline:

1. **Detects and tracks all players** frame-by-frame using YOLOv8x, filtering to up to 4 active players closest to the court, and assigns them fixed labels of **Player 1**, **Player 2**, **Player 3**, and **Player 4**.
2. **Detects the Padel ball** using a custom-trained YOLO model, with Pandas-based interpolation to fill in frames where the ball is occluded or motion-blurred.
3. **Detects court keypoints** using a fine-tuned YOLO model, defining the court boundaries, service boxes, and net position.
4. **Projects player and ball positions** onto a live **2D mini-court overlay** rendered in the top-right corner of every video frame using a homography-based perspective transform.
5. **Detects ball shots** by finding direction changes in the ball's vertical position across frames, producing a list of shot timestamps.
6. **Calculates per-player match statistics** including: total shots, shot speed (km/h), max shot speed, player movement speed, and distance covered for all 4 players.
7. **Renders a real-time stats HUD** at the bottom of every video frame.
8. **Generates court coverage heatmaps** (one per player) using Gaussian kernel density estimation.
9. **Generates a ball shot map** showing the full ball trajectory, colour-coded by relative speed.
10. **Detects highlights** by identifying fast smashes and long rallies, each with start/end frames and timestamps.

The processed output video, heatmaps, shot map, and JSON stats file are all saved and served back through the web dashboard.

---

## Key Features

| Feature | Details |
|---|---|
| Player Tracking | YOLOv8x + persistent tracking, filtered to 4 players, remapped to Player 1–4 |
| Ball Tracking | Custom YOLO `padel_ball_detector.pt`, confidence threshold 0.15, Pandas interpolation for missing frames |
| Court Keypoint Detection | Fine-tuned YOLO `padel_court_keypoints.pt` outputting multiple court landmarks per frame |
| 2D Mini-Court | Real-time homography-based perspective projection of players & ball onto a top-down Padel court diagram |
| Shot Detection | Rolling delta-Y analysis on ball trajectory to find direction changes |
| Match Stats | Shot speed, player speed, distance, shot count — per player (up to 4), per rally segment |
| Stats HUD | OpenCV text overlay rendered on every video frame |
| Heatmaps | SciPy Gaussian filter applied to accumulated player positions on a Padel court schematic |
| Shot Map | Trajectory line + scatter plot coloured by relative ball speed |
| Highlights | Fast smash detection + long rally grouping with padding frames |
| Web Upload | Drag-and-drop video upload → `POST /api/upload` → background pipeline |
| Live Progress | Frontend polls `GET /api/status/{job_id}` every 2.5 seconds, shows stage + % |
| Dashboard | Video player, head-to-head stats, heatmaps, shot map, clickable highlight timeline |

---

## Technology Stack

- **Player Detection:** YOLOv8x (`yolov8x.pt` via Ultralytics)
- **Ball Detection:** Custom fine-tuned YOLO (`models/padel_ball_detector.pt`)
- **Court Keypoint Detection:** Fine-tuned YOLO (`models/padel_court_keypoints.pt`)
- **Image Processing:** OpenCV, NumPy
- **Data / Interpolation:** Pandas, SciPy
- **Spatial Transform:** OpenCV homography (`CourtMapper`)
- **Backend API:** FastAPI + SQLite job store
- **Background Processing:** FastAPI `BackgroundTasks`
- **Frontend:** Next.js 16 (React 19), TypeScript, Tailwind CSS v4, Lucide icons

---

## Repository Structure

```
PadelVision/
├── README.md                  # This file
├── LICENSE
├── output_video_img.png       # Sample output preview image
│
├── docs/
│   └── overview.md            # Detailed technical breakdown of the pipeline
│
├── backend/
│   ├── main.py                # CLI entrypoint — runs the full pipeline on a local video
│   ├── requirements.txt       # Python dependencies (pinned versions)
│   ├── start.bat              # Windows quick-start script for the backend
│   ├── yolo_inference.py      # Shared YOLO inference helper
│   ├── court_mapper.py        # Homography transform: video coords → mini-court coords
│   │
│   ├── app/
│   │   ├── main.py            # FastAPI application: CORS, static file serving, lifespan
│   │   └── schemas.py         # Pydantic v2 request/response models
│   │
│   ├── routes/
│   │   ├── upload.py          # POST /api/upload — accepts video, creates job, starts pipeline
│   │   ├── analysis.py        # GET /api/status/{id}, GET /api/analysis/{id}
│   │   └── video.py           # GET /api/health — model availability check
│   │
│   ├── pipeline/
│   │   ├── processor.py       # Orchestrates the full pipeline end-to-end
│   │   ├── detectors.py       # Thin wrappers: YOLOv8 player detection + ball detection
│   │   ├── trackers.py        # Post-processing: filter players, interpolate ball, mini-court conversion
│   │   ├── analytics.py       # Calculates per-player stats from tracking data
│   │   ├── heatmap.py         # Generates Gaussian heatmaps per player on Padel court schematic
│   │   ├── shot_map.py        # Generates ball trajectory map colour-coded by speed
│   │   └── highlights.py      # Detects fast smashes + long rallies
│   │
│   ├── trackers/
│   │   ├── player_tracker.py  # YOLOv8 player detection, track ID filtering + remapping to P1–P4
│   │   └── ball_tracker.py    # Custom YOLO ball detection + interpolation + shot frame detection
│   │
│   ├── court_line_detector/
│   │   └── court_line_detector.py  # YOLO inference → court keypoint landmarks
│   │
│   ├── mini_court/
│   │   └── mini_court.py      # 2D mini-court projection, rendering, point drawing
│   │
│   ├── jobs/
│   │   └── store.py           # SQLite-backed job status store (queued → processing → done/error)
│   │
│   ├── constants/             # Padel court dimension constants (metres)
│   │   └── __init__.py
│   │
│   ├── utils/                 # Helper functions
│   │   ├── bbox_utils.py      # Bounding box centre, distance, foot position
│   │   ├── conversions.py     # Pixel ↔ metre conversion helpers
│   │   ├── player_stats_drawer_utils.py  # OpenCV HUD overlay renderer
│   │   └── video_utils.py     # Video read/write via OpenCV
│   │
│   ├── models/                # Model weights (not in git — download separately)
│   │   ├── padel_ball_detector.pt
│   │   └── padel_court_keypoints.pt
│   │
│   ├── uploads/               # Incoming uploaded videos (per job_id)
│   └── outputs/               # Pipeline outputs per job_id (video, heatmaps, JSON)
│
└── frontend/
    ├── package.json           # Next.js 16, React 19, Tailwind v4
    ├── next.config.ts
    ├── tsconfig.json
    ├── postcss.config.mjs
    ├── eslint.config.mjs
    │
    ├── app/
    │   ├── layout.tsx         # Root layout with Navbar
    │   ├── page.tsx           # Home page — upload UI + SaaS feature cards
    │   ├── globals.css        # Global styles + custom animations
    │   ├── favicon.ico
    │   └── analysis/[id]/
    │       └── page.tsx       # Analysis dashboard page (3-column SaaS layout)
    │
    ├── components/
    │   ├── Navbar.tsx          # Top navigation bar
    │   ├── Sidebar.tsx         # Left tab navigation (Home, Shot Explorer, Stats, Leaderboards)
    │   ├── BreadcrumbHeader.tsx # Breadcrumb navigation header on dashboard
    │   ├── Uploader.tsx        # Drag-and-drop video uploader with status states
    │   ├── StatusPoller.tsx    # Polls /api/status/{id}, triggers data fetch on completion
    │   ├── VideoPlayer.tsx     # Custom HTML5 video player with controls
    │   ├── StatsPanel.tsx      # Head-to-head player stats comparison with progress bars
    │   ├── HeatmapView.tsx     # Renders heatmap images for each player
    │   ├── ShotMap.tsx         # Renders the ball shot map image
    │   └── HighlightTimeline.tsx # Clickable list of highlight events (seeks video on click)
    │
    └── lib/
        └── api.ts             # API base URL + uploadVideo(), getStatus(), getAnalysis() helpers
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- Node.js 18+

### 1. Create Python Virtual Environment

```bash
# From the project root:
python -m venv venv

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux:
source venv/bin/activate
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** If you are on Python 3.11 and `scipy` freezes on import, upgrade it: `pip install scipy==1.13.1`

### 3. Place Model Weights

Download and place the following model files:

| File | Location |
|---|---|
| `yolov8x.pt` | `backend/yolov8x.pt` (download from Ultralytics) |
| `padel_ball_detector.pt` | `backend/models/padel_ball_detector.pt` |
| `padel_court_keypoints.pt` | `backend/models/padel_court_keypoints.pt` |

**Download links for the custom Padel models:**
- `padel_ball_detector.pt` → [Google Drive](https://drive.google.com/file/d/1GNuhEyX-Akpb9UwbCrFBMS3EI-VQx0lx/view?usp=sharing)
- `padel_court_keypoints.pt` → [Google Drive](https://drive.google.com/file/d/1aXIFR855U27v2QAKy07q0HFOd7KCRlYu/view?usp=sharing)

---

## Running the Web Application

### Start the Backend

```bash
cd backend
..\venv\Scripts\Activate.ps1      # Windows
uvicorn app.main:app --reload --port 8000
```

### Start the Frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Running the CLI Pipeline (No Web UI)

If you want to run the analysis directly without the web interface, place your video at `backend/input_videos/input_video1.mp4` and run:

```bash
cd backend
python main.py
```

Output files will be saved to `backend/outputs/cli_run_{id}/`.

---

## Running on Google Colab (Free GPU)

Use this to avoid using your own CPU/GPU. Paste this into a single cell in a Colab notebook with a **T4 GPU runtime**:

```python
# 1. Clone repo
!rm -rf PadelVision
!git clone https://github.com/Duggineniakhil/PadelVision.git
%cd PadelVision/backend

# 2. Install dependencies (Colab already has PyTorch, OpenCV, Pandas, Numpy)
!pip install ultralytics==8.1.29 supervision gdown lapx

# 3. Download model weights from Google Drive
import os
os.makedirs('models', exist_ok=True)
!gdown "1GNuhEyX-Akpb9UwbCrFBMS3EI-VQx0lx" -O models/padel_ball_detector.pt
!gdown "1aXIFR855U27v2QAKy07q0HFOd7KCRlYu" -O models/padel_court_keypoints.pt

# 4. Upload your video
from google.colab import files
uploaded = files.upload()
video_filename = list(uploaded.keys())[0]

# 5. Run the pipeline
import sys
sys.path.append('.')
from pipeline.processor import process_video
result = process_video(input_path=video_filename, job_id="colab_run", use_stubs=False)

# 6. Download outputs
files.download(result['video_path'])
files.download(result['analysis_path'])
files.download(result['heatmap_paths']['p1'])
files.download(result['heatmap_paths']['p2'])
files.download(result['shot_map_path'])
```

---

## Video Requirements

- **Camera angle:** Must be a high-angle, broadcast-style or behind-the-baseline view. The court must be fully visible including all 4 corners.
- **Net-level or side-on footage:** Will not work. The court keypoint model was trained on high-angle frames and the homography transform requires all court corners to be visible.
- **Recommended resolution:** 720p or higher.
- **Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv` (up to 500MB via web upload).

---

## Architecture

### Backend Pipeline Flow

```
Video Upload (multipart)
        │
        ▼
  SQLite Job Created (queued)
        │
        ▼
  BackgroundTasks starts process_video()
        │
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Stage 1  │ Read video frames (OpenCV)                      │
  │  Stage 2  │ Detect players (YOLOv8x .track())               │
  │  Stage 3  │ Detect ball (custom YOLO .predict())            │
  │  Stage 4  │ Detect court keypoints (YOLO on first frame)    │
  │  Stage 5  │ Filter players → 4 closest to court             │
  │  Stage 6  │ Interpolate ball positions (Pandas)             │
  │  Stage 7  │ Detect shot frames (rolling delta-Y analysis)    │
  │  Stage 8  │ Build mini-court + homography transform          │
  │  Stage 9  │ Convert all positions to mini-court coords       │
  │  Stage 10 │ Compute per-player statistics                   │
  │  Stage 11 │ Generate heatmaps (SciPy Gaussian KDE)          │
  │  Stage 12 │ Generate shot map (trajectory + speed colouring) │
  │  Stage 13 │ Detect highlights (fast smashes + long rallies)  │
  │  Stage 14 │ Render annotated output video (HUD + overlays)   │
  │  Stage 15 │ Save video + analysis.json                      │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
  Job marked done → result stored in SQLite
        │
        ▼
  Frontend fetches analysis → renders dashboard
```

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Accept video file, create job, start background pipeline |
| `GET` | `/api/status/{job_id}` | Poll current processing stage + progress % |
| `GET` | `/api/analysis/{job_id}` | Fetch full result (stats, URLs, highlights) when done |
| `GET` | `/api/health` | Model availability health check |
| `GET` | `/outputs/{job_id}/...` | Static file serving for video, heatmaps, shot map |

### Frontend Pages

| Route | Purpose |
|---|---|
| `/` | Home — drag-and-drop upload + feature cards |
| `/analysis/{job_id}` | Analysis dashboard — video, stats, heatmaps, highlights |

---

## Acknowledgment

This project extends an open-source Padel analysis pipeline into a full-stack AI analytics platform with a FastAPI backend, Next.js frontend, asynchronous background processing, a SQLite job store, and an interactive analytics dashboard.
