"""Quick debug script to check player detections on the last processed video."""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch as _torch
_torch_load_original = _torch.load
def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_original(*args, **kwargs)
_torch.load = _torch_load_patched

import cv2
import glob

# Find the latest uploaded video
uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
videos = glob.glob(os.path.join(uploads_dir, '*.mp4'))
if not videos:
    print("No uploaded videos found!")
    sys.exit(1)
videos.sort(key=os.path.getmtime, reverse=True)
video_path = videos[0]
print(f"Using video: {video_path}")

# Read first few frames
cap = cv2.VideoCapture(video_path)
frames = []
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"Read {len(frames)} frames")

# Test player detection
from trackers import PlayerTracker
tracker = PlayerTracker(model_path="yolov8x.pt")
detections = []
for i, frame in enumerate(frames):
    det = tracker.detect_frame(frame)
    detections.append(det)
    print(f"Frame {i}: {len(det)} persons detected, track_ids: {list(det.keys())}")

# Test court keypoints
from court_line_detector import CourtLineDetector
court_det = CourtLineDetector(model_path="models/padel_court_keypoints.pt")
court_kps = court_det.predict(frames[0])
print(f"\nCourt keypoints: {court_kps.keys() if court_kps else 'EMPTY'}")
for k, v in court_kps.items():
    print(f"  {k}: {len(v)} points")

# Test choose_and_filter_players
filtered = tracker.choose_and_filter_players(court_kps, detections)
print(f"\nFiltered detections (10 frames):")
for i, fd in enumerate(filtered):
    print(f"  Frame {i}: players={list(fd.keys())}, bboxes={bool(fd)}")

# Count total non-empty frames
non_empty = sum(1 for fd in filtered if fd)
print(f"\nNon-empty filtered frames: {non_empty} / {len(filtered)}")

# Check chosen players
chosen = tracker.choose_players(court_kps, detections)
print(f"Chosen track IDs: {chosen}")
