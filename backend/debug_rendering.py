"""Debug: trace the rendering pipeline step by step."""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch as _torch
_torch_load_original = _torch.load
def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_original(*args, **kwargs)
_torch.load = _torch_load_patched

import cv2
import numpy as np
import glob

# Find latest video
uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
videos = sorted(glob.glob(os.path.join(uploads_dir, '*.mp4')), key=os.path.getmtime, reverse=True)
video_path = videos[0]
print(f"Using: {video_path}")

# Read 3 frames
cap = cv2.VideoCapture(video_path)
frames = []
for i in range(3):
    ret, frame = cap.read()
    if ret: frames.append(frame)
cap.release()
print(f"Read {len(frames)} frames, shape: {frames[0].shape}")

# Detect players
from trackers import PlayerTracker
player_tracker = PlayerTracker(model_path="yolov8x.pt")
player_detections = player_tracker.detect_frames(frames)
print(f"Raw detections: {[list(d.keys()) for d in player_detections]}")

# Detect court
from court_line_detector import CourtLineDetector
court_detector = CourtLineDetector(model_path="models/padel_court_keypoints.pt")
court_keypoints = court_detector.predict(frames[0])

# Filter
filtered = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
print(f"Filtered: {[list(d.keys()) for d in filtered]}")
print(f"Filtered bboxes frame 0: {filtered[0]}")

# Now trace the drawing pipeline, saving intermediate frames
frame_idx = 0

# Step 1: player bboxes
step1_frames = player_tracker.draw_bboxes(list(frames), filtered)
cv2.imwrite("debug_step1_player.png", step1_frames[frame_idx])
print(f"Step 1 (player bboxes) saved. Has red pixels: {np.any(step1_frames[frame_idx][:,:,2] > 200)}")

# Step 2: ball bboxes (just use empty detections for debug)
from trackers import BallTracker
ball_tracker = BallTracker(model_path="models/padel_ball_detector.pt")
ball_detections = ball_tracker.detect_frames(frames)
ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
step2_frames = ball_tracker.draw_bboxes(step1_frames, ball_detections)
cv2.imwrite("debug_step2_ball.png", step2_frames[frame_idx])

# Step 3: court keypoints
step3_frames = court_detector.draw_keypoints_on_video(step2_frames, court_keypoints)
cv2.imwrite("debug_step3_court.png", step3_frames[frame_idx])

# Step 4: mini court
from mini_court import MiniCourt
mini_court = MiniCourt(frames[0])
step4_frames = mini_court.draw_mini_court(step3_frames)
cv2.imwrite("debug_step4_minicourt.png", step4_frames[frame_idx])

# Check if player bboxes survive to step4
# Look at where we know player 1 bbox should be
bbox = filtered[0].get(1)
if bbox:
    x1, y1, x2, y2 = [int(c) for c in bbox]
    # Check if step4 has red text/rect near that area
    region_step1 = step1_frames[frame_idx][max(0,y1-15):y1, x1:x1+100]
    region_step4 = step4_frames[frame_idx][max(0,y1-15):y1, x1:x1+100]
    print(f"\nPlayer 1 bbox: {bbox}")
    print(f"Label region step1 mean BGR: {region_step1.mean(axis=(0,1)) if region_step1.size > 0 else 'empty'}")
    print(f"Label region step4 mean BGR: {region_step4.mean(axis=(0,1)) if region_step4.size > 0 else 'empty'}")
    # Are they the same?
    if region_step1.size > 0 and region_step4.size > 0:
        diff = np.abs(region_step1.astype(float) - region_step4.astype(float)).mean()
        print(f"Mean pixel diff between step1 and step4 label regions: {diff:.2f}")

print("\nDone - check debug_step*.png files")
