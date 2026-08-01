"""Full pipeline debug: trace player detection counts through the entire pipeline."""
import sys, os, logging
sys.path.append(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

import torch as _torch
_torch_load_original = _torch.load
def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_original(*args, **kwargs)
_torch.load = _torch_load_patched

import cv2
import glob

# Find latest video
uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
videos = sorted(glob.glob(os.path.join(uploads_dir, '*.mp4')), key=os.path.getmtime, reverse=True)
video_path = videos[0]
print(f"Using: {video_path}")

# Read ALL frames (like the real pipeline does)
from utils import read_video
video_frames = read_video(video_path)
print(f"Total frames: {len(video_frames)}")

# Detect players (like the real pipeline)
from pipeline.detectors import detect_players
player_detections, player_tracker = detect_players(video_frames, model_path="yolov8x.pt")

# Count raw detections
raw_non_empty = sum(1 for d in player_detections if d)
raw_total_persons = sum(len(d) for d in player_detections)
print(f"\nRaw detections: {raw_non_empty}/{len(player_detections)} frames have persons")
print(f"Total person detections across all frames: {raw_total_persons}")

# Show first 5 frames
for i in range(min(5, len(player_detections))):
    print(f"  Frame {i}: track_ids={list(player_detections[i].keys())}")

# Detect court keypoints
from pipeline.trackers import detect_court_keypoints, filter_players
court_keypoints, court_detector = detect_court_keypoints(video_frames[0])
print(f"\nCourt keypoints: {court_keypoints.keys() if court_keypoints else 'EMPTY'}")

# Filter players
filtered_detections = filter_players(player_tracker, court_keypoints, player_detections)

# Count filtered detections
filt_non_empty = sum(1 for d in filtered_detections if d)
filt_total = sum(len(d) for d in filtered_detections)
print(f"\nFiltered detections: {filt_non_empty}/{len(filtered_detections)} frames have players")
print(f"Total player entries across all frames: {filt_total}")

# Show first 5 and last 5 frames  
for i in range(min(5, len(filtered_detections))):
    print(f"  Frame {i}: players={list(filtered_detections[i].keys())}, has_data={bool(filtered_detections[i])}")
print("  ...")
for i in range(max(0, len(filtered_detections)-5), len(filtered_detections)):
    print(f"  Frame {i}: players={list(filtered_detections[i].keys())}, has_data={bool(filtered_detections[i])}")

# Test draw_bboxes on filtered detections
output_frames = player_tracker.draw_bboxes(list(video_frames), filtered_detections)
cv2.imwrite("debug_full_step1_frame0.png", output_frames[0])
cv2.imwrite("debug_full_step1_frame67.png", output_frames[67])
print(f"\nSaved step1 frames 0 and 67 with player bboxes drawn")

# Now simulate the FULL rendering pipeline
from pipeline.detectors import detect_ball
from pipeline.trackers import interpolate_ball, get_shot_frames, build_mini_court, convert_to_mini_court_coordinates
from pipeline.analytics import generate_statistics
from utils.player_stats_drawer_utils import draw_player_stats
from utils import get_video_fps

fps = get_video_fps(video_path)
ball_detections, ball_tracker = detect_ball(video_frames)
ball_detections = interpolate_ball(ball_tracker, ball_detections)

mini_court = build_mini_court(video_frames[0])
player_mini_positions, ball_mini_positions = convert_to_mini_court_coordinates(
    mini_court, filtered_detections, ball_detections, court_keypoints
)

ball_shot_frames = get_shot_frames(ball_tracker, ball_detections)
stats_result = generate_statistics(player_mini_positions, ball_mini_positions, ball_shot_frames, mini_court, fps=fps)
frame_stats_df = stats_result.pop("frame_stats_df")

# Now the actual rendering chain
print("\n--- Rendering chain ---")
output_frames = player_tracker.draw_bboxes(list(video_frames), filtered_detections)
cv2.imwrite("debug_chain_step1.png", output_frames[0])
print(f"After player draw: id(output_frames[0])={id(output_frames[0])}")

output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
cv2.imwrite("debug_chain_step2.png", output_frames[0])
print(f"After ball draw: id(output_frames[0])={id(output_frames[0])}")

output_frames = court_detector.draw_keypoints_on_video(output_frames, court_keypoints)
cv2.imwrite("debug_chain_step3.png", output_frames[0])
print(f"After court draw: id(output_frames[0])={id(output_frames[0])}")

output_frames = mini_court.draw_mini_court(output_frames)
cv2.imwrite("debug_chain_step4.png", output_frames[0])
print(f"After mini court: id(output_frames[0])={id(output_frames[0])}")

print("\nDone! Check debug_chain_step*.png and debug_full_step1_*.png")
