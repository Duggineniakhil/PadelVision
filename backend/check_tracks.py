import cv2
from trackers import PlayerTracker
from utils import read_video
import os

video_path = r"c:\Akhil\PadelVision-main\backend\..\uploads\f1efbba8-629f-4c9d-b82c-bb1e07a31dcb.mp4"
frames = read_video(video_path)

tracker = PlayerTracker("yolov8x.pt")
player_detections = tracker.detect_frames(frames)

all_track_ids = set()
for d in player_detections:
    all_track_ids.update(d.keys())

print(f"Total track IDs found over {len(frames)} frames: {len(all_track_ids)}")
print(f"Track IDs: {sorted(list(all_track_ids))}")
