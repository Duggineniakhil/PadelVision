from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def _flatten_court_landmarks(self, court_landmarks):
        points = []
        if isinstance(court_landmarks, dict):
            values = court_landmarks.values()
        else:
            values = court_landmarks or []

        for value in values:
            if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                points.append(value)
            elif isinstance(value, (list, tuple)):
                for point in value:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        points.append((point[0], point[1]))
        return points

    def choose_and_filter_players(self, court_landmarks, player_detections):
        chosen_players = self.choose_players(court_landmarks, player_detections)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {}
            for i, track_id in enumerate(chosen_players):
                if track_id in player_dict:
                    filtered_player_dict[i + 1] = player_dict[track_id]
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_landmarks, player_detections):
        # Calculate rough court bounding box
        court_points = self._flatten_court_landmarks(court_landmarks)
        court_xs = [pt[0] for pt in court_points]
        court_ys = [pt[1] for pt in court_points]
        
        if not court_xs or not court_ys:
            # Fallback if court landmarks are missing
            min_x, max_x, min_y, max_y = 0, 10000, 0, 10000
        else:
            pad_x = 200
            pad_y = 200
            min_x, max_x = min(court_xs) - pad_x, max(court_xs) + pad_x
            min_y, max_y = min(court_ys) - pad_y, max(court_ys) + pad_y

        track_lifespans = {}
        for player_dict in player_detections:
            for track_id, bbox in player_dict.items():
                center = get_center_of_bbox(bbox)
                cx, cy = center
                # Check if player is near/inside court
                if min_x <= cx <= max_x and min_y <= cy <= max_y:
                    track_lifespans[track_id] = track_lifespans.get(track_id, 0) + 1
        
        # Sort by longest lived in court area
        sorted_tracks = sorted(track_lifespans.items(), key=lambda x: x[1], reverse=True)
        # Choose the top 4 tracks
        chosen_players = [t[0] for t in sorted_tracks[:4]]
        return chosen_players

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True, verbose=False)[0]
        id_name_dict = results.names

        player_dict = {}
        if results.boxes is not None:
            for box in results.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name == "person":
                    player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # We copy the frame to avoid drawing on original if passed around
            frame_copy = frame.copy()
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                label = f"Player {track_id}"
                cv2.putText(frame_copy, label, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame_copy)
        return output_video_frames
