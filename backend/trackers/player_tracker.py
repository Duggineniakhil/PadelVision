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
        court_points = self._flatten_court_landmarks(court_landmarks)
        court_xs = [pt[0] for pt in court_points]
        court_ys = [pt[1] for pt in court_points]
        
        if not court_xs or not court_ys:
            min_x, max_x, min_y, max_y = 0, 10000, 0, 10000
        else:
            pad_x = 200
            pad_y = 200
            min_x, max_x = min(court_xs) - pad_x, max(court_xs) + pad_x
            min_y, max_y = min(court_ys) - pad_y, max(court_ys) + pad_y

        filtered_player_detections = []
        player_last_positions = {1: None, 2: None, 3: None, 4: None}
        current_track_ids = {1: None, 2: None, 3: None, 4: None}
        
        for player_dict in player_detections:
            # Filter tracks that are inside the court area
            valid_tracks = {}
            for track_id, bbox in player_dict.items():
                cx, cy = get_center_of_bbox(bbox)
                if min_x <= cx <= max_x and min_y <= cy <= max_y:
                    valid_tracks[track_id] = bbox
            
            filtered_frame_dict = {}
            available_tracks = set(valid_tracks.keys())
            
            # 1. Keep existing track IDs if they are still valid
            for p_id in range(1, 5):
                t_id = current_track_ids[p_id]
                if t_id is not None and t_id in available_tracks:
                    filtered_frame_dict[p_id] = valid_tracks[t_id]
                    player_last_positions[p_id] = get_center_of_bbox(valid_tracks[t_id])
                    available_tracks.remove(t_id)
            
            # 2. For unassigned player IDs, find the closest available track
            unassigned_pids = [p for p in range(1, 5) if p not in filtered_frame_dict]
            
            # If this is the very first assignment, sort available tracks spatially 
            # to consistently assign 1=TopLeft, 2=TopRight, 3=BotLeft, 4=BotRight
            if len(unassigned_pids) == 4 and all(pos is None for pos in player_last_positions.values()) and len(available_tracks) >= 4:
                sorted_tracks = sorted(list(available_tracks), key=lambda tid: (get_center_of_bbox(valid_tracks[tid])[1], get_center_of_bbox(valid_tracks[tid])[0]))
                for i, p_id in enumerate(range(1, 5)):
                    t_id = sorted_tracks[i]
                    filtered_frame_dict[p_id] = valid_tracks[t_id]
                    player_last_positions[p_id] = get_center_of_bbox(valid_tracks[t_id])
                    current_track_ids[p_id] = t_id
                    available_tracks.remove(t_id)
            else:
                for p_id in unassigned_pids:
                    last_pos = player_last_positions[p_id]
                    if last_pos is None:
                        # If no last position, just pick an available track
                        if available_tracks:
                            t_id = available_tracks.pop()
                            filtered_frame_dict[p_id] = valid_tracks[t_id]
                            player_last_positions[p_id] = get_center_of_bbox(valid_tracks[t_id])
                            current_track_ids[p_id] = t_id
                    else:
                        # Find closest available track
                        best_tid = None
                        best_dist = float('inf')
                        for t_id in available_tracks:
                            cx, cy = get_center_of_bbox(valid_tracks[t_id])
                            dist = ((cx - last_pos[0])**2 + (cy - last_pos[1])**2)**0.5
                            if dist < best_dist:
                                best_dist = dist
                                best_tid = t_id
                        
                        # Distance threshold to avoid jumping to wrong track
                        if best_tid is not None and best_dist < 300:
                            filtered_frame_dict[p_id] = valid_tracks[best_tid]
                            player_last_positions[p_id] = get_center_of_bbox(valid_tracks[best_tid])
                            current_track_ids[p_id] = best_tid
                            available_tracks.remove(best_tid)
                        else:
                            # Track lost for this player
                            current_track_ids[p_id] = None
            
            filtered_player_detections.append(filtered_frame_dict)
            
        return filtered_player_detections

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
