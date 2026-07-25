from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BallTracker:
    MAX_BALL_SIZE = 80
    MIN_BALL_SIZE = 3
    MAX_JUMP_PX = 400

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        detected = df_ball_positions.notna().all(axis=1).sum()
        total = len(df_ball_positions)
        logger.info("Ball detected in %d / %d frames (%.1f%%)", detected, total,
                     100 * detected / max(total, 1))

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        last_pos = None
        for frame in frames:
            ball_dict = self.detect_frame(frame, last_pos)
            ball_detections.append(ball_dict)
            if 1 in ball_dict:
                bbox = ball_dict[1]
                last_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame, last_pos=None):
        results = self.model.predict(frame, conf=0.15, verbose=False)[0]

        best_box = None
        best_conf = 0.0

        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w > self.MAX_BALL_SIZE or h > self.MAX_BALL_SIZE:
                continue
            if w < self.MIN_BALL_SIZE or h < self.MIN_BALL_SIZE:
                continue

            conf = float(box.conf)

            if last_pos is not None:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                dist = ((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2) ** 0.5
                if dist > self.MAX_JUMP_PX:
                    continue

            if conf > best_conf:
                best_conf = conf
                best_box = bbox

        ball_dict = {}
        if best_box is not None:
            ball_dict[1] = best_box
        return ball_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    