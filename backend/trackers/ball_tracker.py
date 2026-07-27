from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BallTracker:
    MAX_BALL_SIZE = 150      # Padel balls can appear larger at close camera distances
    MIN_BALL_SIZE = 3
    MAX_JUMP_PX = 600        # Allow larger jumps between frames
    LOST_RESET_FRAMES = 10   # Reset last_pos after this many consecutive misses

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        detected = df_ball_positions.notna().all(axis=1).sum()
        total = len(df_ball_positions)
        logger.info("Ball detected in %d / %d frames (%.1f%%)", detected, total,
                     100 * detected / max(total, 1))

        # Limit interpolation to gaps of at most 30 frames to avoid
        # wild straight-line hallucinations across long missing stretches.
        df_ball_positions = df_ball_positions.interpolate(limit=30)
        df_ball_positions = df_ball_positions.bfill(limit=30)

        after_interp = df_ball_positions.notna().all(axis=1).sum()
        logger.info("After interpolation: %d / %d frames have ball data (%.1f%%)",
                     after_interp, total, 100 * after_interp / max(total, 1))

        ball_positions = [{1: x} if not any(pd.isna(v) for v in x) else {}
                          for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        if df.empty or df['y1'].isna().all():
            logger.warning("No ball positions available for shot detection")
            return []

        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df['delta_y'] = df['mid_y_rolling_mean'].diff()

        total_frames = len(df)
        minimum_change_frames_for_hit = max(5, min(25, total_frames // 20))

        look_ahead = int(minimum_change_frames_for_hit * 1.2)
        for i in range(1, total_frames - look_ahead):
            neg_change = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i + 1] < 0
            pos_change = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i + 1] > 0

            if neg_change or pos_change:
                change_count = 0
                for j in range(i + 1, i + look_ahead + 1):
                    neg_following = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[j] < 0
                    pos_following = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[j] > 0

                    if (neg_change and neg_following) or (pos_change and pos_following):
                        change_count += 1

                if change_count >= minimum_change_frames_for_hit - 1:
                    df.loc[i, 'ball_hit'] = 1

        hits = df[df['ball_hit'] == 1].index.tolist()
        logger.info("Shot detection: %d shots found (threshold=%d frames)", len(hits), minimum_change_frames_for_hit)
        return hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # ── Pass 1: normal detection with jump constraint + recovery ──────
        last_pos = None
        lost_count = 0
        for frame in frames:
            ball_dict = self.detect_frame(frame, last_pos)
            if 1 in ball_dict:
                bbox = ball_dict[1]
                last_pos = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                lost_count = 0
            else:
                lost_count += 1
                # After too many consecutive misses, reset anchor so any
                # detection can re-acquire the ball.
                if lost_count >= self.LOST_RESET_FRAMES:
                    last_pos = None
            ball_detections.append(ball_dict)

        # ── Pass 2: retry missed frames without jump constraint ───────────
        # For frames that had no detection at all, try again with no last_pos
        # (unconstrained) so the ball can be picked up anywhere in the frame.
        pass1_detected = sum(1 for d in ball_detections if d)
        pass1_total = len(ball_detections)
        logger.info("Ball pass-1: detected %d / %d (%.1f%%)",
                     pass1_detected, pass1_total,
                     100 * pass1_detected / max(pass1_total, 1))

        retry_count = 0
        for i, (frame, det) in enumerate(zip(frames, ball_detections)):
            if not det:
                retry = self.detect_frame(frame, last_pos=None)
                if retry:
                    ball_detections[i] = retry
                    retry_count += 1
        if retry_count:
            logger.info("Ball pass-2 (unconstrained retry): recovered %d extra frames", retry_count)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame, last_pos=None):
        results = self.model.predict(frame, conf=0.10, verbose=False)[0]

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