import cv2
from ultralytics import YOLO

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image):
        results = self.model(image, verbose=False)[0]
        court_landmarks = {}

        if results.boxes is None or results.keypoints is None:
            return court_landmarks

        for box, keypoints in zip(results.boxes, results.keypoints.xy):
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            if class_name not in court_landmarks:
                court_landmarks[class_name] = []
            for kp in keypoints:
                x, y = float(kp[0]), float(kp[1])
                if x > 0 or y > 0:
                    court_landmarks[class_name].append((x, y))

        return court_landmarks

    def draw_keypoints(self, image, court_landmarks):
        for name, points in court_landmarks.items():
            for i, (x, y) in enumerate(points):
                x, y = int(x), int(y)
                label = f"{name}_{i}" if len(points) > 1 else name
                cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, court_landmarks):
        output_video_frames = []
        for frame in video_frames:
            frame_copy = frame.copy()
            frame_copy = self.draw_keypoints(frame_copy, court_landmarks)
            output_video_frames.append(frame_copy)
        return output_video_frames