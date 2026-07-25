import cv2

DEFAULT_FPS = 24.0

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 1:
        return DEFAULT_FPS
    return float(fps)

def save_video(output_video_frames, output_video_path, fps=DEFAULT_FPS):
    if not output_video_frames:
        raise ValueError("Cannot save an empty video.")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def save_highlights_video(video_frames, highlights, output_video_path, fps=DEFAULT_FPS):
    """
    Slices the video frames according to the highlight ranges and saves them to a new video file.
    Adds a brief black screen or text transition between clips.
    """
    if not highlights or not video_frames:
        return
    
    highlight_frames = []
    height, width = video_frames[0].shape[:2]
    
    for idx, highlight in enumerate(highlights):
        start = highlight.get("start_frame", 0)
        end = highlight.get("end_frame", len(video_frames)-1)
        label = highlight.get("label", f"Highlight {idx+1}")
        
        # Add transition frame (black screen with text) for 1 second
        transition_frames = int(fps)
        transition_frame = cv2.resize(cv2.imread(""), (width, height)) if False else None # empty frame
        if transition_frame is None:
            import numpy as np
            transition_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(transition_frame, f"Clip {idx+1}: {label}", (width//2 - 300, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
        highlight_frames.extend([transition_frame.copy() for _ in range(transition_frames)])
        
        # Add the actual highlight clip
        clip = video_frames[start:end+1]
        highlight_frames.extend(clip)
        
    save_video(highlight_frames, output_video_path, fps=fps)
