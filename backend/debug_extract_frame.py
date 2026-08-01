"""Extract frame 67 from the output video to check if player bboxes are there."""
import cv2

video_path = r"c:\Akhil\PadelVision-main\backend\outputs\f1efbba8-629f-4c9d-b82c-bb1e07a31dcb\video.mp4"
cap = cv2.VideoCapture(video_path)

# Seek to frame 67
cap.set(cv2.CAP_PROP_POS_FRAMES, 67)
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite(r"c:\Akhil\PadelVision-main\backend\debug_frame67_output.png", frame)
    print(f"Frame 67 saved, shape: {frame.shape}")
    
    # Also check frame 0
    cap2 = cv2.VideoCapture(video_path)
    ret2, frame0 = cap2.read()
    cap2.release()
    if ret2:
        cv2.imwrite(r"c:\Akhil\PadelVision-main\backend\debug_frame0_output.png", frame0)
        print(f"Frame 0 saved, shape: {frame0.shape}")
else:
    print("Failed to read frame 67")
