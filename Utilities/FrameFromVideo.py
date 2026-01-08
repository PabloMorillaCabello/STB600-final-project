import cv2

target_time_sec = 6.7 # time in seconds to capture the frame

# path video
# video_path = 'videos/basler_recording_4.avi'
video_path = 'videos/straight_mixed.avi'

cap = cv2.VideoCapture(video_path)

# Get the FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# Calculate which frame corresponds to 7 seconds
frame_index = int(target_time_sec * fps)

# Jump to that frame number
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

# Read the frame
ret, frame = cap.read()

if ret:
    cv2.imwrite(f"frame_at_{target_time_sec}s.png", frame)
    print(f"Saved frame at {target_time_sec} seconds.")
else:
    print("Could not read the frame at that time.")

cap.release()