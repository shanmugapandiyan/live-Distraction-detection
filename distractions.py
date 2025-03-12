import cv2
import torch
import numpy as np
from inference import predict_emotion  # Import your emotion detection function

# Load Video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
frame_count = 0
distraction_log = []  # Store timestamps of distraction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    frame_count += 1
    current_time = frame_count / fps  # Convert frame number to seconds

    # Convert frame to grayscale (if needed) and predict emotion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotion = predict_emotion(gray)  # Your emotion detection function

    # List of emotions that indicate distraction
    distracted_emotions = ["bored", "disgust", "angry"]

    if emotion in distracted_emotions:
        distraction_log.append(current_time)

    # Display the frame
    cv2.imshow("Video Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Convert seconds to minutes and display timestamps
print("\nDistraction Detected at:")
for time_sec in distraction_log:
    minutes = int(time_sec // 60)
    seconds = int(time_sec % 60)
    print(f"⏳ {minutes} min {seconds} sec")

