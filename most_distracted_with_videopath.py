import cv2
import torch
import numpy as np
import torch.nn.functional as F
import collections
from torchvision import transforms
from model import DeepCNN  # Import your trained model
from PIL import Image
from collections import defaultdict

# Load the trained model
model_path = "E:\\Arul\\Edison\\FER-2013\\fer2013_cnn.pth"  # Adjust path
num_classes = 7  # Based on your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DeepCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing function
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((48, 48)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

# Load the recorded video
video_path = "video.mp4"  # Replace with your actual video file
cap = cv2.VideoCapture(video_path)

# Dictionary to store number of distractions per timestamp
distraction_counts = defaultdict(int)

# Function to check if a person is distracted
def is_distracted(emotion):
    """Returns True if the emotion is considered distracted."""
    return emotion in ["Neutral", "Sad", "Fear", "Disgust"]  # Define distraction emotions

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    distraction_count = 0  # Count of distracted people in this frame

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)

        # Preprocess the face
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            emotion = emotion_labels[predicted_class]

        # Check if this person is distracted
        if is_distracted(emotion):
            distraction_count += 1

    # Get current timestamp (in seconds)
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

    # Store the count of distracted people for this timestamp
    if distraction_count > 0:
        distraction_counts[timestamp] += distraction_count  # Aggregate distraction count

    # Show video frame with detections
    cv2.imshow("Distraction Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Find the timestamp with the highest distractions
most_distracted_time = max(distraction_counts, key=distraction_counts.get, default=None)
max_distraction_count = distraction_counts[most_distracted_time] if most_distracted_time else 0

# Save results to a file
with open("most_distracted_time.txt", "w") as f:
    if most_distracted_time:
        f.write(f"Most distracted moment: {most_distracted_time} seconds, {max_distraction_count} people distracted.\n")
    else:
        f.write("No distractions detected.")

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print results
if most_distracted_time:
    print(f"Most distracted moment: {most_distracted_time} seconds, {max_distraction_count} people distracted.")
else:
    print("No distractions detected.")
