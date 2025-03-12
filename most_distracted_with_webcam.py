import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from model import DeepCNN  # Import your trained model
from PIL import Image
import time

# Load the trained model
model_path = "E:\\Arul\\Edison\\FER-2013\\fer2013_cnn.pth"  # Path to your saved model
num_classes = 7  # Adjust this based on your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = DeepCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing function
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Start capturing video
cap = cv2.VideoCapture(0)
start_time = time.time()
distraction_counts = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            emotion = emotion_labels[predicted_class]

        if emotion == "Angry":
            elapsed_time = int(time.time() - start_time)
            distraction_counts[elapsed_time] = distraction_counts.get(elapsed_time, 0) + 1
            print(f"Detected distraction at {elapsed_time} seconds")  # Debugging print

    cv2.imshow("Live Distraction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Check if any distractions were recorded
if distraction_counts:
    most_distracted_time = max(distraction_counts, key=distraction_counts.get)
    print(f"Most distracted moment: {most_distracted_time} seconds.")
else:
    print("No distractions detected.")
