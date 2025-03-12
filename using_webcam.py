import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from model import DeepCNN  # Import your trained model
from PIL import Image

# Load the trained model
model_path = "E:\Arul\Edison\FER-2013\\fer2013_cnn.pth"  # Path to your saved model
num_classes = 7  # Adjust this based on your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = DeepCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Define emotion labels (Modify based on your dataset)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize the face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing function
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
    transforms.Resize((48, 48)),  # Resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust if needed)
])

# Start capturing video
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extract the face
        face_pil = Image.fromarray(face)  # Convert to PIL image

        # Preprocess the face
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            emotion = emotion_labels[predicted_class]

        # Draw a rectangle around the face and display the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
