import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from model import DeepCNN  # Import your trained model
from PIL import Image
import time
import streamlit as st

# Streamlit UI
st.title("Live Distraction Detection")
st.write("Detects the most distracted moment based on facial emotions.")

# Load the trained model
model_path = "E:\\Arul\\Edison\\FER-2013\\fer2013_cnn.pth"  # Path to your trained model
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = DeepCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocessing function
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow to avoid OpenCV errors on Windows

# Start time tracking
start_time = time.time()

# Store distraction counts in session state
if "distraction_counts" not in st.session_state:
    st.session_state.distraction_counts = {}

# Streamlit placeholders
frame_placeholder = st.empty()
most_distracted_placeholder = st.empty()

# Add the stop button outside the loop
stop = st.button("Stop", key="stop_button")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Error: Unable to access the webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
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

        # If detected emotion is "Angry," log the timestamp
        if emotion == "Angry":
            elapsed_time = int(time.time() - start_time)
            st.session_state.distraction_counts[elapsed_time] = st.session_state.distraction_counts.get(elapsed_time, 0) + 1

    # Convert frame to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Update the most distracted moment
    if st.session_state.distraction_counts:
        most_distracted_time = max(st.session_state.distraction_counts, key=st.session_state.distraction_counts.get)
        most_distracted_placeholder.write(f"**Most Distracted Moment: {most_distracted_time} seconds.**")

    # Stop the loop if button is pressed
    if stop:
        break

cap.release()
cv2.destroyAllWindows()

# Display final most distracted moment
if st.session_state.distraction_counts:
    most_distracted_time = max(st.session_state.distraction_counts, key=st.session_state.distraction_counts.get)
    st.write(f"Final Most Distracted Moment: {most_distracted_time} seconds.")
else:
    st.write("No distractions detected.")
