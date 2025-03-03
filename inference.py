import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DeepCNN

# Emotion Labels Mapping
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

model = DeepCNN(num_classes=len(emotion_labels)).to(device)
model.load_state_dict(torch.load("E:\Arul\Edison\FER-2013\\fer2013_cnn.pth"))
model.eval()

# Load and Predict on Image
def predict_emotion(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return emotion_labels[predicted.item()]  # Return the emotion name instead of index

# Test with an Image
image_path = "E:\Arul\Edison\FER-2013\images.jpeg"
emotion = predict_emotion(image_path)
print(f"Predicted Emotion: {emotion}")