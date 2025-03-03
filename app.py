import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
from model import DeepCNN  # Import your CNN model class
# Initialize Flask app
app = Flask(__name__)

app = Flask(__name__, template_folder="templates")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels (modify based on your dataset)
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load trained model
num_classes = len(class_labels)
model_path = "E:\Arul\Edison\FER-2013\\fer2013_cnn.pth"

model = DeepCNN(num_classes=num_classes)  # Ensure num_classes is passed
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Home route (optional, to test if Flask is running)
@app.route('/')
def index():
    return render_template("index.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Load image
        image = Image.open(file).convert('L')  # Convert to grayscale
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Predict
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_label = class_labels[predicted_class]
            confidence = probabilities[0][predicted_class].item() * 100

        return jsonify({'emotion': predicted_label, 'confidence': f"{confidence:.2f}%"})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
