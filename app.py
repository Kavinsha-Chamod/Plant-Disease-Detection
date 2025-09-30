import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import os
from flask import Flask, request, jsonify, render_template
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

device = torch.device("cpu")
logger.info(f"Using device: {device}")

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
    'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]
num_classes = len(class_names)
logger.info(f"Loaded {num_classes} classes: {class_names[:5]}...")

class_label_map = {
    'Apple___healthy': 'Healthy Apple',
    'Apple___Apple_scab': 'Apple Scab',
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple___Cedar_apple_rust': 'Apple Cedar Rust',
    'Tomato___healthy': 'Healthy Tomato',
    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Two-Spotted Spider Mites',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    # Add other mappings as needed
}

# Load model
model = models.resnet18(weights=None).to(device)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Try multiple model paths for different deployment environments
model_paths = [
    'resnet18.pth',  # Local development
    '/app/resnet18.pth',  # Render deployment
    './resnet18.pth'  # Alternative path
]

model_loaded = False
for model_path in model_paths:
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            logger.info(f"Model loaded successfully from {model_path}!")
            model_loaded = True
            break
    except Exception as e:
        logger.warning(f"Failed to load model from {model_path}: {e}")
        continue

if not model_loaded:
    logger.error("Could not load model from any path. Available files: " + str(os.listdir('.')))
    raise FileNotFoundError("Model file not found in any expected location")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    upload_dir = 'static/uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    image_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(image_path)
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            raw_prediction = class_names[predicted.item()]
            confidence_score = confidence.item()

            friendly_prediction = class_label_map.get(raw_prediction, raw_prediction.replace('_', ' ').title())
            message = f"The {friendly_prediction.lower()} condition is detected with {confidence_score:.1%} confidence."
        
        os.remove(image_path)
        
        return jsonify({'message': message, 'confidence': f"{confidence_score:.4f}"})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)