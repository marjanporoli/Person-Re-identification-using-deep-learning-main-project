import os
import shutil
from flask import Flask, render_template, request
import numpy as np
from skimage import io
import torch
from torchvision import transforms
from PIL import Image
from model import APN_Model
import base64

app = Flask(__name__)

# Load the trained model and define preprocess transformation
# Assuming APN_Model is your PyTorch model
model = APN_Model()
model.load_state_dict(torch.load('reidentification.pth', map_location=torch.device('cpu')))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
GALLERY_FOLDER = 'static/gallery'  # Gallery image folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GALLERY_FOLDER'] = GALLERY_FOLDER

# Define functions
def euclidean_distance(query, gallery):
    return np.linalg.norm(query - gallery, axis=1)

def get_closest_images(query_feature, gallery_features, gallery_paths, top_k=5):
    distances = euclidean_distance(query_feature, gallery_features)
    closest_indices = np.argsort(distances)[:top_k]
    closest_paths = [gallery_paths[i] for i in closest_indices]
    return closest_paths

def extract_features(image_paths, model):
    features = []
    for path in image_paths:
        img = io.imread(path)
        pil_img = Image.fromarray(img)
        img = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            feature = output.numpy().flatten()
        features.append(feature)
    return np.array(features)

@app.route('/')
def home():
    return render_template('upload.html')

# Update the Flask route to handle both query image and gallery images
@app.route('/predict', methods=['POST'])
def predict():
    # Handle query image
    if 'file' not in request.files:
        return render_template('result.html', prediction_text="Error: No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', prediction_text="Error: No selected file")
    
    # Save query image
    query_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(query_image_path)

    # Handle gallery images
    gallery_image_files = request.files.getlist('gallery[]')
    if not gallery_image_files:
        return render_template('result.html', prediction_text="Error: No gallery images uploaded")

    # Save gallery images
    gallery_image_paths = []
    for gallery_image_file in gallery_image_files:
        if gallery_image_file.filename == '':
            continue
        gallery_image_path = os.path.join(app.config['GALLERY_FOLDER'], gallery_image_file.filename)
        gallery_image_file.save(gallery_image_path)
        gallery_image_paths.append(gallery_image_path)

    # Extract features from the query image
    pil_img = Image.open(query_image_path)
    img = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        query_img_features = output.numpy().flatten()

    # Extract features from the gallery images
    gallery_features = extract_features(gallery_image_paths, model)

    # Find closest images
    closest_image_paths = get_closest_images(query_img_features, gallery_features, gallery_image_paths)

    # Encode closest images as base64 strings
    closest_images_base64 = []
    for img_path in closest_image_paths:
        with open(img_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            closest_images_base64.append(encoded_string)
    
    return render_template('result.html', closest_images_base64=closest_images_base64)

if __name__ == '__main__':
    app.run(port=8000)
