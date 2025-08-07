from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
from PIL import Image
import cv2
import io
import os

app = Flask(__name__)

# Load the trained model
model_path = r"models/image_classifier.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

def get_features(img):
    """Extract RGB mean values from image (same as training)"""
    return list(cv2.mean(img)[:-1])

def preprocess_image(image):
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Extract features using the same method as training
    features = get_features(img_cv)
    return np.array(features).reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            try:
                image = Image.open(file.stream)
                data = preprocess_image(image)
                prediction = model.predict(data)[0]
                label = "Lemon" if prediction == 0 else "Melon"
                return render_template("result.html", label=label)
            except Exception as e:
                return f"Error processing image: {str(e)}"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000) 