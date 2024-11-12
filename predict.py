import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from rembg import remove
from PIL import Image
import io

# Function to extract color features (average RGB values)
def extract_color_features(image):
    avg_rgb = np.mean(image, axis=(0, 1))  # Get average R, G, B values
    return avg_rgb  # Returns a 3-element array

# Function to extract texture features using GLCM
def extract_texture_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return [contrast, homogeneity, correlation, energy]  # 4-element array

# Load the trained KNN model and scaler
knn_model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Function to remove background from an image
def remove_background(image_path):
    with open(image_path, 'rb') as input_file:
        input_image = input_file.read()
        output_image = remove(input_image)
    
    # Load the image without background into PIL and convert to RGB
    image_no_bg = Image.open(io.BytesIO(output_image)).convert("RGB")
    return np.array(image_no_bg)

# Function to predict the quality of the potato from an image
def predict_potato_quality(image_path):
    try:
        # Remove background from the image
        image_no_bg = remove_background(image_path)

        # Convert to grayscale for texture features
        image_gray = cv2.cvtColor(image_no_bg, cv2.COLOR_RGB2GRAY)

        # Extract color and texture features
        color_features = extract_color_features(image_no_bg)
        texture_features = extract_texture_features(image_gray)

        # Combine features into a single array with 7 elements
        features = np.hstack((color_features, texture_features)).reshape(1, -1)

        # Normalize input features
        features_scaled = scaler.transform(features)

        # Predict potato quality
        prediction = knn_model.predict(features_scaled)
        quality = "Normal" if prediction[0] == 0 else "Defective"

        return quality

    except Exception as e:
        return f"Error processing image: {e}"

# Test the code with a new image
image_path = "testing_img/normal.jpg"  # Replace with your test image path
result = predict_potato_quality(image_path)
print(f"Kualitas kentang pada gambar uji: {result}")
