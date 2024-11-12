import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Function to resize the image to 640x640 pixels
def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# Function for color feature extraction (average RGB and histogram)
def extract_color_features(image):
    # Calculate histogram and average RGB values
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    avg_rgb = np.mean(image, axis=(0, 1))  # Get average R, G, B
    return avg_rgb, hist.flatten()

# Function for texture feature extraction using GLCM
def extract_texture_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return [contrast, homogeneity, correlation, energy]

# Initialize data
data = []
labels = []
label_names = []
image_names = []

# Loop to read images from the dataset folder
dataset_dir = "dataset/testing"  # Change to your dataset path
classes = ["normal", "defective"]

for label, category in enumerate(classes):
    folder_path = os.path.join(dataset_dir, category)
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Resize the image to 640x640 pixels
        image_resized = resize_image(image)

        # Convert to grayscale for texture features
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        # Extract color and texture features
        avg_rgb, color_features = extract_color_features(image_resized)
        texture_features = extract_texture_features(image_gray)

        # Prepare data row with average RGB, texture features, label, and image name
        features = np.hstack((avg_rgb, texture_features))
        image_names.append(filename)  # Add the image filename
        data.append(features)
        label_names.append(category)
        labels.append(label)  # 0 = normal, 1 = defective

# Convert to DataFrame with specific columns
df = pd.DataFrame(data, columns=['avg_red', 'avg_green', 'avg_blue', 'contrast', 'homogeneity', 'correlation', 'energy'])
df['label'] = labels  # Add the image name column
df['image_name'] = image_names  # Add the image name column
df['label_name'] = label_names

# Save features and labels to Excel
output_dir = "feature_extraction"
os.makedirs(output_dir, exist_ok=True)
df.to_excel(os.path.join(output_dir, "testing_features.xlsx"), index=False)
