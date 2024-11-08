import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Function to resize the image to 640x640 pixels
def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# Function for color feature extraction (histogram)
def extract_color_features(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# Function for texture feature extraction using GLCM
def extract_texture_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, homogeneity]

# Initialize data
data = []
labels = []

# Loop to read images from the dataset folder
dataset_dir = "dataset/training"  # Change to your dataset path
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
        color_features = extract_color_features(image_resized)
        texture_features = extract_texture_features(image_gray)

        # Combine features
        features = np.hstack((color_features, texture_features))
        data.append(features)
        labels.append(label)  # 0 = normal, 1 = defective

# Save features and labels to Excel
df = pd.DataFrame(data)
df['label'] = labels
df.to_excel("feature_extraction/training_features.xlsx", index=False)
