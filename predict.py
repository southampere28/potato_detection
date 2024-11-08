import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from rembg import remove
from PIL import Image
import io

# Fungsi untuk ekstraksi fitur warna (histogram)
def extract_color_features(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

# Fungsi untuk ekstraksi fitur tekstur menggunakan GLCM
def extract_texture_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, homogeneity]

# Load model KNN dan scaler yang sudah dilatih
knn_model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Fungsi untuk menghapus latar belakang dari gambar
def remove_background(image_path):
    with open(image_path, 'rb') as input_file:
        input_image = input_file.read()
        output_image = remove(input_image)
        
    # Simpan gambar tanpa latar belakang ke objek PIL
    image_no_bg = Image.open(io.BytesIO(output_image)).convert("RGBA")
    return np.array(image_no_bg)[:, :, :3]  # Mengambil hanya tiga saluran RGB

# Fungsi untuk memprediksi kualitas kentang dari gambar
def predict_potato_quality(image_path):
    # Hapus latar belakang gambar
    image_no_bg = remove_background(image_path)
    
    # Baca gambar yang sudah dihapus latar belakang
    image_gray = cv2.cvtColor(image_no_bg, cv2.COLOR_RGBA2GRAY)
    
    # Ekstraksi fitur warna dan tekstur
    color_features = extract_color_features(image_no_bg)
    texture_features = extract_texture_features(image_gray)
    
    # Gabungkan fitur menjadi satu array
    features = np.hstack((color_features, texture_features)).reshape(1, -1)
    
    # Normalisasi fitur input
    features_scaled = scaler.transform(features)
    
    # Prediksi kualitas kentang
    prediction = knn_model.predict(features_scaled)
    quality = "Normal" if prediction == 0 else "Defective"
    
    return quality

# Uji gambar baru
image_path = "testing_img/Potato.jpg"  # Ganti dengan path gambar uji
result = predict_potato_quality(image_path)
print(f"Kualitas kentang pada gambar uji: {result}")
