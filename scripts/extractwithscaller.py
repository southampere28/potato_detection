import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ... [kode pelatihan Anda sebelumnya] ...

# Load training features from the Excel file
training_data = pd.read_excel("feature_extraction/training_features.xlsx")

# Define features and labels based on your column structure
X_train = training_data[['avg_red', 'avg_green', 'avg_blue', 'contrast', 'homogeneity', 'correlation', 'energy']].values
y_train = training_data['label'].values  # Use numeric labels (0, 1)

# Normalize or standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Ekstrak mean dan std dev dari scaler
means = scaler.mean_
std_devs = scaler.scale_

# Membuat DataFrame untuk parameter scaler
scaler_params = pd.DataFrame({
    'feature': ['avg_red', 'avg_green', 'avg_blue', 'contrast', 'homogeneity', 'correlation', 'energy'],
    'mean': means,
    'std_dev': std_devs
})

# Menyimpan parameter scaler ke Excel
scaler_params.to_excel("model/scaler_parameters.xlsx", index=False)

print("Parameter scaler berhasil disimpan ke 'scaler_parameters.xlsx'")
