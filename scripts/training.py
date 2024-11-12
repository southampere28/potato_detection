import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

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

# Save the model and scaler
joblib.dump(knn, "model/knn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
