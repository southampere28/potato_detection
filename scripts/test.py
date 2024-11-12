import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load model and scaler
knn = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Load testing data
test_data = pd.read_excel("feature_extraction/testing_features.xlsx")
X_test = test_data[['avg_red', 'avg_green', 'avg_blue', 'contrast', 'homogeneity', 'correlation', 'energy']].values
y_test = test_data['label'].values  # Use numeric labels (0, 1)

# Normalize the test data using the same scaler
X_test = scaler.transform(X_test)

# Predict and evaluate
y_pred = knn.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["normal", "defective"])

# Print the results
print(report)
