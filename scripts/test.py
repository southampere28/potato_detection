import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Baca model dan data testing
knn = joblib.load("model/knn_model.pkl")
test_data = pd.read_excel("feature_extraction/testing_features.xlsx")
X_test = test_data.iloc[:, :-1].values
y_test = test_data['label'].values

# Prediksi dan evaluasi
y_pred = knn.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["normal", "defective"])

# Cetak hasil
print(report)
