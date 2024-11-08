import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Baca fitur training dari file Excel
data = pd.read_excel("feature_extraction/training_features.xlsx")
X = data.iloc[:, :-1].values  # Fitur
y = data['label'].values       # Label

# Normalisasi atau standarisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Bagi data untuk training dan validasi
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi model pada data validasi
y_pred = knn.predict(X_val)
report = classification_report(y_val, y_pred, target_names=["Normal", "Defective"])

# Cross-Validation (opsional, untuk evaluasi tambahan)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

# Simpan hasil evaluasi dan model
with open("model/classification_report.txt", "w") as f:
    f.write(report)
    f.write("\nCross-validation scores: " + str(cv_scores))
    f.write("\nAverage CV score: " + str(cv_scores.mean()))

# Simpan model dan scaler
joblib.dump(knn, "model/knn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
