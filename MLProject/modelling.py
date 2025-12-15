import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# MLflow Project otomatis membuat run
mlflow.set_experiment("clustering-experiment")

# Load dataset (HARUS ADA DI FOLDER MLProject)
data = pd.read_csv("bank_transactions_preprocessed.csv")
data_numeric = data.select_dtypes(include=["int64", "float64"])

# Training
model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(data_numeric)

# Evaluasi
sil_score = silhouette_score(data_numeric, labels)

# Logging (INI BOLEH)
mlflow.log_param("n_clusters", 3)
mlflow.log_metric("silhouette_score", sil_score)

# Simpan model (INI WAJIB UNTUK NILAI)
mlflow.sklearn.log_model(model, artifact_path="model")

print("MLflow Project training selesai")
