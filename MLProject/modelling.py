import os
import pandas as pd
import joblib

import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer


# Konfigurasi MLflow (local)
mlflow.set_experiment("clustering-experiment")

# Aktifkan AUTOLOG
mlflow.sklearn.autolog()

with mlflow.start_run():

    # 1. Load dataset preprocessing
    data = pd.read_csv("bank_transactions_preprocessed.csv")

    print("\n=== DATA PREVIEW ===")
    print(data.head())

    print("\n=== DATA INFO ===")
    print(data.info())

    print("\n=== DATA SUMMARY ===")
    print(data.describe())

    
    # 2. Elbow Method (Analisis)
    
    data_numeric = data.select_dtypes(include=["int64", "float64"])

    elbow = KElbowVisualizer(
        KMeans(random_state=42),
        k=(2, 10),
        metric="silhouette",
        timings=False
    )
    elbow.fit(data_numeric)

    
    # 3. Training Model
    
    model = KMeans(
        n_clusters=3,
        random_state=42
    )

    labels = model.fit_predict(data_numeric)

    
    # 4. Evaluasi (Silhouette Score)
    
    sil_score = silhouette_score(data_numeric, labels)
    print("Silhouette Score:", sil_score)

    # Manual logging (BOLEH walau autolog aktif)
    mlflow.log_metric("silhouette_score", sil_score)

    print("Training dan evaluasi selesai.")

    # 5. Simpan model ke artifacts
    os.makedirs("artifacts", exist_ok=True)
    
    model_path = "artifacts/model_clustering.pkl"
    joblib.dump(model, model_path)
    
    print(f"Model disimpan di {model_path}")
