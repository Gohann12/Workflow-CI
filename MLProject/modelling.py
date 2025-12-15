import os
import argparse
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer


def main(n_clusters, random_state):
    # Konfigurasi MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("clustering-experiment")

    with mlflow.start_run():

        # 1. Load dataset preprocessing
        data = pd.read_csv("bank_transactions_preprocessed.csv")

        print("\n=== DATA PREVIEW ===")
        print(data.head())

        print("\n=== DATA INFO ===")
        print(data.info())

        print("\n=== DATA SUMMARY ===")
        print(data.describe())

        
        # 2. Elbow Method (Analisis) - TANPA AUTOLOG
        
        data_numeric = data.select_dtypes(include=["int64", "float64"])

        # Matikan autolog sementara untuk elbow analysis
        mlflow.sklearn.autolog(disable=True)
        
        elbow = KElbowVisualizer(
            KMeans(random_state=random_state),
            k=(2, 10),
            metric="silhouette",
            timings=False
        )
        elbow.fit(data_numeric)
        
        # Aktifkan autolog lagi untuk model training
        mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

        
        # 3. Training Model
        
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )

        labels = model.fit_predict(data_numeric)

        
        # 4. Evaluasi (Silhouette Score)
        
        sil_score = silhouette_score(data_numeric, labels)
        print(f"\nSilhouette Score: {sil_score}")

        # Manual logging (hanya metric, params sudah di-handle autolog)
        mlflow.log_metric("silhouette_score", sil_score)

        print("\n✅ Training dan evaluasi selesai.")
        print(f"📊 Silhouette Score: {sil_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering Bank Transactions")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    
    args = parser.parse_args()
    
    main(args.n_clusters, args.random_state)
