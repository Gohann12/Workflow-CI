import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

mlflow.set_experiment("clustering-experiment")

with mlflow.start_run(run_name="ci-training"):

    data = pd.read_csv("bank_transactions_preprocessed.csv")
    data_numeric = data.select_dtypes(include=["int64", "float64"])

    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(data_numeric)

    sil_score = silhouette_score(data_numeric, labels)

    mlflow.log_param("n_clusters", 3)
    mlflow.log_metric("silhouette_score", sil_score)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("Training & logging selesai")
