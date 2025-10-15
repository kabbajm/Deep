# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import numpy as np


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.set_tag("version", "1.0")
    mlflow.set_tag("projet", "iris")

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
x = [[8., 7., 2., 1.]]
ypred = model.predict(x)

print(model.score(X, y))
mlflow.log_metric("accuracy", model.score(X, y))

joblib.dump(model, "artifacts/model.joblib")

mlflow.log_artifact("artifacts/model.joblib")
print("✅ Modèle sauvegardé !")
