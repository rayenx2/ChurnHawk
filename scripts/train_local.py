#!/usr/bin/env python3
"""
Standalone local training script (no MLflow / DVC required).
Trains an XGBoost churn classifier on data/raw/churn_reference.csv and
saves model.pkl + feature_engineer.pkl + metrics.json to models/.

Used to bootstrap model artifacts so the FastAPI inference service can
load a real model without requiring the full Airflow/MLflow/MinIO stack.
"""
import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.feature_engineering import FeatureEngineer  # noqa: E402


def main():
    data_path = ROOT / "data" / "raw" / "churn_reference.csv"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fe = FeatureEngineer(scale_features=True)
    X_train_t = fe.fit_transform(X_train)
    X_test_t = fe.transform(X_test)

    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train_t, y_train, verbose=False)

    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    joblib.dump(model, models_dir / "model.pkl")
    fe.save(str(models_dir / "feature_engineer.pkl"))
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
