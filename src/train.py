import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Import modules
from src.config import settings
from src.feature_engineering import FeatureEngineer
from src.monitoring import monitor
from src.utils import dvc_push, ensure_dvc_data, load_params, save_json, setup_logging

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Plot and save feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} Feature Importances")
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Feature importance saved to {save_path}")


def train_model(X_train, y_train, params=None):
    """
    Train XGBoost model with given parameters

    Args:
        X_train: Training features
        y_train: Training labels
        params: Model hyperparameters (optional)

    Returns:
        Trained model
    """
    logger.info("Starting model training...")

    if params is None:
        # Load from params.yaml via DVC
        dvc_params = load_params()
        params = dvc_params.get("train", {})

        # Fallback to config if not in params.yaml
        params = {
            "max_depth": params.get("max_depth", settings.XGBOOST_MAX_DEPTH),
            "learning_rate": params.get("learning_rate", settings.XGBOOST_LEARNING_RATE),
            "n_estimators": params.get("n_estimators", settings.XGBOOST_N_ESTIMATORS),
            "min_child_weight": params.get("min_child_weight", settings.XGBOOST_MIN_CHILD_WEIGHT),
            "subsample": params.get("subsample", settings.XGBOOST_SUBSAMPLE),
            "colsample_bytree": params.get("colsample_bytree", settings.XGBOOST_COLSAMPLE_BYTREE),
            "objective": params.get("objective", "binary:logistic"),
            "eval_metric": params.get("eval_metric", "auc"),
            "random_state": params.get("random_state", settings.RANDOM_STATE),
        }

    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    logger.info("Model training complete")
    return model, params


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return metrics

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }

    logger.info(f"Model Metrics: {metrics}")

    # Update monitoring metrics
    monitor.model_accuracy.set(metrics["accuracy"])
    monitor.model_auc_roc.set(metrics["roc_auc"])

    return metrics, y_pred, y_pred_proba


def save_model(model, path):
    """
    Save model to disk

    Args:
        model: Model to save
        path: Path to save model
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def train_and_log_model(data_path=None):
    """
    Complete training pipeline with MLflow logging and DVC integration

    Args:
        data_path: Path to training data (optional)

    Returns:
        Dictionary with model and metrics
    """
    logger.info("Starting complete training pipeline with DVC...")

    # Ensure training data is available
    if data_path is None:
        data_path = Path(settings.DATA_PROCESSED_PATH) / "train.csv"

    if not ensure_dvc_data(str(data_path)):
        logger.error(f"Failed to get training data: {data_path}")
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Setup MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(["Churn"], axis=1, errors="ignore")
    if "customerID" in X.columns:
        X = X.drop("customerID", axis=1)

    y = df["Churn"] if df["Churn"].dtype == "int64" else df["Churn"].map({"Yes": 1, "No": 0})

    # Feature engineering
    logger.info("Applying feature engineering...")
    dvc_params = load_params()
    fe_params = dvc_params.get("feature_engineering", {})

    feature_engineer = FeatureEngineer(scale_features=fe_params.get("scale_features", True))
    X_transformed = feature_engineer.fit_transform(X)

    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):

        # Train model
        model, params = train_model(X_transformed, y)

        # Evaluate model (using same data for training metrics)
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_transformed, y)

        # Log parameters to MLflow
        mlflow.log_params(params)
        mlflow.log_params(fe_params)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Create plots directory
        plots_dir = Path(settings.MODEL_PATH) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot confusion matrix
        cm_path = plots_dir / "confusion_matrix.png"
        plot_confusion_matrix(y, y_pred, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Plot feature importance
        fi_path = plots_dir / "feature_importance.png"
        plot_feature_importance(model, X_transformed.columns.tolist(), fi_path)
        mlflow.log_artifact(str(fi_path))

        # Save model
        model_path = Path(settings.MODEL_PATH) / "model.pkl"
        save_model(model, model_path)
        mlflow.log_artifact(str(model_path))

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save feature engineer
        fe_path = Path(settings.MODEL_PATH) / "feature_engineer.pkl"
        feature_engineer.save(str(fe_path))
        mlflow.log_artifact(str(fe_path))

        # Save metrics to file (for DVC)
        metrics_path = Path(settings.MODEL_PATH) / "metrics.json"
        save_json(metrics, str(metrics_path))

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # Push model artifacts to DVC
        logger.info("Pushing model artifacts to DVC...")
        dvc_push(str(settings.MODEL_PATH))

        return {
            "model": model,
            "feature_engineer": feature_engineer,
            "metrics": metrics,
            "run_id": run_id,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to training data CSV file"
    )
    args = parser.parse_args()

    # Run training pipeline
    result = train_and_log_model(data_path=args.data_path)
    print(f"Training complete! Run ID: {result['run_id']}")
    print(f"Metrics: {result['metrics']}")
