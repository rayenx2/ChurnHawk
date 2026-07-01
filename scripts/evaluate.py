#!/usr/bin/env python3
"""
Model evaluation script for DVC pipeline
Evaluates trained model on test set
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings  # noqa: E402
from src.feature_engineering import FeatureEngineer  # noqa: E402
from src.utils import dvc_push, ensure_dvc_data, save_json, setup_logging  # noqa: E402

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix - Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(
                j + 0.5,
                i + 0.7,
                f"{cm[i, j]/total*100:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_proba, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    logger.info(f"ROC curve saved to {save_path}")


def evaluate_model(model_path, feature_engineer_path, test_data_path, output_dir):
    """
    Evaluate model on test set

    Args:
        model_path: Path to trained model
        feature_engineer_path: Path to feature engineer
        test_data_path: Path to test data
        output_dir: Directory to save evaluation results
    """
    logger.info("Starting model evaluation on test set...")

    # Ensure all required files are available
    for path in [model_path, feature_engineer_path, test_data_path]:
        if not ensure_dvc_data(path):
            logger.error(f"Failed to get required file: {path}")
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load model and feature engineer
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading feature engineer from {feature_engineer_path}")
    feature_engineer = FeatureEngineer.load(feature_engineer_path)

    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    # Separate features and target
    X_test = test_df.drop(["Churn"], axis=1, errors="ignore")
    if "customerID" in X_test.columns:
        X_test = X_test.drop("customerID", axis=1)

    y_test = test_df["Churn"]
    if y_test.dtype == "object":
        y_test = y_test.map({"Yes": 1, "No": 0})

    logger.info(f"Test set size: {len(X_test)} samples")

    # Transform features
    logger.info("Transforming test features...")
    X_test_transformed = feature_engineer.transform(X_test)

    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "test_samples": len(y_test),
        "churn_rate": float(y_test.mean()),
        "predicted_churn_rate": float(y_pred.mean()),
    }

    logger.info("Test Set Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    # Create output directory
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = output_path / "test_metrics.json"
    save_json(metrics, str(metrics_path))
    logger.info(f"Metrics saved to {metrics_path}")

    # Plot confusion matrix
    cm_path = plots_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path)

    # Plot ROC curve
    roc_path = plots_dir / "test_roc_curve.png"
    plot_roc_curve(y_test, y_pred_proba, roc_path)

    # Push results to DVC
    logger.info("Pushing evaluation results to DVC...")
    dvc_push(str(output_dir))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument(
        "--model", type=str, default="models/model.pkl", help="Path to trained model"
    )
    parser.add_argument(
        "--feature-engineer",
        type=str,
        default="models/feature_engineer.pkl",
        help="Path to feature engineer",
    )
    parser.add_argument(
        "--test-data", type=str, default="data/processed/test.csv", help="Path to test data"
    )
    parser.add_argument(
        "--output", type=str, default="models", help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model,
        feature_engineer_path=args.feature_engineer,
        test_data_path=args.test_data,
        output_dir=args.output,
    )

    print("\nModel evaluation complete!")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
