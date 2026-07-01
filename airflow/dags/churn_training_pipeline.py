import os
import sys
from datetime import datetime, timedelta

from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG  # type: ignore

# Add src to path
sys.path.insert(0, os.path.abspath("/opt/airflow"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from src.feature_engineering import FeatureEngineer  # noqa: E402
from src.validate import run_validation  # noqa: E402

# Default arguments
default_args = {
    "owner": "rayen",
    "depends_on_past": False,
    "email": [""],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def validate_data_task(**context):
    """Task 1: Validate input data quality"""
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Starting data validation...")

    raw_data_path = "data/raw/churn.csv"
    validation_results = run_validation(raw_data_path)

    context["ti"].xcom_push(key="validation_results", value=validation_results)

    if not validation_results["is_valid"]:
        raise ValueError(f"Data validation failed: {validation_results['errors']}")

    logger.info("Data validation passed successfully")
    return validation_results


def check_data_drift_task(**context):
    """Task 2: Check for data drift"""
    import logging

    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    logger = logging.getLogger(__name__)
    logger.info("Checking for data drift...")

    # Load current and reference data
    current_data = pd.read_csv("/opt/airflow/data/raw/churn.csv")
    reference_data = pd.read_csv("/opt/airflow/data/raw/churn_reference.csv")

    # Create drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Save report
    report.save_html("/opt/airflow/data/reports/data_drift_report.html")

    # Check drift
    drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]

    context["ti"].xcom_push(key="drift_detected", value=drift_detected)

    if drift_detected:
        logger.warning("Data drift detected! Review required.")
    else:
        logger.info("No significant data drift detected")

    return drift_detected


def feature_engineering_task(**context):
    """Task 3: Feature engineering"""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting feature engineering...")

    # Load data
    df = pd.read_csv("/opt/airflow/data/raw/churn.csv")

    # Separate features and target
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Initialize and fit feature engineer
    fe = FeatureEngineer(scale_features=True)
    X_transformed = fe.fit_transform(X)

    # Save feature engineer
    fe.save("/opt/airflow/models/feature_engineer.pkl")

    # Save processed data
    X_transformed.to_csv("/opt/airflow/data/processed/X_train.csv", index=False)
    y.to_csv("/opt/airflow/data/processed/y_train.csv", index=False)

    logger.info(f"Feature engineering complete. Shape: {X_transformed.shape}")

    return {"n_features": X_transformed.shape[1], "n_samples": X_transformed.shape[0]}


def train_model_task(**context):
    """Task 4: Train XGBoost model"""
    import logging

    import joblib
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")

    # Load processed data
    X = pd.read_csv("/opt/airflow/data/processed/X_train.csv")
    y = pd.read_csv("/opt/airflow/data/processed/y_train.csv").values.ravel()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model parameters
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
    }

    # Train model
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Save model
    model_path = "/opt/airflow/models/xgboost_model.pkl"
    joblib.dump(model, model_path)

    logger.info("Model training complete")

    # Store model path in XCom
    context["ti"].xcom_push(key="model_path", value=model_path)

    return model_path


def evaluate_model_task(**context):
    """Task 5: Evaluate model performance"""
    import logging

    import joblib
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")

    # Load model
    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")
    model = joblib.load(model_path)

    # Load validation data
    X = pd.read_csv("/opt/airflow/data/processed/X_train.csv")
    y = pd.read_csv("/opt/airflow/data/processed/y_train.csv").values.ravel()

    from sklearn.model_selection import train_test_split

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1_score": float(f1_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_pred_proba)),
    }

    logger.info(f"Model Metrics: {metrics}")

    # Store metrics in XCom
    context["ti"].xcom_push(key="metrics", value=metrics)

    return metrics


def log_to_mlflow_task(**context):
    """Task 6: Log experiment to MLflow"""
    import logging

    import joblib

    logger = logging.getLogger(__name__)
    logger.info("Logging to MLflow...")

    # Get metrics from previous task
    metrics = context["ti"].xcom_pull(key="metrics", task_ids="evaluate_model")
    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")

    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("churn_production_s3")
    # Load model
    model = joblib.load(model_path)

    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Log parameters
        mlflow.log_params(
            {
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
                "n_estimators": model.n_estimators,
            }
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log feature engineer
        mlflow.log_artifact("/opt/airflow/models/feature_engineer.pkl")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        context["ti"].xcom_push(key="mlflow_run_id", value=run_id)

    return run_id


def decide_deployment_task(**context):
    """Task 7: Decide if model should be deployed"""
    import logging

    logger = logging.getLogger(__name__)

    # Get current metrics
    metrics = context["ti"].xcom_pull(key="metrics", task_ids="evaluate_model")
    current_accuracy = metrics["accuracy"]
    current_auc = metrics["roc_auc"]

    # Define thresholds
    MIN_ACCURACY = 0.75
    MIN_AUC = 0.80

    logger.info(f"Current model - Accuracy: {current_accuracy:.4f}, AUC: {current_auc:.4f}")
    logger.info(f"Thresholds - Accuracy: {MIN_ACCURACY}, AUC: {MIN_AUC}")

    # Decide based on thresholds
    if current_accuracy >= MIN_ACCURACY and current_auc >= MIN_AUC:
        logger.info("✅ Model meets deployment criteria")
        return "register_model"
    else:
        logger.warning("❌ Model does NOT meet deployment criteria")
        return "skip_deployment"


def register_model_task(**context):
    """Task 8: Register model in MLflow Model Registry"""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Registering model to Model Registry...")

    # Get MLflow run ID
    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="log_to_mlflow")

    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Register model
    model_uri = f"runs:/{run_id}/model"
    model_name = "churn_prediction_model"

    try:
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered: {model_name} version {model_version.version}")

        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Production"
        )

        logger.info(f"Model version {model_version.version} transitioned to Production")

        return model_version.version

    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise


def skip_deployment_task(**context):
    """Task 9: Skip deployment"""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("⏭️ Skipping model deployment - criteria not met")


def notify_success_task(**context):
    """Task 10: Send success notification"""
    import logging

    logger = logging.getLogger(__name__)

    metrics = context["ti"].xcom_pull(key="metrics", task_ids="evaluate_model")

    message = f"""
    ✅ Churn Prediction Training Pipeline Completed Successfully!

    Metrics:
    - Accuracy: {metrics['accuracy']:.4f}
    - ROC-AUC: {metrics['roc_auc']:.4f}
    - Precision: {metrics['precision']:.4f}
    - Recall: {metrics['recall']:.4f}

    Execution Date: {context['ds']}
    """

    logger.info(message)
    # Here you could send email, Slack notification, etc.


# Define DAG
with DAG(
    dag_id="churn_training_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for churn prediction",
    schedule_interval="@weekly",  # Run every Sunday
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "churn", "training", "production"],
    max_active_runs=1,
) as dag:

    # Task 1: Validate Data
    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data_task,
        provide_context=True,
    )

    # Task 2: Check Data Drift
    check_drift = PythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift_task,
        provide_context=True,
    )

    # Task 3: Feature Engineering
    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_task,
        provide_context=True,
    )

    # Task 4: Train Model
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
        provide_context=True,
    )

    # Task 5: Evaluate Model
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_task,
        provide_context=True,
    )

    # Task 6: Log to MLflow
    log_to_mlflow = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow_task,
        provide_context=True,
    )

    # Task 7: Decide Deployment
    decide_deployment = BranchPythonOperator(
        task_id="decide_deployment",
        python_callable=decide_deployment_task,
        provide_context=True,
    )

    # Task 8: Register Model (if approved)
    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task,
        provide_context=True,
    )

    # Task 9: Skip Deployment
    skip_deployment = PythonOperator(
        task_id="skip_deployment",
        python_callable=skip_deployment_task,
        provide_context=True,
    )

    # Task 10: Notify Success
    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success_task,
        provide_context=True,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # Define task dependencies
    validate_data >> check_drift >> feature_engineering >> train_model
    train_model >> evaluate_model >> log_to_mlflow >> decide_deployment
    decide_deployment >> [register_model, skip_deployment]
    [register_model, skip_deployment] >> notify_success
