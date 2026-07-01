from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG  # type: ignore


def check_model_performance(**context):
    """Check if model performance has degraded"""
    import logging

    import mlflow

    logger = logging.getLogger(__name__)
    logger.info("Checking model performance...")

    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Get production model
    client = mlflow.tracking.MlflowClient()
    model_name = "churn_prediction_model"

    try:
        # Get latest production version
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.warning("No production model found")
            return

        latest_version = versions[0]
        logger.info(f"Monitoring model version: {latest_version.version}")

        # TODO: Implement performance monitoring logic
        # - Compare current metrics with baseline
        # - Check prediction distribution
        # - Alert if degradation detected

    except Exception as e:
        logger.error(f"Error checking model performance: {str(e)}")


def monitor_data_quality(**context):
    """Monitor incoming data quality"""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Monitoring data quality...")

    # TODO: Implement data quality monitoring
    # - Check for missing values
    # - Validate data schema
    # - Check for anomalies


default_args = {
    "owner": "rayen",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="churn_monitoring",
    default_args=default_args,
    description="Monitor model performance and data quality",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["monitoring", "ml", "churn"],
) as dag:

    check_performance = PythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
        provide_context=True,
    )

    monitor_quality = PythonOperator(
        task_id="monitor_data_quality",
        python_callable=monitor_data_quality,
        provide_context=True,
    )

    check_performance >> monitor_quality
