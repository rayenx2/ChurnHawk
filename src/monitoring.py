import logging
import time
from functools import wraps
from typing import Any, Callable

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info

# Setup logger
logger = logging.getLogger(__name__)


def get_metric(metric_type, name, description, **kwargs):
    """Safe metric initialization to prevent ValueError in multi-worker environments."""
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return metric_type(name, description, **kwargs)


class ModelMonitor:
    def __init__(self):
        # --- 1. Metrics for Inference (API) ---
        self.prediction_latency = get_metric(
            Histogram, "churn_prediction_latency_seconds", "Time spent processing prediction"
        )

        self.prediction_counter = get_metric(
            Counter,
            "churn_predictions_total",
            "Total number of churn predictions made",
            labelnames=["prediction", "model_version"],
        )

        self.error_counter = get_metric(
            Counter,
            "churn_prediction_errors_total",
            "Total number of prediction errors",
            labelnames=["error_type"],
        )

        self.api_requests_total = get_metric(
            Counter,
            "churn_api_requests_total",
            "Total API requests",
            labelnames=["endpoint", "method", "status"],
        )

        self.api_request_duration = get_metric(
            Histogram,
            "churn_api_request_duration_seconds",
            "API request duration",
            labelnames=["endpoint", "method"],
        )

        # --- 2. Metrics for Validation (validate.py) ---
        self.data_quality_score = get_metric(
            Gauge, "churn_data_quality_score", "Data Quality Score (0-100)"
        )

        self.data_drift_detected = get_metric(
            Gauge, "churn_data_drift_detected", "Data Drift Detected (1=Yes, 0=No)"
        )

        # --- 3. Metrics for Training (train.py) ---
        self.model_accuracy = get_metric(Gauge, "churn_model_accuracy", "Current model accuracy")

        self.model_auc_roc = get_metric(Gauge, "churn_model_auc_roc", "Current model AUC-ROC score")

        self.model_info = get_metric(
            Info, "churn_model_info", "Information about the current model"
        )

    def record_prediction(self, prediction: Any, model_version: str = "unknown"):
        self.prediction_counter.labels(
            prediction=str(prediction), model_version=str(model_version)
        ).inc()

    def record_error(self, error_type: str):
        self.error_counter.labels(error_type=error_type).inc()


# Global Instance
monitor = ModelMonitor()

# --- Decorators ---


def track_prediction_time(func: Callable) -> Callable:
    """Decorator to track prediction latency and record prediction values automatically"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            monitor.prediction_latency.observe(duration)

            if isinstance(result, dict) and "churn_prediction" in result:
                monitor.record_prediction(
                    prediction=result["churn_prediction"],
                    model_version=result.get("model_version", "unknown"),
                )
            return result
        except Exception as e:
            monitor.record_error(type(e).__name__)
            raise e

    return wrapper


def track_api_request(endpoint: str, method: str):
    """Decorator for tracking generic API endpoint metrics"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                monitor.api_request_duration.labels(endpoint=endpoint, method=method).observe(
                    duration
                )
                monitor.api_requests_total.labels(
                    endpoint=endpoint, method=method, status=status
                ).inc()

        return wrapper

    return decorator
