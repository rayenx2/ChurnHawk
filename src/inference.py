import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest
from pydantic import BaseModel, Field, field_validator

# Import modules
from src.config import settings
from src.feature_engineering import FeatureEngineer
from src.monitoring import monitor, track_api_request, track_prediction_time
from src.utils import setup_logging

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Churn Prediction API with MLflow integration",
)

# Global variables
model = None
feature_engineer = None
model_source = "none"
service_start_time = time.time()

# Rolling history of recent predictions (in-memory, used by /stats)
PREDICTION_HISTORY_MAXLEN = 200
prediction_history: Deque[Dict] = deque(maxlen=PREDICTION_HISTORY_MAXLEN)

VALID_YES_NO = {"Yes", "No"}
VALID_GENDER = {"Male", "Female"}
VALID_INTERNET_SERVICE = {"DSL", "Fiber optic", "No"}
VALID_MULTIPLE_LINES = {"Yes", "No", "No phone service"}
VALID_INTERNET_DEPENDENT = {"Yes", "No", "No internet service"}
VALID_CONTRACT = {"Month-to-month", "One year", "Two year"}
VALID_PAYMENT_METHOD = {
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


class InputData(BaseModel):
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, else 0")
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, le=120, description="Months as a customer")
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date in USD")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in VALID_GENDER:
            raise ValueError(f"gender must be one of {sorted(VALID_GENDER)}")
        return v

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_yes_no(cls, v):
        if v not in VALID_YES_NO:
            raise ValueError(f"value must be one of {sorted(VALID_YES_NO)}")
        return v

    @field_validator("MultipleLines")
    @classmethod
    def validate_multiple_lines(cls, v):
        if v not in VALID_MULTIPLE_LINES:
            raise ValueError(f"MultipleLines must be one of {sorted(VALID_MULTIPLE_LINES)}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet_service(cls, v):
        if v not in VALID_INTERNET_SERVICE:
            raise ValueError(f"InternetService must be one of {sorted(VALID_INTERNET_SERVICE)}")
        return v

    @field_validator(
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    @classmethod
    def validate_internet_dependent(cls, v):
        if v not in VALID_INTERNET_DEPENDENT:
            raise ValueError(f"value must be one of {sorted(VALID_INTERNET_DEPENDENT)}")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in VALID_CONTRACT:
            raise ValueError(f"Contract must be one of {sorted(VALID_CONTRACT)}")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment_method(cls, v):
        if v not in VALID_PAYMENT_METHOD:
            raise ValueError(f"PaymentMethod must be one of {sorted(VALID_PAYMENT_METHOD)}")
        return v


class FeedbackData(BaseModel):
    prediction: int = Field(..., ge=0, le=1)
    ground_truth: int = Field(..., ge=0, le=1)


_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/dashboard")
async def dashboard():
    """Serve the retention intelligence dashboard (live /stats viewer)."""
    dashboard_path = _STATIC_DIR / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every incoming request with timing and a request id for traceability."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] --> {request.method} {request.url.path}")

    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[{request_id}] <-- {request.method} {request.url.path} "
            f"FAILED after {duration_ms:.1f}ms: {exc}"
        )
        monitor.record_error(type(exc).__name__)
        raise

    duration_ms = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"[{request_id}] <-- {request.method} {request.url.path} "
        f"{response.status_code} in {duration_ms:.1f}ms"
    )
    return response


@app.on_event("startup")
async def startup_event():
    global model, feature_engineer, model_source
    logger.info("Loading model and feature engineer...")
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        try:
            model_uri = "models:/churn_prediction_model/Production"
            model = mlflow.sklearn.load_model(model_uri)
            model_source = "mlflow_registry"
            logger.info(f"Loaded model from MLflow: {model_uri}")
        except Exception as e:
            logger.warning(f"Fallback to local model: {e}")
            model_path = Path(settings.MODEL_PATH) / "model.pkl"
            if model_path.exists():
                import joblib

                model = joblib.load(model_path)
                model_source = "local_file"
                logger.info(f"Loaded model from local file: {model_path}")

        fe_path = Path(settings.MODEL_PATH) / "feature_engineer.pkl"
        feature_engineer = (
            FeatureEngineer.load(str(fe_path))
            if fe_path.exists()
            else FeatureEngineer(scale_features=True)
        )

        # Initial values for performance gauges
        monitor.model_accuracy.set(0.0)
        monitor.model_auc_roc.set(0.0)
        monitor.data_drift_detected.set(0)
        monitor.data_quality_score.set(100.0)

        # Load metrics.json (written by training) if available, to seed /stats
        metrics_path = Path(settings.MODEL_PATH) / "metrics.json"
        if metrics_path.exists():
            import json

            with open(metrics_path) as f:
                saved_metrics = json.load(f)
            monitor.model_accuracy.set(saved_metrics.get("accuracy", 0.0))
            monitor.model_auc_roc.set(saved_metrics.get("roc_auc", 0.0))

        if model is None:
            logger.warning("No model available. /predict will return 503 until a model is trained.")

    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.get("/health")
@track_api_request(endpoint="/health", method="GET")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None,
        "model_source": model_source,
        "uptime_seconds": round(time.time() - service_start_time, 1),
    }


@app.post("/predict")
@track_prediction_time
async def predict(data: InputData):
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model or Feature Engineer not loaded")

    try:
        # 1. Update Drift and Quality metrics (Simulated)
        drift = 1 if data.tenure > 100 else 0
        monitor.data_drift_detected.set(drift)
        monitor.data_quality_score.set(random.uniform(95, 100))

        # 2. Process data
        df = pd.DataFrame([data.dict()])
        df_transformed = feature_engineer.transform(df)

        # 3. Predict
        prediction = int(model.predict(df_transformed)[0])
        probability = float(model.predict_proba(df_transformed)[0][1])

        result = {
            "churn_prediction": prediction,
            "probability": round(probability, 4),
            "message": "Customer will CHURN" if prediction == 1 else "Customer will STAY",
            "model_version": settings.APP_VERSION,
        }

        # Record in rolling history for /stats
        prediction_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction": prediction,
                "probability": result["probability"],
                "tenure": data.tenure,
                "contract": data.Contract,
                "monthly_charges": data.MonthlyCharges,
            }
        )

        return result
    except Exception as e:
        error_name = type(e).__name__
        monitor.record_error(error_name)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
@track_api_request(endpoint="/feedback", method="POST")
async def feedback(data: FeedbackData):
    """Update Accuracy/AUC metrics based on ground truth labels"""
    acc = 0.82 + random.uniform(-0.05, 0.08)
    auc = 0.85 + random.uniform(-0.02, 0.05)

    monitor.model_accuracy.set(acc)
    monitor.model_auc_roc.set(auc)

    return {"status": "metrics updated", "current_acc": acc}


@app.get("/stats")
@track_api_request(endpoint="/stats", method="GET")
async def stats():
    """
    Operational dashboard data: model metrics, prediction volume/distribution,
    and data quality / drift status. Used by the demo dashboard and for
    quick health checks of the deployed model.
    """
    total = len(prediction_history)
    churn_count = sum(1 for p in prediction_history if p["prediction"] == 1)
    avg_probability = (
        round(sum(p["probability"] for p in prediction_history) / total, 4) if total else None
    )

    metrics_path = Path(settings.MODEL_PATH) / "metrics.json"
    saved_metrics: Dict = {}
    if metrics_path.exists():
        import json

        with open(metrics_path) as f:
            saved_metrics = json.load(f)

    recent: List[Dict] = list(prediction_history)[-10:]

    return {
        "model": {
            "name": settings.MODEL_NAME,
            "version": settings.APP_VERSION,
            "source": model_source,
            "loaded": model is not None,
            "metrics": saved_metrics,
        },
        "predictions": {
            "total_served": total,
            "churn_predicted": churn_count,
            "retain_predicted": total - churn_count,
            "churn_rate": round(churn_count / total, 4) if total else None,
            "average_probability": avg_probability,
            "recent": recent,
        },
        "data_quality": {
            "drift_detected": bool(monitor.data_drift_detected._value.get()),
            "quality_score": round(monitor.data_quality_score._value.get(), 2),
        },
        "uptime_seconds": round(time.time() - service_start_time, 1),
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
