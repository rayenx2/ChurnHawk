from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation"""

    # Application
    APP_NAME: str = "Churn Prediction Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # DVC Configuration
    DVC_REMOTE: str = "minio"
    DVC_CACHE_DIR: str = ".dvc/cache"

    # MLflow Configuration
    # BaseSettings automatically maps env vars matching field names
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "churn_prediction"
    MLFLOW_ARTIFACT_LOCATION: Optional[str] = None

    # MinIO / S3 Configuration
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "dvc-storage"
    MINIO_SECURE: bool = False

    # DVC S3 Configuration
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"

    # Database Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "airflow"
    POSTGRES_PASSWORD: str = "airflow"
    POSTGRES_DB: str = "airflow"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Model Configuration
    MODEL_NAME: str = "xgboost_churn_classifier"
    MODEL_STAGE: str = "Production"
    MODEL_PATH: str = "models/"

    # Training Configuration
    TRAIN_TEST_SPLIT: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5

    # XGBoost Parameters
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.1
    XGBOOST_N_ESTIMATORS: int = 100
    XGBOOST_MIN_CHILD_WEIGHT: int = 1
    XGBOOST_SUBSAMPLE: float = 0.8
    XGBOOST_COLSAMPLE_BYTREE: float = 0.8

    # Data Paths
    DATA_RAW_PATH: str = "data/raw/"
    DATA_PROCESSED_PATH: str = "data/processed/"
    DATA_REPORTS_PATH: str = "data/reports/"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # Monitoring
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Pydantic V2 Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra env vars to prevent validation errors
    )


# Singleton instance
settings = Settings()
