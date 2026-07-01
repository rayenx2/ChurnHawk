FROM python:3.11-slim

LABEL maintainer=""
LABEL description="MLflow Tracking Server"

WORKDIR /mlflow

RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mlflow==2.8.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.29.7 \
    cryptography==41.0.7

RUN useradd -m -u 1000 mlflow && \
    chown -R mlflow:mlflow /mlflow

USER mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}"]
