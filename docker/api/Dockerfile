FROM python:3.11-slim
WORKDIR /app

RUN pip install fastapi pandas scikit-learn mlflow==2.14.1 boto3 pydantic xgboost granian

COPY src/inference.py /app/inference.py

CMD ["granian", "--interface", "asgi", "inference:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
