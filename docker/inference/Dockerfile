FROM python:3.11-slim

LABEL maintainer=""
LABEL description="Churn Prediction Inference API"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker: prediction_history/stats in src/inference.py is an in-process
# deque with no shared store (Redis/DB) behind it. Multiple workers each get
# their own isolated copy, so /stats and the dashboard's "recent predictions"
# randomly reflect whichever worker handled the request. Fine for a portfolio
# demo's request volume; scaling this for real traffic needs the history moved
# to Redis/Postgres first.
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
