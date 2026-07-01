FROM apache/airflow:2.7.3-python3.10

USER root

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /mlflow && chmod 777 /mlflow
RUN mkdir -p /opt/airflow/data/processed /opt/airflow/data/reports /opt/airflow/models && \
    chown -R airflow:root /opt/airflow/data /opt/airflow/models

USER airflow

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=airflow:root airflow/dags /opt/airflow/dags
COPY --chown=airflow:root src/ /opt/airflow/src/

ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=30

ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"
