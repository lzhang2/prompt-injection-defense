# MLflow tracking server + model registry. Keep the version in step with the
# clients that write to it (the mlflow pinned in ray/requirements.txt).
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.18.0
WORKDIR /mlflow
EXPOSE 5000
# Keep it small: it shares a ~4 GB Docker VM with the Ray cluster. Job execution
# runs 8 background workers (~110 MB each) for MLflow GenAI jobs we do not use.
ENV MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false
# --allowed-hosts: other containers call this as "mlflow:5000", which MLflow's
# DNS-rebinding protection rejects unless listed. Data lives on the mounted volume.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "2", \
     "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
     "--artifacts-destination", "/mlflow/data/mlartifacts", "--serve-artifacts", \
     "--allowed-hosts", "mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*"]
