# genai-safety app image = the platform's Ray base + this repo's code.
# The Ray cluster (in ../ml-platform) runs THIS image, so its replicas can import
# safety.serving.app. Build the base first: see ../ml-platform/README.md.
ARG RAY_BASE=ml-platform/ray-base:2.60.0
FROM ${RAY_BASE}

USER root
COPY docker/requirements.txt /tmp/app-requirements.txt
RUN pip install -r /tmp/app-requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY pipelines ./pipelines
RUN pip install --no-deps .
COPY configs/config.yaml ./configs/config.yaml
COPY docker/serve_config.yaml docker/deploy_app.sh docker/submit_pipeline.sh ./docker/
RUN chown -R app /app
# Recorded as the app_version tag on every pipeline run in MLflow.
#   docker compose build --build-arg APP_VERSION=$(git rev-parse --short HEAD)
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}
USER app
# CMD is inherited from the base: a Ray head node with no app on it.
