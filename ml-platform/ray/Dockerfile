# Base image for Ray clusters on this platform: ml-platform/ray-base:2.60.0
#   docker build -t ml-platform/ray-base:2.60.0 ray/
#
# Apps build FROM this and add their code, because Ray replicas import app code
# from the image the cluster runs (`serve deploy` sends config, not code).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/app/.cache/huggingface
WORKDIR /app

# CPU-only PyTorch: a few hundred MB instead of several GB of CUDA libraries.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/ray-base-requirements.txt
RUN pip install -r /tmp/ray-base-requirements.txt

# Convention for apps: code under /app, writable output under /app/artifacts.
RUN useradd --create-home app && mkdir -p /app/artifacts /app/.cache && chown -R app /app
USER app

EXPOSE 8000 8265
# A Ray head node in the foreground, with no app on it until something deploys one.
CMD ["ray", "start", "--head", "--block", "--dashboard-host", "0.0.0.0", "--disable-usage-stats"]
