#!/bin/sh
# Run pipeline stages as a Ray Job on the ml-platform cluster.
#   docker compose run --rm train                       # mine,retrain,compare
#   docker compose run --rm train train,eval_classifier
#   docker compose run --rm train retrain,compare \
#       --set classifier.dataset=my-org/new-data --set classifier.epochs=3
#
# The job runs on ray-head, with the code and configs/config.yaml baked into the
# image the cluster runs (rebuild + restart ray-head after changing them). The
# pipeline state comes from MLflow, so nothing on this machine is needed.
# Ctrl-C stops following the logs; the job keeps running (see the dashboard).
set -eu
STAGES="${1:-mine,retrain,compare}"
[ $# -gt 0 ] && shift                # the rest (--set k=v ...) goes to the pipeline
RAY_DASHBOARD="${RAY_DASHBOARD_ADDRESS:-http://ray-head:8265}"

exec ray job submit --address "$RAY_DASHBOARD" \
  --submission-id "pipeline-$(date +%Y%m%d-%H%M%S)" \
  -- python -m pipelines.run_pipeline --config /app/configs/config.yaml --stages "$STAGES" "$@"
