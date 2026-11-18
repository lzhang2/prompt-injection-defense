#!/bin/sh
# Demo of the retrain loop, sized for a laptop (~5-10 min on CPU):
#   Ray Job trains a candidate -> MLflow registers it -> compare gates it against
#   @champion -> deploy rolls it out if it was promoted.
# No LLM calls (no mine/attack_eval), so no API key or rate limits involved.
#
#   sh scripts/demo_retrain.sh            # 1 epoch
#   EPOCHS=3 sh scripts/demo_retrain.sh
set -eu
cd "$(dirname "$0")/.."
API="${API_URL:-http://127.0.0.1:8100}"

serving() {
  curl -fs "$API/health" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["classifier"]["model_version"])' \
    2>/dev/null || echo "not running"
}

BEFORE=$(serving)
echo "== Serving model version before: $BEFORE"

echo "== Retraining on the Ray cluster (Jobs tab: http://127.0.0.1:8365)"
docker compose run --rm train retrain,compare \
  --set classifier.epochs="${EPOCHS:-1}" \
  --set classifier.max_length=128 \
  --set classifier.batch_size=8 \
  --set classifier.ray.num_workers=1 \
  --set classifier.ray.cpus_per_worker=2

echo "== Rolling out @champion"
docker compose run --rm deploy

echo "== Serving model version: $BEFORE -> $(serving)"
echo "   Runs and versions: http://127.0.0.1:5100"
