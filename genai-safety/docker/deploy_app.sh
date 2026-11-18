#!/usr/bin/env sh
# Step 3: deploy (or update) the Serve app onto an already running Ray cluster,
# then wait until the API answers. Safe to re-run: `serve deploy` applies the
# config as a rolling update instead of restarting the cluster.
#
# Inside compose: docker compose run --rm deploy
set -eu
RAY_DASHBOARD=${RAY_DASHBOARD_ADDRESS:-http://ray-head:8265}
API=${API_URL:-http://ray-head:8000}
CONFIG=${SERVE_CONFIG:-docker/serve_config.yaml}
TIMEOUT=${DEPLOY_TIMEOUT_S:-300}

# Resolve serving.model (e.g. "@champion") to a version number now and write it
# into the deployed config, so a moved alias is a real config change.
RESOLVED=/tmp/serve_config.resolved.yaml
python - "$CONFIG" "$RESOLVED" <<'PY'
import sys, yaml, mlflow
from safety.config import load_config
from safety.classifier.predict import resolve_version
src, dst = sys.argv[1], sys.argv[2]
serve_cfg = yaml.safe_load(open(src))
for app in serve_cfg["applications"]:
    args = app.setdefault("args", {})
    cfg = load_config(args.get("config", "configs/config.yaml"))
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    ref = str(args.get("model") or cfg["serving"].get("model", "@champion"))
    args["model"] = resolve_version(cfg, ref)
    print(f"{app['name']}: {ref} -> version {args['model']}")
yaml.safe_dump(serve_cfg, open(dst, "w"), sort_keys=False)
PY

echo "Deploying to $RAY_DASHBOARD"
serve deploy "$RESOLVED" --address "$RAY_DASHBOARD"

echo "Waiting up to ${TIMEOUT}s for $API/health (first start loads the model)..."
python - "$API/health" "$TIMEOUT" <<'PY'
import json, sys, time, urllib.request
url, deadline = sys.argv[1], time.time() + float(sys.argv[2])
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            print("App is up:", json.dumps(json.load(r)))
            sys.exit(0)
    except Exception:
        time.sleep(5)
sys.exit(f"Timed out waiting for {url}; check `serve status` or the replica logs")
PY
