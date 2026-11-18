#!/usr/bin/env bash
# Copy an existing local MLflow registry into data/mlflow so the MLflow container
# serves the same runs and models. No retraining.
#
#   scripts/seed_mlflow.sh ../genai-safety/mlflow.db ../genai-safety/mlartifacts
#   FORCE=1 scripts/seed_mlflow.sh ...      # overwrite an existing copy
#
# Works because artifact paths in mlflow.db are relative to the server
# ("mlflow-artifacts:/1/models/..."), not absolute paths on this machine.
set -euo pipefail
SRC_DB=${1:?usage: seed_mlflow.sh <mlflow.db> <mlartifacts dir>}
SRC_ARTIFACTS=${2:?usage: seed_mlflow.sh <mlflow.db> <mlartifacts dir>}
SRC_DB=$(cd "$(dirname "$SRC_DB")" && pwd)/$(basename "$SRC_DB")
SRC_ARTIFACTS=$(cd "$SRC_ARTIFACTS" && pwd)
cd "$(dirname "$0")/.."
DEST=data/mlflow
PY=${PYTHON:-python}

if [[ -e "$DEST/mlflow.db" && "${FORCE:-0}" != 1 ]]; then
  echo "$DEST/mlflow.db already exists. Run with FORCE=1 to overwrite it." >&2
  exit 1
fi
mkdir -p "$DEST"
# sqlite backup API: a consistent copy even while the source server is running.
"$PY" - "$SRC_DB" "$DEST/mlflow.db" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close()
PY
rsync -a --delete "$SRC_ARTIFACTS/" "$DEST/mlartifacts/"
echo "Seeded $DEST from $SRC_DB and $SRC_ARTIFACTS/"
