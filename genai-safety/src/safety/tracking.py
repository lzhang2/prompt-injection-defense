"""MLflow helpers shared by every stage."""
import json
import tempfile
from contextlib import contextmanager

import mlflow


def setup_mlflow(cfg: dict) -> None:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])


@contextmanager
def stage_run(name: str):
    """One nested MLflow run per pipeline stage."""
    with mlflow.start_run(run_name=name, nested=True) as run:
        yield run


def log_numeric(metrics: dict, prefix: str = "") -> None:
    clean = {f"{prefix}{k}": float(v) for k, v in metrics.items()
             if isinstance(v, (int, float)) and v == v}  # v == v drops NaN
    if clean:
        mlflow.log_metrics(clean)


def latest_version(model_name: str) -> str:
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No registered versions of {model_name}")
    return str(max(int(v.version) for v in versions))


CHAMPION = "champion"   # the registry alias the serving API loads


def get_alias(model_name: str, alias: str = CHAMPION):
    """Version number an alias points to, or None if the alias is not set."""
    try:
        return str(mlflow.MlflowClient().get_model_version_by_alias(model_name, alias).version)
    except Exception:  # MlflowException when the alias does not exist
        return None


def set_alias(model_name: str, version: str, alias: str = CHAMPION) -> None:
    mlflow.MlflowClient().set_registered_model_alias(model_name, alias, str(version))


# --- Pipeline state --------------------------------------------------------
# The state dict (model versions, mined-data URI, rater kappa, ...) lives in
# MLflow, not on one machine's disk, so a stage can run anywhere: on this Mac,
# in a container, or as a Ray Job on the cluster. Every pipeline run logs the
# state it ended with as state.json and is tagged; the next run starts from the
# most recent tagged run. Old snapshots stay on their runs as history.
STATE_TAG = "pipeline_state"
STATE_FILE = "state.json"


def load_state(experiment: str) -> dict:
    """State saved by the most recent pipeline run, or {} if there is none."""
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return {}
    runs = mlflow.MlflowClient().search_runs(
        [exp.experiment_id], filter_string=f"tags.{STATE_TAG} = 'true'",
        order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return {}
    with tempfile.TemporaryDirectory() as d:
        path = mlflow.artifacts.download_artifacts(
            run_id=runs[0].info.run_id, artifact_path=STATE_FILE, dst_path=d)
        with open(path) as f:
            return json.load(f)


def save_state(state: dict) -> None:
    """Save `state` on the active run (call it on the pipeline's parent run)."""
    mlflow.log_dict(state, STATE_FILE)
    mlflow.set_tag(STATE_TAG, "true")
