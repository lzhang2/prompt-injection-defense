"""Run the whole loop, or selected stages.

    python -m pipelines.run_pipeline                      # everything
    python -m pipelines.run_pipeline --stages train,eval_classifier
    python -m pipelines.run_pipeline --stages attack_eval  # reuses state saved in MLflow
    python -m pipelines.run_pipeline --stages retrain,compare \\
        --set classifier.dataset=my-org/new-data --set classifier.epochs=3
"""
import argparse
import json
import logging
import os

import mlflow

from pipelines.stages import STAGES
from safety.config import apply_overrides, load_config, redacted
from safety.tracking import load_state, save_state, setup_mlflow, stage_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stages", default="all",
                        help=f"comma-separated, from: {', '.join(STAGES)}")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value for this run, e.g. "
                             "classifier.learning_rate=3e-5 (repeatable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    overrides = apply_overrides(cfg, args.set)
    names = list(STAGES) if args.stages == "all" else args.stages.split(",")
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        parser.error(f"Unknown stages: {unknown}")

    # MLflow holds the state; artifacts/state.json is only a local copy to read.
    setup_mlflow(cfg)
    state_path = os.path.join(cfg["paths"]["artifacts"], "state.json")
    state = load_state(cfg["mlflow"]["experiment"])
    if not state and os.path.exists(state_path):   # one-time move from disk
        with open(state_path) as f:
            state = json.load(f)
        logging.info("No state in MLflow yet; starting from %s", state_path)
    logging.info("Starting state: %s", state)

    with mlflow.start_run(run_name="pipeline:" + "+".join(names)):
        # Record exactly what this run used, so any model traces back to it.
        mlflow.set_tags({"stages": ",".join(names),
                         "app_version": os.environ.get("APP_VERSION", "dev")})
        mlflow.log_params({f"set.{k}": v for k, v in overrides.items()})
        mlflow.log_dict(redacted(cfg), "config.yaml")
        if overrides:
            logging.info("Overrides: %s", overrides)
        for name in names:
            logging.info("=== stage: %s ===", name)
            with stage_run(name):
                STAGES[name](cfg, state)
            save_state(state)                  # after every stage
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
    logging.info("Done. State: %s", state)


if __name__ == "__main__":
    main()
