"""Point the serving alias at a model version by hand (compare does this automatically).

    python -m safety.serving.promote 2
"""
import argparse

import mlflow

from safety.config import load_config
from safety.tracking import CHAMPION, get_alias, set_alias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version", help="registered model version number, e.g. 2")
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["model_name"]
    old = get_alias(name)
    set_alias(name, args.version)
    print(f"{name}@{CHAMPION}: {old} -> {args.version}. Reload the API to serve it.")


if __name__ == "__main__":
    main()
