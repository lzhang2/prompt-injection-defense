"""Load the YAML config once and pass the dict around."""
import os
from pathlib import Path

import yaml


LLM_SECTIONS = ("rater", "assistant")
KEY_NAMES = {"gemini": "gemini_api_key", "claude": "anthropic_api_key"}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Deployments set addresses through the environment (e.g. in Docker the
    # registry is http://mlflow:5000, not the 127.0.0.1 address in the file).
    if os.environ.get("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    # Relative paths are relative to the repo root (the folder above configs/),
    # not the working directory, so a Ray Job started from any cwd writes in the
    # same place as a local run.
    artifacts = Path(cfg["paths"]["artifacts"])
    if not artifacts.is_absolute():
        artifacts = Path(path).resolve().parent.parent / artifacts
    cfg["paths"]["artifacts"] = str(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    _add_api_keys(cfg, Path(path).parent / cfg.get("secrets_file", "secrets.yaml"))
    return cfg


def _add_api_keys(cfg: dict, secrets_path: Path) -> None:
    """Give each LLM section the key for its provider, from the secrets file.
    A key set directly in the section wins; if neither is set, the client falls
    back to the provider's environment variable."""
    if not secrets_path.exists():
        return
    with open(secrets_path) as f:
        secrets = yaml.safe_load(f) or {}
    for name in LLM_SECTIONS:
        section = cfg.get(name) or {}
        provider = section.get("provider", "gemini")
        key = secrets.get(KEY_NAMES.get(provider, f"{provider}_api_key"))
        if key and not section.get("api_key"):
            section["api_key"] = key


def apply_overrides(cfg: dict, pairs) -> dict:
    """Apply "section.key=value" overrides, e.g. classifier.learning_rate=3e-5.
    Values are parsed as YAML (3e-5 is a float, null is None, true is a bool).
    The key must already exist in config.yaml, so a typo fails loudly instead of
    silently training with the default."""
    applied = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--set needs key=value, got {pair!r}")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for p in parents:
            if not isinstance(node.get(p), dict):
                raise KeyError(f"--set {key}: no section {p!r} in the config")
            node = node[p]
        if leaf not in node:
            raise KeyError(f"--set {key}: no key {leaf!r} in the config")
        node[leaf] = _parse_value(raw)
        applied[key.strip()] = node[leaf]
    return applied


def _parse_value(raw: str):
    value = yaml.safe_load(raw)
    if isinstance(value, str):          # YAML 1.1 reads "3e-5" as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def redacted(cfg: dict) -> dict:
    """Copy of the config that is safe to log: API keys removed."""
    def clean(v):
        if isinstance(v, dict):
            return {k: ("***" if "api_key" in k else clean(x)) for k, x in v.items()}
        return v
    return clean(cfg)
