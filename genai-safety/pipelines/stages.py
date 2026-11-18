"""Pipeline stages. Each stage reads and writes a shared state dict, so stages
can run together or one at a time."""
import json
import logging
import os
import random

import mlflow

from safety.assistant.emails import BENIGN_EMAILS, load_attacks, split_attacks
from safety.tracking import CHAMPION, get_alias, log_numeric, set_alias

log = logging.getLogger(__name__)

DEFENSE_CONFIGS = ["none", "spotlight", "classifier", "judge", "all"]
TRAIN_NEGATIVES, EVAL_NEGATIVES = BENIGN_EMAILS[:4], BENIGN_EMAILS[4:]


def _artifact(cfg, name):
    return os.path.join(cfg["paths"]["artifacts"], name)


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _classifier(cfg, state, key=None):
    from safety.classifier.predict import InjectionClassifier
    key = key or ("v2" if "v2" in state else "v1")
    return InjectionClassifier.from_registry(cfg, state[key]), key


def _rater(cfg):
    from safety.llm import make_client
    from safety.rater.rater import AutoRater
    return AutoRater(make_client(cfg["rater"]), cfg["rater"]["max_workers"])


def _attack_halves(cfg):
    return split_attacks(load_attacks(cfg["attacks"]), cfg["attacks"]["seed"])


# --- Project 1 -------------------------------------------------------------

def train(cfg, state):
    """Train classifier v1 on the base dataset."""
    from safety.classifier.train import run_training
    state["v1"] = run_training(cfg, "classifier-v1")
    mlflow.log_param("model_version", state["v1"])
    if get_alias(cfg["mlflow"]["model_name"]) is None:   # first model: serve it
        _promote(cfg, state, state["v1"], "first trained model")


def eval_classifier(cfg, state):
    """Score the human-labeled test split."""
    from safety.classifier.data import load_splits
    from safety.classifier.evaluate import classifier_report
    clf, key = _classifier(cfg, state)
    test = load_splits(cfg["classifier"])["test"]
    report = classifier_report(list(test["label"]), clf.score(list(test["text"])),
                               cfg["classifier"]["max_fpr"],
                               cfg["classifier"]["threshold"])
    mlflow.log_param("model_key", key)
    log_numeric(report)
    log.info("Classifier %s on test split: %s", key, report)


# --- Project 2 -------------------------------------------------------------

def rater_agreement(cfg, state):
    """Check the rater against human labels before trusting its labels."""
    from safety.classifier.data import load_splits
    from safety.rater.agreement import agreement_report
    test = load_splits(cfg["classifier"])["test"]
    idx = list(range(len(test)))
    random.Random(0).shuffle(idx)
    idx = idx[:cfg["rater"]["agreement_sample"]]
    texts = [test[i]["text"] for i in idx]
    human = [int(test[i]["label"]) for i in idx]

    ratings = _rater(cfg).rate_inputs(texts)
    report = agreement_report(human, [r.label if r else None for r in ratings])
    log_numeric(report)
    log.info("Rater agreement: %s", report)

    # Disagreements are what you read to improve the rubric.
    path = _artifact(cfg, "rater_disagreements.jsonl")
    _write_jsonl(path, [{"text": t, "human": h, "rater": r.label, "reason": r.reason}
                        for t, h, r in zip(texts, human, ratings)
                        if r and r.label != h])
    mlflow.log_artifact(path)
    state["rater_kappa"] = report["kappa"]


# --- Project 3 -------------------------------------------------------------

def attack_eval(cfg, state):
    """ASR for each defense configuration, on the held-out attack half."""
    from safety.assistant.attack_eval import evaluate_config
    from safety.assistant.defenses import build_defenses
    from safety.llm import make_client
    _, holdout = _attack_halves(cfg)
    limit = cfg["assistant"].get("eval_max_attacks")
    if limit:
        holdout = holdout[:limit]
    llm = make_client(cfg["assistant"])
    clf, key = _classifier(cfg, state)
    rater = _rater(cfg)
    mlflow.log_param("model_key", key)

    all_rows = {}
    for name in DEFENSE_CONFIGS:
        result = evaluate_config(
            llm,
            lambda n=name: build_defenses(n, clf.score, rater,
                                          cfg["classifier"]["threshold"]),
            holdout, cfg["assistant"]["top_k"])
        log_numeric(result["metrics"], prefix=f"{name}.")
        all_rows[name] = result["rows"]
        log.info("Defense %-10s %s", name, result["metrics"])
    path = _artifact(cfg, f"attack_eval_{key}.json")
    with open(path, "w") as f:
        json.dump(all_rows, f, indent=2)
    mlflow.log_artifact(path)


# --- Feedback loop ---------------------------------------------------------

def mine(cfg, state):
    """Find attacks the classifier misses, confirm them with the rater, and
    write them out as new training data."""
    mine_half, _ = _attack_halves(cfg)
    clf, _ = _classifier(cfg, state, "v1")
    scores = clf.score(mine_half)
    missed = [t for t, s in zip(mine_half, scores)
              if s < cfg["classifier"]["threshold"]]

    ratings = _rater(cfg).rate_inputs(missed)
    confirmed = [t for t, r in zip(missed, ratings)
                 if r and r.label == 1 and r.confidence == "high"]
    needs_human = [t for t, r in zip(missed, ratings)
                   if not r or r.label == 0 or r.confidence != "high"]

    rows = [{"text": t, "label": 1} for t in confirmed]
    rows += [{"text": e.body, "label": 0} for e in TRAIN_NEGATIVES]  # keep balance
    state["mined_path"] = _artifact(cfg, "mined_train.jsonl")
    _write_jsonl(state["mined_path"], rows)
    _write_jsonl(_artifact(cfg, "needs_human_review.jsonl"),
                 [{"text": t} for t in needs_human])
    mlflow.log_artifact(state["mined_path"])
    # The URI, not the local path, is what other machines (a Ray Job) can read.
    state["mined_uri"] = mlflow.get_artifact_uri(os.path.basename(state["mined_path"]))
    log_numeric({"n_candidates": len(mine_half), "n_missed": len(missed),
                 "n_confirmed": len(confirmed), "n_needs_human": len(needs_human)})


def retrain(cfg, state):
    """Train a candidate (v2) on the configured dataset plus the mined examples,
    if any. Use --set classifier.dataset=... to retrain on new data."""
    from safety.classifier.train import run_training
    state["v2"] = run_training(cfg, "classifier-v2", _mined_data(cfg, state))
    mlflow.log_param("model_version", state["v2"])


def _mined_data(cfg, state):
    """Local path to the mined training data, downloaded from MLflow if needed."""
    if state.get("mined_uri"):
        return mlflow.artifacts.download_artifacts(
            artifact_uri=state["mined_uri"], dst_path=cfg["paths"]["artifacts"])
    if state.get("mined_path") and os.path.exists(state["mined_path"]):
        return state["mined_path"]       # state from before mined_uri existed
    log.warning("No mined data in the pipeline state; training on the dataset only.")
    return None


def compare(cfg, state):
    """The candidate (v2) vs the model serving now (@champion; v1 if none), on
    data neither trained on. Comparing against v1 instead would let a second
    retrain replace a good champion with something only better than v1."""
    from safety.classifier.data import load_splits
    from safety.classifier.evaluate import classifier_report
    from safety.classifier.predict import InjectionClassifier
    _, holdout = _attack_halves(cfg)
    test = load_splits(cfg["classifier"])["test"]
    thr = cfg["classifier"]["threshold"]
    versions = {"v1": get_alias(cfg["mlflow"]["model_name"]) or state["v1"],
                "v2": state["v2"]}
    mlflow.log_params({"champion_before": versions["v1"], "candidate": versions["v2"]})
    if versions["v1"] == versions["v2"]:
        log.info("Candidate %s is already @%s; nothing to compare.", versions["v2"], CHAMPION)
        return
    reports = {}
    for key in ("v1", "v2"):               # v1 = current champion, v2 = candidate
        clf = InjectionClassifier.from_registry(cfg, versions[key])
        report = classifier_report(list(test["label"]), clf.score(list(test["text"])),
                                   cfg["classifier"]["max_fpr"], thr)
        attack_scores = clf.score(holdout)
        benign_scores = clf.score([e.body for e in EVAL_NEGATIVES])
        report["holdout_attack_detection"] = (
            sum(s >= thr for s in attack_scores) / max(len(attack_scores), 1))
        report["benign_email_fpr"] = (
            sum(s >= thr for s in benign_scores) / len(benign_scores))
        log_numeric(report, prefix=f"{key}.")
        log.info("%s (version %s): %s", key, versions[key], report)
        reports[key] = report

    if _v2_is_better(reports["v1"], reports["v2"]):
        _promote(cfg, state, versions["v2"], "candidate beat the champion")
    else:
        _promote(cfg, state, versions["v1"], "candidate did not beat the champion")


def _v2_is_better(v1, v2, pr_auc_tolerance=0.01):
    """Promote v2 only if it catches at least as many held-out attacks, flags no
    more benign emails, and does not lose more than a little PR-AUC. Any NaN
    (too little data to tell) keeps v1."""
    try:
        return (v2["holdout_attack_detection"] >= v1["holdout_attack_detection"]
                and v2["benign_email_fpr"] <= v1["benign_email_fpr"]
                and v2["pr_auc"] >= v1["pr_auc"] - pr_auc_tolerance)
    except (KeyError, TypeError):
        return False


def _promote(cfg, state, version, why):
    """Point the serving alias at `version`. The API picks it up on its next reload."""
    set_alias(cfg["mlflow"]["model_name"], version)
    state["champion"] = str(version)
    mlflow.log_param("champion_version", version)
    log.info("@%s -> version %s (%s)", CHAMPION, version, why)


STAGES = {
    "train": train,
    "eval_classifier": eval_classifier,
    "rater_agreement": rater_agreement,
    "attack_eval": attack_eval,
    "mine": mine,
    "retrain": retrain,
    "compare": compare,
}
