"""Ray Serve app: a FastAPI gateway in front of a classifier deployment.

    serve run safety.serving.app:build_app                         # configs/config.yaml
    serve run safety.serving.app:build_app config=other.yaml

Then open http://127.0.0.1:8000/docs.
"""
import asyncio
import logging
import os
from typing import List

from fastapi import FastAPI, HTTPException
from ray import serve
from ray.serve.handle import DeploymentHandle

from safety.serving.schemas import (AssistantRequest, AssistantResponse,
                                    ScoreRequest, ScoreResponse)

log = logging.getLogger("ray.serve")
app = FastAPI(title="genai-safety", version="0.1.0",
              description="Prompt-injection scoring and a defended email assistant.")


@serve.deployment
class Classifier:
    """Holds the DeBERTa model. Scales separately from the gateway."""

    def __init__(self, cfg: dict):
        self.cfg, self.clf, self.version = cfg, None, None

    def reconfigure(self, user_config: dict) -> None:
        """Ray calls this at start and whenever user_config changes (serve deploy)."""
        import mlflow
        from safety.classifier.predict import InjectionClassifier, resolve_version

        mlflow.set_tracking_uri(self.cfg["mlflow"]["tracking_uri"])
        ref = str(user_config.get("model", "@champion"))
        # Pin the number once, so a replica never changes model mid-flight when the
        # alias moves; reload to pick up a new champion.
        version = resolve_version(self.cfg, ref)
        self.clf = InjectionClassifier.from_registry(self.cfg, version)
        self.ref, self.version = ref, version
        log.info("Classifier loaded %s -> version %s", ref, version)

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def _score_batch(self, texts: List[str]) -> List[float]:
        return self.clf.score(texts)   # one forward pass for texts from many requests

    async def score(self, texts: List[str]) -> List[float]:
        return list(await asyncio.gather(*(self._score_batch(t) for t in texts)))

    def info(self) -> dict:
        return {"model_ref": self.ref, "model_version": self.version}


@serve.deployment
@serve.ingress(app)
class Gateway:
    """Public HTTP API. Runs the assistant logic and calls the classifier."""

    def __init__(self, cfg: dict, classifier: DeploymentHandle):
        from safety.llm import make_client
        from safety.rater.rater import AutoRater
        from safety.serving.request_log import RequestLog
        from safety.serving.service import SafetyService

        s = cfg["serving"]
        self.classifier = classifier
        self.llm_info = {name: {"provider": cfg[name].get("provider", "gemini"),
                                "model": cfg[name]["model"]}
                         for name in ("assistant", "rater")}   # never includes the key
        self.service = SafetyService(
            llm=make_client(cfg["assistant"]),
            rater=AutoRater(make_client(cfg["rater"]), cfg["rater"]["max_workers"]),
            scorer=self._score_blocking,
            threshold=cfg["classifier"]["threshold"],
            top_k=cfg["assistant"]["top_k"],
            default_defense=s.get("defense", "all"),
            request_log=RequestLog(os.path.join(cfg["paths"]["artifacts"], s["log_file"])
                                   if s.get("log_file") else None,
                                   log_text=bool(s.get("log_text", False))))

    def _score_blocking(self, texts):
        # Only called from worker threads (asyncio.to_thread below), never the event loop.
        return self.classifier.score.remote(list(texts)).result()

    async def _version(self) -> str:
        return (await self.classifier.info.remote())["model_version"]

    @app.get("/health")
    async def health(self) -> dict:
        return {"status": "ok", "classifier": await self.classifier.info.remote(),
                "llm": self.llm_info}

    @app.post("/score", response_model=ScoreResponse)
    async def score(self, req: ScoreRequest):
        version = await self._version()
        return await asyncio.to_thread(self.service.score, req.texts, version)

    @app.post("/assistant/run", response_model=AssistantResponse)
    async def run(self, req: AssistantRequest):
        from safety.assistant.emails import Email

        emails = [Email(**e.model_dump()) for e in req.emails] if req.emails else None
        version = await self._version()
        log.info("assistant/run defense=%s llm=%s", req.defense or "default",
                 self.llm_info["assistant"]["model"])
        try:
            return await asyncio.to_thread(self.service.run_assistant, req.request,
                                           emails, req.defense, version)
        except ValueError as e:                       # bad input, e.g. duplicate ids
            raise HTTPException(422, str(e))
        except Exception as e:                        # LLM provider error, quota, etc.
            log.exception("assistant/run failed")
            raise HTTPException(502, f"Upstream model error: {type(e).__name__}")


def build_app(args: dict) -> serve.Application:
    """Ray Serve app builder: `serve run safety.serving.app:build_app config=...`."""
    from safety.config import load_config

    cfg = load_config(os.path.abspath(args.get("config", "configs/config.yaml")))
    # Replicas may not share the driver's working directory, so use absolute paths.
    cfg["paths"]["artifacts"] = os.path.abspath(cfg["paths"]["artifacts"])
    s = cfg["serving"]
    # args["model"] (a version number, set by docker/deploy_app.sh) beats the config.
    # Pinning the number in the deployed config means a moved @champion changes the
    # config, so `serve deploy` rolls it out instead of treating it as a no-op.
    model = str(args.get("model") or s.get("model", "@champion"))
    classifier = Classifier.options(
        num_replicas=s.get("classifier_replicas", 1),
        ray_actor_options={"num_cpus": s.get("classifier_cpus", 1)},
        user_config={"model": model},
    ).bind(cfg)
    return Gateway.options(num_replicas=s.get("gateway_replicas", 1)).bind(cfg, classifier)
