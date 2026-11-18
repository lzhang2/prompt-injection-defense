"""What the API does for each request. No Ray or FastAPI in here, so the
tests can run it with stub models."""
import time
import uuid
from typing import Callable, List, Optional, Sequence

from safety.assistant.agent import EmailAssistant
from safety.assistant.defenses import build_defenses
from safety.assistant.emails import BENIGN_EMAILS, Email
from safety.assistant.retriever import Retriever
from safety.llm import LLMClient
from safety.serving.request_log import RequestLog, sha256

Scorer = Callable[[Sequence[str]], List[float]]


class SafetyService:
    def __init__(self, llm: LLMClient, rater, scorer: Scorer, threshold: float,
                 top_k: int, default_defense: str = "all",
                 request_log: Optional[RequestLog] = None):
        self.llm, self.rater, self.scorer = llm, rater, scorer
        self.threshold, self.top_k = threshold, top_k
        self.default_defense = default_defense
        self.log = request_log or RequestLog(None)

    def score(self, texts: Sequence[str], model_version: str) -> dict:
        start, request_id = time.time(), uuid.uuid4().hex
        scores = self.scorer(list(texts))
        results = [{"score": s, "injection": s >= self.threshold} for s in scores]
        self.log.write(
            {"request_id": request_id, "endpoint": "score",
             "model_version": model_version,
             "latency_ms": round((time.time() - start) * 1000, 1),
             "items": [{"sha256": sha256(t), "score": r["score"],
                        "injection": r["injection"]} for t, r in zip(texts, results)]},
            texts={sha256(t): t for t, r in zip(texts, results) if r["injection"]})
        return {"request_id": request_id, "results": results,
                "threshold": self.threshold, "model_version": model_version}

    def run_assistant(self, request: str, emails: Optional[List[Email]],
                      defense: Optional[str], model_version: str) -> dict:
        start, request_id = time.time(), uuid.uuid4().hex
        defense = defense or self.default_defense
        inbox = list(emails) if emails else list(BENIGN_EMAILS)
        by_id = {e.id: e for e in inbox}
        if len(by_id) != len(inbox):
            raise ValueError("Email ids must be unique")

        seen = {}   # body -> score, so we can report what the classifier saw

        def recording_scorer(texts):
            out = self.scorer(list(texts))
            seen.update(zip(texts, out))
            return out

        defenses = build_defenses(defense, recording_scorer, self.rater, self.threshold)
        result = EmailAssistant(self.llm, Retriever(inbox), defenses,
                                self.top_k).run(request)

        kept = set(result.context_ids)
        email_scores = [{"id": i, "score": seen.get(by_id[i].body), "dropped": i not in kept}
                        for i in result.retrieved_ids]
        response = {
            "request_id": request_id, "answer": result.answer,
            "tool_call": result.tool_call, "tool_call_allowed": result.executed,
            "blocked_by": result.blocked_by, "emails": email_scores,
            "defense": defense, "model_version": model_version,
        }

        # Text worth keeping for later mining (only stored if log_text is on):
        # emails the classifier dropped, and, when the judge blocked an action,
        # the emails that were in context, since one of them got past the classifier.
        flagged = [e["id"] for e in email_scores if e["dropped"]]
        if result.blocked_by == "judge":
            flagged += [i for i in result.context_ids if i not in flagged]
        self.log.write(
            {"request_id": request_id, "endpoint": "assistant/run",
             "model_version": model_version, "defense": defense,
             "latency_ms": round((time.time() - start) * 1000, 1),
             "request_sha256": sha256(request),
             "emails": [{**e, "sha256": sha256(by_id[e["id"]].body)} for e in email_scores],
             "tool_call_name": (result.tool_call or {}).get("name"),
             "tool_call_allowed": result.executed, "blocked_by": result.blocked_by,
             "flagged_ids": flagged},
            texts={"request": request,
                   "emails": {i: by_id[i].body for i in flagged}} if flagged else None)
        return response
