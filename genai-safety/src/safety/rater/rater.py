"""LLM auto-rater: one LLM, two rubrics (input and outcome)."""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence

from safety.llm import LLMClient, parse_json
from safety.rater.rubrics import INPUT_RUBRIC, OUTCOME_RUBRIC

log = logging.getLogger(__name__)


@dataclass
class Rating:
    label: int          # 1 = injection / unauthorized, 0 = benign
    confidence: str     # "high" or "low"
    reason: str


class AutoRater:
    def __init__(self, llm: LLMClient, max_workers: int = 4):
        self.llm, self.max_workers = llm, max_workers

    def _rate(self, prompt: str) -> Rating:
        data = parse_json(self.llm.complete(prompt))
        label = int(data["label"])
        if label not in (0, 1):
            raise ValueError(f"Bad label: {label}")
        return Rating(label, str(data.get("confidence", "low")).lower(),
                      str(data.get("reason", "")))

    def rate_input(self, text: str) -> Rating:
        return self._rate(INPUT_RUBRIC.format(text=text))

    def rate_outcome(self, user_request: str, emails: str, action: str) -> Rating:
        return self._rate(OUTCOME_RUBRIC.format(
            user_request=user_request, emails=emails, action=action))

    def rate_inputs(self, texts: Sequence[str]) -> List[Optional[Rating]]:
        """Rate many inputs in parallel. A failed call returns None, not a guess."""
        def safe(text):
            try:
                return self.rate_input(text)
            except Exception as e:  # API error or unparseable reply
                log.warning("Rater failed on one example: %s", e)
                return None

        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(safe, texts))
