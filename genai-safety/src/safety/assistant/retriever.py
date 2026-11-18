"""Tiny keyword retriever. Swap for embeddings + a vector store in real use."""
import re
from typing import List

from safety.assistant.emails import Email


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Retriever:
    def __init__(self, inbox: List[Email]):
        self.inbox = inbox  # ordered oldest to newest

    def search(self, query: str, k: int = 3) -> List[Email]:
        """Rank by word overlap with the query; ties go to the newest email."""
        q = _tokens(query)
        ranked = sorted(
            enumerate(self.inbox),
            key=lambda pair: (len(q & _tokens(pair[1].text)), pair[0]),
            reverse=True)
        return [email for _, email in ranked[:k]]
