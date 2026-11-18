"""Append-only request log. Metadata only by default: no email or request text.

One JSON object per line, so the offline pipeline can read it later. Several
replicas may append to the same local file; that is fine for a demo, but in
production send these records to a queue or table instead.
"""
import hashlib
import json
import os
import threading
import time
from typing import Optional


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RequestLog:
    def __init__(self, path: Optional[str], log_text: bool = False):
        self.path, self.log_text = path, log_text
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, record: dict, texts: Optional[dict] = None) -> None:
        """`texts` is only stored when log_text is on (opt-in, see the README)."""
        if not self.path:
            return
        record = {"ts": time.time(), **record}
        if self.log_text and texts:
            record["texts"] = texts
        line = json.dumps(record, default=str) + "\n"
        with self._lock, open(self.path, "a") as f:
            f.write(line)
