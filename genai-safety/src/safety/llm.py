"""Thin LLM client wrapper so the rest of the code never imports a vendor SDK."""
import json
import os
import re
from typing import Optional, Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, system: Optional[str] = None) -> str: ...


class ClaudeClient:
    """Calls the Anthropic Messages API. Key from the config, else ANTHROPIC_API_KEY."""

    def __init__(self, model: str, max_tokens: int = 600, temperature: float = 0.0,
                 api_key: Optional[str] = None):
        import anthropic  # imported lazily so tests run without the SDK

        self._client = anthropic.Anthropic(api_key=api_key or None, max_retries=3)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        if self.temperature is not None:
            kwargs["extra_body"] = {"temperature": self.temperature}  # older models only; newer ones reject it
        message = self._client.messages.create(**kwargs)
        return "".join(b.text for b in message.content if b.type == "text")

class GeminiClient:
    """Calls the Gemini API. Key from the config, else GEMINI_API_KEY / GOOGLE_API_KEY."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        from google import genai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("No Gemini API key: set gemini_api_key in configs/secrets.yaml "
                               "or export GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        import time
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        for attempt in range(6):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config)
                return response.text or ""
            except errors.ClientError as e:
                if e.code != 429 or attempt == 5:
                    raise
                # Free tier is per-minute; honor the server's "retry in Ns" hint.
                m = re.search(r"retry in ([\d.]+)s", str(e))
                time.sleep(float(m.group(1)) + 1 if m else 60)
PROVIDERS = {"claude": ClaudeClient, "gemini": GeminiClient}

def make_client(section: dict) -> LLMClient:
    provider = section.get("provider", "gemini")
    print(f"Using LLM provider {provider!r} with model {section.get('model')!r}")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(PROVIDERS)}")
    opts = {k: section[k] for k in ("max_tokens", "temperature", "api_key")
            if section.get(k) is not None}
    return PROVIDERS[provider](section["model"], **opts)

def parse_json(reply: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating code fences and extra text."""
    text = re.sub(r"```(?:json)?", "", reply).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in reply: {reply[:200]!r}")
    return json.loads(text[start:end + 1])
