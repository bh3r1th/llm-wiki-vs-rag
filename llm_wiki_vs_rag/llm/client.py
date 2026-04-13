"""Minimal LLM client abstraction."""

import json
import os
from typing import Any
from urllib import error, request

from llm_wiki_vs_rag.config import LLMConfig


class _OpenAICompatibleAdapter:
    """HTTP adapter for OpenAI-compatible chat completions APIs."""

    def __init__(self, *, base_url: str, api_key: str, model_name: str, temperature: float, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise ValueError(f"OpenAI-compatible request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
            return str(parsed["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("OpenAI-compatible response missing choices[0].message.content") from exc


class LLMClient:
    """Thin abstraction around text and JSON generation APIs."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._mock_mode = config.mock_mode or config.provider.lower() == "mock"

        if self._mock_mode:
            self._adapter: _OpenAICompatibleAdapter | None = None
            return

        provider = config.provider.lower()
        if provider in {"stub", "mock"}:
            raise ValueError("Stub/mock provider is disabled outside deterministic mock_mode tests.")
        if provider != "openai-compatible":
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        base_url = config.base_url or os.getenv("LLM_BASE_URL")
        api_key = config.api_key or os.getenv("LLM_API_KEY")
        model_name = config.model_name or os.getenv("LLM_MODEL")
        if not base_url:
            raise ValueError("Missing LLM configuration: set llm.base_url or LLM_BASE_URL.")
        if not api_key:
            raise ValueError("Missing LLM configuration: set llm.api_key or LLM_API_KEY.")
        if not model_name:
            raise ValueError("Missing LLM configuration: set llm.model_name or LLM_MODEL.")

        self._adapter = _OpenAICompatibleAdapter(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )

    def generate(self, prompt: str) -> str:
        """Generate text output for a prompt."""
        if self._mock_mode:
            return self.config.mock_response
        if self._adapter is None:
            raise ValueError("LLM adapter is not initialized.")
        return self._adapter.generate(prompt)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Generate JSON output for a prompt."""
        text = self.generate(prompt)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Model output is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model output JSON must be an object.")
        return parsed
