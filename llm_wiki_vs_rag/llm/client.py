"""Minimal LLM client abstraction."""

import json
from typing import Any

from llm_wiki_vs_rag.config import LLMConfig


class LLMClient:
    """Thin abstraction around text and JSON generation APIs."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> str:
        """Generate text output for a prompt."""
        _ = prompt
        return f"[{self.config.provider}:{self.config.model_name}] generated response"

    def generate_json(self, prompt: str) -> dict[str, Any]:
        """Generate JSON output for a prompt."""
        text = self.generate(prompt)
        return {"response": text, "prompt_length": len(prompt), "raw": json.dumps({"text": text})}
