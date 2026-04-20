"""LLM provider abstractions for Ask My Docs generation."""

from __future__ import annotations

from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM completion used by the RAG pipeline."""

    def complete(self, system: str, user: str) -> str:
        """Return the assistant's reply given a system and user message."""


class OpenAIProvider:
    """LLM provider backed by the OpenAI chat completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        """Call OpenAI chat completions and return the response text."""

        logger.info("openai_complete", model=self._model)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider:
    """LLM provider backed by the Anthropic messages API."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package is required: pip install anthropic") from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        """Call Anthropic messages API and return the response text."""

        logger.info("anthropic_complete", model=self._model)
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        block = response.content[0]
        if not hasattr(block, "text"):
            raise ValueError(f"Unexpected response block type: {type(block)}")
        return str(block.text)


class OllamaProvider:
    """LLM provider backed by a local Ollama instance."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 1024,
    ) -> None:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests package is required: pip install requests") from exc

        self._requests = requests
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        """Call Ollama chat endpoint and return the response text."""

        logger.info("ollama_complete", model=self._model)
        response = self._requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"num_predict": self._max_tokens},
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"])
