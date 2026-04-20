"""Unit tests for LLM provider implementations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from amd.generation.providers import AnthropicProvider, OllamaProvider, OpenAIProvider

# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


def test_openai_provider_returns_content() -> None:
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="justice is harmony"))]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        result = provider.complete("system prompt", "user message")

    assert result == "justice is harmony"


def test_openai_provider_passes_correct_messages() -> None:
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="key")
        provider.complete("sys", "usr")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "usr"}


def test_openai_provider_returns_empty_string_on_none_content() -> None:
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        provider = OpenAIProvider(api_key="key")
        result = provider.complete("sys", "usr")

    assert result == ""


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


def test_anthropic_provider_returns_text() -> None:
    text_block = SimpleNamespace(text="virtue is the mean")
    mock_response = SimpleNamespace(content=[text_block])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", api_key="key")
        result = provider.complete("system prompt", "user message")

    assert result == "virtue is the mean"


def test_anthropic_provider_passes_system_prompt() -> None:
    text_block = SimpleNamespace(text="answer")
    mock_response = SimpleNamespace(content=[text_block])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        provider = AnthropicProvider(api_key="key")
        provider.complete("my system prompt", "user msg")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "my system prompt"


def test_anthropic_provider_raises_on_non_text_block() -> None:
    non_text_block = SimpleNamespace()  # no .text attribute
    mock_response = SimpleNamespace(content=[non_text_block])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        provider = AnthropicProvider(api_key="key")
        with pytest.raises(ValueError, match="Unexpected response block type"):
            provider.complete("sys", "usr")


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


def test_ollama_provider_returns_content() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "equanimity is key"}}
    mock_requests = MagicMock()
    mock_requests.post.return_value = mock_response

    with patch("requests.post", mock_requests.post):
        provider = OllamaProvider(model="llama3")
        provider._requests = mock_requests
        result = provider.complete("system", "user")

    assert result == "equanimity is key"


def test_ollama_provider_posts_to_chat_endpoint() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "answer"}}
    mock_requests = MagicMock()
    mock_requests.post.return_value = mock_response

    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
    provider._requests = mock_requests
    provider.complete("sys", "usr")

    call_args = mock_requests.post.call_args
    assert call_args.args[0] == "http://localhost:11434/api/chat"
    payload = call_args.kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["stream"] is False
