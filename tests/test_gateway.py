"""Tests for the model gateway (mocked — no real API calls)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sec_analyzer.gateway import get_gateway
from sec_analyzer.gateway.anthropic_backend import AnthropicGateway
from sec_analyzer.gateway.base import Message, ModelResponse
from sec_analyzer.gateway.ollama_backend import OllamaGateway


# ── factory ───────────────────────────────────────────────────────────────────

def test_get_gateway_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    gw = get_gateway()
    assert isinstance(gw, OllamaGateway)


def test_get_gateway_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_gateway(), AnthropicGateway)


def test_get_gateway_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert isinstance(get_gateway(), OllamaGateway)


def test_get_gateway_unknown():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_gateway("cohere")


# ── Anthropic backend ─────────────────────────────────────────────────────────

async def test_anthropic_complete():
    # Build a minimal fake final message
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "The answer is 42."

    fake_msg = MagicMock()
    fake_msg.model = "claude-opus-4-8"
    fake_msg.content = [text_block]
    fake_msg.usage.input_tokens = 10
    fake_msg.usage.output_tokens = 5

    fake_stream = AsyncMock()
    fake_stream.__aenter__ = AsyncMock(return_value=fake_stream)
    fake_stream.__aexit__ = AsyncMock(return_value=False)
    fake_stream.get_final_message = AsyncMock(return_value=fake_msg)

    with patch("sec_analyzer.gateway.anthropic_backend.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = fake_stream
        mock_cls.return_value = mock_client

        gw = AnthropicGateway(api_key="test-key")
        messages: list[Message] = [{"role": "user", "content": "What is 6×7?"}]
        result = await gw.complete(messages, system="You are helpful.")

    assert result.text == "The answer is 42."
    assert result.model == "claude-opus-4-8"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


# ── Ollama backend ────────────────────────────────────────────────────────────

async def test_ollama_complete():
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "message": {"role": "assistant", "content": "Qwen says hello."},
        "eval_count": 7,
        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
    }

    fake_client = AsyncMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_response)

    with patch("sec_analyzer.gateway.ollama_backend.httpx.AsyncClient", return_value=fake_client):
        gw = OllamaGateway(model="qwen2.5:7b", base_url="http://localhost:11434")
        messages: list[Message] = [{"role": "user", "content": "Hi"}]
        result = await gw.complete(messages)

    assert result.text == "Qwen says hello."
    assert result.model == "qwen2.5:7b"
    assert result.input_tokens == 8
    assert result.output_tokens == 4
