"""Model gateway — route LLM calls to Anthropic or Ollama+Qwen."""
from __future__ import annotations

import os

from .base import LLMGateway, Message, ModelResponse
from .anthropic_backend import AnthropicGateway
from .ollama_backend import OllamaGateway


def get_gateway(provider: str | None = None) -> LLMGateway:
    """Return the gateway for *provider*, defaulting to $LLM_PROVIDER or 'anthropic'."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()
    if provider == "anthropic":
        return AnthropicGateway()
    if provider == "ollama":
        return OllamaGateway()
    raise ValueError(f"Unknown LLM provider: {provider!r}. Choose 'anthropic' or 'ollama'.")


__all__ = ["get_gateway", "LLMGateway", "Message", "ModelResponse"]
