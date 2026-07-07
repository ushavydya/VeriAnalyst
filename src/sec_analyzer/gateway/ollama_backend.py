"""Ollama backend — calls the local Ollama REST API with a Qwen model."""
from __future__ import annotations

import os

import httpx

from .base import LLMGateway, Message, ModelResponse

_DEFAULT_MODEL = "qwen2.5:7b"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaGateway(LLMGateway):
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self._model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
        self._base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> ModelResponse:
        payload_messages: list[dict] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

        payload = {
            "model": self._model,
            "messages": payload_messages,
            "stream": False,
            "think": False,          # disable extended thinking (Qwen3 / thinking models)
            "options": {"num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        text: str = data.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return ModelResponse(
            text=text,
            model=self._model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            extra={"eval_count": data.get("eval_count", 0)},
        )
