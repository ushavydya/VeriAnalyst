"""Anthropic backend using claude-opus-4-8 with streaming + adaptive thinking."""
from __future__ import annotations

import os

import anthropic

from .base import LLMGateway, Message, ModelResponse

_DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicGateway(LLMGateway):
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,  # not used by Anthropic; prompt-based JSON already reliable
    ) -> ModelResponse:
        kwargs: dict = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            thinking={"type": "adaptive"},
        )
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            final = await stream.get_final_message()

        text = next(
            (b.text for b in final.content if b.type == "text"), ""
        )
        return ModelResponse(
            text=text,
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
