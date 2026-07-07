"""Shared types for the model gateway."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypedDict


class Message(TypedDict):
    """A single conversation turn compatible with both Anthropic and Ollama."""
    role: str    # "user" | "assistant" | "system"
    content: str


@dataclass
class ModelResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)


class LLMGateway(ABC):
    """Abstract gateway — call complete() regardless of which backend is active."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> ModelResponse:
        """Send *messages* and return the assistant reply."""
