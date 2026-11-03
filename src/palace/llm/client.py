"""Dual-tier Ollama chat client with clean structured and streaming outputs."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, Literal, TypeVar

import httpx
import requests
from pydantic import BaseModel

from palace.config import Settings, get_settings

LLMTier = Literal["fast", "reasoning"]
logging.getLogger("httpx").setLevel(logging.WARNING)

ModelT = TypeVar("ModelT", bound=BaseModel)

SYSTEM_INSTRUCTION = (
    "You are MindPalace, a personal second-brain assistant. Answer directly and "
    "concisely based on the provided context. Do NOT generate internal thought "
    "traces, <think> blocks, or scratchpad reasoning."
)
DEFAULT_KEEP_ALIVE = "10m"


def _field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _visible_token(chunk: Any) -> str:
    """Return only displayable content from dict- or object-shaped chunks."""
    message = _field(chunk, "message")
    if message is not None:
        token = _field(message, "content") or _field(message, "response")
    else:
        token = _field(chunk, "response") or _field(chunk, "content")
    return str(token) if token not in (None, "") else ""


class _ThinkingFilter:
    """Remove think blocks even when their tags span streamed chunks."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    @staticmethod
    def _partial_prefix_length(text: str, marker: str) -> int:
        lowered = text.lower()
        for length in range(min(len(text), len(marker) - 1), 0, -1):
            if lowered.endswith(marker[:length]):
                return length
        return 0

    def feed(self, token: str) -> str:
        if not token:
            return ""
        self._buffer += token
        visible: list[str] = []

        while self._buffer:
            lowered = self._buffer.lower()
            if self._inside:
                closing = lowered.find(self._CLOSE)
                if closing < 0:
                    self._buffer = self._buffer[-(len(self._CLOSE) - 1) :]
                    return "".join(visible)
                self._buffer = self._buffer[closing + len(self._CLOSE) :]
                self._inside = False
                continue

            opening = lowered.find(self._OPEN)
            stray_close = lowered.find(self._CLOSE)
            if stray_close >= 0 and (opening < 0 or stray_close < opening):
                visible.append(self._buffer[:stray_close])
                self._buffer = self._buffer[stray_close + len(self._CLOSE) :]
                continue
            if opening >= 0:
                visible.append(self._buffer[:opening])
                self._buffer = self._buffer[opening + len(self._OPEN) :]
                self._inside = True
                continue

            partial = self._partial_prefix_length(self._buffer, self._OPEN)
            safe_length = len(self._buffer) - partial
            visible.append(self._buffer[:safe_length])
            self._buffer = self._buffer[safe_length:]
            break

        return "".join(visible)

    def finish(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        trailing = self._buffer
        self._buffer = ""
        return trailing


def _clean_visible_text(text: str) -> str:
    filter_ = _ThinkingFilter()
    return filter_.feed(text) + filter_.finish()


class OllamaClient:
    """Route requests by tier and keep Ollama models warm between turns."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
        async_client: httpx.AsyncClient | None = None,
        *,
        keep_alive: str | int | None = None,
        fast_model: str | None = None,
        reasoning_model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._session = session or requests.Session()
        self._async_client = async_client
        self._owns_async_client = async_client is None
        self._url = f"{self.settings.llm.ollama_host.rstrip('/')}/api/chat"
        self.keep_alive = keep_alive or os.getenv(
            "PALACE_OLLAMA_KEEP_ALIVE",
            DEFAULT_KEEP_ALIVE,
        )
        self._models = {
            "fast": fast_model or os.getenv(
                "PALACE_LLM_FAST_MODEL",
                self.settings.llm.fast_tier_model,
            ),
            "reasoning": reasoning_model or os.getenv(
                "PALACE_LLM_REASONING_MODEL",
                self.settings.llm.reasoning_tier_model,
            ),
        }

    def _model_for(self, tier: LLMTier) -> str:
        try:
            return self._models[tier]
        except KeyError as exc:
            raise ValueError(f"unsupported LLM tier: {tier!r}") from exc

    def _temperature_for(self, tier: LLMTier) -> float:
        if tier == "fast":
            return self.settings.llm.fast_tier_temperature
        if tier == "reasoning":
            return self.settings.llm.reasoning_tier_temperature
        raise ValueError(f"unsupported LLM tier: {tier!r}")

    @staticmethod
    def _guarded_messages(
        messages: Sequence[dict[str, str]],
    ) -> list[dict[str, str]]:
        if (
            messages
            and messages[0].get("role") == "system"
            and messages[0].get("content", "").startswith(SYSTEM_INSTRUCTION)
        ):
            return list(messages)
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            *messages,
        ]

    def _payload(
        self,
        messages: Sequence[dict[str, str]],
        tier: LLMTier,
        *,
        stream: bool,
        response_format: dict[str, Any] | str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_for(tier),
            "messages": self._guarded_messages(messages),
            "stream": stream,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self._temperature_for(tier)
                if temperature is None
                else temperature
            },
        }
        if response_format is not None:
            payload["format"] = response_format
        return payload

    def complete_messages(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tier: LLMTier = "reasoning",
    ) -> str:
        response = self._session.post(
            self._url,
            json=self._payload(messages, tier, stream=False),
            timeout=self.settings.llm.request_timeout_seconds,
        )
        response.raise_for_status()
        content = _visible_token(response.json())
        if not content:
            raise ValueError("Ollama response did not contain user-facing content")
        return _clean_visible_text(content)

    def complete(
        self,
        prompt: str,
        *,
        tier: LLMTier = "reasoning",
        system: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete_messages(messages, tier=tier)

    def structured(
        self,
        prompt: str,
        response_model: type[ModelT],
        *,
        tier: LLMTier = "fast",
        system: str | None = None,
        temperature: float | None = None,
    ) -> ModelT:
        schema = response_model.model_json_schema()
        grounded_prompt = (
            f"{prompt}\n\nReturn only JSON matching this schema:\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": grounded_prompt})
        response = self._session.post(
            self._url,
            json=self._payload(
                messages,
                tier,
                stream=False,
                response_format=schema,
                temperature=temperature,
            ),
            timeout=self.settings.llm.request_timeout_seconds,
        )
        response.raise_for_status()
        content = _visible_token(response.json())
        if not content:
            raise ValueError("Ollama response did not contain structured content")
        return response_model.model_validate_json(_clean_visible_text(content))

    def stream_messages(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tier: LLMTier = "reasoning",
    ) -> Iterator[str]:
        """Synchronous compatibility stream with thinking content removed."""
        response = self._session.post(
            self._url,
            json=self._payload(messages, tier, stream=True),
            timeout=self.settings.llm.request_timeout_seconds,
            stream=True,
        )
        response.raise_for_status()
        filter_ = _ThinkingFilter()
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                token = _visible_token(json.loads(line))
                visible = filter_.feed(token)
                if visible:
                    yield visible
            trailing = filter_.finish()
            if trailing:
                yield trailing
        finally:
            response.close()

    async def astream_messages(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tier: LLMTier = "reasoning",
    ) -> AsyncIterator[str]:
        """Stream display tokens without blocking the active event loop."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.settings.llm.request_timeout_seconds,
            )
        filter_ = _ThinkingFilter()
        async with self._async_client.stream(
            "POST",
            self._url,
            json=self._payload(messages, tier, stream=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                token = _visible_token(json.loads(line))
                visible = filter_.feed(token)
                if visible:
                    yield visible
        trailing = filter_.finish()
        if trailing:
            yield trailing

    async def aclose(self) -> None:
        if self._owns_async_client and self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
