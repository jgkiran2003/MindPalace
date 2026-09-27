from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from palace.config import get_settings
from palace.llm.client import OllamaClient, SYSTEM_INSTRUCTION, _visible_token


class _StreamResponse:
    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self.lines = [json.dumps(line).encode() for line in lines]
        self.closed = False

    def raise_for_status(self) -> None:
        pass

    def iter_lines(self) -> list[bytes]:
        return self.lines

    def close(self) -> None:
        self.closed = True


class _SyncSession:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None

    def post(self, url: str, **kwargs: Any) -> _StreamResponse:
        self.request = {"url": url, **kwargs}
        return self.response


def test_stream_discards_thinking_fields_and_split_think_blocks() -> None:
    response = _StreamResponse(
        [
            {"thinking": "never display this"},
            {"message": {"thinking": "also hidden", "content": ""}},
            {"response": "Hello "},
            {"message": {"content": "<thi"}},
            {"message": {"content": "nk>private chain"}},
            {"message": {"content": " of thought</think>world"}},
            {"response": "!"},
            {"response": ""},
        ]
    )
    session = _SyncSession(response)
    client = OllamaClient(
        settings=get_settings(),
        session=session,  # type: ignore[arg-type]
        keep_alive="10m",
    )

    assert "".join(client.stream_messages([{"role": "user", "content": "Hi"}])) == (
        "Hello world!"
    )
    assert response.closed is True
    assert session.request is not None
    payload = session.request["json"]
    assert payload["think"] is False
    assert payload["keep_alive"] == "10m"
    assert payload["messages"][0] == {
        "role": "system",
        "content": SYSTEM_INSTRUCTION,
    }


def test_visible_token_supports_dict_and_object_chunks() -> None:
    assert _visible_token({"response": "dict token", "thinking": "hidden"}) == (
        "dict token"
    )
    assert _visible_token(
        SimpleNamespace(response="object token", thinking="hidden")
    ) == "object token"
    assert _visible_token(
        SimpleNamespace(
            message=SimpleNamespace(content="message token", thinking="hidden")
        )
    ) == "message token"
    assert _visible_token({"thinking": "thinking only"}) == ""


class _AsyncResponse:
    def __init__(self, lines: list[dict[str, Any]]) -> None:
        self.lines = [json.dumps(line) for line in lines]

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _AsyncStreamContext:
    def __init__(self, response: _AsyncResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _AsyncResponse:
        return self.response

    async def __aexit__(self, *args: Any) -> None:
        pass


class _AsyncClient:
    def __init__(self, response: _AsyncResponse) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None

    def stream(self, method: str, url: str, **kwargs: Any) -> _AsyncStreamContext:
        self.request = {"method": method, "url": url, **kwargs}
        return _AsyncStreamContext(self.response)


@pytest.mark.asyncio
async def test_async_stream_yields_only_visible_tokens() -> None:
    transport = _AsyncClient(
        _AsyncResponse(
            [
                {"message": {"thinking": "hidden", "content": ""}},
                {"message": {"content": "Clean"}},
                {"message": {"content": " output"}},
            ]
        )
    )
    client = OllamaClient(
        settings=get_settings(),
        async_client=transport,  # type: ignore[arg-type]
    )

    tokens = [
        token
        async for token in client.astream_messages(
            [{"role": "user", "content": "Hi"}]
        )
    ]

    assert tokens == ["Clean", " output"]
    assert transport.request is not None
    assert transport.request["json"]["stream"] is True
