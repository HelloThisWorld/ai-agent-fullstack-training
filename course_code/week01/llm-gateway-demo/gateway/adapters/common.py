from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.adapters.sse import error_message, iter_sse
from gateway.errors import ProviderNotConfiguredError, ProviderRequestError
from gateway.schemas import ToolCall, UnifiedRequest, Usage
from gateway.secrets.base import SecretStore


def content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, str):
        return content
    if not content:
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
    )


def parse_usage(payload: dict[str, Any] | None) -> Usage:
    payload = payload or {}
    return Usage(
        prompt_tokens=int(payload.get("prompt_tokens", payload.get("input_tokens", 0)) or 0),
        completion_tokens=int(
            payload.get("completion_tokens", payload.get("output_tokens", 0)) or 0
        ),
        total_tokens=int(payload.get("total_tokens", 0) or 0),
    )


def parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, call in enumerate(raw or []):
        function = call.get("function", call)
        calls.append(
            ToolCall(
                id=str(call.get("id", f"tool_call_{index}")),
                name=str(function.get("name", "")),
                arguments=str(function.get("arguments", "{}")),
            )
        )
    return calls


class SecretBackedAdapterMixin:
    def __init__(self, secret_store: SecretStore, secret_ref: str):
        self.secret_store = secret_store
        self.secret_ref = secret_ref

    async def _api_key(self) -> str:
        value = await self.secret_store.get(self.secret_ref)
        if not value:
            raise ProviderNotConfiguredError(self.provider)
        return value


async def ensure_response(response: httpx.Response, provider: str) -> None:
    if response.is_error:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        raise ProviderRequestError(provider, error_message(payload), response.status_code)


async def stream_error_guard(response: httpx.Response, provider: str) -> AsyncIterator[Any]:
    if response.is_error:
        try:
            payload = await response.aread()
            detail = payload.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - defensive network path
            detail = str(exc)
        raise ProviderRequestError(provider, detail, response.status_code)
    async for item in iter_sse(response):
        yield item
