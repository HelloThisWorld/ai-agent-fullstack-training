from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.adapters.base import LLMAdapter
from gateway.adapters.common import (
    ensure_response,
    parse_tool_calls,
    parse_usage,
    stream_error_guard,
)
from gateway.schemas import ProviderCapabilities, StreamEvent, UnifiedRequest, UnifiedResponse


class LlamaCppAdapter(LLMAdapter):
    provider = "local"

    def __init__(self, client: httpx.AsyncClient, base_url: str, model_id: str):
        super().__init__(client)
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id

    def _payload(self, request: UnifiedRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": request.stream,
        }
        for field in ("temperature", "top_p", "max_tokens", "stop", "tool_choice", "response_format"):
            value = getattr(request, field)
            if value is not None:
                payload[field] = value
        if request.tools:
            payload["tools"] = request.tools
        # Qwen3-style models may spend a small local request entirely on hidden
        # thinking tokens. Keep the chat clients useful by default, while still
        # allowing an explicit request override.
        payload["chat_template_kwargs"] = request.chat_template_kwargs or {
            "enable_thinking": False
        }
        return payload

    async def generate(self, request: UnifiedRequest) -> UnifiedResponse | AsyncIterator[StreamEvent]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._payload(request)
        if request.stream:
            return self._stream(url, payload)

        response = await self.client.post(url, json=payload)
        await ensure_response(response, self.provider)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return UnifiedResponse(
            id=str(data.get("id", f"local_{uuid.uuid4().hex}")),
            model=request.model,
            provider=self.provider,
            output_text=str(message.get("content") or ""),
            reasoning_text=message.get("reasoning_content"),
            tool_calls=parse_tool_calls(message.get("tool_calls")),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage=parse_usage(data.get("usage")),
            created=int(data.get("created", time.time())),
        )

    async def _stream(self, url: str, payload: dict[str, Any]) -> AsyncIterator[StreamEvent]:
        async with self.client.stream("POST", url, json=payload) as response:
            finish_reason: str | None = None
            async for chunk in stream_error_guard(response, self.provider):
                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield StreamEvent(type="text.delta", text=str(delta["content"]))
                    if delta.get("reasoning_content"):
                        yield StreamEvent(
                            type="reasoning.delta", reasoning=str(delta["reasoning_content"])
                        )
                    if delta.get("tool_calls"):
                        yield StreamEvent(type="tool_call.delta", tool_call=delta["tool_calls"])
                    finish_reason = choice.get("finish_reason") or finish_reason
                if chunk.get("usage"):
                    yield StreamEvent(type="usage", usage=parse_usage(chunk["usage"]))
            yield StreamEvent(type="completed", finish_reason=finish_reason or "stop")

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(tools=True, json_schema=True)
