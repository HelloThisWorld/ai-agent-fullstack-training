from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.adapters.base import LLMAdapter
from gateway.adapters.common import SecretBackedAdapterMixin, ensure_response, parse_usage, stream_error_guard
from gateway.schemas import ProviderCapabilities, StreamEvent, ToolCall, UnifiedRequest, UnifiedResponse
from gateway.secrets.base import SecretStore


class AnthropicAdapter(SecretBackedAdapterMixin, LLMAdapter):
    provider = "anthropic"

    def __init__(self, client: httpx.AsyncClient, base_url: str, secret_store: SecretStore, secret_ref: str):
        LLMAdapter.__init__(self, client)
        SecretBackedAdapterMixin.__init__(self, secret_store, secret_ref)
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _provider_model(model: str) -> str:
        return model.split("/", 1)[1] if "/" in model else model

    @staticmethod
    def _message_content(content: Any) -> Any:
        if isinstance(content, str):
            return content
        return content or ""

    def _payload(self, request: UnifiedRequest) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system" or message.role == "developer":
                if isinstance(message.content, str):
                    system_parts.append(message.content)
                continue
            messages.append(
                {
                    "role": "user" if message.role == "tool" else message.role,
                    "content": self._message_content(message.content),
                }
            )

        payload: dict[str, Any] = {
            "model": self._provider_model(request.model),
            "messages": messages,
            "max_tokens": request.max_tokens or 512,
            "stream": request.stream,
        }
        if request.instructions:
            system_parts.insert(0, request.instructions)
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        for field in ("temperature", "top_p", "stop"):
            value = getattr(request, field)
            if value is not None:
                payload["stop_sequences" if field == "stop" else field] = value
        if request.tools:
            payload["tools"] = [tool.get("function", tool) for tool in request.tools]
        return payload

    async def generate(self, request: UnifiedRequest) -> UnifiedResponse | AsyncIterator[StreamEvent]:
        key = await self._api_key()
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/v1/messages"
        payload = self._payload(request)
        if request.stream:
            return self._stream(url, headers, payload)

        response = await self.client.post(url, headers=headers, json=payload)
        await ensure_response(response, self.provider)
        data = response.json()
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif block.get("type") in {"thinking", "redacted_thinking"}:
                reasoning_parts.append(str(block.get("thinking", "")))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id", f"tool_call_{len(tool_calls)}")),
                        name=str(block.get("name", "")),
                        arguments=str(block.get("input", {})),
                    )
                )
        usage = parse_usage(data.get("usage"))
        return UnifiedResponse(
            id=str(data.get("id", f"msg_{uuid.uuid4().hex}")),
            model=request.model,
            provider=self.provider,
            output_text="".join(text_parts),
            reasoning_text="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            finish_reason=str(data.get("stop_reason") or "stop"),
            usage=usage,
            created=int(time.time()),
        )

    async def _stream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        async with self.client.stream("POST", url, headers=headers, json=payload) as response:
            finish_reason: str | None = None
            async for event in stream_error_guard(response, self.provider):
                event_type = event.get("type")
                delta = event.get("delta") or {}
                if event_type == "content_block_delta":
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield StreamEvent(type="text.delta", text=str(delta["text"]))
                    elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield StreamEvent(type="reasoning.delta", reasoning=str(delta["thinking"]))
                    elif delta.get("type") == "input_json_delta":
                        yield StreamEvent(type="tool_call.delta", tool_call=delta)
                elif event_type == "message_delta":
                    finish_reason = delta.get("stop_reason") or finish_reason
                    if delta.get("usage"):
                        yield StreamEvent(type="usage", usage=parse_usage(delta["usage"]))
            yield StreamEvent(type="completed", finish_reason=finish_reason or "stop")

    async def health_check(self) -> bool:
        try:
            return bool(await self._api_key())
        except Exception:
            return False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(tools=True, json_schema=False, vision=True)
