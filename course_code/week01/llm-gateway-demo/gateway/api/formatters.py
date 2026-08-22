from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from gateway.schemas import StreamEvent, UnifiedResponse


def _usage(value: Any) -> dict[str, int]:
    if value is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return value.model_dump()


def _tool_calls(response: UnifiedResponse) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for call in response.tool_calls
    ]


def chat_response(response: UnifiedResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.output_text}
    if response.reasoning_text:
        message["reasoning_content"] = response.reasoning_text
    if response.tool_calls:
        message["tool_calls"] = _tool_calls(response)
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason,
            }
        ],
        "usage": _usage(response.usage),
    }


def chat_chunk(
    response_id: str,
    model: str,
    event: StreamEvent,
    first: bool,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if first:
        delta["role"] = "assistant"
    if event.type == "text.delta":
        delta["content"] = event.text or ""
    elif event.type == "reasoning.delta":
        delta["reasoning_content"] = event.reasoning or ""
    elif event.type == "tool_call.delta":
        delta["tool_calls"] = event.tool_call or []
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": event.finish_reason if event.type == "completed" else None,
            }
        ],
        "usage": _usage(event.usage) if event.type == "usage" else None,
    }


async def chat_sse(
    stream: AsyncIterator[StreamEvent], response_id: str, model: str
) -> AsyncIterator[str]:
    first = True
    async for event in stream:
        if event.type not in {"text.delta", "reasoning.delta", "tool_call.delta", "usage", "completed"}:
            continue
        payload = chat_chunk(response_id, model, event, first)
        first = False
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def responses_response(response: UnifiedResponse) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if response.output_text:
        output.append(
            {
                "id": f"msg_{response.id}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response.output_text,
                        "annotations": [],
                    }
                ],
            }
        )
    return {
        "id": response.id,
        "object": "response",
        "created_at": response.created,
        "status": "completed",
        "model": response.model,
        "output": output,
        "output_text": response.output_text,
        "usage": _usage(response.usage),
        "metadata": {},
    }


async def responses_sse(
    stream: AsyncIterator[StreamEvent], response_id: str, model: str
) -> AsyncIterator[str]:
    started = {
        "type": "response.created",
        "response": {"id": response_id, "object": "response", "model": model},
    }
    yield f"event: response.created\ndata: {json.dumps(started, ensure_ascii=False)}\n\n"
    async for event in stream:
        if event.type == "text.delta":
            payload = {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "delta": event.text or "",
            }
            yield f"event: response.output_text.delta\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event.type == "reasoning.delta":
            payload = {
                "type": "response.reasoning.delta",
                "response_id": response_id,
                "delta": event.reasoning or "",
            }
            yield f"event: response.reasoning.delta\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        elif event.type == "completed":
            payload = {
                "type": "response.completed",
                "response": {"id": response_id, "object": "response", "model": model},
                "finish_reason": event.finish_reason or "stop",
            }
            yield f"event: response.completed\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
