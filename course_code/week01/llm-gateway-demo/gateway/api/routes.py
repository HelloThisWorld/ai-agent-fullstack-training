from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.api.formatters import chat_response, chat_sse, responses_response, responses_sse
from gateway.errors import GatewayError
from gateway.schemas import ChatCompletionRequest, Message, ResponsesRequest, UnifiedRequest

router = APIRouter()


def _runtime(request: Request):
    return request.app.state.runtime


def _chat_unified(payload: ChatCompletionRequest) -> UnifiedRequest:
    max_tokens = payload.max_completion_tokens or payload.max_tokens
    return UnifiedRequest(
        model=payload.model,
        messages=payload.messages,
        stream=payload.stream,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=max_tokens,
        stop=payload.stop,
        tools=[tool.model_dump(exclude_none=True) for tool in payload.tools or []] or None,
        tool_choice=payload.tool_choice,
        response_format=payload.response_format,
        chat_template_kwargs=payload.chat_template_kwargs,
        metadata={"endpoint": "/v1/chat/completions"},
    )


def _responses_messages(payload: ResponsesRequest) -> list[Message]:
    if isinstance(payload.input, str):
        return [Message(role="user", content=payload.input)]
    return [item if isinstance(item, Message) else Message.model_validate(item) for item in payload.input]


def _responses_unified(payload: ResponsesRequest) -> UnifiedRequest:
    tools = payload.tools
    response_format = None
    if payload.text and payload.text.get("format"):
        response_format = payload.text["format"]
    return UnifiedRequest(
        model=payload.model,
        messages=_responses_messages(payload),
        instructions=payload.instructions,
        stream=payload.stream,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_output_tokens,
        tools=tools,
        response_format=response_format,
        metadata=payload.metadata or {"endpoint": "/v1/responses"},
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    runtime = _runtime(request)
    local = runtime.registry.resolve(runtime.settings.local_model).adapter
    if await local.health_check():
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "not_ready", "reason": "local_model_unavailable"}, status_code=503)


@router.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    return {"object": "list", "data": runtime.registry.models(runtime.settings.local_model)}


@router.get("/v1/providers")
async def providers(request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    result: list[dict[str, Any]] = []
    for item in runtime.registry.items():
        # This returns only operational status. It never returns credentials or headers.
        result.append(
            {
                "provider": item.adapter.provider,
                "configured": await item.adapter.health_check(),
                "capabilities": item.adapter.capabilities().model_dump(),
            }
        )
    return {"data": result}


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    runtime = _runtime(request)
    unified = _chat_unified(payload)
    result = await runtime.service.generate(unified)
    if payload.stream:
        response_id = f"chatcmpl_{uuid.uuid4().hex}"
        return StreamingResponse(
            chat_sse(result, response_id, payload.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return chat_response(result)


@router.post("/v1/responses")
async def responses(payload: ResponsesRequest, request: Request):
    runtime = _runtime(request)
    unified = _responses_unified(payload)
    result = await runtime.service.generate(unified)
    if payload.stream:
        response_id = f"resp_{uuid.uuid4().hex}"
        return StreamingResponse(
            responses_sse(result, response_id, payload.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return responses_response(result)


async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
