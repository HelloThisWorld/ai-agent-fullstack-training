from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.api.formatters import chat_response, chat_sse, responses_response, responses_sse
from gateway.errors import GatewayError, TemplateError
from gateway.schemas import (
    ChatCompletionRequest,
    Message,
    ResponsesRequest,
    TemplateRenderRequest,
    TemplateUpsertRequest,
    UnifiedRequest,
)
from gateway.templates import validate_template_name

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
        template=payload.template,
        variables=payload.variables,
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
        chat_template_kwargs=payload.chat_template_kwargs,
        template=payload.template,
        variables=payload.variables,
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


@router.get("/v1/usage")
async def usage(
    request: Request,
    model: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    return await _runtime(request).store.usage(model=model, limit=limit)


@router.get("/v1/templates")
async def templates(request: Request) -> dict[str, Any]:
    return {"data": await _runtime(request).store.list_templates()}


@router.post("/v1/templates")
async def create_template(payload: TemplateUpsertRequest, request: Request) -> dict[str, Any]:
    validate_template_name(payload.name)
    return await _runtime(request).store.upsert_template(
        payload.name, payload.content, payload.description
    )


@router.get("/v1/templates/{name}")
async def get_template(name: str, request: Request) -> dict[str, Any]:
    validate_template_name(name)
    template = await _runtime(request).store.get_template(name)
    if template is None:
        raise TemplateError("template_not_found", f"Template '{name}' was not found.", 404)
    return template


@router.put("/v1/templates/{name}")
async def update_template(
    name: str, payload: TemplateUpsertRequest, request: Request
) -> dict[str, Any]:
    validate_template_name(name)
    if payload.name != name:
        raise TemplateError("template_name_mismatch", "Path name and payload name must match.", 400)
    return await _runtime(request).store.upsert_template(
        name, payload.content, payload.description
    )


@router.post("/v1/templates/{name}/render")
async def render_template(
    name: str, payload: TemplateRenderRequest, request: Request
) -> dict[str, Any]:
    validate_template_name(name)
    rendered = await _runtime(request).service.templates.render(name, payload.variables)
    return {"name": name, "rendered": rendered}


@router.delete("/v1/templates/{name}")
async def delete_template(name: str, request: Request) -> dict[str, Any]:
    validate_template_name(name)
    deleted = await _runtime(request).store.delete_template(name)
    if not deleted:
        raise TemplateError("template_not_found", f"Template '{name}' was not found.", 404)
    return {"deleted": True, "name": name}


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
    payload: dict[str, Any] = {"code": exc.code, "message": exc.message}
    if exc.details:
        payload["details"] = exc.details
    headers = {}
    if "retry_after_seconds" in exc.details:
        headers["Retry-After"] = str(exc.details["retry_after_seconds"])
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": payload},
        headers=headers,
    )
