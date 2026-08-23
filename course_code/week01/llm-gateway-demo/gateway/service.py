from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from gateway.errors import GatewayError
from gateway.policies import ModelRateLimiter, RetryPolicy, normalize_provider_exception
from gateway.registry import AdapterRegistry
from gateway.schemas import Message, StreamEvent, UnifiedRequest, UnifiedResponse, Usage, UsageRecord
from gateway.storage import GatewayStore
from gateway.structured import validate_structured_text, validate_response_format
from gateway.templates import PromptTemplateEngine


class CompletionService:
    def __init__(
        self,
        registry: AdapterRegistry,
        store: GatewayStore,
        templates: PromptTemplateEngine,
        limiter: ModelRateLimiter,
        retry_policy: RetryPolicy,
    ):
        self.registry = registry
        self.store = store
        self.templates = templates
        self.limiter = limiter
        self.retry_policy = retry_policy

    async def generate(
        self, request: UnifiedRequest
    ) -> UnifiedResponse | AsyncIterator[StreamEvent]:
        started = time.perf_counter()
        registered = None
        retries = 0
        try:
            validate_response_format(request.response_format)
            registered = self.registry.resolve(request.model)
            await self.limiter.acquire(request.model)
            request = await self._apply_template(request)
            if request.stream:
                return self._stream_with_retry(request, registered.adapter, started)

            result, retries = await self._nonstream_with_retry(request, registered.adapter)
            if request.response_format:
                result.output_text = validate_structured_text(
                    result.output_text, request.response_format
                )
            await self._record_success(request, result, started, retries)
            return result
        except Exception as raw:
            error = normalize_provider_exception(raw)
            if registered is not None:
                await self._record_error(request, registered.adapter.provider, started, retries, error)
            raise error

    async def _apply_template(self, request: UnifiedRequest) -> UnifiedRequest:
        if not request.template:
            return request
        rendered = await self.templates.render(request.template, request.variables)
        return request.model_copy(
            update={
                "messages": [
                    Message(role="system", content=rendered),
                    *request.messages,
                ],
            }
        )

    async def _nonstream_with_retry(self, request: UnifiedRequest, adapter: Any):
        retries = 0
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                result = await adapter.generate(request)
                if isinstance(result, UnifiedResponse):
                    return result, retries
                raise GatewayError("internal_error", "Provider returned an unexpected stream result.")
            except Exception as raw:
                error = normalize_provider_exception(raw)
                if not error.retryable or attempt >= self.retry_policy.max_retries:
                    raise error
                retries += 1
                await asyncio.sleep(self.retry_policy.delay(retries))
        raise GatewayError("internal_error", "Retry loop exited unexpectedly.")

    async def _stream_with_retry(
        self, request: UnifiedRequest, adapter: Any, started: float
    ) -> AsyncIterator[StreamEvent]:
        retries = 0
        usage = Usage()
        ttft_ms: float | None = None
        collected_text: list[str] = []
        pending_completed: StreamEvent | None = None

        for attempt in range(self.retry_policy.max_retries + 1):
            emitted = False
            try:
                stream = await adapter.generate(request)
                if isinstance(stream, UnifiedResponse):
                    if request.response_format:
                        stream.output_text = validate_structured_text(
                            stream.output_text, request.response_format
                        )
                    yield StreamEvent(type="text.delta", text=stream.output_text)
                    yield StreamEvent(type="completed", finish_reason=stream.finish_reason)
                    usage = stream.usage
                else:
                    async for event in stream:
                        if event.type in {"text.delta", "reasoning.delta", "tool_call.delta"}:
                            emitted = True
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - started) * 1000
                        if event.type == "text.delta" and event.text:
                            collected_text.append(event.text)
                        if event.usage:
                            usage = event.usage
                        if event.type == "completed":
                            pending_completed = event
                            continue
                        yield event

                    if request.response_format:
                        validate_structured_text("".join(collected_text), request.response_format)
                    if pending_completed is not None:
                        yield pending_completed

                await self._record_usage(
                    request,
                    provider=adapter.provider,
                    status="success",
                    status_code=200,
                    error_code=None,
                    usage=usage,
                    started=started,
                    ttft_ms=ttft_ms,
                    retries=retries,
                )
                return
            except Exception as raw:
                error = normalize_provider_exception(raw)
                if emitted or not error.retryable or attempt >= self.retry_policy.max_retries:
                    await self._record_error(
                        request, adapter.provider, started, retries, error, usage, ttft_ms
                    )
                    raise error
                retries += 1
                await asyncio.sleep(self.retry_policy.delay(retries))

    async def _record_success(
        self, request: UnifiedRequest, response: UnifiedResponse, started: float, retries: int
    ) -> None:
        await self._record_usage(
            request,
            response.provider,
            "success",
            200,
            None,
            response.usage,
            started,
            (time.perf_counter() - started) * 1000,
            retries,
        )

    async def _record_error(
        self,
        request: UnifiedRequest,
        provider: str,
        started: float,
        retries: int,
        error: GatewayError,
        usage: Usage | None = None,
        ttft_ms: float | None = None,
    ) -> None:
        await self._record_usage(
            request,
            provider,
            "error",
            error.status_code,
            error.code,
            usage or Usage(),
            started,
            ttft_ms,
            retries,
        )

    async def _record_usage(
        self,
        request: UnifiedRequest,
        provider: str,
        status: str,
        status_code: int,
        error_code: str | None,
        usage: Usage,
        started: float,
        ttft_ms: float | None,
        retries: int,
    ) -> None:
        await self.store.record_usage(
            UsageRecord(
                request_id=request.request_id,
                model=request.model,
                provider=provider,
                endpoint=str(request.metadata.get("endpoint", "unknown")),
                stream=request.stream,
                status=status,
                status_code=status_code,
                error_code=error_code,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cached_prompt_tokens=usage.cached_prompt_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=(time.perf_counter() - started) * 1000,
                ttft_ms=ttft_ms,
                retries=retries,
            )
        )
