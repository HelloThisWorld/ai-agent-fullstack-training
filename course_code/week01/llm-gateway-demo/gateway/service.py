from __future__ import annotations

from collections.abc import AsyncIterator

from gateway.registry import AdapterRegistry
from gateway.schemas import StreamEvent, UnifiedRequest, UnifiedResponse


class CompletionService:
    def __init__(self, registry: AdapterRegistry):
        self.registry = registry

    async def generate(
        self, request: UnifiedRequest
    ) -> UnifiedResponse | AsyncIterator[StreamEvent]:
        registered = self.registry.resolve(request.model)
        return await registered.adapter.generate(request)
