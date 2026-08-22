from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx

from gateway.schemas import (
    ProviderCapabilities,
    StreamEvent,
    UnifiedRequest,
    UnifiedResponse,
)


class LLMAdapter(ABC):
    provider: str

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @abstractmethod
    async def generate(
        self, request: UnifiedRequest
    ) -> UnifiedResponse | AsyncIterator[StreamEvent]:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError
