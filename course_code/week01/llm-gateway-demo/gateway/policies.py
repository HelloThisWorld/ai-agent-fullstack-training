from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from gateway.errors import GatewayError, RateLimitExceededError


class ModelRateLimiter:
    """Sliding-window request limiter with an independent bucket per model."""

    def __init__(self, requests: int = 30, window_seconds: float = 60.0):
        if requests < 1 or window_seconds <= 0:
            raise ValueError("Rate limiter requests and window must be positive.")
        self.requests = requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def acquire(self, model: str) -> None:
        now = time.monotonic()
        async with self._lock:
            events = self._events[model]
            while events and now - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= self.requests:
                retry_after = self.window_seconds - (now - events[0])
                raise RateLimitExceededError(model, max(retry_after, 0.001))
            events.append(now)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 4.0

    def delay(self, retry_number: int) -> float:
        exponential = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** max(retry_number - 1, 0)),
        )
        return exponential + random.uniform(0, min(0.1, exponential / 4))


def normalize_provider_exception(exc: Exception) -> GatewayError:
    """Convert transport/provider exceptions into stable public error codes."""

    if isinstance(exc, GatewayError):
        return exc

    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return GatewayError(
                "provider_timeout",
                "The provider request timed out.",
                status_code=504,
                retryable=True,
            )
        if isinstance(exc, httpx.HTTPError):
            return GatewayError(
                "provider_unavailable",
                "The provider connection failed.",
                status_code=502,
                retryable=True,
            )
    except ImportError:  # pragma: no cover - dependencies are required at runtime
        pass

    return GatewayError("internal_error", "The gateway encountered an unexpected error.")
