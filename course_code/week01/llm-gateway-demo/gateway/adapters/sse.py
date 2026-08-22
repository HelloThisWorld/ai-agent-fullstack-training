from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


async def iter_sse(response: httpx.Response) -> AsyncIterator[Any]:
    """Yield decoded `data:` payloads from an SSE response."""

    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                return


def error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            return str(error.get("message", error))
        return str(error)
    return str(payload)
