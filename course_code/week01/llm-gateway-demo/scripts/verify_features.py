"""End-to-end verification for the five requested Gateway feature groups.

This script uses an in-memory fake provider and a temporary database. It never reads
or writes API keys, provider credentials, or files inside the project workspace.

Run after installing the project dependencies:

    python scripts/verify_features.py
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.main import create_app
from gateway.registry import AdapterRegistry
from gateway.schemas import ProviderCapabilities, StreamEvent, UnifiedRequest, UnifiedResponse, Usage


class VerificationAdapter:
    provider = "verification"

    def __init__(self) -> None:
        self.calls = 0
        self.last_request: UnifiedRequest | None = None
        self.failures_remaining = 0

    async def generate(self, request: UnifiedRequest):
        self.calls += 1
        self.last_request = request
        if self.failures_remaining:
            self.failures_remaining -= 1
            from gateway.errors import ProviderRequestError

            raise ProviderRequestError("verification", "temporary failure", 503)
        output = json.dumps({"answer": "verified"}, ensure_ascii=False)
        if request.stream:
            async def stream() -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type="text.delta", text=output[:10])
                yield StreamEvent(type="text.delta", text=output[10:])
                yield StreamEvent(
                    type="usage",
                    usage=Usage(prompt_tokens=4, completion_tokens=3, total_tokens=7),
                )
                yield StreamEvent(type="completed", finish_reason="stop")

            return stream()
        return UnifiedResponse(
            model=request.model,
            provider=self.provider,
            output_text=output,
            usage=Usage(prompt_tokens=4, completion_tokens=3, total_tokens=7),
        )

    async def health_check(self) -> bool:
        return True

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, json_schema=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llm-gateway-verify-") as directory:
        app = create_app(
            Settings(
                data_path=f"{directory}/verification.sqlite3",
                rate_limit_requests=1,
                rate_limit_window_seconds=60,
                max_retries=3,
                retry_base_delay_seconds=0,
                retry_max_delay_seconds=0,
            )
        )
        with TestClient(app) as client:
            adapter = VerificationAdapter()
            runtime = app.state.runtime
            runtime.registry._items.clear()
            runtime.registry.register("fake", adapter, "verification adapter")

            assert client.post(
                "/v1/templates",
                json={"name": "base", "content": "You are {{ persona }}."},
            ).status_code == 200
            assert client.post(
                "/v1/templates",
                json={"name": "child", "content": "{{>base}} Answer {{ user.name }}."},
            ).status_code == 200
            rendered = client.post(
                "/v1/templates/child/render",
                json={"variables": {"persona": "helpful", "user": {"name": "Ada"}}},
            ).json()["rendered"]
            assert rendered == "You are helpful. Answer Ada."

            nonstream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake/nonstream",
                    "messages": [{"role": "user", "content": "hello"}],
                    "template": "child",
                    "variables": {"persona": "helpful", "user": {"name": "Ada"}},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            )
            assert nonstream.status_code == 200, nonstream.text
            assert json.loads(nonstream.json()["choices"][0]["message"]["content"])
            assert adapter.last_request is not None
            assert adapter.last_request.messages[0].role == "system"

            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "fake/stream",
                    "messages": [{"role": "user", "content": "stream"}],
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                stream_body = "".join(response.iter_text())
            assert "data: [DONE]" in stream_body
            assert '"content": "{\\"answer\\"' in stream_body

            adapter.failures_remaining = 2
            retry = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake/retry",
                    "messages": [{"role": "user", "content": "retry"}],
                },
            )
            assert retry.status_code == 200
            assert adapter.failures_remaining == 0

            limited_first = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake/limited",
                    "messages": [{"role": "user", "content": "one"}],
                },
            )
            limited_second = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake/limited",
                    "messages": [{"role": "user", "content": "two"}],
                },
            )
            other_model = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake/other",
                    "messages": [{"role": "user", "content": "three"}],
                },
            )
            assert limited_first.status_code == 200
            assert limited_second.status_code == 492
            assert other_model.status_code == 200

            usage = client.get("/v1/usage").json()
            assert usage["records"]
            assert any(record["ttft_ms"] is not None for record in usage["records"] if record["stream"])
            assert any(record["retries"] == 2 for record in usage["records"])
            assert any(record["status_code"] == 492 for record in usage["records"])

    print("Feature verification passed: SSE, JSON output, templates, usage/TTFT, retry, and 492 rate limit.")


if __name__ == "__main__":
    main()
