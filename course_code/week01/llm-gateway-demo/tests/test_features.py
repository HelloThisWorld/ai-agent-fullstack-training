import asyncio
import tempfile
import unittest
from collections.abc import AsyncIterator

from gateway.errors import ProviderRequestError, RateLimitExceededError, StructuredOutputError
from gateway.policies import ModelRateLimiter, RetryPolicy
from gateway.registry import AdapterRegistry
from gateway.schemas import ProviderCapabilities, StreamEvent, UnifiedRequest, UnifiedResponse, Usage
from gateway.service import CompletionService
from gateway.storage import GatewayStore
from gateway.templates import PromptTemplateEngine


class FakeAdapter:
    provider = "fake"

    def __init__(self, failures: int = 0, output: str = "hello"):
        self.failures = failures
        self.calls = 0
        self.output = output

    async def generate(self, request: UnifiedRequest):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderRequestError(self.provider, "temporary", status_code=503)
        if request.stream:
            async def stream() -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type="text.delta", text=self.output)
                yield StreamEvent(
                    type="usage",
                    usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                )
                yield StreamEvent(type="completed", finish_reason="stop")

            return stream()
        return UnifiedResponse(
            model=request.model,
            provider=self.provider,
            output_text=self.output,
            usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    async def health_check(self) -> bool:
        return True

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, json_schema=True)


def make_service(
    store: GatewayStore,
    adapter: FakeAdapter,
    *,
    requests: int = 30,
    max_retries: int = 3,
) -> CompletionService:
    registry = AdapterRegistry()
    registry.register("fake", adapter, "test")
    return CompletionService(
        registry,
        store,
        PromptTemplateEngine(store),
        ModelRateLimiter(requests, 60),
        RetryPolicy(max_retries=max_retries, base_delay_seconds=0, max_delay_seconds=0),
    )


class FeatureTests(unittest.TestCase):
    def run_async(self, coroutine):
        return asyncio.run(coroutine)

    def test_template_references_and_variables(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = GatewayStore(f"{directory}/gateway.sqlite3")
                await store.initialize()
                await store.upsert_template("base", "You are {{ persona }}.")
                await store.upsert_template("child", "{{>base}}\nAnswer {{ user.name }}.")
                rendered = await PromptTemplateEngine(store).render(
                    "child", {"persona": "helpful", "user": {"name": "Ada"}}
                )
                self.assertEqual(rendered, "You are helpful.\nAnswer Ada.")

        self.run_async(run())

    def test_json_response_format_is_validated(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = GatewayStore(f"{directory}/gateway.sqlite3")
                await store.initialize()
                service = make_service(store, FakeAdapter(output='{"answer":"ok"}'))
                result = await service.generate(
                    UnifiedRequest(
                        model="fake/model",
                        messages=[{"role": "user", "content": "json"}],
                        response_format={"type": "json_object"},
                    )
                )
                self.assertEqual(result.output_text, '{"answer":"ok"}')

        self.run_async(run())

    def test_json_schema_response_format_and_invalid_output(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = GatewayStore(f"{directory}/gateway.sqlite3")
                await store.initialize()
                schema_format = {
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
                }
                service = make_service(store, FakeAdapter(output='{"answer":"ok"}'))
                result = await service.generate(
                    UnifiedRequest(
                        model="fake/schema",
                        messages=[{"role": "user", "content": "json"}],
                        response_format=schema_format,
                    )
                )
                self.assertEqual(result.output_text, '{"answer":"ok"}')

                invalid_service = make_service(store, FakeAdapter(output='{"wrong":1}'))
                with self.assertRaises(StructuredOutputError):
                    await invalid_service.generate(
                        UnifiedRequest(
                            model="fake/invalid-schema",
                            messages=[{"role": "user", "content": "json"}],
                            response_format=schema_format,
                        )
                    )

        self.run_async(run())

    def test_retry_and_usage_record(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = GatewayStore(f"{directory}/gateway.sqlite3")
                await store.initialize()
                adapter = FakeAdapter(failures=2)
                service = make_service(store, adapter)
                result = await service.generate(
                    UnifiedRequest(
                        model="fake/model",
                        messages=[{"role": "user", "content": "retry"}],
                    )
                )
                usage = await store.usage()
                self.assertEqual(result.output_text, "hello")
                self.assertEqual(adapter.calls, 3)
                self.assertEqual(usage["records"][0]["retries"], 2)

        self.run_async(run())

    def test_streaming_records_ttft_and_categories(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = GatewayStore(f"{directory}/gateway.sqlite3")
                await store.initialize()
                service = make_service(store, FakeAdapter())
                stream = await service.generate(
                    UnifiedRequest(
                        model="fake/model",
                        messages=[{"role": "user", "content": "stream"}],
                        stream=True,
                    )
                )
                events = [event async for event in stream]
                usage = await store.usage()
                self.assertEqual(events[0].type, "text.delta")
                self.assertIsNotNone(usage["records"][0]["ttft_ms"])
                self.assertEqual(usage["records"][0]["prompt_tokens"], 3)

        self.run_async(run())

    def test_model_rate_limit_returns_492_and_is_model_scoped(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                store = GatewayStore(f"{directory}/gateway.sqlite3")
                await store.initialize()
                service = make_service(store, FakeAdapter(), requests=1)
                request = UnifiedRequest(
                    model="fake/model-a", messages=[{"role": "user", "content": "one"}]
                )
                await service.generate(request)
                with self.assertRaises(RateLimitExceededError) as caught:
                    await service.generate(request.model_copy(update={"request_id": "second"}))
                self.assertEqual(caught.exception.status_code, 492)
                other = await service.generate(
                    UnifiedRequest(
                        model="fake/model-b", messages=[{"role": "user", "content": "two"}]
                    )
                )
                self.assertEqual(other.output_text, "hello")

        self.run_async(run())


if __name__ == "__main__":
    unittest.main()
