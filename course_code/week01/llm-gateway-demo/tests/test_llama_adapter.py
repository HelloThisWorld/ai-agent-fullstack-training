import asyncio
import json
import unittest

import httpx

from gateway.adapters.llama_cpp import LlamaCppAdapter
from gateway.schemas import StreamEvent, UnifiedRequest


class LlamaAdapterTests(unittest.TestCase):
    def test_non_stream_response_is_normalized(self):
        async def run() -> None:
            def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content)
                self.assertEqual(payload["model"], "qwen3.8-27b-q5ks")
                self.assertEqual(payload["messages"][0]["role"], "user")
                return httpx.Response(
                    200,
                    json={
                        "id": "local-test",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": "hello"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    },
                    request=request,
                )

            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                adapter = LlamaCppAdapter(client, "http://llama.test", "qwen3.8-27b-q5ks")
                result = await adapter.generate(
                    UnifiedRequest(
                        model="local/qwen3.8-27b-q5ks",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                )
                self.assertEqual(result.output_text, "hello")
                self.assertEqual(result.usage.total_tokens, 3)

        asyncio.run(run())

    def test_stream_response_emits_text_and_completed_events(self):
        async def run() -> None:
            def handler(request: httpx.Request) -> httpx.Response:
                body = (
                    b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body,
                    request=request,
                )

            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                adapter = LlamaCppAdapter(client, "http://llama.test", "qwen3.8-27b-q5ks")
                result = await adapter.generate(
                    UnifiedRequest(
                        model="local/qwen3.8-27b-q5ks",
                        messages=[{"role": "user", "content": "hi"}],
                        stream=True,
                    )
                )
                events = [event async for event in result]
                self.assertEqual(events[0].type, "text.delta")
                self.assertEqual(events[0].text, "hi")
                self.assertIsInstance(events[-1], StreamEvent)
                self.assertEqual(events[-1].type, "completed")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
