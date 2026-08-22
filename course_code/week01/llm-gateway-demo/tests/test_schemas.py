import unittest

from gateway.schemas import ChatCompletionRequest, Message, ResponsesRequest, UnifiedRequest


class SchemaTests(unittest.TestCase):
    def test_chat_request_normalizes_messages(self):
        request = ChatCompletionRequest(
            model="local/qwen3.8-27b-q5ks",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=16,
        )
        self.assertIsInstance(request.messages[0], Message)
        self.assertEqual(request.messages[0].content, "hello")

    def test_responses_request_accepts_string_input(self):
        request = ResponsesRequest(model="local/qwen3.8-27b-q5ks", input="hello")
        self.assertEqual(request.input, "hello")

    def test_unified_request_has_no_secret_fields(self):
        request = UnifiedRequest(
            model="local/qwen3.8-27b-q5ks",
            messages=[Message(role="user", content="hello")],
        )
        self.assertNotIn("api_key", request.model_dump())
        self.assertNotIn("authorization", request.model_dump())


if __name__ == "__main__":
    unittest.main()
