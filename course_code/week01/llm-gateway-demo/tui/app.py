from __future__ import annotations

import os

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Label, RichLog


class GatewayTui(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #toolbar { height: 3; padding: 0 1; }
    #model { width: 42; }
    #prompt { dock: bottom; margin: 1; }
    #chat { margin: 1; border: solid #31415e; }
    .muted { color: #7284a3; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.gateway_url = os.getenv("LLM_GATEWAY_URL", "http://127.0.0.1:9000")
        self.history: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="toolbar"):
            yield Label("Model: ")
            yield Input("local/qwen3.8-27b-q5ks", id="model")
            yield Label("Gateway client only; provider credentials remain server-side.", classes="muted")
        yield RichLog(id="chat", highlight=True, markup=False)
        yield Input(placeholder="輸入訊息，Enter 發送；Ctrl+Q 離開", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        self.query_one("#chat", RichLog).write("LLM Gateway TUI ready.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt" or not event.value.strip():
            return
        prompt = event.value.strip()
        event.input.value = ""
        model = self.query_one("#model", Input).value.strip()
        chat = self.query_one("#chat", RichLog)
        self.history.append({"role": "user", "content": prompt})
        chat.write(f"You: {prompt}")
        chat.write("Assistant:")

        payload = {"model": model, "messages": self.history, "stream": True, "max_tokens": 512}
        assistant_text = ""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", f"{self.gateway_url}/v1/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        data = __import__("json").loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            assistant_text += text
                            chat.write(text)
            chat.write("")
            self.history.append({"role": "assistant", "content": assistant_text})
        except Exception as exc:
            chat.write(f"\nRequest failed: {exc}")


def main() -> None:
    GatewayTui().run()


if __name__ == "__main__":
    main()
