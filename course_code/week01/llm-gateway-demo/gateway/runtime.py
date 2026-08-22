from __future__ import annotations

import httpx

from gateway.adapters.anthropic import AnthropicAdapter
from gateway.adapters.deepseek import DeepSeekAdapter
from gateway.adapters.llama_cpp import LlamaCppAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.config import Settings
from gateway.registry import AdapterRegistry
from gateway.secrets.windows_credential import WindowsCredentialStore
from gateway.service import CompletionService
from gateway.process_manager import LlamaProcessManager


class GatewayRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process_manager = LlamaProcessManager(settings)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds, connect=10.0)
        )
        secret_store = WindowsCredentialStore()
        self.registry = AdapterRegistry()

        self.registry.register(
            "local",
            LlamaCppAdapter(self.client, settings.local_base_url, settings.local_model),
            "Local llama.cpp model; no provider credential is used.",
        )
        self.registry.register(
            "openai",
            OpenAIAdapter(
                self.client,
                settings.openai_base_url,
                secret_store,
                settings.openai_secret_ref,
            ),
            "OpenAI Chat Completions adapter; credential is external.",
        )
        self.registry.register(
            "anthropic",
            AnthropicAdapter(
                self.client,
                settings.anthropic_base_url,
                secret_store,
                settings.anthropic_secret_ref,
            ),
            "Anthropic Messages adapter; credential is external.",
        )
        self.registry.register(
            "deepseek",
            DeepSeekAdapter(
                self.client,
                settings.deepseek_base_url,
                secret_store,
                settings.deepseek_secret_ref,
            ),
            "DeepSeek Chat Completions adapter; credential is external.",
        )
        self.service = CompletionService(self.registry)

    async def start(self) -> None:
        await self.process_manager.start()

    async def close(self) -> None:
        await self.process_manager.stop()
        await self.client.aclose()
