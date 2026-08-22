from __future__ import annotations

import httpx

from gateway.adapters.openai import OpenAIAdapter
from gateway.schemas import ProviderCapabilities


class DeepSeekAdapter(OpenAIAdapter):
    provider = "deepseek"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(tools=True, json_schema=True)
