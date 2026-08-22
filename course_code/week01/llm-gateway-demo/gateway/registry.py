from __future__ import annotations

from dataclasses import dataclass

from gateway.adapters.base import LLMAdapter
from gateway.errors import UnsupportedModelError


@dataclass(frozen=True, slots=True)
class RegisteredAdapter:
    prefix: str
    adapter: LLMAdapter
    description: str


class AdapterRegistry:
    def __init__(self) -> None:
        self._items: dict[str, RegisteredAdapter] = {}

    def register(self, prefix: str, adapter: LLMAdapter, description: str) -> None:
        self._items[prefix] = RegisteredAdapter(prefix, adapter, description)

    def resolve(self, model: str) -> RegisteredAdapter:
        prefix = model.split("/", 1)[0] if "/" in model else model
        item = self._items.get(prefix)
        if item is None:
            raise UnsupportedModelError(model)
        return item

    def models(self, local_model: str) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for prefix, item in self._items.items():
            model_id = local_model if prefix == "local" else f"{prefix}/*"
            result.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": item.adapter.provider,
                    "description": item.description,
                }
            )
        return result

    def items(self) -> list[RegisteredAdapter]:
        return list(self._items.values())
