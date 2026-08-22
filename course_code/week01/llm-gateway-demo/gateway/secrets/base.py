from __future__ import annotations

from typing import Protocol


class SecretStore(Protocol):
    """External secret lookup boundary.

    Implementations must never read credential values from files in the workspace.
    """

    async def get(self, secret_ref: str) -> str | None:
        ...


class EmptySecretStore:
    """Useful for local-only mode where only llama.cpp is enabled."""

    async def get(self, secret_ref: str) -> str | None:
        return None
