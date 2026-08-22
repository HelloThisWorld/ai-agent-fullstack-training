from __future__ import annotations

import asyncio
from urllib.parse import urlparse


class WindowsCredentialStore:
    """Windows Credential Manager backend through the `keyring` package.

    The optional import is intentionally delayed so local llama.cpp-only mode does
    not require a credential backend to be present.
    """

    scheme = "windows-credential"

    @classmethod
    def _parts(cls, secret_ref: str) -> tuple[str, str]:
        parsed = urlparse(secret_ref)
        if parsed.scheme != cls.scheme or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("secret_ref must use windows-credential://service/account")
        return parsed.netloc, parsed.path.strip("/")

    async def get(self, secret_ref: str) -> str | None:
        service, account = self._parts(secret_ref)
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError(
                "The keyring package is required for external provider credentials."
            ) from exc

        return await asyncio.to_thread(keyring.get_password, service, account)

    async def set(self, secret_ref: str, value: str) -> None:
        service, account = self._parts(secret_ref)
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError(
                "The keyring package is required for external provider credentials."
            ) from exc

        await asyncio.to_thread(keyring.set_password, service, account, value)

    async def delete(self, secret_ref: str) -> None:
        service, account = self._parts(secret_ref)
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError(
                "The keyring package is required for external provider credentials."
            ) from exc

        await asyncio.to_thread(keyring.delete_password, service, account)
