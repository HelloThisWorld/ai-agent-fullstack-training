from __future__ import annotations

import argparse
import asyncio
import getpass

from gateway.secrets.windows_credential import WindowsCredentialStore


SECRET_REFS = {
    "openai": "windows-credential://llm-gateway/openai",
    "anthropic": "windows-credential://llm-gateway/anthropic",
    "deepseek": "windows-credential://llm-gateway/deepseek",
}


async def _set(provider: str) -> None:
    value = getpass.getpass(f"Enter the {provider} credential (hidden): ")
    if not value:
        raise SystemExit("Credential input was empty; nothing was written.")
    await WindowsCredentialStore().set(SECRET_REFS[provider], value)
    print(f"Stored the {provider} credential in Windows Credential Manager.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage external LLM credentials outside the workspace.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    set_parser = subparsers.add_parser("set", help="set a provider credential interactively")
    set_parser.add_argument("provider", choices=sorted(SECRET_REFS))
    args = parser.parse_args()

    if args.command == "set":
        asyncio.run(_set(args.provider))


if __name__ == "__main__":
    main()
