from __future__ import annotations

import re
from typing import Any

from gateway.errors import TemplateError
from gateway.schemas import Message
from gateway.storage import GatewayStore


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REFERENCE = re.compile(r"{{\s*>\s*([A-Za-z0-9][A-Za-z0-9._-]{0,63})\s*}}")
_VARIABLE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*}}")


def validate_template_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise TemplateError(
            "invalid_template_name",
            "Template names must contain only letters, numbers, '.', '_' or '-'.",
            status_code=400,
        )
    return name


class PromptTemplateEngine:
    def __init__(self, store: GatewayStore):
        self.store = store

    async def render(self, name: str, variables: dict[str, Any] | None = None) -> str:
        validate_template_name(name)
        return await self._render_named(name, variables or {}, [])

    async def _render_named(
        self, name: str, variables: dict[str, Any], stack: list[str]
    ) -> str:
        if name in stack:
            cycle = " -> ".join([*stack, name])
            raise TemplateError("template_reference_cycle", f"Template reference cycle: {cycle}")
        template = await self.store.get_template(name)
        if template is None:
            raise TemplateError("template_not_found", f"Template '{name}' was not found.", 404)
        return await self._render_content(str(template["content"]), variables, [*stack, name])

    async def _render_content(
        self, content: str, variables: dict[str, Any], stack: list[str]
    ) -> str:
        async def replace_reference(match: re.Match[str]) -> str:
            return await self._render_named(match.group(1), variables, stack)

        references = list(_REFERENCE.finditer(content))
        if references:
            chunks: list[str] = []
            cursor = 0
            for reference in references:
                chunks.append(content[cursor : reference.start()])
                chunks.append(await replace_reference(reference))
                cursor = reference.end()
            chunks.append(content[cursor:])
            content = "".join(chunks)

        def replace_variable(match: re.Match[str]) -> str:
            path = match.group(1)
            value: Any = variables
            for segment in path.split("."):
                if not isinstance(value, dict) or segment not in value:
                    raise TemplateError(
                        "template_variable_missing",
                        f"Variable '{path}' was not provided.",
                    )
                value = value[segment]
            if isinstance(value, (dict, list)):
                import json

                return json.dumps(value, ensure_ascii=False)
            return str(value)

        return _VARIABLE.sub(replace_variable, content)

    async def render_as_system_message(
        self, name: str, variables: dict[str, Any] | None = None
    ) -> Message:
        return Message(role="system", content=await self.render(name, variables))
