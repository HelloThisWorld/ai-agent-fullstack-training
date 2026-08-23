from __future__ import annotations

import json
from typing import Any

from gateway.errors import GatewayError, StructuredOutputError


def validate_response_format(response_format: dict[str, Any] | None) -> None:
    if response_format is None:
        return
    format_type = response_format.get("type")
    if format_type == "json_object":
        return
    if format_type == "json_schema":
        schema = response_format.get("schema")
        if schema is None:
            schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise GatewayError(
                "invalid_response_format",
                "json_schema response_format requires a JSON Schema object.",
                status_code=400,
            )
        return
    raise GatewayError(
        "invalid_response_format",
        "response_format.type must be 'json_object' or 'json_schema'.",
        status_code=400,
    )


def _schema(response_format: dict[str, Any]) -> dict[str, Any] | None:
    if response_format.get("type") != "json_schema":
        return None
    return response_format.get("schema") or (response_format.get("json_schema") or {}).get("schema")


def normalize_json_text(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Model output is not valid JSON: {exc.msg}.") from exc

    schema = None
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def validate_structured_text(text: str, response_format: dict[str, Any] | None) -> str:
    validate_response_format(response_format)
    if response_format is None:
        return text
    normalized = normalize_json_text(text)
    schema = _schema(response_format or {})
    if schema is not None:
        try:
            import jsonschema

            jsonschema.validate(json.loads(normalized), schema)
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise StructuredOutputError("JSON Schema validation dependency is unavailable.") from exc
        except Exception as exc:
            if exc.__class__.__name__ == "ValidationError":
                raise StructuredOutputError(f"Model output failed JSON Schema validation: {exc}.") from exc
            raise
    return normalized
