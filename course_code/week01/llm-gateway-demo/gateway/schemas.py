from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ContentPart = dict[str, Any]
MessageRole = Literal["system", "user", "assistant", "tool", "developer"]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: MessageRole
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "function"
    function: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    user: str | None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[Message | dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_output_tokens: int | None = Field(default=None, gt=0)
    tools: list[dict[str, Any]] | None = None
    text: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class UnifiedRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    instructions: str | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class UnifiedResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    model: str
    provider: str
    output_text: str = ""
    reasoning_text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)
    created: int = Field(default_factory=lambda: int(time.time()))


class StreamEvent(BaseModel):
    type: Literal[
        "text.delta",
        "reasoning.delta",
        "tool_call.delta",
        "usage",
        "completed",
    ]
    text: str | None = None
    reasoning: str | None = None
    tool_call: dict[str, Any] | list[dict[str, Any]] | None = None
    usage: Usage | None = None
    finish_reason: str | None = None


class ProviderCapabilities(BaseModel):
    chat_completions: bool = True
    responses: bool = True
    streaming: bool = True
    tools: bool = False
    json_schema: bool = False
    vision: bool = False


def new_request_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
