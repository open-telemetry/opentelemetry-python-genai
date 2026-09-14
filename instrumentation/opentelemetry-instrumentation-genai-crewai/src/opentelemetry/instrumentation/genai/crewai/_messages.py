# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert CrewAI values to the public ``opentelemetry-util-genai`` models."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    TextPart,
    ToolDefinition,
)
from opentelemetry.util.types import AnyValue

_logger = logging.getLogger(__name__)

# CrewAI < 1.15 rewrites ``BaseTool.description`` at construction time into
# the LLM-facing block "Tool Name: ...\nTool Arguments: ...\nTool
# Description: <authored>"; newer versions keep the authored text and
# compose it on demand.
_COMPOSITE_DESCRIPTION_RE = re.compile(
    r"^Tool Name:.*\nTool Arguments:.*\nTool Description:\s*", re.DOTALL
)


def tool_description(tool: Any) -> str | None:
    """Return the authored description of a CrewAI tool."""
    description = getattr(tool, "description", None)
    if description is None:
        return None
    return _COMPOSITE_DESCRIPTION_RE.sub("", str(description), count=1)


def bind_call_arguments(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bind an intercepted call without invoking or validating it."""
    try:
        bound = inspect.signature(wrapped).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        _logger.debug("Failed to bind CrewAI call arguments", exc_info=True)
        return {}


def agent_tool_definitions(
    tools: Iterable[Any] | None,
) -> list[ToolDefinition] | None:
    """Convert CrewAI tools to semantic-convention tool definitions."""
    if tools is None:
        return None
    try:
        definitions: list[ToolDefinition] = []
        for tool in tools:
            name = str(getattr(tool, "name", None) or type(tool).__name__)
            description = tool_description(tool)
            parameters: Any = None
            schema = getattr(tool, "args_schema", None)
            if schema is not None:
                try:
                    parameters = schema.model_json_schema()
                except Exception:
                    _logger.debug(
                        "Failed to collect CrewAI tool schema", exc_info=True
                    )
            definitions.append(
                FunctionToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )
        return definitions
    except Exception:
        _logger.debug(
            "Failed to collect CrewAI tool definitions", exc_info=True
        )
        return None


def task_to_input_messages(
    task: Any, *, context: Any = None
) -> list[InputMessage]:
    """Convert a CrewAI task and its optional context to one user message."""
    try:
        parts: list[TextPart] = []
        description = getattr(task, "description", None)
        if description is not None:
            parts.append(TextPart(content=str(description)))
        expected_output = getattr(task, "expected_output", None)
        if expected_output is not None:
            parts.append(
                TextPart(content=f"Expected output: {expected_output}")
            )
        if context is not None:
            parts.append(TextPart(content=f"Context: {context}"))
        return [InputMessage(role="user", parts=list(parts))] if parts else []
    except Exception:
        _logger.debug("Failed to convert CrewAI task", exc_info=True)
        return []


def messages_to_input_messages(messages: Any) -> list[InputMessage]:
    """Convert standalone ``Agent.kickoff`` messages."""
    try:
        if isinstance(messages, str):
            return [
                InputMessage(role="user", parts=[TextPart(content=messages)])
            ]
        if not isinstance(messages, (list, tuple)):
            return []

        converted: list[InputMessage] = []
        message_values = cast(list[Any] | tuple[Any, ...], messages)
        for message in message_values:
            if isinstance(message, Mapping):
                message_mapping = cast(Mapping[str, Any], message)
                role = str(message_mapping.get("role") or "user")
                content = message_mapping.get("content")
                name_value = message_mapping.get("name")
            else:
                role = str(getattr(message, "role", None) or "user")
                content = getattr(message, "content", None)
                name_value = getattr(message, "name", None)
            if content is None:
                continue
            converted.append(
                InputMessage(
                    role=role,
                    parts=[TextPart(content=str(content))],
                    name=str(name_value) if name_value is not None else None,
                )
            )
        return converted
    except Exception:
        _logger.debug("Failed to convert CrewAI messages", exc_info=True)
        return []


def output_to_output_messages(output: Any) -> list[OutputMessage]:
    """Convert an agent result to one assistant output message."""
    try:
        raw = getattr(output, "raw", output)
        if raw is None:
            return []
        return [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=_content_to_text(raw))],
            )
        ]
    except Exception:
        _logger.debug("Failed to convert CrewAI output", exc_info=True)
        return []


def tool_args_to_arguments(
    tool: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> AnyValue | None:
    """Bind ``BaseTool.run`` arguments against the underlying tool function."""
    try:
        implementation = getattr(tool, "_run", None)
        if callable(implementation):
            bound = bind_call_arguments(implementation, args, kwargs)
            if bound:
                return _to_any_value(bound)
        return _to_any_value({"args": list(args), "kwargs": dict(kwargs)})
    except Exception:
        _logger.debug("Failed to collect CrewAI tool arguments", exc_info=True)
        return None


def structured_tool_input_to_arguments(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AnyValue | None:
    """Collect the payload forwarded by ``CrewStructuredTool.invoke``."""
    try:
        bound = bind_call_arguments(wrapped, args, kwargs)
        payload = bound.get("input")
        forwarded = bound.get("kwargs")
        forwarded_kwargs: dict[str, Any] = (
            dict(cast(Mapping[str, Any], forwarded))
            if isinstance(forwarded, Mapping)
            else {}
        )

        if isinstance(payload, Mapping):
            payload_mapping = dict(cast(Mapping[str, Any], payload))
            return _to_any_value(
                _without_fingerprint(payload_mapping) | forwarded_kwargs
            )
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except (TypeError, ValueError):
                parsed = payload
            if isinstance(parsed, Mapping):
                parsed_mapping = dict(cast(Mapping[str, Any], parsed))
                return _to_any_value(
                    _without_fingerprint(parsed_mapping) | forwarded_kwargs
                )
            if forwarded_kwargs:
                return _to_any_value(
                    {"input": parsed, "kwargs": forwarded_kwargs}
                )
            return _to_any_value(parsed)
        if forwarded_kwargs:
            return _to_any_value(
                {"input": payload, "kwargs": forwarded_kwargs}
            )
        return _to_any_value(payload)
    except Exception:
        _logger.debug(
            "Failed to collect CrewAI structured tool arguments", exc_info=True
        )
        return None


def _without_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    # CrewAI < 1.15 injects its agent/task fingerprints into the tool input as
    # ``security_context``; the tool's args schema drops it before the call.
    return {
        key: value
        for key, value in payload.items()
        if key != "security_context"
    }


def tool_result_to_result(result: Any) -> AnyValue | None:
    """Preserve JSON-safe tool results and stringify other values."""
    try:
        return _to_any_value(result)
    except Exception:
        _logger.debug("Failed to convert CrewAI tool result", exc_info=True)
        return None


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return json.dumps(model_dump(), ensure_ascii=False, default=str)
        except Exception:
            pass
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _to_any_value(value: Any) -> AnyValue | None:
    if value is None or isinstance(value, (bool, str, bytes, int, float)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _to_any_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return [_to_any_value(item) for item in sequence]
    return str(value)
