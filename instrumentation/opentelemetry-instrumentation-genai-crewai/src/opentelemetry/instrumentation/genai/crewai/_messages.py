# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert CrewAI values to the public ``opentelemetry-util-genai`` models.

Every converter is total: it never raises, and on any failure it logs at
debug level and returns an empty or ``None`` result so that telemetry
extraction can never break the instrumented application call. Agents are
read attribute by attribute rather than serialized, because CrewAI's models
reference each other cyclically.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeAlias, cast

from crewai.lite_agent_output import LiteAgentOutput
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.types import LLMMessage
from pydantic import BaseModel

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericPart,
    InputMessage,
    MessagePart,
    OutputMessage,
    TextPart,
    ToolDefinition,
)
from opentelemetry.util.types import AnyValue

_logger = logging.getLogger(__name__)

CrewAITool: TypeAlias = BaseTool | CrewStructuredTool
"""Either CrewAI tool representation: the authored tool or its ReAct adapter."""

# CrewAI < 1.15 rewrites ``BaseTool.description`` at construction time into
# the LLM-facing block "Tool Name: ...\nTool Arguments: ...\nTool
# Description: <authored>"; newer versions keep the authored text and
# compose it on demand.
_COMPOSITE_DESCRIPTION_RE = re.compile(
    r"^Tool Name:.*\nTool Arguments:.*\nTool Description:\s*", re.DOTALL
)


def tool_description(tool: CrewAITool) -> str:
    """Return the authored description of a CrewAI tool.

    Args:
        tool: A ``BaseTool`` or ``CrewStructuredTool`` instance.

    Returns:
        The tool's ``description`` with any prompt-facing composite prefix
        removed.
    """
    return _COMPOSITE_DESCRIPTION_RE.sub("", tool.description, count=1)


def bind_call_arguments(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bind an intercepted call without invoking or validating it.

    Args:
        wrapped: The original callable whose signature is used for binding.
        args: Positional arguments of the intercepted call.
        kwargs: Keyword arguments of the intercepted call.

    Returns:
        Parameter names mapped to values, with defaults applied, so that
        positional and keyword call styles are recorded identically. Empty if
        the arguments do not match the signature.
    """
    try:
        bound = inspect.signature(wrapped).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        _logger.debug("Failed to bind CrewAI call arguments", exc_info=True)
        return {}


def crewai_tools(value: object) -> list[CrewAITool] | None:
    """Narrow a dynamically bound ``tools`` argument to CrewAI tools.

    Args:
        value: The ``tools`` value of an intercepted call.

    Returns:
        The CrewAI tools contained in ``value`` when it is a sequence
        (non-tool entries are skipped), or ``None`` for anything else,
        including ``None`` itself.
    """
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    items = cast(Sequence[object], value)
    return [item for item in items if isinstance(item, CrewAITool)]


def agent_tool_definitions(
    tools: Iterable[CrewAITool] | None,
) -> list[ToolDefinition] | None:
    """Convert CrewAI tools to semantic-convention tool definitions.

    Args:
        tools: The tools available to an agent, or ``None`` when unknown.

    Returns:
        One ``FunctionToolDefinition`` per tool carrying its name, authored
        description and the JSON schema of its ``args_schema`` (``None`` when
        the tool has no schema or generating it fails). ``None`` if ``tools``
        is ``None`` or iterating it fails; an empty list for no tools, which
        the handler omits from the span.
    """
    if tools is None:
        return None
    try:
        definitions: list[ToolDefinition] = []
        for tool in tools:
            parameters: dict[str, Any] | None = None
            schema = tool.args_schema
            if schema is not None:
                try:
                    parameters = schema.model_json_schema()
                except Exception:
                    _logger.debug(
                        "Failed to collect CrewAI tool schema", exc_info=True
                    )
            definitions.append(
                FunctionToolDefinition(
                    name=tool.name,
                    description=tool_description(tool),
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
    task: Task, *, context: str | None = None
) -> list[InputMessage]:
    """Convert a CrewAI task and its optional context to one user message.

    Args:
        task: The ``Task`` being executed; its ``description`` and
            ``expected_output`` are read.
        context: Output of earlier tasks passed to ``execute_task``, if any.

    Returns:
        A single ``user`` ``InputMessage`` whose text parts are the task
        description, ``Expected output: ...`` and ``Context: ...`` (each only
        when present), or an empty list if nothing could be extracted.
    """
    try:
        parts: list[TextPart] = []
        if task.description:
            parts.append(TextPart(content=task.description))
        if task.expected_output:
            parts.append(
                TextPart(content=f"Expected output: {task.expected_output}")
            )
        if context is not None:
            parts.append(TextPart(content=f"Context: {context}"))
        return [InputMessage(role="user", parts=list(parts))] if parts else []
    except Exception:
        _logger.debug("Failed to convert CrewAI task", exc_info=True)
        return []


def messages_to_input_messages(
    messages: str | Sequence[LLMMessage],
) -> list[InputMessage]:
    """Convert standalone ``Agent.kickoff`` messages.

    Args:
        messages: The ``messages`` argument of ``Agent.kickoff``: a plain
            prompt string, or a sequence of ``LLMMessage`` dictionaries.

    Returns:
        A ``user`` message for a string, or one ``InputMessage`` per entry
        that has content (the role defaults to ``user``). String content
        becomes a single text part; a content-part list is mapped block by
        block (see ``_content_parts``).
    """
    try:
        if isinstance(messages, str):
            return [
                InputMessage(role="user", parts=[TextPart(content=messages)])
            ]
        converted: list[InputMessage] = []
        for message in messages:
            content = message["content"]
            if content is None:
                continue
            # Subscripting rather than ``.get``: the TypedDict's ``get``
            # overloads reference an unresolved type and fail strict pyright.
            name: str | None = None
            if "name" in message:
                name = message["name"]
            converted.append(
                InputMessage(
                    role=message["role"] or "user",
                    parts=_content_parts(content),
                    name=name,
                )
            )
        return converted
    except Exception:
        _logger.debug("Failed to convert CrewAI messages", exc_info=True)
        return []


def output_to_output_messages(output: object) -> list[OutputMessage]:
    """Convert an agent result to one assistant output message.

    Args:
        output: The value returned by ``Agent.execute_task`` (a string or a
            structured pydantic model) or by ``Agent.kickoff`` (a
            ``LiteAgentOutput`` whose ``raw`` attribute is used).

    Returns:
        A single ``assistant`` ``OutputMessage`` with the result as text
        (structured results are JSON-encoded), or an empty list if the
        result is ``None``. No finish reason is set: agent completion is
        not a model finish.
    """
    try:
        raw = output.raw if isinstance(output, LiteAgentOutput) else output
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
    tool: BaseTool, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> AnyValue | None:
    """Bind ``BaseTool.run`` arguments against the underlying tool function.

    Args:
        tool: The ``BaseTool`` whose ``_run`` signature names the parameters.
        args: Positional arguments passed to ``run``.
        kwargs: Keyword arguments passed to ``run``.

    Returns:
        The arguments keyed by parameter name when they bind to ``_run``;
        otherwise an ``{"args": [...], "kwargs": {...}}`` envelope so mixed
        calls are still recorded. ``None`` if conversion fails.
    """
    try:
        bound = bind_call_arguments(tool._run, args, kwargs)
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
    """Collect the payload forwarded by ``CrewStructuredTool.invoke``.

    Args:
        wrapped: The original ``invoke`` bound method, used to bind the call.
        args: Positional arguments passed to ``invoke``.
        kwargs: Keyword arguments passed to ``invoke``.

    Returns:
        The ``input`` payload merged with any extra keyword arguments the
        adapter forwards to the tool. A JSON string input is parsed; the
        adapter's ``config`` and CrewAI's ``security_context`` fingerprint
        are excluded. Unparsable input is kept as-is, wrapped in an
        ``{"input": ..., "kwargs": {...}}`` envelope when extra keyword
        arguments are present. ``None`` if conversion fails.
    """
    try:
        bound = bind_call_arguments(wrapped, args, kwargs)
        payload: object = bound.get("input")
        forwarded: object = bound.get("kwargs")
        forwarded_kwargs = (
            dict(cast(Mapping[str, object], forwarded))
            if isinstance(forwarded, Mapping)
            else {}
        )

        if isinstance(payload, Mapping):
            payload_mapping = cast(Mapping[str, object], payload)
            return _to_any_value(
                _without_fingerprint(payload_mapping) | forwarded_kwargs
            )
        if isinstance(payload, str):
            parsed: object
            try:
                parsed = json.loads(payload)
            except (TypeError, ValueError):
                parsed = payload
            if isinstance(parsed, Mapping):
                parsed_mapping = cast(Mapping[str, object], parsed)
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


def _without_fingerprint(payload: Mapping[str, object]) -> dict[str, object]:
    """Drop CrewAI's fingerprint metadata from a structured tool payload.

    CrewAI < 1.15 injects its agent/task fingerprints into the tool input as
    ``security_context``; the tool's args schema drops it before the call, so
    it is never a real argument.

    Args:
        payload: The ``input`` mapping passed to ``CrewStructuredTool.invoke``.

    Returns:
        A copy of ``payload`` without the ``security_context`` key.
    """
    return {
        key: value
        for key, value in payload.items()
        if key != "security_context"
    }


def tool_result_to_result(result: object) -> AnyValue | None:
    """Preserve JSON-safe tool results and stringify other values.

    Args:
        result: The value returned by the tool.

    Returns:
        The result as an ``AnyValue`` (primitives, lists and string-keyed
        mappings are kept, anything else is coerced with ``str``), or
        ``None`` if conversion fails.
    """
    try:
        return _to_any_value(result)
    except Exception:
        _logger.debug("Failed to convert CrewAI tool result", exc_info=True)
        return None


def _content_parts(content: str | list[dict[str, Any]]) -> list[MessagePart]:
    """Map ``LLMMessage`` content to semantic-convention message parts.

    Args:
        content: A plain string, or the provider-style content-part list
            CrewAI builds for multimodal input (``{"type": "text", ...}``,
            ``{"type": "image_url", ...}``, ...).

    Returns:
        One ``TextPart`` for a string or for each ``text`` block; every other
        block becomes a ``GenericPart`` carrying only its ``type``, so media
        keeps its semantic shape without exposing provider payloads. Blocks
        without a type, or ``text`` blocks whose text is not a string, are
        skipped, as CrewAI itself skips them.
    """
    if isinstance(content, str):
        return [TextPart(content=content)]
    parts: list[MessagePart] = []
    for block in content:
        block_type: object = block.get("type")
        if block_type == "text":
            text: object = block.get("text")
            if isinstance(text, str):
                parts.append(TextPart(content=text))
        elif isinstance(block_type, str) and block_type:
            # Media blocks (``image_url``, ``input_audio``, ``file``, ...)
            # only come from the optional ``crewai-files`` extra and are
            # provider-shaped, so they are not yet mapped to ``UriPart`` /
            # ``BlobPart`` / ``FilePart``; only their type is recorded.
            parts.append(GenericPart(type=block_type))
    return parts


def _content_to_text(value: object) -> str:
    """Render an agent result or message content as text.

    Args:
        value: A string, a pydantic model, or any JSON-serializable value.

    Returns:
        Strings unchanged; pydantic models and other values JSON-encoded,
        falling back to ``str(value)``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        try:
            return json.dumps(
                value.model_dump(), ensure_ascii=False, default=str
            )
        except Exception:
            pass
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _to_any_value(value: object) -> AnyValue | None:
    """Recursively coerce a value to the ``AnyValue`` attribute type.

    Args:
        value: Arbitrary Python value.

    Returns:
        Primitives unchanged, mappings with keys coerced to ``str``, lists
        and tuples as lists, and anything else as ``str(value)``.
    """
    if value is None or isinstance(value, (bool, str, bytes, int, float)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _to_any_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[object], value)
        return [_to_any_value(item) for item in sequence]
    return str(value)
