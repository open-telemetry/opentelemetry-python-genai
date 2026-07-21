# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch functions for AgentScope v1 tool instrumentation."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import ToolInvocation

logger = logging.getLogger(__name__)


def _get_tool_description(instance: Any, tool_name: str | None) -> str | None:
    """Get tool description from toolkit."""
    if (
        not tool_name
        or not hasattr(instance, "tools")
        or not isinstance(instance.tools, dict)
    ):
        return None

    tool_obj = instance.tools.get(tool_name)
    if not tool_obj:
        return None

    json_schema = getattr(tool_obj, "json_schema", None)
    if isinstance(json_schema, dict):
        func_dict = json_schema.get("function", {})
        if isinstance(func_dict, dict):
            description = func_dict.get("description")
            if description:
                return description

    return getattr(tool_obj, "description", None)


def _get_tool_result(chunk: Any) -> Any:
    """Extract tool result from chunk."""
    if chunk is None:
        return None
    if hasattr(chunk, "content"):
        return chunk.content
    return chunk


async def _trace_async_generator_wrapper(
    result_generator: Any, invocation: ToolInvocation
) -> Any:
    """Trace tool execution by wrapping the async generator it returns.

    Collects the last yielded chunk to populate the tool result, then
    finalizes the invocation once the generator is fully drained.
    """
    last_chunk = None
    try:
        async for chunk in result_generator:
            last_chunk = chunk
            yield chunk
    except GeneratorExit:
        invocation.stop()
        raise
    except Exception as e:
        invocation.fail(e)
        raise
    else:
        if (
            last_chunk is not None
            and invocation.should_capture_content_on_span
        ):
            result_content = _get_tool_result(last_chunk)
            if result_content is not None:
                invocation.tool_result = result_content
        invocation.stop()


async def wrap_tool_call(
    wrapped: Any,
    instance: Any,
    args: Any,
    kwargs: Any,
    handler: TelemetryHandler,
) -> Any:
    """Async wrapper for ``Toolkit.call_tool_function``."""
    tool_call = args[0] if args else kwargs.get("tool_call", {})
    tool_name = (
        tool_call.get("name", "unknown_tool")
        if isinstance(tool_call, dict)
        else "unknown_tool"
    )
    tool_id = tool_call.get("id") if isinstance(tool_call, dict) else None
    tool_args = (
        tool_call.get("input", {}) if isinstance(tool_call, dict) else {}
    )

    tool_description = _get_tool_description(instance, tool_name)

    # NOTE: tool_type is set to "function" as agentscope currently only
    # supports function-type tools. Update when other types are supported.
    invocation = handler.tool(
        tool_name,
        tool_call_id=tool_id,
        tool_type="function",
        tool_description=tool_description,
    )
    if invocation.should_capture_content_on_span:
        invocation.arguments = tool_args

    try:
        result_generator = await wrapped(*args, **kwargs)
    except Exception as error:
        invocation.fail(error)
        raise

    return _trace_async_generator_wrapper(result_generator, invocation)
