# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from collections.abc import Callable
from typing import Any

from google.genai.types import (
    ToolListUnion,
    ToolListUnionDict,
    ToolOrDict,
)

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.utils import (
    bind_arguments,
    object_to_any_value,
)

ToolFunction = Callable[..., Any]


def _wrap_tool_function(
    tool_function: ToolFunction,
    telemetry_handler: TelemetryHandler,
) -> ToolFunction:
    if inspect.iscoroutinefunction(tool_function):

        @functools.wraps(tool_function)
        async def async_wrapped_function(
            *args: object, **kwargs: object
        ) -> Any:
            with telemetry_handler.tool(
                tool_function.__name__,
            ) as tool_invocation:
                tool_invocation.tool_description = tool_function.__doc__
                # Snapshot before the call: the tool may mutate its inputs.
                if tool_invocation.should_capture_content:
                    tool_invocation.arguments = object_to_any_value(
                        bind_arguments(tool_function, args, kwargs)
                    )
                result = await tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content:
                    tool_invocation.tool_result = object_to_any_value(result)
            return result

        return async_wrapped_function
    else:

        @functools.wraps(tool_function)
        def wrapped_function(*args: object, **kwargs: object) -> Any:
            with telemetry_handler.tool(
                tool_function.__name__,
            ) as tool_invocation:
                tool_invocation.tool_description = tool_function.__doc__
                # Snapshot before the call: the tool may mutate its inputs.
                if tool_invocation.should_capture_content:
                    tool_invocation.arguments = object_to_any_value(
                        bind_arguments(tool_function, args, kwargs)
                    )
                result = tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content:
                    tool_invocation.tool_result = object_to_any_value(result)
            return result

    return wrapped_function


def wrapped_tool(
    tool_or_tools: ToolFunction
    | ToolOrDict
    | ToolListUnion
    | ToolListUnionDict
    | None,
    telemetry_handler: TelemetryHandler,
):
    if tool_or_tools is None:
        return None
    if isinstance(tool_or_tools, list):
        return [
            wrapped_tool(tool, telemetry_handler) for tool in tool_or_tools
        ]
    if isinstance(tool_or_tools, dict):
        return {
            key: wrapped_tool(tool, telemetry_handler)
            for (key, tool) in tool_or_tools.items()
        }
    if callable(tool_or_tools):
        return _wrap_tool_function(tool_or_tools, telemetry_handler)
    return tool_or_tools
