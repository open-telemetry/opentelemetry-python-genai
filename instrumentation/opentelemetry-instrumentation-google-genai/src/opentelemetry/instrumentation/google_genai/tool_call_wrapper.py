# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import json
import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

from google.genai.types import (
    ToolListUnion,
    ToolListUnionDict,
    ToolOrDict,
)

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.utils import bind_arguments
from opentelemetry.util.types import AnyValue

ToolFunction = Callable[..., Any]
_logger = logging.getLogger(__name__)


def _is_primitive(value):
    return isinstance(value, (str, int, bool, float))


def _to_otel_value(python_value):
    """Coerces parameters to something representable with Open Telemetry."""
    if python_value is None or _is_primitive(python_value):
        return python_value
    if isinstance(python_value, list):
        return [_to_otel_value(x) for x in python_value]
    if isinstance(python_value, dict):
        return {
            key: _to_otel_value(val) for (key, val) in python_value.items()
        }
    if hasattr(python_value, "model_dump"):
        return python_value.model_dump()
    if hasattr(python_value, "__dict__"):
        return _to_otel_value(python_value.__dict__)
    return repr(python_value)


def _snapshot_tool_arguments(
    tool_function: ToolFunction,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, AnyValue] | None:
    # ToolInvocation serializes at span end, after the tool may mutate its inputs.
    try:
        bound = deepcopy(
            bind_arguments(tool_function, args, kwargs, apply_defaults=False)
        )
        return {
            name: _normalize_tool_argument(value)
            for name, value in bound.items()
        }
    except Exception:
        _logger.warning("Failed to snapshot tool arguments", exc_info=True)
        return None


def _normalize_tool_argument(value: object) -> AnyValue:
    if value is None or isinstance(value, (str, int, bool, float, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_tool_argument(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_tool_argument(item) for key, item in value.items()
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalize_tool_argument(model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_tool_argument(asdict(value))
    if hasattr(value, "__dict__"):
        return _normalize_tool_argument(value.__dict__)
    return _to_otel_value(value)


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
                # Do this before calling the tool in case that crashes.
                if tool_invocation.should_capture_content:
                    tool_invocation.arguments = _snapshot_tool_arguments(
                        tool_function, args, kwargs
                    )
                result = await tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content:
                    tool_invocation.tool_result = json.dumps(
                        _to_otel_value(result)
                    )
            return result

        return async_wrapped_function
    else:

        @functools.wraps(tool_function)
        def wrapped_function(*args: object, **kwargs: object) -> Any:
            with telemetry_handler.tool(
                tool_function.__name__,
            ) as tool_invocation:
                tool_invocation.tool_description = tool_function.__doc__
                # Do this before calling the tool in case that crashes.
                if tool_invocation.should_capture_content:
                    tool_invocation.arguments = _snapshot_tool_arguments(
                        tool_function, args, kwargs
                    )
                result = tool_function(*args, **kwargs)
                if tool_invocation.should_capture_content:
                    tool_invocation.tool_result = json.dumps(
                        _to_otel_value(result)
                    )
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
