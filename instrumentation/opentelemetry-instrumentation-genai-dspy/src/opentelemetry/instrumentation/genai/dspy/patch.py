# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for DSPy instrumentation."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.dspy.utils import (
    extract_input_content,
    extract_output_content,
    prepare_tool_definitions,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    TextPart,
)

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.primitives.module import Module
    from dspy.primitives.prediction import Prediction

_REACT_MODULE = "dspy.predict.react"
_REACT_CLASS = "ReAct"

_REACT_V2_MODULE = "dspy.predict.react_v2"
_REACT_V2_CLASS = "ReActV2"


def patch_dspy(handler: TelemetryHandler) -> None:
    """Apply patches to DSPy Tool and ReAct classes."""
    import dspy
    import dspy.predict.react

    tool_module = dspy.Tool.__module__
    tool_name = dspy.Tool.__name__
    wrap_function_wrapper(
        tool_module,
        f"{tool_name}.__call__",
        _tool_call(handler),
    )
    if hasattr(dspy.Tool, "acall"):
        wrap_function_wrapper(
            tool_module,
            f"{tool_name}.acall",
            _tool_acall(handler),
        )

    wrap_function_wrapper(
        _REACT_MODULE,
        f"{_REACT_CLASS}.forward",
        _react_forward(handler, "dspy.ReAct"),
    )
    if hasattr(dspy.predict.react.ReAct, "aforward"):
        wrap_function_wrapper(
            _REACT_MODULE,
            f"{_REACT_CLASS}.aforward",
            _react_aforward(handler, "dspy.ReAct"),
        )

    # ReActV2 was added in DSPy 3.0+; earlier supported versions (2.6.x) do not ship this module.
    try:
        import dspy.predict.react_v2  # pylint: disable=import-outside-toplevel

        react_v2_cls = getattr(dspy.predict.react_v2, "ReActV2", None)
        if react_v2_cls is not None:
            if hasattr(react_v2_cls, "forward"):
                wrap_function_wrapper(
                    _REACT_V2_MODULE,
                    f"{_REACT_V2_CLASS}.forward",
                    _react_forward(handler, "dspy.ReActV2"),
                )
            if "aforward" in getattr(react_v2_cls, "__dict__", {}):
                wrap_function_wrapper(
                    _REACT_V2_MODULE,
                    f"{_REACT_V2_CLASS}.aforward",
                    _react_aforward(handler, "dspy.ReActV2"),
                )
    except (ImportError, AttributeError):
        pass


def unpatch_dspy() -> None:
    """Remove patches from DSPy classes."""
    import dspy
    import dspy.predict.react

    unwrap(dspy.Tool, "__call__")
    if hasattr(dspy.Tool, "acall"):
        unwrap(dspy.Tool, "acall")

    unwrap(dspy.predict.react.ReAct, "forward")
    if hasattr(dspy.predict.react.ReAct, "aforward"):
        unwrap(dspy.predict.react.ReAct, "aforward")

    # ReActV2 was added in DSPy 3.0+; earlier supported versions (2.6.x) do not ship this module.
    try:
        import dspy.predict.react_v2  # pylint: disable=import-outside-toplevel

        react_v2_cls = getattr(dspy.predict.react_v2, "ReActV2", None)
        if react_v2_cls is not None:
            unwrap(react_v2_cls, "forward")
            if "aforward" in getattr(react_v2_cls, "__dict__", {}):
                unwrap(react_v2_cls, "aforward")
    except (ImportError, AttributeError):
        pass


def _extract_tool_arguments(
    instance: Tool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    func: Any = getattr(instance, "func", None)
    if func is not None and callable(func):
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        except (TypeError, ValueError):
            pass

    if args and kwargs:
        return {"args": list(args), "kwargs": dict(kwargs)}
    if kwargs:
        return dict(kwargs)
    if args:
        return list(args)
    return None


def _start_tool_invocation(
    handler: TelemetryHandler,
    instance: Tool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ToolInvocation:
    tool_name: Any = getattr(instance, "name", None)
    if not tool_name:
        func: Any = getattr(instance, "func", None)
        tool_name = getattr(func, "__name__", None)
    if not tool_name:
        tool_name = "tool"

    tool_desc: Any = getattr(instance, "desc", None) or getattr(
        instance, "description", None
    )

    invocation = handler.tool(
        name=str(tool_name),
        tool_type="function",
        tool_description=str(tool_desc) if tool_desc is not None else None,
    )
    invocation.arguments = _extract_tool_arguments(instance, args, kwargs)
    return invocation


def _tool_call(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Tool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_tool_invocation(handler, instance, args, kwargs)
        with invocation:
            result = wrapped(*args, **kwargs)
            invocation.tool_result = result
            return result

    return traced_method


def _tool_acall(handler: TelemetryHandler) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Tool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_tool_invocation(handler, instance, args, kwargs)
        with invocation:
            result = await wrapped(*args, **kwargs)
            invocation.tool_result = result
            return result

    return traced_method


def _start_agent_invocation(
    handler: TelemetryHandler,
    instance: Module,
    kwargs: dict[str, Any],
    agent_name: str,
) -> AgentInvocation:
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    if kwargs:
        content_str = extract_input_content(kwargs)
        invocation.input_messages = [
            InputMessage(role="user", parts=[TextPart(content=content_str)])
        ]

    tools: Any = getattr(instance, "tools", None)
    invocation.tool_definitions = prepare_tool_definitions(tools)
    return invocation


def _set_agent_invocation_output(
    invocation: AgentInvocation,
    instance: Module,
    result: Prediction | None,
) -> None:
    signature: Any = getattr(instance, "signature", None)
    output_str = extract_output_content(result, signature)
    invocation.output_messages = [
        OutputMessage(
            role="assistant",
            parts=[TextPart(content=output_str)],
            finish_reason="stop",
        )
    ]


def _react_forward(
    handler: TelemetryHandler,
    agent_name: str,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_agent_invocation(
            handler, instance, dict(kwargs), agent_name
        )
        with invocation:
            result = wrapped(*args, **kwargs)
            _set_agent_invocation_output(invocation, instance, result)
            return result

    return traced_method


def _react_aforward(
    handler: TelemetryHandler,
    agent_name: str,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_agent_invocation(
            handler, instance, dict(kwargs), agent_name
        )
        with invocation:
            result = await wrapped(*args, **kwargs)
            _set_agent_invocation_output(invocation, instance, result)
            return result

    return traced_method
