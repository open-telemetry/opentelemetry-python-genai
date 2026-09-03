# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for DSPy instrumentation."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy, deepcopy
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from wrapt import (
    BoundFunctionWrapper,
    FunctionWrapper,
    apply_patch,
    resolve_path,
)

from opentelemetry.instrumentation.genai.dspy.utils import (
    extract_input_content,
    extract_output_content,
    prepare_tool_definitions,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    RetrievalInvocation,
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
    from dspy.retrievers.retrieve import Retrieve

_REACT_MODULE = "dspy.predict.react"
_REACT_CLASS = "ReAct"

_REACT_V2_MODULE = "dspy.predict.react_v2"
_REACT_V2_CLASS = "ReActV2"


if TYPE_CHECKING:
    _BoundFunctionWrapper = BoundFunctionWrapper[Any, Any]
    _FunctionWrapper = FunctionWrapper[Any, Any]
else:
    _BoundFunctionWrapper = BoundFunctionWrapper
    _FunctionWrapper = FunctionWrapper


class _CopyableBoundFunctionWrapper(_BoundFunctionWrapper):
    """BoundFunctionWrapper that supports copy and deepcopy."""

    def __init__(
        self,
        wrapped: Any,
        instance: Any = None,
        wrapper: Any = None,
        enabled: Any = None,
        binding: str = "callable",
        parent: Any = None,
        owner: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        init_args: tuple[Any, ...] = (
            wrapped,
            instance,
            wrapper,
            enabled,
            binding,
            parent,
            owner,
            *args,
        )
        super().__init__(*init_args, **kwargs)

    def __copy__(self) -> _CopyableBoundFunctionWrapper:
        return _CopyableBoundFunctionWrapper(
            self.__wrapped__,
            self._self_instance,
            self._self_wrapper,
            self._self_enabled,
            self._self_binding,
            self._self_parent,
            self._self_owner,
        )

    def __deepcopy__(self, memo: dict[Any, Any]) -> Any:
        if self._self_instance is not None:
            copied_instance: Any = deepcopy(self._self_instance, memo)
            attr_name: str | None = getattr(self, "__name__", None)
            if attr_name and hasattr(copied_instance, attr_name):
                return getattr(copied_instance, attr_name)
            return _CopyableBoundFunctionWrapper(
                deepcopy(self.__wrapped__, memo),
                copied_instance,
                self._self_wrapper,
                self._self_enabled,
                self._self_binding,
                self._self_parent,
                self._self_owner,
            )
        return _CopyableBoundFunctionWrapper(
            deepcopy(self.__wrapped__, memo),
            None,
            self._self_wrapper,
            self._self_enabled,
            self._self_binding,
            self._self_parent,
            self._self_owner,
        )


class _CopyableFunctionWrapper(_FunctionWrapper):
    """FunctionWrapper that supports copy and deepcopy."""

    __bound_function_wrapper__ = _CopyableBoundFunctionWrapper

    def __copy__(self) -> _CopyableFunctionWrapper:
        wrapped: Any = self.__wrapped__
        wrapper: Any = self._self_wrapper
        return _CopyableFunctionWrapper(
            copy(wrapped),
            wrapper,
            self._self_enabled,
        )

    def __deepcopy__(self, memo: dict[Any, Any]) -> _CopyableFunctionWrapper:
        wrapped: Any = self.__wrapped__
        wrapper: Any = self._self_wrapper
        return _CopyableFunctionWrapper(
            deepcopy(wrapped, memo),
            wrapper,
            self._self_enabled,
        )


def _wrap_function(
    target: Any,
    name: str,
    wrapper: Callable[..., Any],
) -> None:
    """Wrap a target attribute with _CopyableFunctionWrapper.

    Resolves target if given as a module name string, traverses dotted attribute
    paths to find the owning object, and applies the wrapper.
    """
    if isinstance(target, str):
        target = import_module(target)

    parent, attribute, original = resolve_path(target, name)
    wrapped = _CopyableFunctionWrapper(original, wrapper)
    apply_patch(parent, attribute, wrapped)


def patch_dspy(handler: TelemetryHandler) -> None:
    """Apply patches to DSPy Tool, ReAct, and Retrieve classes."""
    import dspy
    import dspy.predict.react

    tool_module = dspy.Tool.__module__
    tool_name = dspy.Tool.__name__
    _wrap_function(
        tool_module,
        f"{tool_name}.__call__",
        _tool_call(handler),
    )
    if hasattr(dspy.Tool, "acall"):
        _wrap_function(
            tool_module,
            f"{tool_name}.acall",
            _tool_acall(handler),
        )

    retrieve_module = dspy.Retrieve.__module__
    retrieve_name = dspy.Retrieve.__name__
    _wrap_function(
        retrieve_module,
        f"{retrieve_name}.forward",
        _retrieve_forward(handler),
    )
    if hasattr(dspy.Retrieve, "aforward"):
        _wrap_function(
            retrieve_module,
            f"{retrieve_name}.aforward",
            _retrieve_aforward(handler),
        )

    _wrap_function(
        _REACT_MODULE,
        f"{_REACT_CLASS}.forward",
        _react_forward(handler, "dspy.ReAct"),
    )
    if hasattr(dspy.predict.react.ReAct, "aforward"):
        _wrap_function(
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
                _wrap_function(
                    _REACT_V2_MODULE,
                    f"{_REACT_V2_CLASS}.forward",
                    _react_forward(handler, "dspy.ReActV2"),
                )
            if "aforward" in getattr(react_v2_cls, "__dict__", {}):
                _wrap_function(
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

    unwrap(dspy.Retrieve, "forward")
    if hasattr(dspy.Retrieve, "aforward"):
        unwrap(dspy.Retrieve, "aforward")

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


def _extract_retrieval_query(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    if "query" in kwargs and kwargs["query"] is not None:
        return str(kwargs["query"])
    if args and args[0] is not None:
        return str(args[0])
    return None


def _extract_retrieval_k(
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> int | None:
    k = kwargs.get("k")
    if k is None and len(args) > 1:
        k = args[1]
    if k is None and hasattr(instance, "k"):
        k = getattr(instance, "k", None)
    if k is not None:
        try:
            return int(k)
        except (ValueError, TypeError):
            return None
    return None


def _start_retrieval_invocation(
    handler: TelemetryHandler,
    instance: Retrieve,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> RetrievalInvocation:
    rm: Any = getattr(instance, "rm", None)
    if rm is None:
        import dspy

        rm = getattr(dspy.settings, "rm", None)

    # DSPy retrieval models lack a uniform identifier schema, so inspect common index and provider
    # attributes across the Retrieve instance and configured RM.
    data_source_id: str | None = (
        getattr(instance, "data_source_id", None)
        or getattr(instance, "index_name", None)
        or (getattr(rm, "data_source_id", None) if rm is not None else None)
        or (getattr(rm, "index_name", None) if rm is not None else None)
        or (getattr(rm, "collection_name", None) if rm is not None else None)
    )
    provider: str | None = (
        getattr(instance, "provider", None)
        or (getattr(rm, "provider", None) if rm is not None else None)
        or (getattr(rm, "provider_name", None) if rm is not None else None)
    )

    invocation = handler.retrieval(
        data_source_id=str(data_source_id)
        if data_source_id is not None
        else None,
        provider=str(provider) if provider is not None else None,
    )

    invocation.query_text = _extract_retrieval_query(args, kwargs)
    invocation.top_k = _extract_retrieval_k(instance, args, kwargs)
    return invocation


def _set_retrieval_invocation_documents(
    handler: TelemetryHandler,
    invocation: RetrievalInvocation,
    result: object,
) -> None:
    if not handler.should_capture_content():
        return

    passages: Sequence[object] | None = None
    if hasattr(result, "passages"):
        attr_val = getattr(result, "passages")
        if isinstance(attr_val, Sequence) and not isinstance(
            attr_val, (str, bytes)
        ):
            passages = cast(Sequence[object], attr_val)
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        passages = cast(Sequence[object], result)
    elif isinstance(result, str):
        passages = [result]

    if passages is None:
        return

    documents: list[dict[str, Any]] = []
    for psg in passages:
        # Standard dspy.Retrieve flattens RM passages to strings via psg.long_text.
        if isinstance(psg, str):
            documents.append({"content": psg})
        elif isinstance(psg, Mapping):
            mapping_psg = cast(Mapping[str, Any], psg)
            content_val = mapping_psg.get("long_text") or mapping_psg.get(
                "content"
            )
            doc: dict[str, Any] = {
                "content": str(
                    content_val if content_val is not None else mapping_psg
                )
            }
            if "id" in mapping_psg:
                doc["id"] = str(mapping_psg["id"])
            if "score" in mapping_psg:
                try:
                    doc["score"] = float(mapping_psg["score"])
                except (ValueError, TypeError):
                    pass
            documents.append(doc)
        elif hasattr(psg, "long_text"):
            long_text = getattr(psg, "long_text")
            doc = {"content": str(long_text)}
            if hasattr(psg, "id"):
                doc["id"] = str(getattr(psg, "id"))
            if hasattr(psg, "score"):
                try:
                    doc["score"] = float(getattr(psg, "score"))
                except (ValueError, TypeError):
                    pass
            documents.append(doc)
        else:
            documents.append({"content": str(psg)})

    invocation.documents = documents


def _retrieve_forward(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Retrieve,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_retrieval_invocation(
            handler, instance, args, kwargs
        )
        with invocation:
            result = wrapped(*args, **kwargs)
            _set_retrieval_invocation_documents(handler, invocation, result)
            return result

    return traced_method


def _retrieve_aforward(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Retrieve,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_retrieval_invocation(
            handler, instance, args, kwargs
        )
        with invocation:
            result = await wrapped(*args, **kwargs)
            _set_retrieval_invocation_documents(handler, invocation, result)
            return result

    return traced_method
