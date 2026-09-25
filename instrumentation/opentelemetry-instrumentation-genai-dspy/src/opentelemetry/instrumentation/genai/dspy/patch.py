# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for DSPy instrumentation."""

from __future__ import annotations

import sys
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
    SENTINEL_TOOL_NAMES,
    extract_embedding_dimension,
    extract_input_content,
    extract_output_content,
    extract_server_address_and_port,
    is_mapping,
    is_sequence,
    prepare_tool_definitions,
    resolve_embedder_provider_and_model,
    safe_int,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    LocalAgentInvocation,
    RetrievalInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    RetrievalDocument,
    TextPart,
)
from opentelemetry.util.genai.utils import bind_arguments

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.clients.embedding import Embedder
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

    tool_module = dspy.Tool.__module__
    tool_name = dspy.Tool.__name__
    _wrap_function(
        tool_module,
        f"{tool_name}.__call__",
        _tool_call(handler),
    )
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

    _wrap_function(
        _REACT_MODULE,
        f"{_REACT_CLASS}.forward",
        _react_forward(handler, "dspy.ReAct"),
    )
    _wrap_function(
        _REACT_MODULE,
        f"{_REACT_CLASS}.aforward",
        _react_aforward(handler, "dspy.ReAct"),
    )

    react_v2_cls = getattr(
        sys.modules.get(_REACT_V2_MODULE), _REACT_V2_CLASS, None
    )
    if react_v2_cls is not None:
        _wrap_function(
            _REACT_V2_MODULE,
            f"{_REACT_V2_CLASS}.forward",
            _react_forward(handler, "dspy.ReActV2"),
        )
        if hasattr(react_v2_cls, "aforward"):
            _wrap_function(
                _REACT_V2_MODULE,
                f"{_REACT_V2_CLASS}.aforward",
                _react_aforward(handler, "dspy.ReActV2"),
            )

    if hasattr(dspy, "Embedder"):
        embedder_module = dspy.Embedder.__module__
        embedder_name = dspy.Embedder.__name__
        _wrap_function(
            embedder_module,
            f"{embedder_name}.__call__",
            _embedder_call(handler),
        )
        _wrap_function(
            embedder_module,
            f"{embedder_name}.acall",
            _embedder_acall(handler),
        )


def unpatch_dspy() -> None:
    """Remove patches from DSPy classes."""
    import dspy
    import dspy.predict.react

    unwrap(dspy.Tool, "__call__")
    unwrap(dspy.Tool, "acall")

    unwrap(dspy.Retrieve, "forward")

    unwrap(dspy.predict.react.ReAct, "forward")
    unwrap(dspy.predict.react.ReAct, "aforward")

    react_v2_cls = getattr(
        sys.modules.get(_REACT_V2_MODULE), _REACT_V2_CLASS, None
    )
    if react_v2_cls is not None:
        unwrap(react_v2_cls, "forward")
        unwrap(react_v2_cls, "aforward")

    if hasattr(dspy, "Embedder"):
        unwrap(dspy.Embedder, "__call__")
        unwrap(dspy.Embedder, "acall")


def _extract_tool_arguments(
    instance: Tool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    func: Any = getattr(instance, "func", None)
    if func is not None and callable(func):
        bound = bind_arguments(func, args, kwargs, apply_defaults=True)
        if args and bound == kwargs:
            return None
        return bound

    if kwargs and not args:
        return dict(kwargs)
    return None


def _get_tool_name(instance: Tool) -> str:
    tool_name: Any = getattr(instance, "name", None)
    if not tool_name:
        func: Any = getattr(instance, "func", None)
        tool_name = getattr(func, "__name__", None)
    return str(tool_name) if tool_name else "tool"


def _start_tool_invocation(
    handler: TelemetryHandler,
    instance: Tool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ToolInvocation:
    tool_name = _get_tool_name(instance)

    tool_desc: Any = getattr(instance, "desc", None) or getattr(
        instance, "description", None
    )

    invocation = handler.tool(
        name=tool_name,
        tool_type="function",
    )
    if tool_desc is not None:
        invocation.tool_description = str(tool_desc)
    if handler.should_capture_content():
        invocation.arguments = _extract_tool_arguments(instance, args, kwargs)
    return invocation


def _tool_call(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Tool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _get_tool_name(instance) in SENTINEL_TOOL_NAMES:
            return wrapped(*args, **kwargs)

        invocation = _start_tool_invocation(handler, instance, args, kwargs)
        with invocation:
            result = wrapped(*args, **kwargs)
            if handler.should_capture_content():
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
        if _get_tool_name(instance) in SENTINEL_TOOL_NAMES:
            return await wrapped(*args, **kwargs)

        invocation = _start_tool_invocation(handler, instance, args, kwargs)
        with invocation:
            result = await wrapped(*args, **kwargs)
            if handler.should_capture_content():
                invocation.tool_result = result
            return result

    return traced_method


def _start_agent_invocation(
    handler: TelemetryHandler,
    instance: Module,
    kwargs: dict[str, Any],
    agent_name: str,
) -> LocalAgentInvocation:
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    if handler.should_capture_content() and kwargs:
        content_str = extract_input_content(kwargs)
        invocation.input_messages = [
            InputMessage(role="user", parts=[TextPart(content=content_str)])
        ]

    tools: Any = getattr(instance, "tools", None)
    invocation.tool_definitions = prepare_tool_definitions(tools)
    return invocation


def _set_agent_invocation_output(
    invocation: LocalAgentInvocation,
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
            if handler.should_capture_content():
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
            if handler.should_capture_content():
                _set_agent_invocation_output(invocation, instance, result)
            return result

    return traced_method


def _extract_retrieval_query(
    bound: Mapping[str, object],
) -> str | None:
    val = bound.get("query")
    return str(val) if val is not None else None


def _extract_retrieval_k(
    instance: Any,
    bound: Mapping[str, object],
) -> int | None:
    k = bound.get("k")
    if k is None and hasattr(instance, "k"):
        k = getattr(instance, "k", None)
    if k is not None:
        try:
            return int(cast(Any, k))
        except (ValueError, TypeError):
            return None
    return None


def _start_retrieval_invocation(
    handler: TelemetryHandler,
    instance: Retrieve,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    wrapped: Callable[..., Any],
) -> RetrievalInvocation:
    rm: Any = getattr(instance, "rm", None)
    if rm is None:
        import dspy

        rm = getattr(dspy.settings, "rm", None)

    # DSPy retrieval models lack a uniform identifier schema, so inspect common index
    # attributes across the Retrieve instance and configured RM.
    data_source_id: str | None = (
        getattr(instance, "data_source_id", None)
        or getattr(instance, "index_name", None)
        or (getattr(rm, "data_source_id", None) if rm is not None else None)
        or (getattr(rm, "index_name", None) if rm is not None else None)
        or (getattr(rm, "collection_name", None) if rm is not None else None)
    )

    invocation = handler.retrieval(
        data_source_id=str(data_source_id)
        if data_source_id is not None
        else None,
    )

    bound = bind_arguments(wrapped, args, kwargs)
    invocation.query_text = _extract_retrieval_query(bound)
    invocation.top_k = _extract_retrieval_k(instance, bound)
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

    # Retrieve returns passage text, without document IDs or scores.
    invocation.documents = [RetrievalDocument() for _ in passages]


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
            handler, instance, args, kwargs, wrapped
        )
        with invocation:
            result = wrapped(*args, **kwargs)
            _set_retrieval_invocation_documents(handler, invocation, result)
            return result

    return traced_method


def _extract_encoding_formats(
    merged_kwargs: Mapping[str, object],
) -> list[str] | None:
    encoding_format = merged_kwargs.get("encoding_format")
    if encoding_format is None:
        encoding_format = merged_kwargs.get("encoding_formats")
    if isinstance(encoding_format, str) and encoding_format:
        return [encoding_format]
    if is_sequence(encoding_format):
        formats = [str(fmt) for fmt in encoding_format if fmt is not None]
        return formats or None
    return None


def _start_embedding_invocation(
    handler: TelemetryHandler,
    instance: Embedder,
    kwargs: dict[str, Any],
) -> EmbeddingInvocation:
    provider, request_model = resolve_embedder_provider_and_model(instance)

    merged_kwargs: dict[str, object] = {}
    # Embedder stores init kwargs in default_kwargs unlike other DSPy modules that use kwargs;
    # both are supported with call-time kwargs taking highest precedence.
    init_kwargs: object = getattr(instance, "kwargs", None)
    if is_mapping(init_kwargs):
        merged_kwargs.update(init_kwargs)
    default_kwargs: object = getattr(instance, "default_kwargs", None)
    if is_mapping(default_kwargs):
        merged_kwargs.update(default_kwargs)
    merged_kwargs.update(kwargs)

    api_base = merged_kwargs.get("api_base")
    if api_base is None:
        api_base = merged_kwargs.get("base_url")
    server_address, server_port = extract_server_address_and_port(api_base)

    invocation = handler.embedding(
        provider=provider,
        request_model=request_model,
        server_address=server_address,
        server_port=server_port,
    )

    invocation.dimension_count = safe_int(merged_kwargs.get("dimensions"))
    invocation.encoding_formats = _extract_encoding_formats(merged_kwargs)

    return invocation


def _embedder_call(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Embedder,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_embedding_invocation(handler, instance, kwargs)
        with invocation:
            result = wrapped(*args, **kwargs)
            invocation.dimension_count = extract_embedding_dimension(result)
            return result

    return traced_method


def _embedder_acall(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Embedder,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_embedding_invocation(handler, instance, kwargs)
        with invocation:
            result = await wrapped(*args, **kwargs)
            invocation.dimension_count = extract_embedding_dimension(result)
            return result

    return traced_method
