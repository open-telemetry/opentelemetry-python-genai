# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for DSPy instrumentation."""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
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
    _safe_float,
    _safe_int,
    _safe_stop_sequences,
    apply_usage_to_invocation,
    extract_input_content,
    extract_lm_input_messages,
    extract_lm_output_messages,
    extract_output_content,
    prepare_tool_definitions,
    resolve_provider,
    resolve_request_model,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
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
from opentelemetry.util.genai.utils import bind_arguments, get_argument

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.clients.lm import LM
    from dspy.core.types import LMResponse
    from dspy.primitives.module import Module
    from dspy.primitives.prediction import Prediction
    from dspy.retrievers.retrieve import Retrieve

_REACT_MODULE = "dspy.predict.react"
_REACT_CLASS = "ReAct"

_REACT_V2_MODULE = "dspy.predict.react_v2"
_REACT_V2_CLASS = "ReActV2"

_current_lm_history_entry: ContextVar[Any] = ContextVar(
    "_current_lm_history_entry", default=None
)


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

    if hasattr(dspy, "LM"):
        lm_module = dspy.LM.__module__
        lm_name = dspy.LM.__name__
        _wrap_function(
            lm_module,
            f"{lm_name}.__call__",
            _lm_call(handler),
        )
        _wrap_function(
            lm_module,
            f"{lm_name}.acall",
            _lm_acall(handler),
        )
        if hasattr(dspy.LM, "update_history"):
            _wrap_function(
                lm_module,
                f"{lm_name}.update_history",
                _lm_update_history(),
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

    if hasattr(dspy, "LM"):
        unwrap(dspy.LM, "__call__")
        unwrap(dspy.LM, "acall")
        if hasattr(dspy.LM, "update_history"):
            unwrap(dspy.LM, "update_history")


def _start_lm_invocation(
    handler: TelemetryHandler,
    instance: LM,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    wrapped: Callable[..., Any],
) -> InferenceInvocation:
    provider = resolve_provider(instance)
    request_model = resolve_request_model(instance)

    invocation = handler.inference(
        provider=provider,
        request_model=request_model,
    )

    merged_kwargs = {**getattr(instance, "kwargs", {}), **kwargs}

    invocation.temperature = _safe_float(merged_kwargs.get("temperature"))
    invocation.max_tokens = _safe_int(merged_kwargs.get("max_tokens"))
    invocation.top_p = _safe_float(
        merged_kwargs.get("top_p")
        if merged_kwargs.get("top_p") is not None
        else merged_kwargs.get("p")
    )
    invocation.frequency_penalty = _safe_float(
        merged_kwargs.get("frequency_penalty")
    )
    invocation.presence_penalty = _safe_float(
        merged_kwargs.get("presence_penalty")
    )
    invocation.seed = _safe_int(merged_kwargs.get("seed"))

    invocation.stop_sequences = _safe_stop_sequences(merged_kwargs.get("stop"))

    choice_count = _safe_int(merged_kwargs.get("n"))
    if choice_count is not None and choice_count != 1:
        invocation.request_choice_count = choice_count

    if handler.should_capture_content():
        bound = bind_arguments(wrapped, args, kwargs)
        invocation.input_messages = extract_lm_input_messages(bound)

    return invocation


def _get_field(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return cast(Mapping[str, Any], obj).get(key)
    return getattr(obj, key, None)


def _lm_update_history() -> Callable[..., Any]:
    """Capture the history entry in context to isolate concurrent calls from shared instance.history."""

    def traced_method(
        wrapped: Callable[..., Any],
        instance: LM,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        entry = get_argument("entry", wrapped, args, kwargs)
        if entry is not None:
            _current_lm_history_entry.set(entry)
        return wrapped(*args, **kwargs)

    return traced_method


def _set_lm_invocation_response(
    handler: TelemetryHandler,
    invocation: InferenceInvocation,
    instance: LM,
    result: LMResponse | list[dict[str, Any] | str],
    history_entry: Any = None,
) -> None:
    if not isinstance(result, list):
        if result.model:
            invocation.response_model_name = str(result.model)
        if result.response_id:
            invocation.response_id = str(result.response_id)

        usage_dict = result.usage_as_dict()
        if usage_dict:
            apply_usage_to_invocation(invocation, usage_dict)

        finish_reasons = [
            out.finish_reason for out in result.outputs if out.finish_reason
        ]
        if finish_reasons:
            invocation.finish_reasons = finish_reasons

        if handler.should_capture_content():
            invocation.output_messages = extract_lm_output_messages(result)
        return

    # DSPy 3.x LM calls return a legacy list by default unless experimental=True
    # or an LMRequest is used. When available, use history_entry captured from
    # LM.update_history for per-call isolation; fall back to instance.history[-1].
    entry = history_entry
    if entry is None:
        history: Sequence[Mapping[str, Any]] | None = getattr(
            instance, "history", None
        )
        if isinstance(history, Sequence) and history:
            entry = history[-1]

    choice_finish_reasons: list[str | None] | None = None
    if entry is not None:
        resp_model = _get_field(entry, "response_model") or _get_field(
            entry, "model"
        )
        if resp_model:
            invocation.response_model_name = str(resp_model)

        usage = _get_field(entry, "usage")
        if isinstance(usage, Mapping):
            apply_usage_to_invocation(
                invocation, cast(Mapping[str, Any], usage)
            )

        resp_obj: Any = _get_field(entry, "response")
        if resp_obj is not None:
            resp_id = _get_field(resp_obj, "id")
            if resp_id:
                invocation.response_id = str(resp_id)

            choices = _get_field(resp_obj, "choices")
            if isinstance(choices, Sequence) and choices:
                choice_finish_reasons = []
                for choice in cast(Sequence[object], choices):
                    fr = _get_field(choice, "finish_reason")
                    choice_finish_reasons.append(
                        str(fr) if fr is not None else None
                    )
                invocation_finish_reasons = [
                    fr for fr in choice_finish_reasons if fr is not None
                ]
                if invocation_finish_reasons:
                    invocation.finish_reasons = invocation_finish_reasons
            else:
                outputs = _get_field(resp_obj, "outputs")
                if isinstance(outputs, Sequence) and outputs:
                    choice_finish_reasons = []
                    for out in cast(Sequence[object], outputs):
                        fr = _get_field(out, "finish_reason")
                        choice_finish_reasons.append(
                            str(fr) if fr is not None else None
                        )
                    invocation_finish_reasons = [
                        fr for fr in choice_finish_reasons if fr is not None
                    ]
                    if invocation_finish_reasons:
                        invocation.finish_reasons = invocation_finish_reasons

    if handler.should_capture_content():
        invocation.output_messages = extract_lm_output_messages(
            result, finish_reasons=choice_finish_reasons
        )


def _lm_call(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: LM,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_lm_invocation(
            handler, instance, args, kwargs, wrapped
        )
        # Isolate per-call history metadata from concurrent calls on the shared LM instance.
        token = _current_lm_history_entry.set(None)
        try:
            with invocation:
                result = wrapped(*args, **kwargs)
                history_entry = _current_lm_history_entry.get()
                _set_lm_invocation_response(
                    handler,
                    invocation,
                    instance,
                    result,
                    history_entry=history_entry,
                )
                return result
        finally:
            _current_lm_history_entry.reset(token)

    return traced_method


def _lm_acall(handler: TelemetryHandler) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: LM,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_lm_invocation(
            handler, instance, args, kwargs, wrapped
        )
        # Isolate per-call history metadata from concurrent calls on the shared LM instance.
        token = _current_lm_history_entry.set(None)
        try:
            with invocation:
                result = await wrapped(*args, **kwargs)
                history_entry = _current_lm_history_entry.get()
                _set_lm_invocation_response(
                    handler,
                    invocation,
                    instance,
                    result,
                    history_entry=history_entry,
                )
                return result
        finally:
            _current_lm_history_entry.reset(token)

    return traced_method


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


def _extract_agent_inputs(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Mapping[str, object]:
    bound = bind_arguments(wrapped, args, kwargs)
    input_args = bound.get("input_args")
    if isinstance(input_args, Mapping):
        return {
            **{k: v for k, v in bound.items() if k != "input_args"},
            **cast(Mapping[str, object], input_args),
        }
    return bound


def _start_agent_invocation(
    handler: TelemetryHandler,
    instance: Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    agent_name: str,
    wrapped: Callable[..., Any],
) -> LocalAgentInvocation:
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    if handler.should_capture_content():
        agent_inputs = _extract_agent_inputs(wrapped, args, kwargs)
        if agent_inputs:
            content_str = extract_input_content(agent_inputs)
            invocation.input_messages = [
                InputMessage(
                    role="user", parts=[TextPart(content=content_str)]
                )
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
            handler, instance, args, kwargs, agent_name, wrapped
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
            handler, instance, args, kwargs, agent_name, wrapped
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
        return _safe_int(k)
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
