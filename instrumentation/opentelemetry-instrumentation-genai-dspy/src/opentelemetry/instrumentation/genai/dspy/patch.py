# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for DSPy instrumentation."""

from __future__ import annotations

import sys
import urllib.parse
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
    extract_input_content,
    extract_output_content,
    prepare_tool_definitions,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
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
    from dspy.primitives.module import Module
    from dspy.primitives.prediction import Prediction

_REACT_MODULE = "dspy.predict.react"
_REACT_CLASS = "ReAct"

_REACT_V2_MODULE = "dspy.predict.react_v2"
_REACT_V2_CLASS = "ReActV2"

_EMBEDDINGS_MODULE = "dspy.retrievers.embeddings"
_EMBEDDINGS_WITH_SCORES_CLASS = "EmbeddingsWithScores"

_in_retrieval_invocation: ContextVar[bool] = ContextVar(
    "_in_retrieval_invocation", default=False
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
        f"{retrieve_name}.__call__",
        _retrieve_forward(handler),
    )
    _wrap_function(
        retrieve_module,
        f"{retrieve_name}.forward",
        _retrieve_forward(handler),
    )

    if hasattr(dspy, "ColBERTv2"):
        colbert_module = dspy.ColBERTv2.__module__
        colbert_name = dspy.ColBERTv2.__name__
        _wrap_function(
            colbert_module,
            f"{colbert_name}.__call__",
            _retrieve_forward(handler),
        )

    if hasattr(dspy, "Embeddings"):
        emb_module = dspy.Embeddings.__module__
        emb_name = dspy.Embeddings.__name__
        _wrap_function(
            emb_module,
            f"{emb_name}.__call__",
            _retrieve_forward(handler),
        )
        _wrap_function(
            emb_module,
            f"{emb_name}.forward",
            _retrieve_forward(handler),
        )

    embeddings_with_scores_cls = getattr(
        sys.modules.get(_EMBEDDINGS_MODULE),
        _EMBEDDINGS_WITH_SCORES_CLASS,
        None,
    )
    if embeddings_with_scores_cls is not None:
        _wrap_function(
            _EMBEDDINGS_MODULE,
            f"{_EMBEDDINGS_WITH_SCORES_CLASS}.forward",
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


def unpatch_dspy() -> None:
    """Remove patches from DSPy classes."""
    import dspy
    import dspy.predict.react

    unwrap(dspy.Tool, "__call__")
    unwrap(dspy.Tool, "acall")

    unwrap(dspy.Retrieve, "__call__")
    unwrap(dspy.Retrieve, "forward")

    if hasattr(dspy, "ColBERTv2"):
        unwrap(dspy.ColBERTv2, "__call__")

    if hasattr(dspy, "Embeddings"):
        unwrap(dspy.Embeddings, "__call__")
        unwrap(dspy.Embeddings, "forward")

    embeddings_with_scores_cls = getattr(
        sys.modules.get(_EMBEDDINGS_MODULE),
        _EMBEDDINGS_WITH_SCORES_CLASS,
        None,
    )
    if embeddings_with_scores_cls is not None:
        unwrap(embeddings_with_scores_cls, "forward")

    unwrap(dspy.predict.react.ReAct, "forward")
    unwrap(dspy.predict.react.ReAct, "aforward")

    react_v2_cls = getattr(
        sys.modules.get(_REACT_V2_MODULE), _REACT_V2_CLASS, None
    )
    if react_v2_cls is not None:
        unwrap(react_v2_cls, "forward")
        unwrap(react_v2_cls, "aforward")


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
    if val is None:
        val = bound.get("query_or_queries")
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, Sequence) and not isinstance(val, bytes):
        str_items = [
            item
            for item in cast(Sequence[object], val)
            if isinstance(item, str)
        ]
        if str_items:
            return ", ".join(str_items)
        return None
    return str(val)


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


def _extract_server_address_and_port(
    instance: Any,
    rm: Any,
) -> tuple[str | None, int | None]:
    url: Any = getattr(instance, "url", None) or (
        getattr(rm, "url", None) if rm is not None else None
    )
    if not isinstance(url, str) or not url:
        return None, None
    parsed = urllib.parse.urlparse(url if "://" in url else f"http://{url}")
    try:
        port = parsed.port
    except ValueError:
        port = None
    return parsed.hostname, port


def _start_retrieval_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    wrapped: Callable[..., Any],
) -> RetrievalInvocation:
    rm: Any = getattr(instance, "rm", None)
    if rm is None and type(instance).__name__ == "Retrieve":
        import dspy

        rm = getattr(dspy.settings, "rm", None)

    # DSPy retrieval models lack a uniform identifier schema, so inspect common index
    # attributes across the retriever instance and configured RM.
    data_source_id: str | None = (
        getattr(instance, "data_source_id", None)
        or getattr(instance, "index_name", None)
        or getattr(instance, "collection_name", None)
        or getattr(instance, "_weaviate_collection_name", None)
        or getattr(instance, "databricks_index_name", None)
        or (getattr(rm, "data_source_id", None) if rm is not None else None)
        or (getattr(rm, "index_name", None) if rm is not None else None)
        or (getattr(rm, "collection_name", None) if rm is not None else None)
        or (
            getattr(rm, "_weaviate_collection_name", None)
            if rm is not None
            else None
        )
        or (
            getattr(rm, "databricks_index_name", None)
            if rm is not None
            else None
        )
    )

    server_address, server_port = _extract_server_address_and_port(
        instance, rm
    )

    invocation = handler.retrieval(
        data_source_id=str(data_source_id)
        if data_source_id is not None
        else None,
        server_address=server_address,
        server_port=server_port,
    )

    target_callable: Callable[..., Any] = wrapped
    if getattr(wrapped, "__name__", None) == "__call__" and hasattr(
        instance, "forward"
    ):
        forward_attr = getattr(instance, "forward", None)
        if callable(forward_attr):
            target_callable = forward_attr

    bound = bind_arguments(target_callable, args, kwargs, apply_defaults=True)
    invocation.query_text = _extract_retrieval_query(bound)
    invocation.top_k = _extract_retrieval_k(instance, bound)
    return invocation


def _extract_doc_score(val: object) -> float | None:
    if isinstance(val, (int, float, str)):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


def _set_retrieval_invocation_documents(
    handler: TelemetryHandler,
    invocation: RetrievalInvocation,
    result: object,
) -> None:
    if not handler.should_capture_content():
        return

    ids_seq: Sequence[object] | None = None
    for id_attr in ("doc_ids", "indices"):
        raw_ids = getattr(result, id_attr, None)
        if isinstance(raw_ids, Sequence) and not isinstance(
            raw_ids, (str, bytes)
        ):
            ids_seq = cast(Sequence[object], raw_ids)
            break

    scores_seq: Sequence[object] | None = None
    raw_scores = getattr(result, "scores", None)
    if isinstance(raw_scores, Sequence) and not isinstance(
        raw_scores, (str, bytes)
    ):
        scores_seq = cast(Sequence[object], raw_scores)

    passages: Sequence[object] | None = None
    for attr_name in ("passages", "docs"):
        if hasattr(result, attr_name):
            attr_val = getattr(result, attr_name)
            if isinstance(attr_val, Sequence) and not isinstance(
                attr_val, (str, bytes)
            ):
                passages = cast(Sequence[object], attr_val)
                break
    if passages is None:
        if isinstance(result, Sequence) and not isinstance(
            result, (str, bytes)
        ):
            passages = cast(Sequence[object], result)
        elif isinstance(result, str):
            passages = [result]

    if passages is None:
        return

    docs: list[RetrievalDocument] = []
    for idx, psg in enumerate(passages):
        doc_id: str | None = None
        if ids_seq is not None and idx < len(ids_seq):
            if ids_seq[idx] is not None:
                doc_id = str(ids_seq[idx])
        elif isinstance(psg, Mapping):
            psg_map = cast(Mapping[str, object], psg)
            for key in ("id", "pid", "doc_id"):
                val = psg_map.get(key)
                if val is not None:
                    doc_id = str(val)
                    break
        elif not isinstance(psg, (str, bytes)):
            for key in ("id", "pid", "doc_id"):
                val = getattr(psg, key, None)
                if val is not None:
                    doc_id = str(val)
                    break

        score: float | None = None
        if scores_seq is not None and idx < len(scores_seq):
            score = _extract_doc_score(scores_seq[idx])
        elif isinstance(psg, Mapping):
            score = _extract_doc_score(
                cast(Mapping[str, object], psg).get("score")
            )
        elif not isinstance(psg, (str, bytes)):
            score = _extract_doc_score(getattr(psg, "score", None))

        docs.append(RetrievalDocument(id=doc_id, score=score))

    invocation.documents = docs


def _retrieve_forward(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _in_retrieval_invocation.get():
            return wrapped(*args, **kwargs)

        invocation = _start_retrieval_invocation(
            handler, instance, args, kwargs, wrapped
        )
        token = _in_retrieval_invocation.set(True)
        try:
            with invocation:
                result = wrapped(*args, **kwargs)
                _set_retrieval_invocation_documents(
                    handler, invocation, result
                )
                return result
        finally:
            _in_retrieval_invocation.reset(token)

    return traced_method
