# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Haystack instrumentation.

Builds ``opentelemetry-util-genai`` invocations around:

- ``haystack.Pipeline.run`` / ``run_async`` -> ``WorkflowInvocation``
- classified component ``run`` / ``run_async`` methods -> ``InferenceInvocation``
  (generators), ``EmbeddingInvocation`` (embedders), or ``RetrievalInvocation``
  (retrievers/rankers)

See ``component_types.py`` for the classification and MIGRATION_REPORT.md for
components/methods this migration deliberately doesn't wrap.
"""

from __future__ import annotations

from inspect import BoundArguments, Parameter, signature
from typing import Any, Callable, Mapping

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    InferenceInvocation,
    RetrievalInvocation,
)

from .component_types import ComponentType
from .message_utils import (
    chat_replies_to_output_messages,
    documents_to_retrieval_documents,
    prompt_to_input_messages,
    text_replies_to_output_messages,
    to_input_messages,
    tools_to_definitions,
)
from .provider import infer_provider

CHAT = GenAI.GenAiOperationNameValues.CHAT.value
TEXT_COMPLETION = GenAI.GenAiOperationNameValues.TEXT_COMPLETION.value


def _bind_arguments(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> BoundArguments:
    sig = signature(func)
    accepts_var_kwargs = any(
        param.kind == Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_var_kwargs or key in sig.parameters
    }
    bound = sig.bind_partial(*args, **valid_kwargs)
    bound.apply_defaults()
    return bound


# ---------------------------------------------------------------------------
# Pipeline.run / Pipeline.run_async -> WorkflowInvocation
# ---------------------------------------------------------------------------
#
# Pipeline.run_async (the true async entry point) internally drains
# run_async_generator() to completion — wrapping both would double-count a
# single logical pipeline execution, so only the two entry points a caller
# invokes directly are wrapped. See MIGRATION_REPORT.md.


def pipeline_run(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        invocation = handler.workflow(name=instance.__class__.__name__)
        try:
            response = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        invocation.stop()
        return response

    return traced_method


def pipeline_run_async(handler: TelemetryHandler) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        invocation = handler.workflow(name=instance.__class__.__name__)
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        invocation.stop()
        return response

    return traced_method


# ---------------------------------------------------------------------------
# Classified component run() / run_async() wrapping
# ---------------------------------------------------------------------------


def _start_generator_invocation(
    handler: TelemetryHandler,
    component: Any,
    bound_arguments: BoundArguments,
    capture_content: bool,
) -> tuple[InferenceInvocation, bool]:
    """Start an InferenceInvocation for a Generator/ChatGenerator ``run``.

    Returns ``(invocation, is_chat)`` — ``is_chat`` selects how the response
    is parsed once the wrapped call returns.
    """
    arguments = bound_arguments.arguments
    # ChatGenerator-style components take a ``messages`` parameter (a
    # ``list[ChatMessage]`` or, on newer Haystack versions, a bare ``str``
    # convenience form) and always reply with ``list[ChatMessage]``. Plain
    # text-completion generators take a ``prompt: str`` and reply with
    # ``list[str]``. Presence of the parameter -- not its runtime type --
    # decides which shape the response will have.
    is_chat = "messages" in arguments
    operation_name = CHAT if is_chat else TEXT_COMPLETION
    messages = arguments.get("messages")
    request_model = getattr(component, "model", None)
    invocation = handler.inference(
        provider=infer_provider(component) or "",
        request_model=request_model,
        operation_name=operation_name,
    )
    if capture_content:
        if is_chat and isinstance(messages, str):
            invocation.input_messages = prompt_to_input_messages(messages)
        elif is_chat:
            invocation.input_messages = to_input_messages(messages)
        elif isinstance(prompt := arguments.get("prompt"), str):
            invocation.input_messages = prompt_to_input_messages(prompt)
        tools = arguments.get("tools")
        if tools:
            invocation.tool_definitions = tools_to_definitions(tools)
    generation_kwargs = arguments.get("generation_kwargs") or {}
    if isinstance(generation_kwargs, Mapping):
        if isinstance(
            temperature := generation_kwargs.get("temperature"), (int, float)
        ):
            invocation.temperature = float(temperature)
        if isinstance(top_p := generation_kwargs.get("top_p"), (int, float)):
            invocation.top_p = float(top_p)
        if isinstance(max_tokens := generation_kwargs.get("max_tokens"), int):
            invocation.max_tokens = max_tokens
    return invocation, is_chat


def _finish_generator_invocation(
    invocation: InferenceInvocation,
    is_chat: bool,
    response: Mapping[str, Any],
    capture_content: bool,
) -> None:
    replies = response.get("replies") or []
    meta = response.get("meta")

    if capture_content:
        if is_chat:
            invocation.output_messages = chat_replies_to_output_messages(
                replies
            )
        else:
            invocation.output_messages = text_replies_to_output_messages(
                replies
            )

    reply_meta: Mapping[str, Any] | None = None
    if isinstance(meta, list) and meta and isinstance(meta[0], Mapping):
        reply_meta = meta[0]
    elif is_chat and replies:
        first_reply_meta = getattr(replies[0], "meta", None)
        if isinstance(first_reply_meta, Mapping):
            reply_meta = first_reply_meta

    if reply_meta is not None:
        if isinstance(model := reply_meta.get("model"), str):
            invocation.response_model_name = model
        usage = reply_meta.get("usage")
        if isinstance(usage, Mapping):
            if isinstance(prompt_tokens := usage.get("prompt_tokens"), int):
                invocation.input_tokens = prompt_tokens
            if isinstance(
                completion_tokens := usage.get("completion_tokens"), int
            ):
                invocation.output_tokens = completion_tokens
        finish_reason = reply_meta.get("finish_reason")
        if isinstance(finish_reason, str):
            invocation.finish_reasons = [finish_reason]
    invocation.stop()


def _start_embedding_invocation(
    handler: TelemetryHandler,
    component: Any,
    capture_content: bool,  # noqa: ARG001 - kept for signature symmetry with generator/retrieval starters
) -> EmbeddingInvocation:
    request_model = getattr(component, "model", None)
    return handler.embedding(
        provider=infer_provider(component) or "",
        request_model=request_model,
    )


def _finish_embedding_invocation(
    invocation: EmbeddingInvocation, response: Mapping[str, Any]
) -> None:
    documents = response.get("documents")
    vector = None
    if isinstance(documents, list) and documents:
        vector = getattr(documents[0], "embedding", None)
    else:
        vector = response.get("embedding")
    # Haystack embedders return the decoded vector as a `list[float]` or a
    # `numpy.ndarray` depending on the provider integration; duck-type on
    # `__len__` rather than requiring a specific sequence type.
    if vector is not None and hasattr(vector, "__len__"):
        invocation.dimension_count = len(vector)

    meta = response.get("meta")
    if isinstance(meta, Mapping):
        usage = meta.get("usage")
        if isinstance(usage, Mapping) and isinstance(
            prompt_tokens := usage.get("prompt_tokens"), int
        ):
            invocation.input_tokens = prompt_tokens
        if isinstance(model := meta.get("model"), str):
            invocation.response_model_name = model
    invocation.stop()


def _start_retrieval_invocation(
    handler: TelemetryHandler,
    component: Any,
    bound_arguments: BoundArguments,
    capture_content: bool,
) -> RetrievalInvocation:
    arguments = bound_arguments.arguments
    invocation = handler.retrieval()
    top_k = arguments.get("top_k")
    if top_k is None:
        top_k = getattr(component, "top_k", None)
    if isinstance(top_k, (int, float)):
        invocation.top_k = float(top_k)
    if capture_content and isinstance(query := arguments.get("query"), str):
        invocation.query_text = query
    return invocation


def _finish_retrieval_invocation(
    invocation: RetrievalInvocation,
    response: Mapping[str, Any],
    capture_content: bool,
) -> None:
    documents = response.get("documents")
    if capture_content and isinstance(documents, list):
        invocation.documents = documents_to_retrieval_documents(documents)
    invocation.stop()


def component_run(
    handler: TelemetryHandler, component_type: ComponentType
) -> Callable[..., Any]:
    """Build a sync wrapper for a component ``run`` method classified as ``component_type``."""
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        bound_arguments = _bind_arguments(wrapped, args, kwargs)
        if component_type is ComponentType.GENERATOR:
            invocation, is_chat = _start_generator_invocation(
                handler, instance, bound_arguments, capture_content
            )
        elif component_type is ComponentType.EMBEDDER:
            invocation = _start_embedding_invocation(
                handler, instance, capture_content
            )
        else:
            invocation = _start_retrieval_invocation(
                handler, instance, bound_arguments, capture_content
            )
        try:
            response = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        if component_type is ComponentType.GENERATOR:
            _finish_generator_invocation(
                invocation, is_chat, response, capture_content
            )
        elif component_type is ComponentType.EMBEDDER:
            _finish_embedding_invocation(invocation, response)
        else:
            _finish_retrieval_invocation(invocation, response, capture_content)
        return response

    return traced_method


def component_run_async(
    handler: TelemetryHandler, component_type: ComponentType
) -> Callable[..., Any]:
    """Build an async wrapper for a component ``run_async`` method classified as ``component_type``."""
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        bound_arguments = _bind_arguments(wrapped, args, kwargs)
        if component_type is ComponentType.GENERATOR:
            invocation, is_chat = _start_generator_invocation(
                handler, instance, bound_arguments, capture_content
            )
        elif component_type is ComponentType.EMBEDDER:
            invocation = _start_embedding_invocation(
                handler, instance, capture_content
            )
        else:
            invocation = _start_retrieval_invocation(
                handler, instance, bound_arguments, capture_content
            )
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        if component_type is ComponentType.GENERATOR:
            _finish_generator_invocation(
                invocation, is_chat, response, capture_content
            )
        elif component_type is ComponentType.EMBEDDER:
            _finish_embedding_invocation(invocation, response)
        else:
            _finish_retrieval_invocation(invocation, response, capture_content)
        return response

    return traced_method
