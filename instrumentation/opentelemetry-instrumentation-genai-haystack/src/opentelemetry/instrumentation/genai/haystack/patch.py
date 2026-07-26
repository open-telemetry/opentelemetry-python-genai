# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Haystack instrumentation.

Builds ``opentelemetry-util-genai`` invocations around:

- ``haystack.Pipeline.run`` / ``run_async`` -> ``WorkflowInvocation``
- classified component ``run`` / ``run_async`` methods -> ``InferenceInvocation``
  (generators), ``EmbeddingInvocation`` (embedders), ``RetrievalInvocation``
  (retrievers/rankers), or ``AgentInvocation`` (``Agent``)
- ``haystack.tools.tool.Tool.invoke`` / ``invoke_async`` -> ``ToolInvocation``

See ``component_types.py`` for the classification and MIGRATION_REPORT.md for
components/methods this migration deliberately doesn't wrap.
"""

from __future__ import annotations

from contextvars import ContextVar
from inspect import BoundArguments, Parameter, signature
from typing import Any, Callable, Mapping

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    EmbeddingInvocation,
    InferenceInvocation,
    RetrievalInvocation,
    ToolInvocation,
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
# Pipeline.run / Pipeline.run_async / Pipeline.run_async_generator -> WorkflowInvocation
# ---------------------------------------------------------------------------
#
# Pipeline.run_async (the true async entry point) internally drains
# run_async_generator() to completion in the same asyncio task. Wrapping
# both unconditionally would double-count a single logical pipeline
# execution, so `_inside_run_async` -- set for the duration of the outer
# call -- lets the run_async_generator wrapper tell "called directly by
# user code" (create a span) from "driven internally by run_async" (already
# covered by the outer span; skip). contextvars propagate across `await`
# within one task, so this holds across the internal `async for`.

_inside_run_async: ContextVar[bool] = ContextVar(
    "_inside_run_async", default=False
)


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
        token = _inside_run_async.set(True)
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        finally:
            _inside_run_async.reset(token)
        invocation.stop()
        return response

    return traced_method


def pipeline_run_async_generator(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if _inside_run_async.get():
            # Driven internally by an already-wrapped run_async() call --
            # its WorkflowInvocation already covers this execution.
            async for item in wrapped(*args, **kwargs):
                yield item
            return
        invocation = handler.workflow(name=instance.__class__.__name__)
        try:
            async for item in wrapped(*args, **kwargs):
                yield item
        except Exception as exc:
            invocation.fail(exc)
            raise
        else:
            invocation.stop()

    return traced_method


# ---------------------------------------------------------------------------
# Classified component run() / run_async() wrapping
# ---------------------------------------------------------------------------


def _server_address_and_port(component: Any) -> tuple[str | None, int | None]:
    """Best-effort ``server.address``/``server.port`` from a component's SDK client.

    Haystack's OpenAI-backed generators/embedders construct their
    underlying SDK client lazily via ``warm_up()`` (``self.client``/
    ``self.async_client`` start as ``None``). ``Pipeline.run()`` calls
    ``warm_up()`` on its components automatically, so this resolves
    correctly for Pipeline-driven calls; a component called *standalone*
    only gets it starting on the instance's second call, since nothing
    else triggers ``warm_up()`` first. See MIGRATION_REPORT.md.
    """
    client = getattr(component, "client", None) or getattr(
        component, "async_client", None
    )
    base_url = getattr(client, "base_url", None)
    if base_url is None:
        return None, None
    return getattr(base_url, "host", None), getattr(base_url, "port", None)


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
    server_address, server_port = _server_address_and_port(component)
    invocation = handler.inference(
        provider=infer_provider(component) or "",
        request_model=request_model,
        operation_name=operation_name,
        server_address=server_address,
        server_port=server_port,
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
        # Best-effort: Haystack's own OpenAIChatGenerator does not copy the
        # provider response id into `reply.meta` (see MIGRATION_REPORT.md),
        # so this only populates for generators/tests that do.
        if isinstance(response_id := reply_meta.get("id"), str):
            invocation.response_id = response_id
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
    server_address, server_port = _server_address_and_port(component)
    return handler.embedding(
        provider=infer_provider(component) or "",
        request_model=request_model,
        server_address=server_address,
        server_port=server_port,
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
        # gen_ai.request.top_k is registered as an int; Haystack's own
        # top_k parameters are always a plain int count of results, so
        # round-trip it as one rather than widening to float.
        invocation.top_k = int(top_k)
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


def _start_agent_invocation(
    handler: TelemetryHandler,
    component: Any,
    bound_arguments: BoundArguments,
    capture_content: bool,
) -> tuple[AgentInvocation, int]:
    """Start an AgentInvocation for a Haystack ``Agent.run``.

    The Agent's own LLM calls are already captured as nested ``chat``
    spans -- the Agent instance's ``chat_generator`` is itself a
    ``GENERATOR``-classified component, wrapped independently at the class
    level regardless of how it's invoked. This invocation only needs to
    provide the outer ``invoke_agent`` span that groups them.

    Returns ``(invocation, input_message_count)`` -- the count is needed at
    finish time to slice the agent's echoed-back conversation history into
    just the newly generated output messages.
    """
    arguments = bound_arguments.arguments
    agent_name = (
        getattr(component, "name", None) or component.__class__.__name__
    )
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    messages = arguments.get("messages")
    input_count = len(messages) if isinstance(messages, list) else 0
    if capture_content and isinstance(messages, list):
        invocation.input_messages = to_input_messages(messages)
    tools = arguments.get("tools")
    if capture_content and tools:
        invocation.tool_definitions = tools_to_definitions(tools)
    return invocation, input_count


def _finish_agent_invocation(
    invocation: AgentInvocation,
    input_count: int,
    response: Mapping[str, Any],
    capture_content: bool,
) -> None:
    if capture_content and isinstance(
        all_messages := response.get("messages"), list
    ):
        new_messages = all_messages[input_count:] or all_messages
        invocation.output_messages = chat_replies_to_output_messages(
            new_messages
        )
    invocation.stop()


def _start_component_invocation(
    handler: TelemetryHandler,
    component_type: ComponentType,
    instance: Any,
    bound_arguments: BoundArguments,
    capture_content: bool,
) -> tuple[Any, Any]:
    """Dispatch to the right ``handler.*()`` factory for ``component_type``.

    Returns ``(invocation, extra)`` -- ``extra`` carries whatever bit of
    start-time state the matching ``_finish_*`` function needs (``is_chat``
    for generators, the input message count for agents, ``None`` otherwise).
    Shared by both the sync and async component wrappers.
    """
    if component_type is ComponentType.GENERATOR:
        return _start_generator_invocation(
            handler, instance, bound_arguments, capture_content
        )
    if component_type is ComponentType.EMBEDDER:
        return (
            _start_embedding_invocation(handler, instance, capture_content),
            None,
        )
    if component_type is ComponentType.AGENT:
        return _start_agent_invocation(
            handler, instance, bound_arguments, capture_content
        )
    # RANKER and RETRIEVER share the same retrieval invocation shape.
    return (
        _start_retrieval_invocation(
            handler, instance, bound_arguments, capture_content
        ),
        None,
    )


def _finish_component_invocation(
    component_type: ComponentType,
    invocation: Any,
    extra: Any,
    response: Mapping[str, Any],
    capture_content: bool,
) -> None:
    if component_type is ComponentType.GENERATOR:
        _finish_generator_invocation(
            invocation, extra, response, capture_content
        )
    elif component_type is ComponentType.EMBEDDER:
        _finish_embedding_invocation(invocation, response)
    elif component_type is ComponentType.AGENT:
        _finish_agent_invocation(invocation, extra, response, capture_content)
    else:
        _finish_retrieval_invocation(invocation, response, capture_content)


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
        invocation, extra = _start_component_invocation(
            handler, component_type, instance, bound_arguments, capture_content
        )
        try:
            response = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        _finish_component_invocation(
            component_type, invocation, extra, response, capture_content
        )
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
        invocation, extra = _start_component_invocation(
            handler, component_type, instance, bound_arguments, capture_content
        )
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        _finish_component_invocation(
            component_type, invocation, extra, response, capture_content
        )
        return response

    return traced_method


# ---------------------------------------------------------------------------
# Tool.invoke / Tool.invoke_async -> ToolInvocation
# ---------------------------------------------------------------------------
#
# haystack.tools.tool.Tool is the single concrete class every Haystack tool
# is built from -- wrapped directly on the class, not via the component
# registry (a Tool is not a Haystack ``Component``). Correlating the span
# with the model's tool_call.id would require hooking the private
# haystack.components.agents.tool_calling._make_context_bound_invoke, which
# is where the id is available; deliberately not done here to avoid
# depending on Haystack internals -- see MIGRATION_REPORT.md.


def _start_tool_invocation(
    handler: TelemetryHandler, instance: Any, kwargs: Mapping[str, Any]
) -> ToolInvocation:
    invocation = handler.tool(
        name=getattr(instance, "name", None) or instance.__class__.__name__,
        tool_type="function",
        tool_description=getattr(instance, "description", None),
    )
    if invocation.should_capture_content_on_span:
        invocation.arguments = dict(kwargs)
    return invocation


def _finish_tool_invocation(invocation: ToolInvocation, result: Any) -> None:
    if invocation.should_capture_content_on_span:
        invocation.tool_result = result
    invocation.stop()


def tool_invoke(handler: TelemetryHandler) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        invocation = _start_tool_invocation(handler, instance, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        _finish_tool_invocation(invocation, result)
        return result

    return traced_method


def tool_invoke_async(handler: TelemetryHandler) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        invocation = _start_tool_invocation(handler, instance, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        _finish_tool_invocation(invocation, result)
        return result

    return traced_method
