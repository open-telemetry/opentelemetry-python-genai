# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridges agents-library tracing callbacks to opentelemetry-util-genai.

The agents library exposes a public extension API
(:func:`agents.tracing.add_trace_processor`) for plugging custom
:class:`TracingProcessor` implementations into its own tracing system.
``Trace.start()`` / ``Span.start()`` invoke the registered processors'
``on_*_start`` callbacks *synchronously* on whichever asyncio task
started the agents-library span:

* Workflow (``Trace``) and agent (``AgentSpanData``) spans start in the
  ``Runner.run`` task itself.
* Function tool (``FunctionSpanData``) spans start inside the per-tool
  ``asyncio.Task`` the agents library creates for tool dispatch. That
  sub-task inherits a snapshot of the run-loop context (so workflow +
  agent are already active in OTel contextvars).

Because every ``*_end`` callback fires on the same task as its
matching ``*_start``, util-genai's auto-``attach()`` / ``detach()`` of
OTel context is balanced and no context tokens leak across tasks.
OTel's natural parent tracking nests the tree:

    workflow > invoke_agent > [chat from openai instrumentation,
                               execute_tool]

LLM-level spans (``chat`` / ``responses`` / ``embeddings``) are not
emitted here — ``opentelemetry-instrumentation-genai-openai`` patches
the openai SDK directly and produces those.
"""

from __future__ import annotations

import weakref
from typing import Any

from agents import RunHooks
from agents.tracing import Span, Trace, TracingProcessor, get_current_span
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
)

from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ErrorTypeValues,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    LocalAgentInvocation,
    ToolInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import Error
from opentelemetry.util.genai.utils import (
    object_to_any_value,
    tool_arguments_to_any_value,
)

# Non-semconv attribute: surfaces the workflow name on the workflow span
# so callers can query/filter by it. util-genai's WorkflowInvocation
# only puts the name in the span name, not as an attribute.
_WORKFLOW_NAME_ATTR = "gen_ai.workflow.name"


class GenAITracingProcessor(TracingProcessor):
    """Translate agents-library tracing into util-genai invocations.

    Stateful only for span lifetime: each in-flight Trace/Span has one
    entry in a :class:`weakref.WeakKeyDictionary` keyed by the
    agents-library object itself. Entries are removed on ``*_end`` or
    garbage-collected with the agents-library span/trace if the library
    drops it before ``end`` (which it shouldn't, but the weak reference
    is belt-and-suspenders against any future leak).
    """

    def __init__(self, handler: TelemetryHandler, provider: str) -> None:
        self._handler = handler
        self._provider = provider
        self._invocations: weakref.WeakKeyDictionary[
            Any, WorkflowInvocation | LocalAgentInvocation | ToolInvocation
        ] = weakref.WeakKeyDictionary()
        self._agent_spans: weakref.WeakValueDictionary[str, Span[Any]] = (
            weakref.WeakValueDictionary()
        )

    def on_trace_start(self, trace: Trace) -> None:
        # ``trace.name`` comes from ``RunConfig.workflow_name`` (default
        # "Agent workflow"). Callers customize it via the agents library's
        # own ``Runner.run(..., run_config=RunConfig(workflow_name=...))``;
        # we don't expose a second knob.
        #
        # ``group_id`` is public on ``TraceImpl`` but not declared on the
        # ``Trace`` ABC, hence the getattr.
        group_id: str | None = getattr(trace, "group_id", None)
        invocation = self._handler.workflow(
            name=trace.name,
            conversation_id=group_id,
        )
        if trace.name:
            invocation.attributes[_WORKFLOW_NAME_ATTR] = trace.name
        self._invocations[trace] = invocation

    def on_trace_end(self, trace: Trace) -> None:
        invocation = self._invocations.pop(trace, None)
        if invocation is not None:
            invocation.stop()

    def on_span_start(self, span: Span[Any]) -> None:
        span_data = span.span_data
        if isinstance(span_data, AgentSpanData):
            invocation = self._handler.invoke_local_agent(
                agent_name=span_data.name,
            )
            self._invocations[span] = invocation
            self._agent_spans[span.span_id] = span
            return
        if span.parent_id is not None:
            agent_span = self._agent_spans.get(span.parent_id)
            if agent_span is not None:
                self._agent_spans[span.span_id] = agent_span
        if isinstance(span_data, FunctionSpanData):
            invocation = self._handler.tool(
                name=span_data.name,
                tool_type="function",
            )
            self._invocations[span] = invocation
            return
        # Other span_data types (GenerationSpanData, ResponseSpanData,
        # HandoffSpanData, GuardrailSpanData, Speech/TranscriptionSpanData)
        # are intentionally ignored. LLM-level spans come from the openai
        # instrumentation; the rest have no semconv yet.

    def on_span_end(self, span: Span[Any]) -> None:
        self._agent_spans.pop(span.span_id, None)
        invocation = self._invocations.pop(span, None)
        if invocation is None:
            return
        if (
            isinstance(invocation, ToolInvocation)
            and isinstance(span.span_data, FunctionSpanData)
            and invocation.should_capture_content
        ):
            arguments = span.span_data.input
            if arguments:
                invocation.arguments = tool_arguments_to_any_value(arguments)
            invocation.tool_result = object_to_any_value(span.span_data.output)
        # SpanError is a mapping, not a raised exception, so it never
        # reaches util-genai's exception path on its own.
        span_error = getattr(span, "error", None)
        if span_error is not None:
            invocation.fail(
                Error(
                    type=ErrorTypeValues.OTHER.value,
                    message=str(span_error),
                )
            )
            return
        invocation.stop()

    def _get_current_agent(self) -> LocalAgentInvocation | None:
        span = get_current_span()
        agent_span = (
            self._agent_spans.get(span.span_id) if span is not None else None
        )
        invocation = (
            self._invocations.get(agent_span)
            if agent_span is not None
            else None
        )
        return (
            invocation
            if isinstance(invocation, LocalAgentInvocation)
            else None
        )

    def shutdown(self) -> None:
        for invocation in list(self._invocations.values()):
            try:
                invocation.stop()
            except Exception:  # pylint: disable=broad-except
                pass
        self._invocations.clear()
        self._agent_spans.clear()

    def force_flush(self) -> None:  # pragma: no cover - nothing to flush
        pass


class GenAIRunHooks(RunHooks[Any]):
    """Count execution events while forwarding application lifecycle hooks."""

    def __init__(
        self,
        hooks: RunHooks[Any] | None,
        processor: GenAITracingProcessor,
    ) -> None:
        self._hooks = hooks if hooks is not None else RunHooks()
        self._processor = processor

    def _get_invocation(self) -> LocalAgentInvocation | None:
        invocation = self._processor._get_current_agent()
        if invocation is not None:
            if invocation.inference_calls is None:
                invocation.inference_calls = 0
            if invocation.tool_calls is None:
                invocation.tool_calls = 0
        return invocation

    async def on_agent_start(self, *args: Any, **kwargs: Any) -> None:
        self._get_invocation()
        await self._hooks.on_agent_start(*args, **kwargs)

    async def on_agent_end(self, *args: Any, **kwargs: Any) -> None:
        await self._hooks.on_agent_end(*args, **kwargs)

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        invocation = self._get_invocation()
        if invocation is not None:
            invocation.inference_calls = (invocation.inference_calls or 0) + 1
        await self._hooks.on_llm_start(*args, **kwargs)

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        await self._hooks.on_llm_end(*args, **kwargs)

    async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        invocation = self._get_invocation()
        if invocation is not None:
            invocation.tool_calls = (invocation.tool_calls or 0) + 1
        await self._hooks.on_tool_start(*args, **kwargs)

    async def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        await self._hooks.on_tool_end(*args, **kwargs)

    async def on_handoff(self, *args: Any, **kwargs: Any) -> None:
        await self._hooks.on_handoff(*args, **kwargs)
