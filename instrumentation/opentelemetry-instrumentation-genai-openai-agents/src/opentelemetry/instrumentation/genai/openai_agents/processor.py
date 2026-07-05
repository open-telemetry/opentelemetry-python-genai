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

from agents.tracing import Span, Trace, TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
)

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    GenAIInvocation,
    ToolInvocation,
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
        self._invocations: weakref.WeakKeyDictionary[Any, GenAIInvocation] = (
            weakref.WeakKeyDictionary()
        )

    def on_trace_start(self, trace: Trace) -> None:
        # ``trace.name`` comes from ``RunConfig.workflow_name`` (default
        # "Agent workflow"). Callers customize it via the agents library's
        # own ``Runner.run(..., run_config=RunConfig(workflow_name=...))``;
        # we don't expose a second knob.
        invocation = self._handler.workflow(name=trace.name)
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
            return
        if isinstance(span_data, FunctionSpanData):
            invocation = self._handler.tool(
                name=span_data.name,
                tool_type="function",
            )

            invocation.arguments = span_data.input

            # ToolInvocation does not include provider in metric attributes
            # by default; set it so gen_ai.client.operation.duration carries
            # the required gen_ai.provider.name attribute.
            invocation.metric_attributes[GenAI.GEN_AI_PROVIDER_NAME] = (
                self._provider
            )
            self._invocations[span] = invocation
            return
        # Other span_data types (GenerationSpanData, ResponseSpanData,
        # HandoffSpanData, GuardrailSpanData, Speech/TranscriptionSpanData)
        # are intentionally ignored. LLM-level spans come from the openai
        # instrumentation; the rest have no semconv yet.

    def on_span_end(self, span: Span[Any]) -> None:
        invocation = self._invocations.pop(span, None)
        if invocation is None:
            return
        if isinstance(invocation, ToolInvocation) and isinstance(
            span.span_data, FunctionSpanData
        ):
            output = span.span_data.output
            if output is not None:
                invocation.tool_result = (
                    output if isinstance(output, str) else str(output)
                )
        invocation.stop()

    def shutdown(self) -> None:
        for invocation in list(self._invocations.values()):
            try:
                invocation.stop()
            except Exception:  # pylint: disable=broad-except
                pass
        self._invocations.clear()

    def force_flush(self) -> None:  # pragma: no cover - nothing to flush
        pass
