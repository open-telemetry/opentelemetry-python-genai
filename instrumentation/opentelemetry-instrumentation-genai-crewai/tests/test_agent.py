# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from crewai import Agent, BaseLLM, Crew, Process, Task
from crewai.tools.agent_tools.agent_tools import AgentTools

from opentelemetry import trace
from opentelemetry.instrumentation.genai.crewai import patch as patch_module
from opentelemetry.instrumentation.genai.crewai.patch import (
    _current_agent_name,
)
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace.status import StatusCode

from .test_operations import AddTool


class ScriptedLLM(BaseLLM):
    """Return deterministic responses without provider access.

    ReAct-format strings drive the text-parsing executor; a list of tool-call
    dicts (with ``native=True``) drives the native function-calling executor.

    State lives in undeclared underscore attributes because ``BaseLLM`` is a
    plain ABC on older CrewAI releases and a pydantic model on newer ones.
    """

    def __init__(self, responses: list[Any], *, native: bool = False) -> None:
        super().__init__(model="scripted-model")
        self._responses = responses
        self._calls = 0
        self._native = native

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        del (
            messages,
            tools,
            callbacks,
            available_functions,
            from_task,
            from_agent,
            response_model,
        )
        response = self._responses[min(self._calls, len(self._responses) - 1)]
        self._calls += 1
        return response

    def supports_function_calling(self) -> bool:
        return self._native

    def supports_stop_words(self) -> bool:
        return False


class FailingLLM(ScriptedLLM):
    def call(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise RuntimeError("model failed")


class InterruptingLLM(ScriptedLLM):
    def call(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        raise KeyboardInterrupt


class FlakyLLM(ScriptedLLM):
    """Fail on the first call, then answer normally."""

    def call(self, *args: Any, **kwargs: Any) -> Any:
        if self._calls == 0:
            self._calls += 1
            raise RuntimeError("flaky")
        return super().call(*args, **kwargs)


class TracingLLM(ScriptedLLM):
    """Stand in for a provider client instrumentation by opening a span."""

    def __init__(
        self, responses: list[Any], *, tracer_provider: TracerProvider
    ) -> None:
        super().__init__(responses)
        self._tracer_provider = tracer_provider

    def call(self, *args: Any, **kwargs: Any) -> Any:
        tracer = self._tracer_provider.get_tracer("provider-client")
        with tracer.start_as_current_span("chat scripted-model"):
            return super().call(*args, **kwargs)


FINAL_ANSWER = "Thought: done.\nFinal Answer: {}"
ADD_ACTION = (
    "Thought: I should add.\nAction: add\nAction Input: {"
    '"left": 2, "right": 3}'
)


def _agent(
    llm: BaseLLM,
    *,
    role: str = "Researcher",
    tools: list[Any] | None = None,
    timeout: int | None = None,
    max_retry_limit: int = 0,
) -> Agent:
    return Agent(
        role=role,
        goal="Produce a reliable answer",
        backstory="An offline test agent",
        llm=llm,
        tools=tools or [],
        max_execution_time=timeout,
        max_retry_limit=max_retry_limit,
        verbose=False,
    )


def _task(
    agent: Agent, description: str = "Do the work", **kwargs: Any
) -> Task:
    return Task(
        description=description,
        expected_output="A short answer",
        agent=agent,
        **kwargs,
    )


def _spans_named(spans: Any, name: str) -> list[ReadableSpan]:
    return [span for span in spans if span.name == name]


def _assert_child_of(child: ReadableSpan, parent: ReadableSpan) -> None:
    assert child.context.trace_id == parent.context.trace_id
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


@pytest.mark.parametrize("timeout", [None, 120])
def test_tool_span_is_child_of_agent_span(
    instrument_crewai_with_content,
    span_exporter,
    timeout: int | None,
) -> None:
    llm = ScriptedLLM([ADD_ACTION, FINAL_ANSWER.format(5)])
    agent = _agent(llm, tools=[AddTool()], timeout=timeout)

    assert agent.execute_task(_task(agent, "Add two and three")) == "5"

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "execute_tool add",
        "invoke_agent Researcher",
    ]
    tool_span, agent_span = spans
    _assert_child_of(tool_span, agent_span)
    assert (
        tool_span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "Researcher"
    )
    assert json.loads(
        tool_span.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
    ) == {"left": 2, "right": 3}
    assert _current_agent_name.get() is None


def test_agent_attributes_and_content(
    instrument_crewai_with_content,
    span_exporter,
) -> None:
    agent = _agent(
        ScriptedLLM([FINAL_ANSWER.format("Hello")]), tools=[AddTool()]
    )

    assert agent.execute_task(_task(agent, "Say hello")) == "Hello"

    span = span_exporter.get_finished_spans()[0]
    assert span.name == "invoke_agent Researcher"
    assert span.kind == trace.SpanKind.INTERNAL
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == "invoke_agent"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "Researcher"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_AGENT_DESCRIPTION]
        == "Produce a reliable answer"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "scripted-model"
    )
    definitions = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    )
    assert definitions[0]["type"] == "function"
    assert definitions[0]["name"] == "add"
    assert definitions[0]["description"] == "Add two integers"
    assert set(definitions[0]["parameters"]["properties"]) == {
        "left",
        "right",
    }
    input_message = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )[0]
    assert input_message["role"] == "user"
    assert [part["content"] for part in input_message["parts"]] == [
        "Say hello",
        "Expected output: A short answer",
    ]
    output_message = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )[0]
    assert output_message["role"] == "assistant"
    assert output_message["parts"][0]["content"] == "Hello"


def test_content_not_captured_by_default(
    instrument_crewai,
    span_exporter,
) -> None:
    agent = _agent(
        ScriptedLLM([FINAL_ANSWER.format("Hello")]), tools=[AddTool()]
    )

    assert agent.execute_task(_task(agent, "Say hello")) == "Hello"

    attributes = span_exporter.get_finished_spans()[0].attributes
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
    # Tool definitions are not message content and are always recorded.
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in attributes


def test_explicit_empty_tools_omits_tool_definitions(
    instrument_crewai,
    span_exporter,
) -> None:
    agent = _agent(
        ScriptedLLM([FINAL_ANSWER.format("Hello")]), tools=[AddTool()]
    )

    assert agent.execute_task(_task(agent), tools=[]) == "Hello"

    attributes = span_exporter.get_finished_spans()[0].attributes
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in attributes


def test_standalone_agent_kickoff(
    instrument_crewai_with_content,
    span_exporter,
) -> None:
    agent = _agent(ScriptedLLM(["Hello from kickoff"]))

    result = agent.kickoff("Say hello")

    assert result.raw == "Hello from kickoff"
    span = span_exporter.get_finished_spans()[0]
    assert span.name == "invoke_agent Researcher"
    assert span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "Researcher"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "scripted-model"
    )
    input_message = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )[0]
    assert input_message["role"] == "user"
    assert input_message["parts"][0]["content"] == "Say hello"
    assert (
        json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])[0][
            "parts"
        ][0]["content"]
        == "Hello from kickoff"
    )


def test_agent_error_is_reraised_and_recorded(
    instrument_crewai,
    span_exporter,
) -> None:
    agent = _agent(FailingLLM([]))

    with pytest.raises(RuntimeError, match="model failed"):
        agent.execute_task(_task(agent))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes["error.type"] == "RuntimeError"
    assert _current_agent_name.get() is None


def test_base_exception_finalizes_span_and_restores_context(
    instrument_crewai,
    span_exporter,
    tracer_provider: TracerProvider,
) -> None:
    agent = _agent(InterruptingLLM([]))
    tracer = tracer_provider.get_tracer("app")

    with tracer.start_as_current_span("app") as app_span:
        with pytest.raises(KeyboardInterrupt):
            agent.execute_task(_task(agent))
        assert trace.get_current_span() is app_span
        assert _current_agent_name.get() is None

    agent_span = _spans_named(
        span_exporter.get_finished_spans(), "invoke_agent Researcher"
    )[0]
    assert agent_span.status.status_code == StatusCode.ERROR
    assert agent_span.attributes["error.type"] == "KeyboardInterrupt"
    assert agent_span.parent is not None
    assert agent_span.parent.span_id == app_span.context.span_id


def test_metadata_failure_does_not_affect_call(
    instrument_crewai_with_content,
    span_exporter,
    tracer_provider: TracerProvider,
) -> None:
    agent = _agent(
        ScriptedLLM([FINAL_ANSWER.format("Hello")]), tools=[AddTool()]
    )
    tracer = tracer_provider.get_tracer("app")

    with (
        patch.object(
            patch_module,
            "agent_tool_definitions",
            side_effect=RuntimeError("definitions failed"),
        ),
        patch.object(
            patch_module,
            "output_to_output_messages",
            side_effect=RuntimeError("output failed"),
        ),
        tracer.start_as_current_span("app") as app_span,
    ):
        assert agent.execute_task(_task(agent)) == "Hello"
        assert trace.get_current_span() is app_span
        assert _current_agent_name.get() is None

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent Researcher", "app"]
    agent_span = spans[0]
    assert agent_span.status.status_code == StatusCode.UNSET
    assert (
        agent_span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME]
        == "Researcher"
    )
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in agent_span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in agent_span.attributes


def test_kickoff_in_running_loop_is_not_instrumented(
    instrument_crewai,
    span_exporter,
) -> None:
    agent = _agent(ScriptedLLM(["Hello from async kickoff"]))

    async def run() -> None:
        result = await agent.kickoff("Say hello")
        assert result.raw == "Hello from async kickoff"

    asyncio.run(run())
    assert span_exporter.get_finished_spans() == ()


def test_agent_instrumentation_suppression(
    instrument_crewai,
    span_exporter,
    metric_reader,
) -> None:
    llm = ScriptedLLM([ADD_ACTION, FINAL_ANSWER.format(5)])
    agent = _agent(llm, tools=[AddTool()])

    with suppress_instrumentation():
        assert agent.execute_task(_task(agent)) == "5"

    assert span_exporter.get_finished_spans() == ()
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is None or not [
        metric
        for resource_metrics in metrics_data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    ]


def _two_agent_crew() -> Crew:
    first = _agent(ScriptedLLM([FINAL_ANSWER.format("one")]), role="First")
    second = _agent(ScriptedLLM([FINAL_ANSWER.format("two")]), role="Second")
    return Crew(
        agents=[first, second],
        tasks=[_task(first, "Step one"), _task(second, "Step two")],
        process=Process.sequential,
        verbose=False,
    )


def test_sequential_crew_emits_one_span_per_agent(
    instrument_crewai,
    span_exporter,
) -> None:
    assert _two_agent_crew().kickoff().raw == "two"

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "invoke_agent First",
        "invoke_agent Second",
    ]
    for span in spans:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
            == "scripted-model"
        )
    # Without a workflow span, each agent invocation starts its own trace.
    first, second = spans
    assert first.parent is None
    assert second.parent is None
    assert first.context.trace_id != second.context.trace_id


def test_sequential_crew_under_application_span(
    instrument_crewai,
    span_exporter,
    tracer_provider: TracerProvider,
) -> None:
    tracer = tracer_provider.get_tracer("app")
    with tracer.start_as_current_span("crew run") as app_span:
        assert _two_agent_crew().kickoff().raw == "two"

    spans = span_exporter.get_finished_spans()
    first, second = spans[:2]
    assert first.name == "invoke_agent First"
    assert second.name == "invoke_agent Second"
    for span in (first, second):
        assert span.context.trace_id == app_span.context.trace_id
        assert span.parent is not None
        assert span.parent.span_id == app_span.context.span_id


def test_parallel_native_tool_calls_stay_parented(
    instrument_crewai_with_content,
    span_exporter,
) -> None:
    llm = ScriptedLLM(
        [
            [
                {
                    "id": "call_1",
                    "function": {
                        "name": "add",
                        "arguments": '{"left": 1, "right": 2}',
                    },
                },
                {
                    "id": "call_2",
                    "function": {
                        "name": "add",
                        "arguments": '{"left": 3, "right": 4}',
                    },
                },
            ],
            "3 and 7",
        ],
        native=True,
    )
    agent = _agent(llm, tools=[AddTool()])

    assert agent.execute_task(_task(agent, "Add twice")) == "3 and 7"

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_named(spans, "invoke_agent Researcher")[0]
    tool_spans = _spans_named(spans, "execute_tool add")
    assert len(tool_spans) == 2
    assert len(spans) == 3
    for tool_span in tool_spans:
        _assert_child_of(tool_span, agent_span)
        assert (
            tool_span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME]
            == "Researcher"
        )
    assert sorted(
        json.loads(
            span.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        )["left"]
        for span in tool_spans
    ) == [1, 3]
    assert sorted(
        span.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT]
        for span in tool_spans
    ) == [3, 7]


def test_delegation_nests_agent_under_tool_span(
    instrument_crewai,
    span_exporter,
) -> None:
    helper = _agent(
        ScriptedLLM([FINAL_ANSWER.format("helped")]), role="Helper"
    )
    lead_llm = ScriptedLLM(
        [
            "Thought: I should delegate.\n"
            "Action: Delegate work to coworker\n"
            'Action Input: {"task": "Do it", "context": "Now", '
            '"coworker": "Helper"}',
            FINAL_ANSWER.format("delegated"),
        ]
    )
    lead = _agent(
        lead_llm, role="Lead", tools=AgentTools(agents=[helper]).tools()
    )

    assert lead.execute_task(_task(lead, "Lead the work")) == "delegated"

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "invoke_agent Helper",
        "execute_tool Delegate work to coworker",
        "invoke_agent Lead",
    ]
    helper_span, tool_span, lead_span = spans
    _assert_child_of(tool_span, lead_span)
    _assert_child_of(helper_span, tool_span)
    assert tool_span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "Lead"
    assert (
        helper_span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "Helper"
    )
    assert _current_agent_name.get() is None


def test_execution_retry_nests_agent_spans(
    instrument_crewai,
    span_exporter,
) -> None:
    agent = _agent(
        FlakyLLM([FINAL_ANSWER.format("recovered")]), max_retry_limit=1
    )

    assert agent.execute_task(_task(agent)) == "recovered"

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "invoke_agent Researcher",
        "invoke_agent Researcher",
    ]
    # The retry re-enters execute_task from inside the failed invocation, and
    # the outer invocation returns the retry's result rather than raising.
    inner, outer = spans
    _assert_child_of(inner, outer)
    assert outer.parent is None
    assert outer.status.status_code == StatusCode.UNSET
    assert inner.status.status_code == StatusCode.UNSET


def test_guardrail_retry_emits_sibling_agent_spans(
    instrument_crewai,
    span_exporter,
    tracer_provider: TracerProvider,
) -> None:
    attempts: list[str] = []

    # CrewAI validates a return annotation against ``Tuple[bool, Any]`` and
    # rejects the stringified form produced by ``from __future__``.
    def guardrail(output: Any):
        attempts.append(output.raw)
        if len(attempts) == 1:
            return False, "Try again"
        return True, output.raw

    agent = _agent(
        ScriptedLLM(
            [FINAL_ANSWER.format("first"), FINAL_ANSWER.format("second")]
        )
    )
    task = _task(agent, guardrail=guardrail, guardrail_max_retries=1)
    tracer = tracer_provider.get_tracer("app")

    with tracer.start_as_current_span("task run") as app_span:
        assert task.execute_sync().raw == "second"

    assert attempts == ["first", "second"]
    agent_spans = _spans_named(
        span_exporter.get_finished_spans(), "invoke_agent Researcher"
    )
    assert len(agent_spans) == 2
    for span in agent_spans:
        assert span.parent is not None
        assert span.parent.span_id == app_span.context.span_id
        assert span.status.status_code == StatusCode.UNSET


@pytest.mark.parametrize("timeout", [None, 120])
def test_provider_span_is_child_of_agent_span(
    instrument_crewai,
    span_exporter,
    tracer_provider: TracerProvider,
    timeout: int | None,
) -> None:
    llm = TracingLLM(
        [ADD_ACTION, FINAL_ANSWER.format(5)], tracer_provider=tracer_provider
    )
    agent = _agent(llm, tools=[AddTool()], timeout=timeout)

    assert agent.execute_task(_task(agent)) == "5"

    spans = span_exporter.get_finished_spans()
    agent_span = _spans_named(spans, "invoke_agent Researcher")[0]
    provider_spans = _spans_named(spans, "chat scripted-model")
    assert len(provider_spans) == 2
    for provider_span in provider_spans:
        _assert_child_of(provider_span, agent_span)
    # CrewAI itself emits no inference telemetry.
    crewai_spans = [
        span
        for span in spans
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name
        == "opentelemetry.instrumentation.genai.crewai"
    ]
    assert {
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        for span in crewai_spans
    } == {"invoke_agent", "execute_tool"}
