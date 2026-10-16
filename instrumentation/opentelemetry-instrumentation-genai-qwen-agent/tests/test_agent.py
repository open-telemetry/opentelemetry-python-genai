# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agent.run() instrumentation (invoke_agent spans)."""
# pylint: disable=redefined-outer-name

import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qwen_agent.agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm import base as qwen_llm_base
from qwen_agent.llm.schema import FunctionCall, Message
from qwen_agent.tools.base import BaseTool, register_tool

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode

# The tool-call cassette was recorded through the DashScope raw API
# (``use_raw_api``); older qwen-agent versions without it emulate function
# calling via prompt templates and cannot parse the recorded response.
_SUPPORTS_RAW_API = "use_raw_api" in inspect.getsource(qwen_llm_base)


class _StubAgent(Agent):
    """A minimal Agent subclass that skips the heavy Agent.__init__."""

    @classmethod
    def create(cls, name="TestAgent", llm=None):
        obj = cls.__new__(cls)
        obj.name = name
        obj.description = "A test agent"
        obj.system_message = "You are a helpful assistant."
        obj.llm = llm
        obj.function_map = {}
        obj.extra_generate_cfg = {}
        return obj

    def _run(self, messages, **kwargs):
        raise NotImplementedError


class _StubLLM:
    model = "qwen-max"
    model_type = "qwen_dashscope"


def _spans_named(span_exporter, name):
    return [s for s in span_exporter.get_finished_spans() if s.name == name]


def _metric_points(metric_reader, name):
    data = metric_reader.get_metrics_data()
    return [
        point
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def test_agent_run_counts_calls_once_per_invocation(
    span_exporter, metric_reader, instrument_no_content
):
    agent = _StubAgent.create(name="CountingBot", llm=_StubLLM())
    agent.llm.chat = lambda **kwargs: iter(
        [[Message(role="assistant", content="ok")]]
    )
    agent.function_map["get_weather"] = SimpleNamespace(
        call=lambda *_args, **_kwargs: "sunny"
    )

    def fake_run(messages, **kwargs):
        for response in agent._call_llm(messages):
            yield response
        agent._call_tool("get_weather", "{}")
        for response in agent._call_llm(messages):
            yield response

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        run = agent.run([Message(role="user", content="Go")])
        assert not span_exporter.get_finished_spans()
        assert len(list(run)) == 2

    assert len(_spans_named(span_exporter, "invoke_agent CountingBot")) == 1
    attrs = dict(
        _spans_named(span_exporter, "invoke_agent CountingBot")[0].attributes
        or {}
    )
    assert GenAIAttributes.GEN_AI_REQUEST_STREAM not in attrs
    for name, expected in (
        ("gen_ai.invoke_agent.inference_calls", 2),
        ("gen_ai.invoke_agent.tool_calls", 1),
    ):
        points = _metric_points(metric_reader, name)
        assert len(points) == 1
        assert points[0].sum == expected
        assert isinstance(points[0].sum, int)
        assert dict(points[0].attributes) == {
            GenAIAttributes.GEN_AI_AGENT_NAME: "CountingBot"
        }


def test_agent_run_counts_failed_inference_and_tool(
    metric_reader, instrument_no_content
):
    agent = _StubAgent.create(name="FailingCounter", llm=_StubLLM())
    agent.llm.chat = lambda **kwargs: iter(())

    def fake_run(messages, **kwargs):
        agent._call_llm(messages)
        agent._call_tool("missing", "{}")
        raise ValueError("agent failed")
        yield  # pragma: no cover

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        with pytest.raises(ValueError, match="agent failed"):
            list(agent.run([Message(role="user", content="Go")]))

    assert (
        _metric_points(metric_reader, "gen_ai.invoke_agent.inference_calls")[
            0
        ].sum
        == 1
    )
    assert (
        _metric_points(metric_reader, "gen_ai.invoke_agent.tool_calls")[0].sum
        == 1
    )


def test_assistant_counts_unknown_and_known_tool_calls_once(
    metric_reader, instrument_no_content
):
    agent = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="AssistantCounter",
    )
    agent.function_map["known"] = SimpleNamespace(
        file_access=False,
        call=lambda *_args, **_kwargs: "ok",
        description="A test tool",
    )

    def fake_run(messages, **kwargs):
        assert (
            agent._call_tool("missing", "{}")
            == "Tool missing does not exists."
        )
        assert agent._call_tool("known", "{}") == "ok"
        yield [Message(role="assistant", content="done")]

    with patch.object(agent, "_run", side_effect=fake_run):
        list(agent.run([Message(role="user", content="Go")]))

    points = _metric_points(metric_reader, "gen_ai.invoke_agent.tool_calls")
    assert len(points) == 1
    assert points[0].sum == 2
    assert isinstance(points[0].sum, int)


def test_nested_agent_call_counts_are_separate(
    metric_reader, instrument_no_content
):
    parent = _StubAgent.create(name="ParentCounter", llm=_StubLLM())
    child = _StubAgent.create(name="ChildCounter", llm=_StubLLM())
    parent.llm.chat = child.llm.chat = lambda **kwargs: iter(())

    def fake_run(self, messages, **kwargs):
        self._call_llm(messages)
        if self is parent:
            yield from child.run([Message(role="user", content="child")])
            self._call_llm(messages)
        yield [Message(role="assistant", content="done")]

    with patch.object(_StubAgent, "_run", autospec=True, side_effect=fake_run):
        list(parent.run([Message(role="user", content="parent")]))

    points = _metric_points(
        metric_reader, "gen_ai.invoke_agent.inference_calls"
    )
    assert {
        point.attributes[GenAIAttributes.GEN_AI_AGENT_NAME]: point.sum
        for point in points
    } == {"ParentCounter": 2, "ChildCounter": 1}


@pytest.mark.vcr()
def test_agent_run(span_exporter, instrument_with_content):
    """Assistant.run() produces an invoke_agent span."""
    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="TestAssistant",
        description="A test assistant.",
        system_message="You are a helpful assistant.",
    )
    responses = list(
        bot.run([{"role": "user", "content": "Hello, what is 1+1?"}])
    )
    assert responses

    agent_spans = _spans_named(span_exporter, "invoke_agent TestAssistant")
    assert len(agent_spans) == 1
    agent_span = agent_spans[0]
    assert agent_span.kind == SpanKind.INTERNAL
    assert agent_span.status.status_code != StatusCode.ERROR

    attrs = dict(agent_span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_NAME] == "TestAssistant"
    assert (
        attrs[GenAIAttributes.GEN_AI_AGENT_DESCRIPTION] == "A test assistant."
    )
    # invoke_local_agent() produces INTERNAL spans without a provider.
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in attrs
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "qwen-max"

    input_messages = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[-1]["role"] == "user"
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    system_instructions = json.loads(
        attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    )
    assert system_instructions == [
        {"content": "You are a helpful assistant.", "type": "text"}
    ]


@pytest.mark.vcr()
def test_agent_run_nonstream(span_exporter, instrument_with_content):
    """run_nonstream() calls run() internally; exactly one invoke_agent span."""
    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="NonStreamAssistant",
    )
    result = bot.run_nonstream(
        [{"role": "user", "content": "Say 'OK' and nothing else."}]
    )
    assert result

    agent_spans = _spans_named(
        span_exporter, "invoke_agent NonStreamAssistant"
    )
    assert len(agent_spans) == 1


@pytest.mark.vcr()
def test_multi_turn_conversation(span_exporter, instrument_with_content):
    """History messages are all captured on the invoke_agent span input."""
    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="MultiTurnAssistant",
    )
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ]
    list(bot.run(messages))

    agent_spans = _spans_named(
        span_exporter, "invoke_agent MultiTurnAssistant"
    )
    assert len(agent_spans) == 1
    attrs = dict(agent_spans[0].attributes or {})
    input_messages = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    roles = [message["role"] for message in input_messages]
    assert roles == ["user", "assistant", "user"]


@pytest.mark.skipif(
    not _SUPPORTS_RAW_API,
    reason="cassette recorded via the DashScope raw API, which this"
    " qwen-agent version does not support",
)
@pytest.mark.vcr()
def test_agent_run_with_tool_call(
    span_exporter, metric_reader, instrument_with_content
):
    """A tool-calling agent run nests execute_tool spans."""

    @register_tool("get_current_weather_test", allow_overwrite=True)
    class _GetCurrentWeatherTool(BaseTool):
        description = "Get the current weather for a given city."
        parameters = [
            {
                "name": "city",
                "type": "string",
                "description": "The city name to get weather for.",
                "required": True,
            }
        ]

        def call(self, params, **kwargs):
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except ValueError:
                    params = {"city": params}
            city = (
                params.get("city", "unknown")
                if isinstance(params, dict)
                else "unknown"
            )
            return f"The weather in {city} is sunny and 22 degrees Celsius."

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="WeatherAgent",
        function_list=["get_current_weather_test"],
    )
    list(
        bot.run(
            [
                {
                    "role": "user",
                    "content": "What is the weather in Beijing right now?",
                }
            ]
        )
    )

    agent_spans = _spans_named(span_exporter, "invoke_agent WeatherAgent")
    assert len(agent_spans) == 1
    agent_span = agent_spans[0]

    tool_spans = _spans_named(
        span_exporter, "execute_tool get_current_weather_test"
    )
    # The recorded run has the model requesting the tool twice before
    # producing the final answer.
    assert len(tool_spans) == 2
    tool_span = tool_spans[0]
    tool_attrs = dict(tool_span.attributes or {})
    assert tool_attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert (
        tool_attrs[GenAIAttributes.GEN_AI_TOOL_NAME]
        == "get_current_weather_test"
    )
    assert tool_attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
    assert (
        tool_attrs[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION]
        == "Get the current weather for a given city."
    )
    arguments = tool_attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
    assert isinstance(arguments, str)
    assert "Beijing" in arguments
    result = tool_attrs[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT]
    assert isinstance(result, str)
    assert "sunny" in result

    # The tool span is nested (transitively) under the invoke_agent span.
    assert tool_span.context.trace_id == agent_span.context.trace_id
    assert tool_span.parent is not None

    for name, expected in (
        ("gen_ai.invoke_agent.inference_calls", 3),
        ("gen_ai.invoke_agent.tool_calls", 2),
    ):
        points = [
            point
            for point in _metric_points(metric_reader, name)
            if point.attributes[GenAIAttributes.GEN_AI_AGENT_NAME]
            == "WeatherAgent"
        ]
        assert len(points) == 1
        assert points[0].sum == expected


def test_agent_run_error(span_exporter, instrument_no_content):
    """An exception during the run re-raises and fails the invoke_agent span."""
    agent = _StubAgent.create(name="FailBot", llm=_StubLLM())

    def fake_run(messages, **kwargs):
        if False:  # pylint: disable=using-constant-test
            yield
        raise ValueError("Agent processing failed")

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        with pytest.raises(ValueError, match="Agent processing failed"):
            list(agent.run([Message(role="user", content="Go")]))

    agent_spans = _spans_named(span_exporter, "invoke_agent FailBot")
    assert len(agent_spans) == 1
    span = agent_spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[ErrorAttributes.ERROR_TYPE] == "ValueError"


def test_agent_run_early_close_finalizes_span(
    span_exporter, instrument_no_content
):
    """Closing the run generator before draining still ends the span."""
    agent = _StubAgent.create(name="EarlyCloseBot", llm=_StubLLM())

    def fake_run(messages, **kwargs):
        yield [Message(role="assistant", content="chunk 1")]
        yield [Message(role="assistant", content="chunk 2")]

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        run_iter = agent.run([Message(role="user", content="Go")])
        next(run_iter)
        run_iter.close()

    agent_spans = _spans_named(span_exporter, "invoke_agent EarlyCloseBot")
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code != StatusCode.ERROR


def test_agent_run_multiple_yields_single_span(
    span_exporter, instrument_no_content
):
    """A run yielding multiple times produces exactly one invoke_agent span."""
    agent = _StubAgent.create(name="MultiYieldBot", llm=_StubLLM())

    def fake_run(messages, **kwargs):
        yield [Message(role="assistant", content="Thinking...")]
        yield [Message(role="assistant", content="Done!")]

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        results = list(agent.run([Message(role="user", content="Go")]))

    assert len(results) == 2
    agent_spans = _spans_named(span_exporter, "invoke_agent MultiYieldBot")
    assert len(agent_spans) == 1


def test_agent_run_records_only_final_output_message(
    span_exporter, instrument_with_content
):
    """Agent output keeps the final answer, not tool calls/results."""
    agent = _StubAgent.create(name="ToolHistoryBot", llm=_StubLLM())

    response_msgs = [
        Message(
            role="assistant",
            content="",
            function_call=FunctionCall(
                name="get_snapshot", arguments='{"incident_id": "INC-1"}'
            ),
        ),
        Message(
            role="function",
            name="get_snapshot",
            content='{"p95_latency_ms": 1840}',
        ),
        Message(role="assistant", content="Final verdict: continue."),
    ]

    def fake_run(messages, **kwargs):
        yield response_msgs

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        list(agent.run([Message(role="user", content="diagnose")]))

    agent_spans = _spans_named(span_exporter, "invoke_agent ToolHistoryBot")
    assert len(agent_spans) == 1
    attrs = dict(agent_spans[0].attributes or {})
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"
    assert output_messages[0]["parts"] == [
        {"content": "Final verdict: continue.", "type": "text"}
    ]


def test_agent_run_input_history_records_tool_call_parts(
    span_exporter, instrument_with_content
):
    """Tool-call history in the run input round-trips onto the invoke_agent
    span as ``tool_call`` and ``tool_call_response`` message parts."""
    agent = _StubAgent.create(name="ToolPartsBot", llm=_StubLLM())

    history = [
        Message(role="user", content="What is the weather in Beijing?"),
        Message(
            role="assistant",
            content="",
            function_call=FunctionCall(
                name="get_weather", arguments='{"city": "Beijing"}'
            ),
        ),
        Message(
            role="function",
            name="get_weather",
            content="Sunny, 22 degrees Celsius.",
        ),
    ]

    def fake_run(messages, **kwargs):
        yield [Message(role="assistant", content="It is sunny in Beijing.")]

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        list(agent.run(history))

    agent_spans = _spans_named(span_exporter, "invoke_agent ToolPartsBot")
    assert len(agent_spans) == 1
    attrs = dict(agent_spans[0].attributes or {})
    input_messages = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    parts = [part for message in input_messages for part in message["parts"]]

    tool_calls = [part for part in parts if part["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert tool_calls[0]["arguments"] == {"city": "Beijing"}

    tool_call_responses = [
        part for part in parts if part["type"] == "tool_call_response"
    ]
    assert len(tool_call_responses) == 1
    assert tool_call_responses[0]["response"] == "Sunny, 22 degrees Celsius."


def test_nested_agent_runs_produce_nested_spans(
    span_exporter, instrument_no_content
):
    """A sub-agent run nests its invoke_agent span under the parent's."""
    parent_agent = _StubAgent.create(name="ParentBot", llm=_StubLLM())
    child_agent = _StubAgent.create(name="ChildBot", llm=_StubLLM())

    def fake_run(self, messages, **kwargs):
        if self is parent_agent:
            yield from child_agent.run(
                [Message(role="user", content="child task")]
            )
            yield [Message(role="assistant", content="parent final")]
        else:
            yield [Message(role="assistant", content="child final")]

    with patch.object(_StubAgent, "_run", autospec=True, side_effect=fake_run):
        list(parent_agent.run([Message(role="user", content="parent task")]))

    parent_spans = _spans_named(span_exporter, "invoke_agent ParentBot")
    child_spans = _spans_named(span_exporter, "invoke_agent ChildBot")
    assert len(parent_spans) == 1
    assert len(child_spans) == 1
    assert child_spans[0].parent is not None
    assert child_spans[0].parent.span_id == parent_spans[0].context.span_id
