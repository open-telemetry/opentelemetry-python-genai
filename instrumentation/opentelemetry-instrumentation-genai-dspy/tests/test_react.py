# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy ReAct (v1) agent instrumentation."""

from __future__ import annotations

import json

import dspy
import pytest

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def async_multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


class MockSyncPredict:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, **kwargs: object) -> object:
        self.count += 1

        class Pred:
            next_thought = "Need to calculate 2 + 2"
            next_tool_name = "add"
            next_tool_args = {"x": 2, "y": 2}

        class FinishPred:
            next_thought = "Done calculation"
            next_tool_name = "finish"
            next_tool_args = {}

        return Pred() if self.count == 1 else FinishPred()


class MockAsyncPredict:
    def __init__(self) -> None:
        self.count = 0

    async def acall(self, **kwargs: object) -> object:
        self.count += 1

        class Pred:
            next_thought = "Need to multiply 3 and 5"
            next_tool_name = "multiply"
            next_tool_args = {"a": 3, "b": 5}

        class FinishPred:
            next_thought = "Done calculation"
            next_tool_name = "finish"
            next_tool_args = {}

        return Pred() if self.count == 1 else FinishPred()


class MockExtract:
    def __call__(self, **kwargs: object) -> dict[str, str]:
        return {"answer": "4"}

    async def acall(self, **kwargs: object) -> dict[str, str]:
        return {"answer": "15"}


def test_react_sync_execution(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        tool = dspy.Tool(add, name="add", desc="Add two numbers.")
        react = dspy.ReAct("question -> answer", tools=[tool])
        react.react = MockSyncPredict()
        react.extract = MockExtract()

        res = react(question="What is 2 + 2?")
        assert res.answer == "4"

    spans = span_exporter.get_finished_spans()
    # Expect 1 execute_tool add, 1 execute_tool finish, 1 invoke_agent dspy.ReAct
    assert len(spans) == 3

    agent_spans = [s for s in spans if s.name == "invoke_agent dspy.ReAct"]
    tool_spans = [s for s in spans if s.name.startswith("execute_tool")]

    assert len(agent_spans) == 1
    agent_span = agent_spans[0]
    assert agent_span.status.status_code == StatusCode.UNSET

    agent_attrs = agent_span.attributes or {}
    assert agent_attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    assert agent_attrs.get(GenAI.GEN_AI_AGENT_NAME) == "dspy.ReAct"

    # Verify input messages
    input_msgs_raw = agent_attrs.get(GenAI.GEN_AI_INPUT_MESSAGES)
    assert input_msgs_raw is not None
    input_msgs = json.loads(str(input_msgs_raw))
    assert input_msgs[0]["parts"][0]["content"] == "What is 2 + 2?"

    # Verify output messages
    output_msgs_raw = agent_attrs.get(GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert output_msgs_raw is not None
    output_msgs = json.loads(str(output_msgs_raw))
    assert output_msgs[0]["parts"][0]["content"] == "4"

    # Verify tool definitions
    tool_defs_raw = agent_attrs.get(GenAI.GEN_AI_TOOL_DEFINITIONS)
    assert tool_defs_raw is not None
    tool_defs = json.loads(str(tool_defs_raw))
    assert any(t["name"] == "add" for t in tool_defs)

    # Verify child tool spans reference agent span
    assert len(tool_spans) == 2
    for t_span in tool_spans:
        assert t_span.parent is not None
        assert t_span.parent.span_id == agent_span.context.span_id


@pytest.mark.skipif(
    not hasattr(dspy.ReAct, "aforward"),
    reason="dspy.ReAct.aforward not available in this DSPy version",
)
@pytest.mark.anyio
async def test_react_async_execution(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        tool = dspy.Tool(
            async_multiply, name="multiply", desc="Multiply numbers."
        )
        react = dspy.ReAct("question -> answer", tools=[tool])
        async_predict = MockAsyncPredict()
        react.react = async_predict
        react.extract = MockExtract()

        res = await react.aforward(question="What is 3 * 5?")
        assert res.answer == "15"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    agent_spans = [s for s in spans if s.name == "invoke_agent dspy.ReAct"]
    assert len(agent_spans) == 1
    agent_span = agent_spans[0]

    agent_attrs = agent_span.attributes or {}
    assert agent_attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    assert agent_attrs.get(GenAI.GEN_AI_AGENT_NAME) == "dspy.ReAct"


def test_react_no_content_capture(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ):
        tool = dspy.Tool(add, name="add", desc="Add two numbers.")
        react = dspy.ReAct("question -> answer", tools=[tool])
        react.react = MockSyncPredict()
        react.extract = MockExtract()

        res = react(question="What is 2 + 2?")
        assert res.answer == "4"

    spans = span_exporter.get_finished_spans()
    agent_spans = [s for s in spans if s.name == "invoke_agent dspy.ReAct"]
    assert len(agent_spans) == 1
    agent_span = agent_spans[0]

    attrs = agent_span.attributes or {}
    assert GenAI.GEN_AI_INPUT_MESSAGES not in attrs
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs
    assert attrs.get(GenAI.GEN_AI_AGENT_NAME) == "dspy.ReAct"


def test_react_error_handling(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        tool = dspy.Tool(add, name="add", desc="Add two numbers.")
        react = dspy.ReAct("question -> answer", tools=[tool])

        class FailingPredict:
            def __call__(self, **kwargs: object) -> object:
                raise RuntimeError("Predict model failure")

        react.react = FailingPredict()

        with pytest.raises(RuntimeError, match="Predict model failure"):
            react(question="What is 2 + 2?")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent dspy.ReAct"
    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "RuntimeError"
