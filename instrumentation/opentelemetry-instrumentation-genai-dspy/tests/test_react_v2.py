# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy ReActV2 agent instrumentation."""

from __future__ import annotations

import json

import dspy
import pytest

try:
    from dspy.adapters.types.tool import ToolCalls
except ImportError:
    ToolCalls = None  # type: ignore[assignment,misc]

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

pytestmark = pytest.mark.skipif(
    not hasattr(dspy, "ReActV2") or ToolCalls is None,
    reason="dspy.ReActV2 not available in this DSPy version",
)


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


class MockSyncPredictV2:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, **kwargs: object) -> object:
        self.count += 1

        class Pred:
            pass

        p = Pred()
        if self.count == 1:
            p.next_thought = "Add 2 and 2"  # type: ignore[attr-defined]
            p.tool_calls = ToolCalls(  # type: ignore[misc]
                tool_calls=[
                    ToolCalls.ToolCall(  # type: ignore[misc]
                        id="call_1", name="add", args={"x": 2, "y": 2}
                    )
                ]
            )
        else:
            p.next_thought = "Submit answer"  # type: ignore[attr-defined]
            p.tool_calls = ToolCalls(  # type: ignore[misc]
                tool_calls=[
                    ToolCalls.ToolCall(  # type: ignore[misc]
                        id="call_2", name="submit", args={"answer": "4"}
                    )
                ]
            )
        return p


def test_react_v2_sync_execution(
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
        react_v2 = dspy.ReActV2("question -> answer", tools=[add])
        react_v2.react = MockSyncPredictV2()

        res = react_v2(question="What is 2 + 2?")
        assert res.answer == "4"

    spans = span_exporter.get_finished_spans()
    # Expect 1 execute_tool add, 1 execute_tool submit, 1 invoke_agent dspy.ReActV2
    assert len(spans) == 3

    agent_spans = [s for s in spans if s.name == "invoke_agent dspy.ReActV2"]
    tool_spans = [s for s in spans if s.name.startswith("execute_tool")]

    assert len(agent_spans) == 1
    agent_span = agent_spans[0]
    assert agent_span.status.status_code == StatusCode.UNSET

    agent_attrs = agent_span.attributes or {}
    assert agent_attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    assert agent_attrs.get(GenAI.GEN_AI_AGENT_NAME) == "dspy.ReActV2"

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


def test_react_v2_no_content_capture(
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
        react_v2 = dspy.ReActV2("question -> answer", tools=[add])
        react_v2.react = MockSyncPredictV2()

        res = react_v2(question="What is 2 + 2?")
        assert res.answer == "4"

    spans = span_exporter.get_finished_spans()
    agent_spans = [s for s in spans if s.name == "invoke_agent dspy.ReActV2"]
    assert len(agent_spans) == 1
    agent_span = agent_spans[0]

    attrs = agent_span.attributes or {}
    assert GenAI.GEN_AI_INPUT_MESSAGES not in attrs
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs
    assert attrs.get(GenAI.GEN_AI_AGENT_NAME) == "dspy.ReActV2"


def test_react_v2_error_handling(
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
        react_v2 = dspy.ReActV2("question -> answer", tools=[add])

        class FailingPredictV2:
            def __call__(self, **kwargs: object) -> object:
                raise RuntimeError("ReActV2 model error")

        react_v2.react = FailingPredictV2()

        with pytest.raises(RuntimeError, match="ReActV2 model error"):
            react_v2(question="What is 2 + 2?")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent dspy.ReActV2"
    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "RuntimeError"
