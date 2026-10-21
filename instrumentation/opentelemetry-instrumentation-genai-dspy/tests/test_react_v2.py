# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy ReActV2 agent instrumentation."""

from __future__ import annotations

import json
from typing import Any

import dspy
import pytest
from dspy.adapters.types.tool import ToolCalls
from dspy.predict.react_v2 import ReActV2
from tests.test_react import (
    MockExtract,
    MockSyncPredict,
    assert_agent_finish_reason,
)

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.instrumentation.genai.dspy.patch import (
    _react_v2_finish_reason,
)
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

# DSPy 3.4+ raises, instead of returning, when the forced submit fails.
_FORCED_SUBMIT_FAILURE_RAISES = tuple(
    int(part) for part in dspy.__version__.split(".")[:2]
) >= (3, 4)


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


class MockSyncPredictV2:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, **kwargs: object) -> object:
        self.count += 1

        class CallPred:
            next_thought = "Add 2 and 2"
            tool_calls = ToolCalls(
                tool_calls=[
                    ToolCalls.ToolCall(
                        id="call_1", name="add", args={"x": 2, "y": 2}
                    )
                ]
            )

        class SubmitPred:
            next_thought = "Submit answer"
            tool_calls = ToolCalls(
                tool_calls=[
                    ToolCalls.ToolCall(
                        id="call_2", name="submit", args={"answer": "4"}
                    )
                ]
            )

        return CallPred() if self.count == 1 else SubmitPred()

    async def acall(self, **kwargs: object) -> object:
        return self(**kwargs)


class MockNoToolCallsPredictV2:
    """Answers without tool calls, submitting only when DSPy forces it."""

    def __init__(self, submit_when_forced: bool = True) -> None:
        self.submit_when_forced = submit_when_forced

    def __call__(self, **kwargs: object) -> object:
        class TextPred:
            next_thought = "The answer is 4"
            tool_calls = ToolCalls(tool_calls=[])

        class SubmitPred:
            next_thought = "Submit answer"
            tool_calls = ToolCalls(
                tool_calls=[
                    ToolCalls.ToolCall(
                        id="call_1", name="submit", args={"answer": "4"}
                    )
                ]
            )

        if self.submit_when_forced and "config" in kwargs:
            return SubmitPred()
        return TextPred()

    async def acall(self, **kwargs: object) -> object:
        return self(**kwargs)


class MockNeverSubmitPredictV2:
    """Keeps calling `add`, even when DSPy forces a final submit."""

    def __call__(self, **kwargs: object) -> object:
        class CallPred:
            next_thought = "Add 2 and 2"
            tool_calls = ToolCalls(
                tool_calls=[
                    ToolCalls.ToolCall(
                        id="call_1", name="add", args={"x": 2, "y": 2}
                    )
                ]
            )

        return CallPred()


class MockParseErrorPredictV2:
    """Fails to parse its first reply, then answers without `submit`."""

    def __call__(self, **kwargs: object) -> object:
        if "config" not in kwargs:
            raise ValueError("Could not parse the reply")

        class TextPred:
            next_thought = "The answer is 4"
            tool_calls = ToolCalls(tool_calls=[])

        return TextPred()


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
    # Expect 1 execute_tool add, 1 invoke_agent dspy.ReActV2 (submit sentinel skipped)
    assert len(spans) == 2

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

    # Verify child tool spans reference agent span and submit is skipped
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "execute_tool add"
    assert not any(s.name == "execute_tool submit" for s in spans)
    assert tool_spans[0].parent is not None
    assert tool_spans[0].parent.span_id == agent_span.context.span_id


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


def test_react_v2_missing_tolerated(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    import dspy.predict.react_v2

    monkeypatch.delattr(dspy.predict.react_v2, "ReActV2", raising=False)

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
    assert len(spans) == 2
    assert any(s.name == "invoke_agent dspy.ReAct" for s in spans)
    assert any(s.name == "execute_tool add" for s in spans)


@pytest.mark.parametrize("content_capture", ["SPAN_ONLY", "NO_CONTENT"])
@pytest.mark.parametrize(
    ("call_kwargs", "finish_reason"),
    [({}, "stop"), ({"max_iters": 1}, "error")],
    ids=["submit", "max_iters"],
)
def test_react_v2_finish_reasons(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    content_capture: str,
    call_kwargs: dict[str, Any],
    finish_reason: str,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture=content_capture,
    ):
        react_v2 = dspy.ReActV2("question -> answer", tools=[add])
        react_v2.react = MockSyncPredictV2()

        res = react_v2(question="What is 2 + 2?", **call_kwargs)
        assert res.answer == "4"

    assert_agent_finish_reason(
        span_exporter, "dspy.ReActV2", finish_reason, content_capture
    )


@pytest.mark.skipif(
    not hasattr(ReActV2, "aforward"),
    reason="ReActV2.aforward not available in this DSPy version",
)
@pytest.mark.anyio
@pytest.mark.parametrize("content_capture", ["SPAN_ONLY", "NO_CONTENT"])
@pytest.mark.parametrize(
    ("max_iters", "finish_reason"),
    [(20, "stop"), (1, "error")],
    ids=["submit", "max_iters"],
)
async def test_react_v2_async_finish_reasons(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    content_capture: str,
    max_iters: int,
    finish_reason: str,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture=content_capture,
    ):
        react_v2 = dspy.ReActV2(
            "question -> answer", tools=[add], max_iters=max_iters
        )
        react_v2.react = MockSyncPredictV2()

        res = await react_v2.aforward(question="What is 2 + 2?")
        assert res.answer == "4"

    assert_agent_finish_reason(
        span_exporter, "dspy.ReActV2", finish_reason, content_capture
    )


def test_react_v2_finish_reason_when_loop_ends_early(
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
        react_v2.react = MockNoToolCallsPredictV2()

        res = react_v2(question="What is 2 + 2?")
        assert res.answer == "4"
        assert res.termination_reason == "forced_submit"

    assert_agent_finish_reason(
        span_exporter, "dspy.ReActV2", "error", "SPAN_ONLY"
    )


@pytest.mark.skipif(
    not hasattr(ReActV2, "aforward"),
    reason="ReActV2.aforward not available in this DSPy version",
)
@pytest.mark.anyio
async def test_react_v2_async_finish_reason_when_loop_ends_early(
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
        react_v2.react = MockNoToolCallsPredictV2()

        res = await react_v2.aforward(question="What is 2 + 2?")
        assert res.answer == "4"
        assert res.termination_reason == "forced_submit"

    assert_agent_finish_reason(
        span_exporter, "dspy.ReActV2", "error", "SPAN_ONLY"
    )


@pytest.mark.parametrize(
    ("termination_reason", "finish_reason"),
    [
        ("submit", "stop"),
        ("forced_submit", "error"),
        ("parse_error", "error"),
        ("context_window_exceeded", "context_window_exceeded"),
        (None, None),
    ],
)
def test_react_v2_finish_reason_mapping(
    termination_reason: str | None, finish_reason: str | None
) -> None:
    result = dspy.Prediction(answer="4", termination_reason=termination_reason)
    assert _react_v2_finish_reason(result) == finish_reason


@pytest.mark.parametrize(
    ("predictor", "termination_reason", "finish_reason"),
    [
        (MockNeverSubmitPredictV2(), "max_iters", "max_iters"),
        (
            MockNoToolCallsPredictV2(submit_when_forced=False),
            "empty_tool_calls",
            "empty_tool_calls",
        ),
        (MockParseErrorPredictV2(), "parse_error", "error"),
    ],
    ids=["max_iters", "empty_tool_calls", "parse_error"],
)
def test_react_v2_finish_reason_when_forced_submit_fails(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    predictor: object,
    termination_reason: str,
    finish_reason: str,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ):
        react_v2 = dspy.ReActV2("question -> answer", tools=[add], max_iters=1)
        react_v2.react = predictor

        if _FORCED_SUBMIT_FAILURE_RAISES:
            with pytest.raises(ValueError, match="failed to produce"):
                react_v2(question="What is 2 + 2?")
        else:
            res = react_v2(question="What is 2 + 2?")
            assert res.termination_reason == termination_reason

    if not _FORCED_SUBMIT_FAILURE_RAISES:
        assert_agent_finish_reason(
            span_exporter, "dspy.ReActV2", finish_reason, "NO_CONTENT"
        )
        return

    agent_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name == "invoke_agent dspy.ReActV2"
    ]
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code == StatusCode.ERROR
    attrs = agent_spans[0].attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "ValueError"
    assert GenAI.GEN_AI_RESPONSE_FINISH_REASONS not in attrs
