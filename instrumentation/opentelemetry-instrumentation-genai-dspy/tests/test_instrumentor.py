# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DSPy instrumentor lifecycle."""

import copy

import dspy
import pytest
from tests.test_react import MockExtract, MockSyncPredict, add
from tests.test_react_v2 import MockSyncPredictV2

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


def test_instrumentation_dependencies() -> None:
    assert DSPyInstrumentor().instrumentation_dependencies() == (
        "dspy >= 3.3.0, < 4",
    )


def test_instrument_uninstrument_cycle(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    instrumentor = DSPyInstrumentor()

    for _ in range(2):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        instrumentor.uninstrument()


def test_instrument_with_global_providers() -> None:
    instrumentor = DSPyInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()


def test_copy_and_deepcopy_wrapped_callables(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    tool = dspy.Tool(add, name="add", desc="Add two numbers.")
    react = dspy.ReAct("question -> answer", tools=[tool])

    tool_copy = copy.copy(tool)
    tool_deepcopy = copy.deepcopy(tool)
    call_copy = copy.copy(tool.__call__)
    call_deepcopy = copy.deepcopy(tool.__call__)

    assert tool_copy(x=2, y=2) == 4
    assert tool_deepcopy(x=2, y=2) == 4
    assert call_copy(x=2, y=2) == 4
    assert call_deepcopy(x=2, y=2) == 4

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert span.name == "execute_tool add"

    react.react = MockSyncPredict()
    react.extract = MockExtract()

    react_copy = copy.copy(react)
    react_deepcopy = copy.deepcopy(react)
    forward_copy = copy.copy(react.forward)
    forward_deepcopy = copy.deepcopy(react.forward)

    assert react_copy is not None
    assert react_deepcopy is not None
    assert forward_copy is not None
    assert forward_deepcopy is not None

    res = react_deepcopy(question="What is 2 + 2?")
    assert res.answer == "4"

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "invoke_agent dspy.ReAct" in span_names
    assert "execute_tool add" in span_names


def test_copy_and_deepcopy_react_v2(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    react_v2_mod = pytest.importorskip("dspy.predict.react_v2")
    react_v2_cls = getattr(react_v2_mod, "ReActV2", None)
    if react_v2_cls is None:
        pytest.skip("ReActV2 not available")

    tool = dspy.Tool(add, name="add", desc="Add two numbers.")
    react_v2 = react_v2_cls("question -> answer", tools=[tool])
    react_v2.react = MockSyncPredictV2()

    react_v2_copy = copy.copy(react_v2)
    react_v2_deepcopy = copy.deepcopy(react_v2)
    assert react_v2_copy is not None
    assert react_v2_deepcopy is not None
    assert copy.copy(react_v2.forward) is not None
    assert copy.deepcopy(react_v2.forward) is not None

    res = react_v2_deepcopy(question="What is 2 + 2?")
    assert res.answer == "4"

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "invoke_agent dspy.ReActV2" in span_names
    assert "execute_tool add" in span_names


def test_deepcopy_agent_as_tool_module_tree(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    base_tool = dspy.Tool(add, name="add", desc="Add two numbers.")
    sub = dspy.ReAct("question -> answer", tools=[base_tool])
    sub.react = MockSyncPredict()
    sub.extract = MockExtract()

    subagent_tool = dspy.Tool(
        func=sub.forward, name="subagent", desc="Subagent tool"
    )
    outer = dspy.ReAct("question -> answer", tools=[subagent_tool])
    outer.react = MockSyncPredict(
        tool_name="subagent", tool_args={"question": "What is 2 + 2?"}
    )
    outer.extract = MockExtract()

    outer_copy = copy.deepcopy(outer)
    res = outer_copy(question="start")
    assert res.answer == "4"

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "execute_tool add" in span_names
    assert "execute_tool subagent" in span_names
    assert span_names.count("invoke_agent dspy.ReAct") == 2


@pytest.mark.skipif(
    not hasattr(dspy.Tool, "acall"),
    reason="dspy.Tool.acall not available in this DSPy version",
)
@pytest.mark.anyio
async def test_copy_and_deepcopy_async_tool(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    async def async_fn(x: str) -> str:
        return f"async:{x}"

    tool = dspy.Tool(
        func=async_fn, name="async_test_tool", desc="An async test tool"
    )
    tool_copy = copy.copy(tool)
    tool_deepcopy = copy.deepcopy(tool)
    acall_copy = copy.copy(tool.acall)
    acall_deepcopy = copy.deepcopy(tool.acall)

    assert await tool_copy.acall(x="foo") == "async:foo"
    assert await tool_deepcopy.acall(x="bar") == "async:bar"
    assert await acall_copy(x="baz") == "async:baz"
    assert await acall_deepcopy(x="qux") == "async:qux"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert span.name == "execute_tool async_test_tool"
