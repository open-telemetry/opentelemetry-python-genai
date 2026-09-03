# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DSPy instrumentor lifecycle."""

import copy
from importlib import import_module

import dspy
import pytest

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


def test_instrumentation_dependencies() -> None:
    assert DSPyInstrumentor().instrumentation_dependencies() == (
        "dspy >= 2.6.0, < 4",
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
    tool = dspy.Tool(
        func=lambda x: f"result:{x}", name="test_tool", desc="A test tool"
    )
    react = dspy.ReAct("question -> answer", tools=[tool])

    tool_copy = copy.copy(tool)
    tool_deepcopy = copy.deepcopy(tool)
    call_copy = copy.copy(tool.__call__)
    call_deepcopy = copy.deepcopy(tool.__call__)

    assert tool_copy(x="foo") == "result:foo"
    assert tool_deepcopy(x="bar") == "result:bar"
    assert call_copy(x="baz") == "result:baz"
    assert call_deepcopy(x="qux") == "result:qux"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert span.name == "execute_tool test_tool"

    react_copy = copy.copy(react)
    react_deepcopy = copy.deepcopy(react)
    forward_copy = copy.copy(react.forward)
    forward_deepcopy = copy.deepcopy(react.forward)

    assert react_copy is not None
    assert react_deepcopy is not None
    assert forward_copy is not None
    assert forward_deepcopy is not None

    try:
        react_v2_mod = import_module("dspy.predict.react_v2")
        react_v2_cls = getattr(react_v2_mod, "ReActV2", None)
        if react_v2_cls is not None:
            react_v2 = react_v2_cls("question -> answer", tools=[tool])
            assert copy.copy(react_v2) is not None
            assert copy.deepcopy(react_v2) is not None
            assert copy.copy(react_v2.forward) is not None
            assert copy.deepcopy(react_v2.forward) is not None
    except (ImportError, AttributeError):
        pass


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
