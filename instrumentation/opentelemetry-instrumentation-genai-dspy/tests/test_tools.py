# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy Tool instrumentation."""

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


def failing_tool(x: int) -> int:
    """Always fails."""
    raise ValueError("Calculation error")


async def async_failing_tool(x: int) -> int:
    """Always fails asynchronously."""
    raise RuntimeError("Async calculation error")


def test_sync_tool_execution(
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
        res = tool(x=3, y=4)
        assert res == 7

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "execute_tool add"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "execute_tool"
    assert attrs.get(GenAI.GEN_AI_TOOL_NAME) == "add"
    assert attrs.get(GenAI.GEN_AI_TOOL_TYPE) == "function"
    assert attrs.get(GenAI.GEN_AI_TOOL_DESCRIPTION) == "Add two numbers."

    args_attr = attrs.get(GenAI.GEN_AI_TOOL_CALL_ARGUMENTS)
    assert args_attr is not None
    assert json.loads(str(args_attr)) == {"x": 3, "y": 4}
    assert attrs.get(GenAI.GEN_AI_TOOL_CALL_RESULT) == 7


def test_sync_tool_positional_and_mixed_args_extraction(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    from opentelemetry.instrumentation.genai.dspy.patch import (
        _extract_tool_arguments,
    )

    tool = dspy.Tool(add, name="add", desc="Add two numbers.")
    extracted = _extract_tool_arguments(tool, (5,), {"y": 10})
    assert extracted == {"x": 5, "y": 10}

    extracted_pos = _extract_tool_arguments(tool, (5, 10), {})
    assert extracted_pos == {"x": 5, "y": 10}


@pytest.mark.skipif(
    not hasattr(dspy.Tool, "acall"),
    reason="dspy.Tool.acall not available in this DSPy version",
)
@pytest.mark.anyio
async def test_async_tool_execution(
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
        res = await tool.acall(a=5, b=6)
        assert res == 30

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "execute_tool multiply"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "execute_tool"
    assert attrs.get(GenAI.GEN_AI_TOOL_NAME) == "multiply"
    assert attrs.get(GenAI.GEN_AI_TOOL_TYPE) == "function"
    assert attrs.get(GenAI.GEN_AI_TOOL_DESCRIPTION) == "Multiply numbers."

    args_attr = attrs.get(GenAI.GEN_AI_TOOL_CALL_ARGUMENTS)
    assert args_attr is not None
    assert json.loads(str(args_attr)) == {"a": 5, "b": 6}
    assert attrs.get(GenAI.GEN_AI_TOOL_CALL_RESULT) == 30


def test_sync_tool_error(
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
        tool = dspy.Tool(failing_tool, name="failing_tool")
        with pytest.raises(ValueError, match="Calculation error"):
            tool(x=10)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "ValueError"
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "execute_tool"
    assert attrs.get(GenAI.GEN_AI_TOOL_NAME) == "failing_tool"


@pytest.mark.skipif(
    not hasattr(dspy.Tool, "acall"),
    reason="dspy.Tool.acall not available in this DSPy version",
)
@pytest.mark.anyio
async def test_async_tool_error(
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
        tool = dspy.Tool(async_failing_tool, name="async_failing_tool")
        with pytest.raises(RuntimeError, match="Async calculation error"):
            await tool.acall(x=10)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "RuntimeError"
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "execute_tool"


def test_tool_content_capture_disabled(
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
        tool = dspy.Tool(add, name="add")
        res = tool(x=1, y=2)
        assert res == 3

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    attrs = span.attributes or {}
    assert GenAI.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
    assert GenAI.GEN_AI_TOOL_CALL_RESULT not in attrs
    assert attrs.get(GenAI.GEN_AI_TOOL_NAME) == "add"


def test_tool_kwargs_not_mutated_by_caller(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    def greet(msg: str) -> str:
        return f"Hello, {msg}"

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        tool = dspy.Tool(greet, name="greet")
        kwargs = {"msg": "world"}
        tool(**kwargs)
        kwargs["msg"] = "mutated"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    args_raw = attrs.get(GenAI.GEN_AI_TOOL_CALL_ARGUMENTS)
    assert args_raw is not None
    args_dict = json.loads(str(args_raw))
    assert args_dict.get("msg") == "world"
