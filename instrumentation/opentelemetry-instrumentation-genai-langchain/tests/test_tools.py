# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool-related support in the LangChain callback handler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool

from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.genai.langchain.utils import (
    _get_property_value,
    prepare_tool_definitions,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import FunctionToolDefinition


@pytest.fixture(autouse=True)
def reset_semconv_stability():
    """Reset the semconv stability cache before and after each test."""
    _OpenTelemetrySemanticConventionStability._initialized = False
    _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING = {}
    yield
    _OpenTelemetrySemanticConventionStability._initialized = False
    _OpenTelemetrySemanticConventionStability._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING = {}


# ---------------------------------------------------------------------------
# Unit tests for _get_property_value
# ---------------------------------------------------------------------------


def test_get_property_value_from_dict():
    assert _get_property_value({"name": "my_tool"}, "name") == "my_tool"


def test_get_property_value_from_dict_missing_key():
    assert _get_property_value({}, "name") is None


def test_get_property_value_from_object():
    obj = MagicMock()
    obj.name = "obj_tool"
    assert _get_property_value(obj, "name") == "obj_tool"


def test_get_property_value_from_object_missing_attr():
    class Plain:
        pass

    assert _get_property_value(Plain(), "missing") is None


# ---------------------------------------------------------------------------
# Unit tests for prepare_tool_definitions
# ---------------------------------------------------------------------------


def test_prepare_tool_definitions_returns_none_for_empty():
    assert prepare_tool_definitions([]) is None


def test_prepare_tool_definitions_dict_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                },
            },
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    defn = result[0]
    assert isinstance(defn, FunctionToolDefinition)
    assert defn.name == "multiply"
    assert defn.description == "Multiply two numbers"
    assert defn.parameters is not None


def test_prepare_tool_definitions_skips_non_function_type():
    tools = [{"type": "retrieval", "retrieval": {}}]
    result = prepare_tool_definitions(tools)
    assert result is None


def test_prepare_tool_definitions_multiple_tools():
    tools = [
        {
            "type": "function",
            "function": {"name": "add", "description": "Add numbers"},
        },
        {
            "type": "function",
            "function": {
                "name": "subtract",
                "description": "Subtract numbers",
            },
        },
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 2
    assert result[0].name == "add"
    assert result[1].name == "subtract"


def test_prepare_tool_definitions_missing_name_defaults_to_empty_string():
    tools = [
        {
            "type": "function",
            "function": {"description": "No name tool"},
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == ""


def test_prepare_tool_definitions_none_description_stays_none():
    tools = [
        {
            "type": "function",
            "function": {"name": "no_desc"},
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert result[0].description is None


def test_prepare_tool_definitions_object_tools():
    """Tools may be objects (e.g. pydantic models) rather than dicts."""

    class FuncDef:
        name = "get_weather"
        description = "Get current weather"
        parameters = {"type": "object"}

    class ToolDef:
        type = "function"
        function = FuncDef()

    result = prepare_tool_definitions([ToolDef()])
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "get_weather"
    assert result[0].description == "Get current weather"


# ---------------------------------------------------------------------------
# Helpers shared by callback-handler integration tests
# ---------------------------------------------------------------------------


def _make_providers():
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    return tracer_provider, span_exporter, logger_provider, meter_provider


def _make_handler(tracer_provider, logger_provider, meter_provider):
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )


def _make_callback_handler(tracer_provider, logger_provider, meter_provider):
    telemetry_handler = _make_handler(
        tracer_provider, logger_provider, meter_provider
    )
    return OpenTelemetryLangChainCallbackHandler(telemetry_handler)


_OPENAI_SERIALIZED: dict[str, Any] = {"name": "ChatOpenAI"}
_OPENAI_INVOCATION_PARAMS: dict[str, Any] = {
    "model_name": "gpt-4",
    "temperature": 0.0,
}
_OPENAI_METADATA: dict[str, Any] = {"ls_provider": "openai"}


# ---------------------------------------------------------------------------
# on_tool_start / on_tool_end
# ---------------------------------------------------------------------------


def test_on_tool_start_and_end_creates_span(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "multiply", "description": "Multiply two numbers"},
        input_str="",
        run_id=run_id,
        inputs={"a": 3, "b": 4},
    )

    output = MagicMock()
    output.content = "12"
    output.tool_call_id = "call_abc"
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool multiply"
    attrs = span.attributes
    assert attrs[gen_ai_attributes.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_NAME] == "multiply"
    assert (
        attrs[gen_ai_attributes.GEN_AI_TOOL_DESCRIPTION]
        == "Multiply two numbers"
    )
    assert (
        attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS] == '{"a":3,"b":4}'
    )


def test_on_tool_start_with_string_input(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "search"},
        input_str="Paris weather",
        run_id=run_id,
    )
    output = MagicMock()
    output.content = "Sunny"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_NAME] == "search"
    assert (
        attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS] == "Paris weather"
    )


def test_on_tool_start_with_no_serialized(monkeypatch):
    """on_tool_start with serialized=None falls back to name='unknown'."""
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized=None,
        input_str="some input",
        run_id=run_id,
    )
    output = MagicMock()
    output.content = "result"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_NAME] == "unknown"


def test_on_tool_error_records_error_type(monkeypatch):
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "failing_tool"},
        input_str="bad input",
        run_id=run_id,
    )
    exc = ValueError("something went wrong")
    handler.on_tool_error(error=exc, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_NAME] == "failing_tool"
    assert attrs[error_attributes.ERROR_TYPE] == "ValueError"


# ---------------------------------------------------------------------------
# on_chat_model_start with tool_definitions
# ---------------------------------------------------------------------------


def test_on_chat_model_start_with_tools_sets_definitions(monkeypatch):
    """Tool definitions passed via invocation_params are captured on the span."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {"type": "object"},
            },
        }
    ]

    handler.on_chat_model_start(
        serialized=_OPENAI_SERIALIZED,
        messages=[[HumanMessage(content="What is 3 * 4?")]],
        run_id=run_id,
        metadata=_OPENAI_METADATA,
        invocation_params={**_OPENAI_INVOCATION_PARAMS, "tools": tools},
    )

    # Finish the span so attributes are flushed
    ai_msg = AIMessage(content="12")
    ai_msg.response_metadata = {"finish_reason": "stop"}
    generation = ChatGeneration(message=ai_msg, text="12")
    generation.generation_info = {"finish_reason": "stop"}
    result = LLMResult(generations=[[generation]])
    handler.on_llm_end(response=result, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    # Tool definitions are serialised into gen_ai.tool.definitions
    assert gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS in attrs
    tool_definitions = attrs[gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS]
    assert "multiply" in tool_definitions
    assert "Multiply two numbers" in tool_definitions


# ---------------------------------------------------------------------------
# on_llm_end with tool_calls finish reason
# ---------------------------------------------------------------------------


def _build_tool_call_llm_result(
    tool_calls: list[dict[str, Any]],
) -> LLMResult:
    """Build a fake LLMResult where the model responded with tool calls."""
    ai_msg = AIMessage(content="")
    ai_msg.tool_calls = tool_calls  # type: ignore[attr-defined]
    ai_msg.response_metadata = {}
    ai_msg.usage_metadata = None  # type: ignore[assignment]
    generation = ChatGeneration(message=ai_msg, text="")
    generation.generation_info = {"finish_reason": "tool_calls"}
    return LLMResult(generations=[[generation]])


def test_on_llm_end_with_tool_calls_records_tool_call_requests(monkeypatch):
    """When finish_reason is tool_calls the output message parts are ToolCallRequests."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_chat_model_start(
        serialized=_OPENAI_SERIALIZED,
        messages=[[HumanMessage(content="What is 3 * 4?")]],
        run_id=run_id,
        metadata=_OPENAI_METADATA,
        invocation_params=_OPENAI_INVOCATION_PARAMS,
    )

    result = _build_tool_call_llm_result(
        [{"name": "multiply", "id": "call_001", "args": {"a": 3, "b": 4}}]
    )
    handler.on_llm_end(response=result, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES in attrs
    output_messages = attrs[gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES]
    assert "multiply" in output_messages
    assert "tool_calls" in output_messages


def test_on_llm_end_with_multiple_tool_calls(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_chat_model_start(
        serialized=_OPENAI_SERIALIZED,
        messages=[[HumanMessage(content="Compute 3*4 and 5+6")]],
        run_id=run_id,
        metadata=_OPENAI_METADATA,
        invocation_params=_OPENAI_INVOCATION_PARAMS,
    )

    result = _build_tool_call_llm_result(
        [
            {"name": "multiply", "id": "call_001", "args": {"a": 3, "b": 4}},
            {"name": "add", "id": "call_002", "args": {"a": 5, "b": 6}},
        ]
    )
    handler.on_llm_end(response=result, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    output_messages = attrs[gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES]
    assert "multiply" in output_messages
    assert "add" in output_messages


_BEDROCK_SERIALIZED: dict[str, Any] = {"name": "ChatBedrock"}
_BEDROCK_INVOCATION_PARAMS: dict[str, Any] = {
    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}
_BEDROCK_METADATA: dict[str, Any] = {
    "ls_provider": "amazon_bedrock",
    "ls_model_type": "chat",
    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


def test_on_llm_end_with_bedrock_tool_use_records_tool_call_requests(
    monkeypatch,
):
    """When finish_reason is 'tool_use' (Bedrock/Anthropic stopReason) the
    output message parts must be ToolCallRequests, same as for OpenAI's
    'tool_calls' finish reason."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_chat_model_start(
        serialized=_BEDROCK_SERIALIZED,
        messages=[[HumanMessage(content="What is 3 * 4?")]],
        run_id=run_id,
        metadata=_BEDROCK_METADATA,
        invocation_params=_BEDROCK_INVOCATION_PARAMS,
    )

    # Bedrock path: generation_info is None; stopReason lives in response_metadata
    tool_call = {
        "name": "multiply",
        "id": "tooluse_001",
        "args": {"a": 3, "b": 4},
    }
    ai_msg = AIMessage(content="")
    ai_msg.tool_calls = [tool_call]  # type: ignore[attr-defined]
    ai_msg.response_metadata = {"stopReason": "tool_use"}
    ai_msg.usage_metadata = None  # type: ignore[assignment]
    gen = ChatGeneration(message=ai_msg, text="")
    gen.generation_info = None
    result = LLMResult(generations=[[gen]])

    handler.on_llm_end(response=result, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES in attrs
    output_messages = attrs[gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES]
    assert "multiply" in output_messages
    assert "tool_use" in output_messages


# ---------------------------------------------------------------------------
# Full LangChain tool invocation via instrumentor (no network)
# ---------------------------------------------------------------------------


def test_tool_span_created_via_instrumentor(monkeypatch):
    """Using LangChainInstrumentor, on_tool_start/end produces an execute_tool span."""

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )

    try:

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        multiply.invoke({"a": 3, "b": 4})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "execute_tool multiply"
        attrs = span.attributes
        assert attrs[gen_ai_attributes.GEN_AI_OPERATION_NAME] == "execute_tool"
        assert attrs[gen_ai_attributes.GEN_AI_TOOL_NAME] == "multiply"
    finally:
        instrumentor.uninstrument()


# ---------------------------------------------------------------------------
# Content capturing off — arguments and result suppressed
# ---------------------------------------------------------------------------


def test_on_tool_start_and_end_no_content_capture_suppresses_arguments(
    monkeypatch,
):
    """Without content capture, arguments and result are absent from the span."""
    # OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT intentionally not set
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "multiply", "description": "Multiply two numbers"},
        input_str="",
        run_id=run_id,
        inputs={"a": 3, "b": 4},
    )
    output = MagicMock()
    output.content = "12"
    output.tool_call_id = "call_abc"
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_NAME] == "multiply"
    assert gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
    assert gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT not in attrs


def test_on_tool_end_captures_result_with_span_only_mode(monkeypatch):
    """tool_result is set on the span when content capture is SPAN_ONLY."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "lookup"},
        input_str="query",
        run_id=run_id,
    )
    output = MagicMock()
    output.content = "result text"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT] == "result text"


# ---------------------------------------------------------------------------
# on_tool_end attribute types
# ---------------------------------------------------------------------------


def test_on_tool_end_sets_tool_call_id_attribute(monkeypatch):
    """tool_call_id from the output object is set on the span."""
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "mytool"},
        input_str="",
        run_id=run_id,
    )
    output = MagicMock()
    output.content = "done"
    output.tool_call_id = "call_xyz"
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] == "call_xyz"


def test_on_tool_end_with_none_tool_call_id_omits_attribute(monkeypatch):
    """tool_call_id is absent when the output carries no call id."""
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "mytool"},
        input_str="",
        run_id=run_id,
    )
    output = MagicMock()
    output.content = "done"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert gen_ai_attributes.GEN_AI_TOOL_CALL_ID not in attrs


# ---------------------------------------------------------------------------
# Complex argument and result values — AnyValue / JSON serialization
# ---------------------------------------------------------------------------


def test_on_tool_start_with_complex_inputs_serializes_to_json(monkeypatch):
    """Complex dict inputs are serialized to a compact JSON string on the span."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "search"},
        input_str="",
        run_id=run_id,
        inputs={
            "query": "Paris weather",
            "filters": {"lang": "en", "limit": 5},
        },
    )
    output = MagicMock()
    output.content = "Sunny"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    arguments = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS]
    assert isinstance(arguments, str)

    parsed = json.loads(arguments)
    assert parsed["query"] == "Paris weather"
    assert parsed["filters"] == {"lang": "en", "limit": 5}


def test_on_tool_end_with_complex_result_serializes_to_json(monkeypatch):
    """Complex dict/list tool results are serialized to a JSON string on the span."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "get_data"},
        input_str="",
        run_id=run_id,
        inputs={"id": 42},
    )
    # Simulate a tool returning a complex structured result
    output = MagicMock()
    output.content = {"status": "ok", "items": [1, 2, 3], "meta": {"count": 3}}
    output.tool_call_id = "call_complex"
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    result = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT]
    assert isinstance(result, str)

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["items"] == [1, 2, 3]
    assert parsed["meta"]["count"] == 3


def test_on_tool_end_with_list_result_serializes_to_json(monkeypatch):
    """List tool results are serialized to a JSON string on the span."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "list_items"},
        input_str="",
        run_id=run_id,
        inputs={"category": "fruits"},
    )
    output = MagicMock()
    output.content = ["apple", "banana", "cherry"]
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    result = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT]
    assert isinstance(result, str)

    parsed = json.loads(result)
    assert parsed == ["apple", "banana", "cherry"]


# ---------------------------------------------------------------------------
# on_tool_end / on_tool_error with unknown run_id — must not raise
# ---------------------------------------------------------------------------


def test_on_tool_end_unknown_run_id_does_not_raise(monkeypatch):
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    output = MagicMock()
    output.content = "result"
    output.tool_call_id = None
    # No on_tool_start was called — should be a no-op
    handler.on_tool_end(output=output, run_id=uuid4())

    assert len(span_exporter.get_finished_spans()) == 0


def test_on_tool_error_unknown_run_id_does_not_raise(monkeypatch):
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    handler.on_tool_error(error=RuntimeError("boom"), run_id=uuid4())

    assert len(span_exporter.get_finished_spans()) == 0


# ---------------------------------------------------------------------------
# on_tool_start: inputs=None falls back to input_str
# ---------------------------------------------------------------------------


def test_on_tool_start_uses_input_str_when_inputs_is_none(monkeypatch):
    """When inputs kwarg is absent (None), input_str is used for arguments."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    # inputs not passed → defaults to None → callback uses input_str
    handler.on_tool_start(
        serialized={"name": "greet"},
        input_str="hello world",
        run_id=run_id,
    )
    output = MagicMock()
    output.content = "hi"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS] == "hello world"


def test_on_tool_start_inputs_takes_priority_over_input_str(monkeypatch):
    """When both inputs dict and input_str are provided, inputs dict wins."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "add"},
        input_str="ignored",
        run_id=run_id,
        inputs={"x": 1, "y": 2},
    )
    output = MagicMock()
    output.content = "3"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert (
        attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS] == '{"x":1,"y":2}'
    )


def test_on_tool_start_json_input_str_is_deserialized(monkeypatch):
    """When inputs is None but input_str is valid JSON, it is deserialized to an object."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "lookup"},
        input_str='{"city": "Berlin"}',
        run_id=run_id,
    )
    output = MagicMock()
    output.content = "result"
    output.tool_call_id = None
    handler.on_tool_end(output=output, run_id=run_id)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    # JSON-deserialized dict is re-serialized to JSON for the span attribute
    assert (
        attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        == '{"city":"Berlin"}'
    )


# ---------------------------------------------------------------------------
# on_chat_model_start: functions key as alternative to tools
# ---------------------------------------------------------------------------


def test_on_chat_model_start_with_functions_key_sets_definitions(monkeypatch):
    """Tool definitions are also picked up from the 'functions' invocation param."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
            },
        }
    ]

    handler.on_chat_model_start(
        serialized=_OPENAI_SERIALIZED,
        messages=[[HumanMessage(content="What's the weather?")]],
        run_id=run_id,
        metadata=_OPENAI_METADATA,
        invocation_params={
            **_OPENAI_INVOCATION_PARAMS,
            "functions": functions,
        },
    )
    ai_msg = AIMessage(content="It is sunny.")
    ai_msg.response_metadata = {"finish_reason": "stop"}
    generation = ChatGeneration(message=ai_msg, text="It is sunny.")
    generation.generation_info = {"finish_reason": "stop"}
    handler.on_llm_end(
        response=LLMResult(generations=[[generation]]), run_id=run_id
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS in attrs
    assert "get_weather" in attrs[gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS]


def test_on_chat_model_start_without_tools_omits_definitions(monkeypatch):
    """No tool_definitions attribute when invocation_params has no tools."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    tracer_provider, span_exporter, logger_provider, meter_provider = (
        _make_providers()
    )
    handler = _make_callback_handler(
        tracer_provider, logger_provider, meter_provider
    )

    run_id = uuid4()
    handler.on_chat_model_start(
        serialized=_OPENAI_SERIALIZED,
        messages=[[HumanMessage(content="Hello")]],
        run_id=run_id,
        metadata=_OPENAI_METADATA,
        invocation_params=_OPENAI_INVOCATION_PARAMS,
    )
    ai_msg = AIMessage(content="Hi")
    ai_msg.response_metadata = {"finish_reason": "stop"}
    generation = ChatGeneration(message=ai_msg, text="Hi")
    generation.generation_info = {"finish_reason": "stop"}
    handler.on_llm_end(
        response=LLMResult(generations=[[generation]]), run_id=run_id
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS not in spans[0].attributes
