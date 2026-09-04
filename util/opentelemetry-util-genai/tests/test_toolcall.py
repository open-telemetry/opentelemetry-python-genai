# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ToolCallRequestPart and ToolInvocation inheritance structure"""

import os
from unittest.mock import patch

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import Decision, SamplingResult
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import GenAIInvocation
from opentelemetry.util.genai.types import (
    CompactionPart,
    InputMessage,
    ServerToolCall,
    ServerToolCallPart,
    ServerToolCallResponse,
    ServerToolCallResponsePart,
    ToolCallRequest,
    ToolCallRequestPart,
    ToolCallResponse,
    ToolCallResponsePart,
)
from opentelemetry.util.genai.utils import gen_ai_json_dumps


def _make_handler() -> TelemetryHandler:
    return TelemetryHandler(tracer_provider=TracerProvider())


def test_toolcallrequest_is_message_part():
    """ToolCallRequestPart is for message parts only"""
    tcr = ToolCallRequestPart(
        arguments={"location": "Paris"}, name="get_weather", id="call_123"
    )
    msg = InputMessage(role="user", parts=[tcr])
    assert len(msg.parts) == 1


def test_toolcall_inherits_from_genaiinvocation():
    """ToolInvocation inherits from GenAIInvocation for lifecycle management"""
    handler = _make_handler()
    tc = handler.tool("get_weather")
    tc.arguments = {"city": "Paris"}
    assert isinstance(tc, GenAIInvocation)
    assert not isinstance(tc, ToolCallRequestPart)
    tc.stop()


def test_toolcall_has_attributes_dict():
    """ToolInvocation inherits attributes dict from GenAIInvocation"""
    handler = _make_handler()
    tc = handler.tool("test")
    tc.attributes["custom.key"] = "value"
    assert tc.attributes["custom.key"] == "value"
    tc.stop()


def test_toolcallrequest_in_message_part_union():
    """ToolCallRequestPart (not ToolInvocation) is the correct type for message parts"""
    tc = ToolCallRequestPart(
        name="get_weather", arguments={"city": "Paris"}, id="call_123"
    )
    msg = InputMessage(role="assistant", parts=[tc])
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], ToolCallRequestPart)
    assert not isinstance(msg.parts[0], GenAIInvocation)


def test_toolcall_operation_name():
    """ToolInvocation operation_name is fixed to execute_tool"""
    handler = _make_handler()
    tc = handler.tool("my_tool")
    assert tc._operation_name == "execute_tool"
    tc.stop()


def test_server_tool_call_basic():
    """ServerToolCallPart can be created with required fields"""
    stc = ServerToolCallPart(
        name="code_interpreter",
        server_tool_call={"type": "code_interpreter", "code": "print(1)"},
    )
    assert stc.name == "code_interpreter"
    assert stc.server_tool_call == {
        "type": "code_interpreter",
        "code": "print(1)",
    }
    assert stc.id is None
    assert stc.type == "server_tool_call"


def test_server_tool_call_with_id():
    """ServerToolCallPart can have an optional id"""
    stc = ServerToolCallPart(
        name="web_search",
        server_tool_call={"type": "web_search", "query": "weather"},
        id="stc_001",
    )
    assert stc.id == "stc_001"


def test_server_tool_call_response_basic():
    """ServerToolCallResponsePart can be created with required fields"""
    stcr = ServerToolCallResponsePart(
        server_tool_call_response={
            "type": "code_interpreter",
            "output": "1\n",
        },
    )
    assert stcr.server_tool_call_response == {
        "type": "code_interpreter",
        "output": "1\n",
    }
    assert stcr.id is None
    assert stcr.type == "server_tool_call_response"


def test_server_tool_call_in_message():
    """ServerToolCallPart and ServerToolCallResponsePart work as MessageParts"""
    stc = ServerToolCallPart(
        name="code_interpreter",
        server_tool_call={"type": "code_interpreter", "code": "x = 1"},
    )
    stcr = ServerToolCallResponsePart(
        server_tool_call_response={"type": "code_interpreter", "output": ""},
        id="stc_001",
    )
    msg = InputMessage(role="assistant", parts=[stc, stcr])
    assert len(msg.parts) == 2
    assert isinstance(msg.parts[0], ServerToolCallPart)
    assert isinstance(msg.parts[1], ServerToolCallResponsePart)


def test_compactionpart_is_message_part():
    """CompactionPart works as a MessagePart alongside other part types."""
    part = CompactionPart(
        id="compact_001", content="Summary of earlier turns."
    )
    assert part.id == "compact_001"
    assert part.content == "Summary of earlier turns."
    assert part.type == "compaction"

    msg = InputMessage(role="assistant", parts=[part])
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], CompactionPart)


def test_deprecated_tool_call_names_are_aliases():
    """The pre-*Part tool call names are aliases of their replacements."""
    assert ToolCallRequest is ToolCallRequestPart
    assert ToolCallResponse is ToolCallResponsePart
    assert ServerToolCall is ServerToolCallPart
    assert ServerToolCallResponse is ServerToolCallResponsePart


def test_deprecated_tool_call_names_build_message_parts():
    """Parts built through the deprecated names are still valid MessageParts."""
    tcr = ToolCallRequest(
        arguments={"location": "Paris"}, name="get_weather", id="call_123"
    )
    stc = ServerToolCall(
        name="code_interpreter",
        server_tool_call={"type": "code_interpreter", "code": "x = 1"},
    )
    msg = InputMessage(role="assistant", parts=[tcr, stc])

    assert isinstance(msg.parts[0], ToolCallRequestPart)
    assert msg.parts[0].type == "tool_call"
    assert isinstance(msg.parts[1], ServerToolCallPart)
    assert msg.parts[1].type == "server_tool_call"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_tool_span_is_internal_kind():
    """execute_tool runs in-process, so its span must be INTERNAL, not CLIENT."""
    span_exporter, handler = _make_span_exporter_and_handler()
    handler.tool("get_weather").stop()

    assert span_exporter.get_finished_spans()[0].kind == SpanKind.INTERNAL


def test_start_tool_passes_sampling_attributes_at_span_creation():
    """Verify that only sampling-relevant attributes are available at start_span() time for tools."""
    captured_attributes = {}

    class AttributeCapturingSampler:  # pylint: disable=no-self-use
        def should_sample(
            self,
            parent_context,
            trace_id,
            name,
            kind=None,
            attributes=None,
            links=None,
        ):
            captured_attributes.update(attributes or {})
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

        def get_description(self):
            return "AttributeCapturingSampler"

    span_exporter = InMemorySpanExporter()
    sampler_provider = TracerProvider(sampler=AttributeCapturingSampler())
    sampler_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    handler = TelemetryHandler(tracer_provider=sampler_provider)

    invocation = handler.tool(
        "get_weather",
        tool_call_id="call_123",
        tool_type="function",
        tool_description="Gets weather for a location",
    )
    invocation.stop()

    assert captured_attributes[GenAI.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert captured_attributes[GenAI.GEN_AI_TOOL_NAME] == "get_weather"
    assert captured_attributes[GenAI.GEN_AI_TOOL_TYPE] == "function"
    assert GenAI.GEN_AI_TOOL_CALL_ID not in captured_attributes
    assert GenAI.GEN_AI_TOOL_DESCRIPTION not in captured_attributes

    finished_attrs = span_exporter.get_finished_spans()[0].attributes
    assert finished_attrs[GenAI.GEN_AI_TOOL_CALL_ID] == "call_123"
    assert (
        finished_attrs[GenAI.GEN_AI_TOOL_DESCRIPTION]
        == "Gets weather for a location"
    )


def test_tool_call_id_and_description_on_invocation_instance():
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.tool("get_weather", tool_type="function")
    invocation.tool_call_id = "call_456"
    invocation.tool_description = "Weather info"
    invocation.stop()

    finished_attrs = span_exporter.get_finished_spans()[0].attributes
    assert finished_attrs[GenAI.GEN_AI_TOOL_CALL_ID] == "call_456"
    assert finished_attrs[GenAI.GEN_AI_TOOL_DESCRIPTION] == "Weather info"


# ---------------------------------------------------------------------------
# _any_value_to_attribute_value — tested via the public ToolInvocation API
# ---------------------------------------------------------------------------


def _make_span_exporter_and_handler():
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    return span_exporter, handler


@patch.dict(
    os.environ,
    {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"},
)
def test_arguments_dict_serialized_to_json():
    """dict arguments are JSON-serialized onto the span attribute."""
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.tool("get_weather")
    invocation.arguments = {"location": "Paris", "unit": "celsius"}
    invocation.stop()

    attrs = span_exporter.get_finished_spans()[0].attributes
    assert GenAI.GEN_AI_TOOL_CALL_ARGUMENTS in attrs
    assert attrs[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS] == gen_ai_json_dumps(
        {"location": "Paris", "unit": "celsius"}
    )


@patch.dict(
    os.environ,
    {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"},
)
def test_arguments_str_passed_through():
    """str arguments are stored as-is (no JSON wrapping)."""
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.tool("echo")
    invocation.arguments = "hello"
    invocation.stop()

    attrs = span_exporter.get_finished_spans()[0].attributes
    assert attrs[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS] == "hello"


@patch.dict(
    os.environ,
    {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"},
)
def test_arguments_int_passed_through():
    """int arguments are stored as-is."""
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.tool("counter")
    invocation.arguments = 42
    invocation.stop()

    attrs = span_exporter.get_finished_spans()[0].attributes
    assert attrs[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS] == 42


@patch.dict(
    os.environ,
    {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"},
)
def test_arguments_none_omits_attribute():
    """None arguments must not produce the attribute on the span."""
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.tool("noop")
    invocation.arguments = None
    invocation.stop()

    attrs = span_exporter.get_finished_spans()[0].attributes
    assert GenAI.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs


def test_arguments_omitted_when_content_capture_disabled():
    """arguments must not appear on the span when content capture is off."""
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.tool("get_weather")
    invocation.arguments = {"location": "Paris"}
    invocation.stop()

    attrs = span_exporter.get_finished_spans()[0].attributes
    assert GenAI.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs


# ---------------------------------------------------------------------------
# agent_name propagation onto execute_tool spans (semconv Conditionally
# Required "When applicable")
# ---------------------------------------------------------------------------


def test_tool_span_carries_agent_name_when_set():
    """When agent_name is passed, `gen_ai.agent.name` is on the execute_tool span."""
    span_exporter, handler = _make_span_exporter_and_handler()
    handler.tool("get_weather", agent_name="weather_agent").stop()

    attrs = span_exporter.get_finished_spans()[0].attributes
    assert attrs[GenAI.GEN_AI_AGENT_NAME] == "weather_agent"


def test_tool_span_omits_agent_name_when_absent():
    """`gen_ai.agent.name` must not appear on the span when no agent context."""
    span_exporter, handler = _make_span_exporter_and_handler()
    handler.tool("get_weather").stop()

    attrs = span_exporter.get_finished_spans()[0].attributes
    assert GenAI.GEN_AI_AGENT_NAME not in attrs


def test_tool_start_attributes_omit_agent_name_at_sampling_time():
    """`gen_ai.agent.name` is not sampling-relevant for `execute_tool` in
    current semconv"""
    captured_attributes: dict[str, object] = {}

    class AttributeCapturingSampler:  # pylint: disable=no-self-use
        def should_sample(
            self,
            parent_context,
            trace_id,
            name,
            kind=None,
            attributes=None,
            links=None,
        ):
            captured_attributes.update(attributes or {})
            return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

        def get_description(self):
            return "AttributeCapturingSampler"

    span_exporter = InMemorySpanExporter()
    sampler_provider = TracerProvider(sampler=AttributeCapturingSampler())
    sampler_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    handler = TelemetryHandler(tracer_provider=sampler_provider)

    handler.tool("get_weather", agent_name="weather_agent").stop()

    assert GenAI.GEN_AI_AGENT_NAME not in captured_attributes
    finalized = span_exporter.get_finished_spans()[0].attributes
    assert finalized[GenAI.GEN_AI_AGENT_NAME] == "weather_agent"
