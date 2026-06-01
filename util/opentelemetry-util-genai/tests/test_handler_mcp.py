# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest import TestCase
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
from opentelemetry.semconv._incubating.attributes import (
    rpc_attributes as Rpc,
)
from opentelemetry.trace import INVALID_SPAN, SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai._mcp_invocation import (
    _CLIENT_ADDRESS,
    _CLIENT_PORT,
    _JSONRPC_PROTOCOL_VERSION,
    _JSONRPC_REQUEST_ID,
    _MCP_METHOD_NAME,
    _MCP_PROTOCOL_VERSION,
    _MCP_RESOURCE_URI,
    _MCP_SESSION_ID,
    _NETWORK_PROTOCOL_NAME,
    _NETWORK_PROTOCOL_VERSION,
    _NETWORK_TRANSPORT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import MCPInvocation
from opentelemetry.util.genai.types import Error


class _MCPTestBase(TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=self.tracer_provider)

    def _get_finished_spans(self):
        return self.span_exporter.get_finished_spans()


class TelemetryHandlerMCPTest(_MCPTestBase):
    def test_mcp_creates_span_and_returns_mcp_invocation(self) -> None:
        invocation = self.handler.mcp("tools/call", tool_name="get_weather")
        self.assertIsInstance(invocation, MCPInvocation)
        self.assertIsNot(invocation.span, INVALID_SPAN)
        invocation.stop()

    def test_span_name_with_tool_name(self) -> None:
        self.handler.mcp("tools/call", tool_name="get_weather").stop()
        self.assertEqual(
            self._get_finished_spans()[0].name, "tools/call get_weather"
        )

    def test_span_name_with_prompt_name(self) -> None:
        self.handler.mcp("prompts/get", prompt_name="summarize").stop()
        self.assertEqual(
            self._get_finished_spans()[0].name, "prompts/get summarize"
        )

    def test_span_name_without_target(self) -> None:
        self.handler.mcp("tools/list").stop()
        self.assertEqual(self._get_finished_spans()[0].name, "tools/list")

    def test_span_name_tool_name_takes_precedence(self) -> None:
        self.handler.mcp("tools/call", tool_name="t", prompt_name="p").stop()
        self.assertEqual(self._get_finished_spans()[0].name, "tools/call t")

    def test_span_kind_client_by_default(self) -> None:
        self.handler.mcp("tools/call", tool_name="x").stop()
        self.assertEqual(self._get_finished_spans()[0].kind, SpanKind.CLIENT)

    def test_span_kind_server(self) -> None:
        self.handler.mcp("tools/call", tool_name="x", is_client=False).stop()
        self.assertEqual(self._get_finished_spans()[0].kind, SpanKind.SERVER)

    def test_mcp_method_name_attribute(self) -> None:
        self.handler.mcp("tools/call", tool_name="x").stop()
        self.assertEqual(
            self._get_finished_spans()[0].attributes[_MCP_METHOD_NAME],
            "tools/call",
        )

    def test_operation_name_execute_tool_for_tools_call(self) -> None:
        self.handler.mcp("tools/call", tool_name="x").stop()
        self.assertEqual(
            self._get_finished_spans()[0].attributes[
                GenAI.GEN_AI_OPERATION_NAME
            ],
            "execute_tool",
        )

    def test_operation_name_absent_for_non_tools_call(self) -> None:
        for method in ("tools/list", "initialize", "prompts/get"):
            self.span_exporter.clear()
            self.handler.mcp(method).stop()
            self.assertNotIn(
                GenAI.GEN_AI_OPERATION_NAME,
                self._get_finished_spans()[0].attributes,
            )

    def test_tool_name_and_prompt_name_attributes(self) -> None:
        self.handler.mcp("tools/call", tool_name="get_weather").stop()
        self.assertEqual(
            self._get_finished_spans()[0].attributes[GenAI.GEN_AI_TOOL_NAME],
            "get_weather",
        )
        self.span_exporter.clear()
        self.handler.mcp("prompts/get", prompt_name="summarize").stop()
        self.assertEqual(
            self._get_finished_spans()[0].attributes[GenAI.GEN_AI_PROMPT_NAME],
            "summarize",
        )

    def test_server_address_for_client_span(self) -> None:
        self.handler.mcp(
            "tools/call",
            tool_name="x",
            server_address="mcp.example.com",
            server_port=443,
        ).stop()
        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(attrs["server.address"], "mcp.example.com")
        self.assertEqual(attrs["server.port"], 443)

    def test_client_address_for_server_span(self) -> None:
        self.handler.mcp(
            "tools/call",
            tool_name="x",
            is_client=False,
            client_address="10.0.0.1",
            client_port=54321,
        ).stop()
        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(attrs[_CLIENT_ADDRESS], "10.0.0.1")
        self.assertEqual(attrs[_CLIENT_PORT], 54321)

    def test_server_attrs_excluded_from_server_span(self) -> None:
        self.handler.mcp(
            "tools/call",
            tool_name="x",
            is_client=False,
            server_address="mcp.example.com",
            server_port=443,
        ).stop()
        attrs = self._get_finished_spans()[0].attributes
        self.assertNotIn("server.address", attrs)
        self.assertNotIn("server.port", attrs)

    def test_client_attrs_excluded_from_client_span(self) -> None:
        self.handler.mcp(
            "tools/call",
            tool_name="x",
            client_address="10.0.0.1",
            client_port=54321,
        ).stop()
        attrs = self._get_finished_spans()[0].attributes
        self.assertNotIn(_CLIENT_ADDRESS, attrs)
        self.assertNotIn(_CLIENT_PORT, attrs)

    def test_mutable_attributes(self) -> None:
        inv = self.handler.mcp("tools/call", tool_name="x")
        inv.mcp_session_id = "session-123"
        inv.mcp_protocol_version = "2025-06-18"
        inv.mcp_resource_uri = "file:///doc.pdf"
        inv.jsonrpc_request_id = "42"
        inv.jsonrpc_protocol_version = "2.0"
        inv.rpc_response_status_code = "-32600"
        inv.network_transport = "tcp"
        inv.network_protocol_name = "http"
        inv.network_protocol_version = "1.1"
        inv.tool_call_id = "call-abc-123"
        inv.stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(attrs[_MCP_SESSION_ID], "session-123")
        self.assertEqual(attrs[_MCP_PROTOCOL_VERSION], "2025-06-18")
        self.assertEqual(attrs[_MCP_RESOURCE_URI], "file:///doc.pdf")
        self.assertEqual(attrs[_JSONRPC_REQUEST_ID], "42")
        self.assertEqual(attrs[_JSONRPC_PROTOCOL_VERSION], "2.0")
        self.assertEqual(attrs[Rpc.RPC_RESPONSE_STATUS_CODE], "-32600")
        self.assertEqual(attrs[_NETWORK_TRANSPORT], "tcp")
        self.assertEqual(attrs[_NETWORK_PROTOCOL_NAME], "http")
        self.assertEqual(attrs[_NETWORK_PROTOCOL_VERSION], "1.1")
        self.assertEqual(attrs[GenAI.GEN_AI_TOOL_CALL_ID], "call-abc-123")

    def test_content_capture_enabled(self) -> None:
        with patch(
            "opentelemetry.util.genai._mcp_invocation.should_capture_content_on_spans",
            return_value=True,
        ):
            inv = self.handler.mcp("tools/call", tool_name="x")
        inv.tool_call_arguments = '{"city": "London"}'
        inv.tool_call_result = '{"temp": 20}'
        inv.stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(
            attrs[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS], '{"city": "London"}'
        )
        self.assertEqual(attrs[GenAI.GEN_AI_TOOL_CALL_RESULT], '{"temp": 20}')

    def test_content_capture_disabled(self) -> None:
        inv = self.handler.mcp("tools/call", tool_name="x")
        inv.tool_call_arguments = '{"city": "London"}'
        inv.tool_call_result = '{"temp": 20}'
        inv.stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertNotIn(GenAI.GEN_AI_TOOL_CALL_ARGUMENTS, attrs)
        self.assertNotIn(GenAI.GEN_AI_TOOL_CALL_RESULT, attrs)

    def test_custom_attributes(self) -> None:
        inv = self.handler.mcp("tools/call", tool_name="x")
        inv.attributes["custom.key"] = "value"
        inv.stop()
        self.assertEqual(
            self._get_finished_spans()[0].attributes["custom.key"], "value"
        )

    def test_none_optional_attrs_omitted(self) -> None:
        self.handler.mcp("tools/list").stop()
        attrs = self._get_finished_spans()[0].attributes
        for key in (
            GenAI.GEN_AI_TOOL_NAME,
            GenAI.GEN_AI_PROMPT_NAME,
            _MCP_SESSION_ID,
            _MCP_PROTOCOL_VERSION,
            _MCP_RESOURCE_URI,
            _JSONRPC_REQUEST_ID,
        ):
            self.assertNotIn(key, attrs)

    def test_fail_sets_error(self) -> None:
        inv = self.handler.mcp("tools/call", tool_name="x")
        inv.fail(Error(message="timeout", type=TimeoutError))

        span = self._get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.status.description, "timeout")
        self.assertEqual(span.attributes["error.type"], "TimeoutError")
        self.assertEqual(span.attributes[_MCP_METHOD_NAME], "tools/call")

    def test_fail_with_exception_instance(self) -> None:
        inv = self.handler.mcp("tools/call", tool_name="x")
        inv.fail(ValueError("oops"))

        span = self._get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "ValueError")


class TelemetryHandlerMCPContextManagerTest(_MCPTestBase):
    def test_context_manager_creates_and_ends_span(self) -> None:
        with self.handler.mcp("tools/call", tool_name="x") as inv:
            self.assertIsNot(inv.span, INVALID_SPAN)
        self.assertEqual(len(self._get_finished_spans()), 1)

    def test_context_manager_success_has_unset_status(self) -> None:
        with self.handler.mcp("tools/call", tool_name="x"):
            pass
        self.assertEqual(
            self._get_finished_spans()[0].status.status_code, StatusCode.UNSET
        )

    def test_context_manager_reraises_exception(self) -> None:
        with pytest.raises(ValueError, match="tool failed"):
            with self.handler.mcp("tools/call", tool_name="x"):
                raise ValueError("tool failed")

    def test_context_manager_marks_error_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with self.handler.mcp("tools/call", tool_name="x"):
                raise RuntimeError("connection lost")
        span = self._get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "RuntimeError")

    def test_context_manager_sets_mutable_attributes(self) -> None:
        with self.handler.mcp("tools/call", tool_name="x") as inv:
            inv.mcp_session_id = "sess-456"
            inv.jsonrpc_request_id = "1"
        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(attrs[_MCP_SESSION_ID], "sess-456")
        self.assertEqual(attrs[_JSONRPC_REQUEST_ID], "1")

    def test_context_manager_server_span(self) -> None:
        with self.handler.mcp(
            "tools/call",
            tool_name="x",
            is_client=False,
            client_address="10.0.0.1",
            client_port=54321,
        ):
            pass
        span = self._get_finished_spans()[0]
        self.assertEqual(span.kind, SpanKind.SERVER)
        self.assertEqual(span.attributes[_CLIENT_ADDRESS], "10.0.0.1")
        self.assertEqual(span.attributes[_CLIENT_PORT], 54321)


class TelemetryHandlerMCPSamplingTest(_MCPTestBase):
    def _make_capturing_handler(self):
        captured: dict = {}

        class _Sampler:
            def should_sample(
                self,
                parent_context,
                trace_id,
                name,
                kind=None,
                attributes=None,
                links=None,
            ):
                captured.update(attributes or {})
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

            def get_description(self):
                return "_Sampler"

        provider = TracerProvider(sampler=_Sampler())
        provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        return TelemetryHandler(tracer_provider=provider), captured

    def test_sampling_attributes_client_span(self) -> None:
        handler, captured = self._make_capturing_handler()
        handler.mcp(
            "tools/call",
            tool_name="get_weather",
            server_address="mcp.example.com",
            server_port=443,
        ).stop()

        self.assertEqual(captured[_MCP_METHOD_NAME], "tools/call")
        self.assertEqual(captured[GenAI.GEN_AI_OPERATION_NAME], "execute_tool")
        self.assertEqual(captured[GenAI.GEN_AI_TOOL_NAME], "get_weather")
        self.assertEqual(captured["server.address"], "mcp.example.com")
        self.assertEqual(captured["server.port"], 443)

    def test_sampling_attributes_server_span(self) -> None:
        handler, captured = self._make_capturing_handler()
        handler.mcp(
            "tools/call",
            tool_name="x",
            is_client=False,
            client_address="10.0.0.1",
            client_port=54321,
        ).stop()

        self.assertEqual(captured[_CLIENT_ADDRESS], "10.0.0.1")
        self.assertEqual(captured[_CLIENT_PORT], 54321)
        self.assertNotIn("server.address", captured)

    def test_sampling_no_operation_name_for_non_tools_call(self) -> None:
        handler, captured = self._make_capturing_handler()
        handler.mcp("tools/list").stop()

        self.assertEqual(captured[_MCP_METHOD_NAME], "tools/list")
        self.assertNotIn(GenAI.GEN_AI_OPERATION_NAME, captured)
