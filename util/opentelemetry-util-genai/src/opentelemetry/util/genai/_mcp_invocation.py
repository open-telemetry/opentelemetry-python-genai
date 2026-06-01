# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from opentelemetry._logs import Logger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.attributes import (
    rpc_attributes as Rpc,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._invocation import Error, GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.utils import should_capture_content_on_spans
from opentelemetry.util.types import AttributeValue

# TODO: Migrate once available in opentelemetry-semantic-conventions package
_MCP_METHOD_NAME = "mcp.method.name"
_MCP_SESSION_ID = "mcp.session.id"
_MCP_PROTOCOL_VERSION = "mcp.protocol.version"
_MCP_RESOURCE_URI = "mcp.resource.uri"
_JSONRPC_REQUEST_ID = "jsonrpc.request.id"
_JSONRPC_PROTOCOL_VERSION = "jsonrpc.protocol.version"
_NETWORK_TRANSPORT = "network.transport"
_NETWORK_PROTOCOL_NAME = "network.protocol.name"
_NETWORK_PROTOCOL_VERSION = "network.protocol.version"
_CLIENT_ADDRESS = "client.address"
_CLIENT_PORT = "client.port"


class MCPInvocation(GenAIInvocation):
    """MCP invocation span (client or server). Use handler.mcp() to create."""

    def __init__(
        self,
        tracer: Tracer,
        metrics_recorder: InvocationMetricsRecorder,
        logger: Logger,
        completion_hook: CompletionHook,
        mcp_method_name: str,
        *,
        tool_name: str | None = None,
        prompt_name: str | None = None,
        is_client: bool = True,
        server_address: str | None = None,
        server_port: int | None = None,
        client_address: str | None = None,
        client_port: int | None = None,
    ) -> None:
        span_kind = SpanKind.CLIENT if is_client else SpanKind.SERVER
        target = tool_name or prompt_name
        span_name = (
            f"{mcp_method_name} {target}" if target else mcp_method_name
        )
        _operation_name = (
            GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
            if mcp_method_name == "tools/call"
            else ""
        )

        super().__init__(
            tracer,
            metrics_recorder,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=span_name,
            span_kind=span_kind,
        )
        self.should_capture_content_on_span = should_capture_content_on_spans()
        self.mcp_method_name = mcp_method_name
        self.tool_name = tool_name
        self.prompt_name = prompt_name
        self.is_client = is_client
        self.server_address = server_address
        self.server_port = server_port
        self.client_address = client_address
        self.client_port = client_port
        self.mcp_session_id: str | None = None
        self.mcp_protocol_version: str | None = None
        self.mcp_resource_uri: str | None = None
        self.jsonrpc_request_id: str | None = None
        self.jsonrpc_protocol_version: str | None = None
        self.rpc_response_status_code: str | None = None
        self.network_transport: str | None = None
        self.network_protocol_name: str | None = None
        self.network_protocol_version: str | None = None
        self.tool_call_id: str | None = None
        self.tool_call_arguments: AttributeValue | None = None
        self.tool_call_result: AttributeValue | None = None

        self._start(self._get_base_attributes())

    def _get_base_attributes(self) -> dict[str, Any]:
        return self._common_attrs()

    def _common_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            _MCP_METHOD_NAME: self.mcp_method_name,
        }
        if self._operation_name:
            attrs[GenAI.GEN_AI_OPERATION_NAME] = self._operation_name

        optional: tuple[tuple[str, Any], ...] = (
            (GenAI.GEN_AI_TOOL_NAME, self.tool_name),
            (GenAI.GEN_AI_PROMPT_NAME, self.prompt_name),
        )
        optional += self._network_endpoint_attrs()
        attrs.update({k: v for k, v in optional if v is not None})
        return attrs

    def _network_endpoint_attrs(self) -> tuple[tuple[str, Any], ...]:
        if self.is_client:
            return (
                (server_attributes.SERVER_ADDRESS, self.server_address),
                (server_attributes.SERVER_PORT, self.server_port),
            )
        return (
            (_CLIENT_ADDRESS, self.client_address),
            (_CLIENT_PORT, self.client_port),
        )

    def _get_metric_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            _MCP_METHOD_NAME: self.mcp_method_name,
        }
        if self._operation_name:
            attrs[GenAI.GEN_AI_OPERATION_NAME] = self._operation_name
        attrs.update(self.metric_attributes)
        return attrs

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)

        attrs = self._common_attrs()

        optional: tuple[tuple[str, Any], ...] = (
            (_MCP_SESSION_ID, self.mcp_session_id),
            (_MCP_PROTOCOL_VERSION, self.mcp_protocol_version),
            (_MCP_RESOURCE_URI, self.mcp_resource_uri),
            (_JSONRPC_REQUEST_ID, self.jsonrpc_request_id),
            (_JSONRPC_PROTOCOL_VERSION, self.jsonrpc_protocol_version),
            (Rpc.RPC_RESPONSE_STATUS_CODE, self.rpc_response_status_code),
            (_NETWORK_TRANSPORT, self.network_transport),
            (_NETWORK_PROTOCOL_NAME, self.network_protocol_name),
            (_NETWORK_PROTOCOL_VERSION, self.network_protocol_version),
            (GenAI.GEN_AI_TOOL_CALL_ID, self.tool_call_id),
            (
                GenAI.GEN_AI_TOOL_CALL_ARGUMENTS,
                self.tool_call_arguments
                if self.should_capture_content_on_span
                else None,
            ),
            (
                GenAI.GEN_AI_TOOL_CALL_RESULT,
                self.tool_call_result
                if self.should_capture_content_on_span
                else None,
            ),
        )
        attrs.update({k: v for k, v in optional if v is not None})
        attrs.update(self.attributes)
        self.span.set_attributes(attrs)
        self._metrics_recorder.record(self)
