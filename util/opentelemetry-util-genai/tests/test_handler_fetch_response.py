# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from unittest import TestCase
from unittest.mock import patch

import pytest

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import INVALID_SPAN, SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import FetchResponseInvocation
from opentelemetry.util.genai.types import (
    Error,
    FunctionToolDefinition,
    OutputMessage,
    Text,
)

# TODO: use the semconv constants once these attributes are released in
# opentelemetry-semantic-conventions.
GEN_AI_REQUEST_STREAM_CURSOR = "gen_ai.request.stream_cursor"
GEN_AI_RESPONSE_STATUS = "gen_ai.response.status"

RESPONSE_ID = "resp_123"


class _FetchResponseTestBase(TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(
            metric_readers=[self.metric_reader]
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def _fetch_response(self, **kwargs) -> FetchResponseInvocation:
        return self.handler.fetch_response(
            "openai", response_id=RESPONSE_ID, **kwargs
        )

    def _get_finished_spans(self):
        return self.span_exporter.get_finished_spans()

    def _get_metrics(self):
        metrics = {}
        data = self.metric_reader.get_metrics_data()
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metrics[metric.name] = metric
        return metrics


class TelemetryHandlerFetchResponseTest(_FetchResponseTestBase):
    # ------------------------------------------------------------------
    # span creation
    # ------------------------------------------------------------------

    def test_fetch_response_creates_span(self) -> None:
        invocation = self._fetch_response()
        self.assertIsNot(invocation.span, INVALID_SPAN)
        invocation.stop()

    def test_span_name_omits_the_high_cardinality_response_id(self) -> None:
        self._fetch_response().stop()

        spans = self._get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "fetch_response")

    def test_span_kind_is_client(self) -> None:
        self._fetch_response().stop()

        self.assertEqual(self._get_finished_spans()[0].kind, SpanKind.CLIENT)

    def test_records_monotonic_start(self) -> None:
        with patch("timeit.default_timer", return_value=42.0):
            invocation = self._fetch_response()
        self.assertEqual(invocation._monotonic_start_s, 42.0)
        invocation.stop()

    # ------------------------------------------------------------------
    # required and conditionally required attributes
    # ------------------------------------------------------------------

    def test_required_attributes_are_set_at_span_creation(self) -> None:
        invocation = self._fetch_response()

        attrs = invocation.span.attributes
        self.assertEqual(attrs[GenAI.GEN_AI_OPERATION_NAME], "fetch_response")
        self.assertEqual(attrs[GenAI.GEN_AI_PROVIDER_NAME], "openai")
        self.assertEqual(attrs[GenAI.GEN_AI_RESPONSE_ID], RESPONSE_ID)
        invocation.stop()

    def test_response_id_is_exposed(self) -> None:
        invocation = self._fetch_response()
        self.assertEqual(invocation.response_id, RESPONSE_ID)
        invocation.stop()

    def test_request_stream_is_set_at_span_creation(self) -> None:
        invocation = self._fetch_response(request_stream=True)

        self.assertIs(
            invocation.span.attributes[GenAI.GEN_AI_REQUEST_STREAM], True
        )
        invocation.stop()

    def test_stop_sets_server_address_and_port(self) -> None:
        self._fetch_response(
            server_address="api.openai.com", server_port=8080
        ).stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(
            attrs[server_attributes.SERVER_ADDRESS], "api.openai.com"
        )
        self.assertEqual(attrs[server_attributes.SERVER_PORT], 8080)

    def test_stop_sets_stream_cursor(self) -> None:
        invocation = self._fetch_response()
        invocation.stream_cursor = "42"
        invocation.stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(attrs[GEN_AI_REQUEST_STREAM_CURSOR], "42")

    # ------------------------------------------------------------------
    # recommended attributes
    # ------------------------------------------------------------------

    def test_stop_sets_response_model_status_and_finish_reasons(self) -> None:
        invocation = self._fetch_response()
        invocation.response_model_name = "gpt-4o-mini-2024-07-18"
        invocation.response_status = "completed"
        invocation.finish_reasons = ["stop"]
        invocation.stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(
            attrs[GenAI.GEN_AI_RESPONSE_MODEL], "gpt-4o-mini-2024-07-18"
        )
        self.assertEqual(attrs[GEN_AI_RESPONSE_STATUS], "completed")
        self.assertEqual(
            attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS], ("stop",)
        )

    def test_stop_omits_unset_attributes(self) -> None:
        self._fetch_response().stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertNotIn(GenAI.GEN_AI_RESPONSE_MODEL, attrs)
        self.assertNotIn(GEN_AI_RESPONSE_STATUS, attrs)
        self.assertNotIn(GenAI.GEN_AI_RESPONSE_FINISH_REASONS, attrs)
        self.assertNotIn(GenAI.GEN_AI_REQUEST_STREAM, attrs)
        self.assertNotIn(GEN_AI_REQUEST_STREAM_CURSOR, attrs)
        self.assertNotIn(server_attributes.SERVER_ADDRESS, attrs)
        self.assertNotIn(server_attributes.SERVER_PORT, attrs)

    def test_stop_sets_custom_attributes(self) -> None:
        invocation = self._fetch_response()
        invocation.attributes["openai.api.type"] = "responses"
        invocation.stop()

        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(attrs["openai.api.type"], "responses")

    # ------------------------------------------------------------------
    # metrics — a fetch performs no inference, so no token usage
    # ------------------------------------------------------------------

    def test_stop_records_duration_without_token_usage(self) -> None:
        invocation = self._fetch_response()
        invocation.response_model_name = "gpt-4o-mini-2024-07-18"
        invocation.stop()

        metrics = self._get_metrics()
        self.assertNotIn("gen_ai.client.token.usage", metrics)

        duration = metrics["gen_ai.client.operation.duration"]
        (point,) = duration.data.data_points
        self.assertEqual(
            point.attributes[GenAI.GEN_AI_OPERATION_NAME], "fetch_response"
        )
        self.assertEqual(
            point.attributes[GenAI.GEN_AI_RESPONSE_MODEL],
            "gpt-4o-mini-2024-07-18",
        )
        # High cardinality — must stay off the metric.
        self.assertNotIn(GenAI.GEN_AI_RESPONSE_ID, point.attributes)

    # ------------------------------------------------------------------
    # fail
    # ------------------------------------------------------------------

    def test_fail_sets_error_status_and_type(self) -> None:
        invocation = self._fetch_response()
        invocation.fail(Error(message="not found", type="NotFoundError"))

        span = self._get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.status.description, "not found")
        self.assertEqual(span.attributes["error.type"], "NotFoundError")

    def test_fail_with_exception_instance(self) -> None:
        invocation = self._fetch_response()
        invocation.fail(ValueError("oops"))

        span = self._get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "ValueError")

    def test_fail_records_error_type_on_duration_metric(self) -> None:
        self._fetch_response().fail(ValueError("oops"))

        duration = self._get_metrics()["gen_ai.client.operation.duration"]
        (point,) = duration.data.data_points
        self.assertEqual(point.attributes["error.type"], "ValueError")

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def test_context_manager_ends_span(self) -> None:
        with self._fetch_response() as invocation:
            self.assertIsInstance(invocation, FetchResponseInvocation)

        spans = self._get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, StatusCode.UNSET)

    def test_context_manager_reraises_exception(self) -> None:
        with pytest.raises(ValueError, match="fetch failed"):
            with self._fetch_response():
                raise ValueError("fetch failed")

        span = self._get_finished_spans()[0]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "ValueError")


class TelemetryHandlerFetchResponseContentTest(_FetchResponseTestBase):
    # ------------------------------------------------------------------
    # opt-in content
    # ------------------------------------------------------------------

    def _fetch_with_content(self) -> None:
        invocation = self._fetch_response()
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content="This is a test.")],
                finish_reason="stop",
            )
        ]
        invocation.system_instruction = [
            Text(content="You are a helpful assistant.")
        ]
        invocation.tool_definitions = [
            FunctionToolDefinition(
                name="get_weather", description=None, parameters=None
            )
        ]
        invocation.stop()

    def test_content_captured_on_span_when_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY"},
        ):
            self._fetch_with_content()

        attrs = self._get_finished_spans()[0].attributes
        self.assertEqual(
            json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES]),
            [
                {
                    "role": "assistant",
                    "parts": [{"content": "This is a test.", "type": "text"}],
                    "finish_reason": "stop",
                }
            ],
        )
        self.assertEqual(
            json.loads(attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS]),
            [{"content": "You are a helpful assistant.", "type": "text"}],
        )
        # A fetched response carries no input messages.
        self.assertNotIn(GenAI.GEN_AI_INPUT_MESSAGES, attrs)

    def test_content_suppressed_on_span_when_disabled(self) -> None:
        self._fetch_with_content()

        attrs = self._get_finished_spans()[0].attributes
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_MESSAGES, attrs)
        self.assertNotIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, attrs)
