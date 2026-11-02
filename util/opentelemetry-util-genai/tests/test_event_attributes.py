# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import TelemetryHandler

from .test_utils import (
    _create_input_message,
    _create_output_message,
    _get_single_span,
)


class TestEventOnlyAttributes(unittest.TestCase):
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )
        self.tracer_provider = tracer_provider
        self.logger_provider = logger_provider

    def tearDown(self):
        self.span_exporter.clear()
        self.log_exporter.clear()

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
            "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true",
        },
    )
    def test_event_attributes_go_to_event_not_span(self):
        """``event_attributes`` reach the operation.details event but never
        the span, and win collisions with ``attributes`` on the event."""
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="event-only-model"
        )
        invocation.input_messages = [_create_input_message("test query")]
        invocation.output_messages = [_create_output_message("test response")]
        invocation.attributes["shared.key"] = "from-attributes"
        invocation.event_attributes["user.id"] = "alice"
        invocation.event_attributes["shared.key"] = "from-event-attributes"
        invocation.stop()

        span = _get_single_span(self.span_exporter)
        span_attrs = span.attributes or {}
        self.assertNotIn("user.id", span_attrs)
        self.assertEqual(span_attrs.get("shared.key"), "from-attributes")

        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        event_attrs = logs[0].log_record.attributes
        self.assertEqual(event_attrs["user.id"], "alice")
        self.assertEqual(event_attrs["shared.key"], "from-event-attributes")
