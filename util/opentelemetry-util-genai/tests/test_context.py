# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from opentelemetry._logs import LogRecord
from opentelemetry.context import attach, detach, get_value, set_value
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
from opentelemetry.util.genai import (
    INFERENCE_EVENT_KEY,
    INFERENCE_SPAN_KEY,
    get_current_inference_event,
    get_current_inference_span,
    set_inference_event_in_context,
    set_inference_span_in_context,
)
from opentelemetry.util.genai.handler import TelemetryHandler


class TestInferenceSpanContext(unittest.TestCase):
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        self.tracer = self.tracer_provider.get_tracer(__name__)

    def test_key_constant_value(self):
        self.assertEqual(
            INFERENCE_SPAN_KEY, "opentelemetry.genai.inference_span"
        )

    def test_get_current_inference_span_none_by_default(self):
        self.assertIsNone(get_current_inference_span())

    def test_set_and_get_inference_span_in_context(self):
        span = self.tracer.start_span("test_span")
        ctx = set_inference_span_in_context(span)
        self.assertIs(get_current_inference_span(ctx), span)
        self.assertIsNone(get_current_inference_span())

        token = attach(ctx)
        try:
            self.assertIs(get_current_inference_span(), span)
        finally:
            detach(token)
        self.assertIsNone(get_current_inference_span())
        span.end()

    def test_plain_string_key_interoperability(self):
        span = self.tracer.start_span("external_native_span")
        # An external native library sets the well-known string key
        ctx = set_value("opentelemetry.genai.inference_span", span)
        # Our helper can read it
        self.assertIs(get_current_inference_span(ctx), span)

        # Our helper sets it, and an external library reading the string key gets it
        ctx2 = set_inference_span_in_context(span)
        self.assertIs(
            get_value("opentelemetry.genai.inference_span", ctx2), span
        )
        self.assertIs(get_value(INFERENCE_SPAN_KEY, ctx2), span)
        span.end()

    def test_inference_invocation_attaches_and_cleans_up_context(self):
        self.assertIsNone(get_current_inference_span())

        invocation = self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        )
        self.assertIs(get_current_inference_span(), invocation.span)

        invocation.stop()
        self.assertIsNone(get_current_inference_span())

    def test_inference_invocation_cleans_up_on_fail(self):
        self.assertIsNone(get_current_inference_span())

        invocation = self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        )
        self.assertIs(get_current_inference_span(), invocation.span)

        invocation.fail(ValueError("test error"))
        self.assertIsNone(get_current_inference_span())

    def test_inference_invocation_context_manager(self):
        self.assertIsNone(get_current_inference_span())

        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            self.assertIs(get_current_inference_span(), invocation.span)

            # Downstream instrumentation can fetch and modify the span
            downstream_span = get_current_inference_span()
            self.assertIsNotNone(downstream_span)
            if downstream_span is not None:
                downstream_span.set_attribute("custom.attribute", "enriched")

        self.assertIsNone(get_current_inference_span())

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes.get("custom.attribute"), "enriched"
        )

    def test_non_inference_invocations_do_not_set_inference_span(self):
        self.assertIsNone(get_current_inference_span())

        with self.handler.invoke_local_agent(agent_name="MathTutor"):
            self.assertIsNone(get_current_inference_span())

            # Nested inference invocation properly sets the inference span
            with self.handler.inference(
                "openai", request_model="gpt-4o-mini"
            ) as inf_inv:
                self.assertIs(get_current_inference_span(), inf_inv.span)

            self.assertIsNone(get_current_inference_span())

    def test_event_key_constant_value(self):
        self.assertEqual(
            INFERENCE_EVENT_KEY, "opentelemetry.genai.inference_event"
        )

    def test_get_current_inference_event_none_by_default(self):
        self.assertIsNone(get_current_inference_event())

    def test_set_and_get_inference_event_in_context(self):
        log_record = LogRecord(event_name="test_event", attributes={"a": "b"})
        ctx = set_inference_event_in_context(log_record)
        self.assertIs(get_current_inference_event(ctx), log_record)
        self.assertIsNone(get_current_inference_event())

        token = attach(ctx)
        try:
            self.assertIs(get_current_inference_event(), log_record)
        finally:
            detach(token)
        self.assertIsNone(get_current_inference_event())

    def test_inference_invocation_attaches_and_enriches_event(self):
        log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(log_exporter)
        )
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=logger_provider,
        )

        with patch.dict(
            os.environ, {"OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true"}
        ):
            self.assertIsNone(get_current_inference_span())
            self.assertIsNone(get_current_inference_event())

            with handler.inference(
                "openai", request_model="gpt-4o-mini"
            ) as invocation:
                self.assertIs(get_current_inference_span(), invocation.span)
                event = get_current_inference_event()
                self.assertIsNotNone(event)
                assert event is not None
                self.assertEqual(
                    event.event_name,
                    "gen_ai.client.inference.operation.details",
                )
                # Downstream enriches the event in context
                if event.attributes is None:
                    event.attributes = {}
                event.attributes["downstream.enriched"] = "enriched_val"

            self.assertIsNone(get_current_inference_span())
            self.assertIsNone(get_current_inference_event())

            logs = log_exporter.get_finished_logs()
            self.assertEqual(len(logs), 1)
            self.assertEqual(
                logs[0].log_record.attributes.get("downstream.enriched"),
                "enriched_val",
            )

        self.assertIsNone(get_current_inference_span())

    def test_already_started_property(self):
        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as root_inv:
            self.assertFalse(root_inv.already_started)

            with self.handler.inference(
                "google", request_model="gemini-2.0-flash"
            ) as nested_inv:
                self.assertTrue(nested_inv.already_started)
                self.assertIs(nested_inv.span, root_inv.span)

    def test_nested_inference_invocation_reuses_span_and_event(self):
        log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(log_exporter)
        )
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=logger_provider,
        )

        with patch.dict(
            os.environ, {"OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true"}
        ):
            with handler.inference(
                "upstream-agent", request_model="gpt-4o"
            ) as root_inv:
                self.assertFalse(root_inv.already_started)

                with handler.inference(
                    "downstream-client",
                    request_model="gpt-4o",
                    server_address="api.openai.com",
                ) as nested_inv:
                    self.assertTrue(nested_inv.already_started)
                    self.assertIs(nested_inv.span, root_inv.span)
                    nested_inv.input_tokens = 15
                    nested_inv.output_tokens = 25

                # After nested exit, span is NOT ended yet (still recording)
                self.assertTrue(root_inv.span.is_recording())
                spans = self.span_exporter.get_finished_spans()
                self.assertEqual(len(spans), 0)

            # After root exit, span is ended and event is emitted
            spans = self.span_exporter.get_finished_spans()
            self.assertEqual(len(spans), 1)
            self.assertEqual(
                spans[0].attributes.get("server.address"), "api.openai.com"
            )
            self.assertEqual(
                spans[0].attributes.get("gen_ai.usage.input_tokens"), 15
            )
            self.assertEqual(
                spans[0].attributes.get("gen_ai.usage.output_tokens"), 25
            )

            logs = log_exporter.get_finished_logs()
            self.assertEqual(len(logs), 1)
            self.assertEqual(
                logs[0].log_record.attributes.get("server.address"),
                "api.openai.com",
            )
            self.assertEqual(
                logs[0].log_record.attributes.get("gen_ai.usage.input_tokens"),
                15,
            )

    def test_nested_inference_invocation_does_not_end_span_on_fail(self):
        with self.handler.inference(
            "upstream", request_model="gpt-4o"
        ) as root_inv:
            with self.assertRaises(ValueError) as handler_error:
                with self.handler.inference(
                    "downstream", request_model="gpt-4o"
                ) as nested_inv:
                    self.assertTrue(nested_inv.already_started)
                    raise ValueError("downstream network failure")

            self.assertEqual(
                str(handler_error.exception), "downstream network failure"
            )
            # Root span must NOT be ended yet
            self.assertTrue(root_inv.span.is_recording())
            self.assertEqual(len(self.span_exporter.get_finished_spans()), 0)

        # After root finishes, the span is ended with error status
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code.name, "ERROR")
