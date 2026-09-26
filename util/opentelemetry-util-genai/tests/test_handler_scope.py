# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
)
from opentelemetry.util.genai.version import __version__

_SCHEMA_URL = Schemas.V1_37_0.value
_UTIL_SCOPE = "opentelemetry.util.genai.handler"


class TelemetryHandlerScopeTest(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.log_exporter = InMemoryLogRecordExporter()
        self.logger_provider = LoggerProvider()
        self.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )

    def _emit(self, handler: TelemetryHandler) -> None:
        with handler.inference("prov", request_model="model"):
            pass

    def test_instrumentation_scope_is_used_for_all_signals(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
                "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true",
            },
        ):
            handler = TelemetryHandler(
                tracer_provider=self.tracer_provider,
                meter_provider=self.meter_provider,
                logger_provider=self.logger_provider,
                instrumentation_scope_name="opentelemetry.instrumentation.genai.example",
                instrumentation_scope_version="4.2b0",
            )
            self._emit(handler)

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(
            span.instrumentation_scope.name,
            "opentelemetry.instrumentation.genai.example",
        )
        self.assertEqual(span.instrumentation_scope.version, "4.2b0")
        self.assertEqual(span.instrumentation_scope.schema_url, _SCHEMA_URL)

        scopes = [
            scope_metric.scope
            for resource_metric in self.memory_metrics_reader.get_metrics_data().resource_metrics
            for scope_metric in resource_metric.scope_metrics
        ]
        self.assertEqual(len(scopes), 1)
        self.assertEqual(
            scopes[0].name, "opentelemetry.instrumentation.genai.example"
        )
        self.assertEqual(scopes[0].version, "4.2b0")
        self.assertEqual(scopes[0].schema_url, _SCHEMA_URL)

        log_scope = self.log_exporter.get_finished_logs()[
            0
        ].instrumentation_scope
        self.assertEqual(
            log_scope.name, "opentelemetry.instrumentation.genai.example"
        )
        self.assertEqual(log_scope.version, "4.2b0")
        self.assertEqual(log_scope.schema_url, _SCHEMA_URL)

    def test_defaults_to_util_scope(self) -> None:
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )
        self._emit(handler)

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(span.instrumentation_scope.name, _UTIL_SCOPE)
        self.assertEqual(span.instrumentation_scope.version, __version__)

    def test_version_without_name_falls_back_to_util_scope(self) -> None:
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            instrumentation_scope_version="4.2b0",
        )
        self._emit(handler)

        span = self.memory_exporter.get_finished_spans()[0]
        self.assertEqual(span.instrumentation_scope.name, _UTIL_SCOPE)
        self.assertEqual(span.instrumentation_scope.version, __version__)
