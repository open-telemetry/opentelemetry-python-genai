# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import Any, ClassVar
from unittest.mock import patch

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.util.genai.handler import TelemetryHandler

from .instrumentor import instrument


class TelemetryHandlerScopeTest:
    instrumentor_class: ClassVar[type[BaseInstrumentor]]
    instrumentation_scope_name: ClassVar[str]
    instrumentation_scope_version: ClassVar[str]

    def test_instrumentation_scope_is_used_for_all_signals(
        self,
        tracer_provider: TracerProvider,
        logger_provider: LoggerProvider,
        meter_provider: MeterProvider,
        span_exporter: InMemorySpanExporter,
        log_exporter: InMemoryLogRecordExporter,
        metric_reader: InMemoryMetricReader,
    ) -> None:
        handlers: list[TelemetryHandler] = []
        original_init = TelemetryHandler.__init__

        @functools.wraps(original_init)
        def capture_handler(
            handler: TelemetryHandler, *args: Any, **kwargs: Any
        ) -> None:
            original_init(handler, *args, **kwargs)
            handlers.append(handler)

        with patch.object(TelemetryHandler, "__init__", capture_handler):
            with instrument(
                self.instrumentor_class(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="EVENT_ONLY",
                emit_event=True,
            ):
                assert len(handlers) == 1
                with handlers[0].inference("provider", request_model="model"):
                    pass

        scope_name = self.instrumentation_scope_name
        scope_version = self.instrumentation_scope_version
        schema_url = Schemas.V1_37_0.value

        (span,) = span_exporter.get_finished_spans()
        assert span.instrumentation_scope.name == scope_name
        assert span.instrumentation_scope.version == scope_version
        assert span.instrumentation_scope.schema_url == schema_url

        scope_metrics = [
            scope_metric
            for resource_metric in metric_reader.get_metrics_data().resource_metrics
            for scope_metric in resource_metric.scope_metrics
        ]
        assert len(scope_metrics) == 1
        assert scope_metrics[0].scope.name == scope_name
        assert scope_metrics[0].scope.version == scope_version
        assert scope_metrics[0].scope.schema_url == schema_url

        (log,) = log_exporter.get_finished_logs()
        assert log.instrumentation_scope.name == scope_name
        assert log.instrumentation_scope.version == scope_version
        assert log.instrumentation_scope.schema_url == schema_url
