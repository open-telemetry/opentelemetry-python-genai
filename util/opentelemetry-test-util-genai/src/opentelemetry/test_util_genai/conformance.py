# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance runner for GenAI instrumentations.

Intended call shape from a per-package ``tests/test_conformance.py``::

    @pytest.mark.parametrize(
        "scenario", [InferenceScenario(), ToolCallingScenario()]
    )
    def test_conformance(scenario, vcr, weaver_live_check):
        report = run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
        # Optionally layer lib-specific assertions on `report` here.

The ``*-conformance`` tox envs point pytest directly at
``tests/test_conformance.py``; the regular ``*-{oldest,latest}`` envs
``--ignore`` it. The OTLP/gRPC exporter and ``weaver_live_check`` only need
to be installed in the conformance envs.

Each ``tests/conformance/<op>.py`` defines a :class:`Scenario` subclass with:

- ``expected_spans`` — ``gen_ai.operation.name`` values that must appear in
  the report's span samples.
- ``expected_metrics`` — metric names that must appear in
  ``statistics.seen_registry_metrics``.
- ``run(*, tracer_provider, meter_provider, logger_provider, vcr)`` — wires
  the instrumentor against the providers and exercises one semconv operation
  type's happy path inside ``vcr.use_cassette(...)``.
- ``validate(report)`` — asserts the emitted telemetry matches the scenario.
  The base implementation enforces ``expected_spans`` / ``expected_metrics``
  presence; per-scenario overrides call ``super().validate(report)`` and
  layer on additional checks against the weaver report.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.test.weaver_live_check import (
    LiveCheckError,
    LiveCheckReport,
    WeaverLiveCheck,
)


class Scenario(ABC):
    """Base class every ``tests/conformance/<op>.py`` scenario must subclass."""

    expected_spans: ClassVar[tuple[str, ...]] = ()
    expected_metrics: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None: ...

    def validate(self, report: LiveCheckReport) -> None:
        """Assert the weaver live-check report matches the scenario.

        The base implementation enforces that every ``expected_spans`` and
        ``expected_metrics`` entry appears at least once. Subclasses should
        override and call ``super().validate(report)`` to layer on extra
        scenario-specific checks against the report.
        """
        expected_spans = set(self.expected_spans)
        seen_spans = _seen_span_operations(report)
        missing_spans = expected_spans - seen_spans
        assert not missing_spans, (
            f"Expected span operations {sorted(expected_spans)} but weaver "
            f"only saw {sorted(seen_spans)} — missing {sorted(missing_spans)}"
        )

        expected_metrics = set(self.expected_metrics)
        seen_metrics = _seen_metric_names(report)
        missing_metrics = expected_metrics - seen_metrics
        assert not missing_metrics, (
            f"Expected metrics {sorted(expected_metrics)} but weaver only "
            f"saw {sorted(seen_metrics)} — missing {sorted(missing_metrics)}"
        )


def _build_providers(
    endpoint: str,
) -> tuple[TracerProvider, MeterProvider, LoggerProvider]:
    # OTLP/gRPC exporters are only installed in the *-conformance tox envs
    # (see dev-requirements-conformance.txt). Import lazily so this module
    # stays importable in regular test envs that exclude conformance tests.
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (  # noqa: PLC0415
        OTLPLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # noqa: PLC0415
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )

    # Disable periodic export — metrics flush via the explicit force_flush()
    # at the end of the scenario, so the report is deterministic.
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint, insecure=True),
        export_interval_millis=2**31 - 1,
    )
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(
            OTLPLogExporter(endpoint=endpoint, insecure=True)
        )
    )

    return tracer_provider, meter_provider, logger_provider


def _seen_metric_names(report: LiveCheckReport) -> set[str]:
    """Names of metrics weaver observed at least one data point for."""
    seen = report["statistics"]["seen_registry_metrics"]
    return {name for name, count in seen.items() if count}


def _seen_span_operations(report: LiveCheckReport) -> set[str]:
    """`gen_ai.operation.name` values observed across the report's span samples."""
    return {
        attr["value"]
        for entry in report["samples"]
        if "span" in entry
        for attr in entry["span"]["attributes"]
        if attr["name"] == "gen_ai.operation.name"
    }


def _dump_report(scenario: Scenario, report: LiveCheckReport) -> None:
    out = Path("weaver_reports") / f"{type(scenario).__name__}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report._report, indent=2, sort_keys=True))  # noqa: SLF001


def run_conformance(
    scenario: Scenario,
    *,
    vcr: Any,
    weaver: WeaverLiveCheck,
) -> LiveCheckReport:
    """Run one conformance scenario and return the weaver report.

    Raises :class:`LiveCheckError` on semconv violations.
    """
    tracer_provider, meter_provider, logger_provider = _build_providers(
        weaver.otlp_endpoint
    )

    try:
        scenario.run(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            vcr=vcr,
        )
        tracer_provider.force_flush()
        meter_provider.force_flush()
        logger_provider.force_flush()

        try:
            report = weaver.end_and_check(timeout=120)
            _dump_report(scenario, report)
        except LiveCheckError as exc:
            _dump_report(scenario, exc.report)
            raise

        scenario.validate(report)
        return report
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()
        logger_provider.shutdown()
