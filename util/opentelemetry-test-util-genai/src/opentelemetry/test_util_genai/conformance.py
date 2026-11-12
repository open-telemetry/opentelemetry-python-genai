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

- ``expected_spans`` — maps each ``gen_ai.operation.name`` to the exact number
  of spans the report must carry it on, with no undeclared operations present.
- ``expected_metrics`` — metric names that must appear in
  ``statistics.seen_registry_metrics``.
- ``expected_violations`` — :class:`ExpectedViolation` entries for known
  gaps. ``run_conformance`` fails on undeclared violations and on declared
  entries weaver no longer reports.
- ``run(*, tracer_provider, meter_provider, logger_provider, vcr)`` — wires
  the instrumentor against the providers and exercises one semconv operation
  type's happy path inside ``vcr.use_cassette(...)``.
- ``validate(report)`` — asserts the emitted telemetry matches the scenario.
  The base implementation enforces the exact ``expected_spans`` operation
  counts and ``expected_metrics`` presence; per-scenario overrides call
  ``super().validate(report)`` and layer on additional checks against the
  weaver report.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test.weaver_live_check import (
    LiveCheckError,
    LiveCheckReport,
    WeaverLiveCheck,
)


@dataclass(frozen=True)
class ExpectedViolation:
    """One known-and-accepted semconv violation.

    Matched by weaver advice ``id`` plus a substring of its ``message``.
    Declared entries the report no longer contains are flagged so
    suppressions don't rot.
    """

    advice_id: str
    message_substring: str

    def matches(self, violation: dict[str, Any]) -> bool:
        return violation.get(
            "id"
        ) == self.advice_id and self.message_substring in str(
            violation.get("message", "")
        )


class Scenario(ABC):
    """Base class every ``tests/conformance/<op>.py`` scenario must subclass."""

    expected_spans: ClassVar[dict[str, int]] = {}
    expected_metrics: ClassVar[tuple[str, ...]] = ()
    expected_violations: ClassVar[tuple[ExpectedViolation, ...]] = ()

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

        ``expected_spans`` maps each ``gen_ai.operation.name`` to the exact
        number of spans that must carry it: the report's spans must match these
        operation counts exactly — no missing, no extra. ``expected_metrics``
        entries must each appear at least once. Subclasses should override and
        call ``super().validate(report)`` to layer on extra scenario-specific
        checks against the report.
        """
        expected_spans = dict(self.expected_spans)
        seen_spans = _seen_span_operations(report)
        assert seen_spans == expected_spans, (
            f"Expected span operation counts {dict(sorted(expected_spans.items()))} "
            f"but weaver saw {dict(sorted(seen_spans.items()))}"
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
) -> tuple[
    TracerProvider,
    MeterProvider,
    LoggerProvider,
    InMemorySpanExporter,
    InMemoryLogRecordExporter,
]:
    # OTLP/gRPC exporters are only installed in the *-conformance tox envs
    # (see dev-requirements-conformance.txt). Import lazily so this module
    # stays importable in regular test envs that exclude conformance tests.
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    span_exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

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
    log_exporter = InMemoryLogRecordExporter()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    return (
        tracer_provider,
        meter_provider,
        logger_provider,
        span_exporter,
        log_exporter,
    )


def _seen_metric_names(report: LiveCheckReport) -> set[str]:
    """Names of metrics weaver observed at least one data point for."""
    seen = report["statistics"]["seen_registry_metrics"]
    return {name for name, count in seen.items() if count}


def _seen_span_operations(report: LiveCheckReport) -> dict[str, int]:
    """`gen_ai.operation.name` counts across the report's span samples."""
    counts: dict[str, int] = {}
    for entry in report["samples"]:
        if "span" not in entry:
            continue
        for attr in entry["span"]["attributes"]:
            if attr["name"] == "gen_ai.operation.name":
                counts[attr["value"]] = counts.get(attr["value"], 0) + 1
                break
    return counts


def _normalize_sample_value(val: Any) -> Any:
    if isinstance(val, (tuple, list)):
        return [_normalize_sample_value(x) for x in val]
    if isinstance(val, dict):
        return {k: _normalize_sample_value(v) for k, v in val.items()}
    return val


def _enrich_attributes(
    sample_attrs: list[dict[str, Any]], attrs: Mapping[str, Any]
) -> None:
    from opentelemetry.test_util_genai.content import (
        SCHEMA_FILES,
        validate_content_attribute,
    )

    for attr_entry in sample_attrs:
        name = attr_entry.get("name")
        if name not in attrs:
            continue
        raw_val = attrs[name]
        if name in SCHEMA_FILES:
            attr_entry["value"] = validate_content_attribute(name, raw_val)
        elif isinstance(raw_val, (dict, list, tuple, bytes)):
            attr_entry["value"] = _normalize_sample_value(raw_val)


def _sample_scalar_attrs(
    sample_attrs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        a["name"]: a.get("value")
        for a in sample_attrs
        if "name" in a and isinstance(a.get("value"), (str, int, float, bool))
    }


def _match_span(
    sample_span: dict[str, Any],
    remaining_spans: list[ReadableSpan],
) -> ReadableSpan | None:
    sample_name = sample_span.get("name")
    sample_scalars = _sample_scalar_attrs(sample_span.get("attributes", []))
    for idx, span in enumerate(remaining_spans):
        if span.name != sample_name:
            continue
        span_attrs = span.attributes or {}
        if all(
            span_attrs.get(k) == v
            for k, v in sample_scalars.items()
            if k in span_attrs
            and isinstance(span_attrs[k], (str, int, float, bool))
        ):
            return remaining_spans.pop(idx)
    for idx, span in enumerate(remaining_spans):
        if span.name == sample_name:
            return remaining_spans.pop(idx)
    return remaining_spans.pop(0) if remaining_spans else None


def _match_log(
    sample_log: dict[str, Any],
    remaining_logs: list[Any],
) -> Any | None:
    sample_span_id = sample_log.get("span_id")
    sample_trace_id = sample_log.get("trace_id")
    sample_event_name = sample_log.get("event_name")
    for idx, log in enumerate(remaining_logs):
        log_record = getattr(log, "log_record", log)
        span_id = getattr(log_record, "span_id", None)
        trace_id = getattr(log_record, "trace_id", None)
        event_name = getattr(log_record, "event_name", None)
        if (
            isinstance(span_id, int)
            and isinstance(trace_id, int)
            and f"{span_id:016x}" == sample_span_id
            and f"{trace_id:032x}" == sample_trace_id
            and (not sample_event_name or event_name == sample_event_name)
        ):
            return remaining_logs.pop(idx)
    return remaining_logs.pop(0) if remaining_logs else None


def _enrich_report_samples(
    report: LiveCheckReport,
    finished_spans: Sequence[ReadableSpan],
    finished_logs: Sequence[Any],
) -> None:
    # Weaver v0.26.1 drops KvlistValue (maps) during OTLP-to-JSON conversion, resulting in
    # empty lists [] or None for structured content attributes (messages, tool definitions).
    # Re-hydrate them from the emitted spans/logs matched by identity and validate
    # structured content attributes against the pinned semconv JSON schemas.
    remaining_spans = list(finished_spans)
    remaining_logs = list(finished_logs)
    for entry in report.get("samples", []):
        if "span" in entry:
            span = _match_span(entry["span"], remaining_spans)
            if span is not None and span.attributes:
                _enrich_attributes(
                    entry["span"].get("attributes", []), span.attributes
                )
        elif "log" in entry:
            log = _match_log(entry["log"], remaining_logs)
            if log is not None:
                log_record = getattr(log, "log_record", log)
                attrs = getattr(log_record, "attributes", None)
                if attrs:
                    _enrich_attributes(
                        entry["log"].get("attributes", []), attrs
                    )


def _dump_report(scenario: Scenario, report: LiveCheckReport) -> None:
    out = Path("weaver_reports") / f"{type(scenario).__name__}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            report._report,
            indent=2,
            sort_keys=True,
            default=lambda o: repr(o) if isinstance(o, bytes) else str(o),
        )
    )


def run_conformance(
    scenario: Scenario,
    *,
    vcr: Any,
    weaver: WeaverLiveCheck,
) -> LiveCheckReport:
    """Run one conformance scenario and return the weaver report.

    Raises :class:`LiveCheckError` on undeclared violations and
    :class:`AssertionError` on declared violations weaver no longer
    reports.
    """
    (
        tracer_provider,
        meter_provider,
        logger_provider,
        span_exporter,
        log_exporter,
    ) = _build_providers(weaver.otlp_endpoint)

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

        report = weaver.end(timeout=120)
        _enrich_report_samples(
            report,
            span_exporter.get_finished_spans(),
            log_exporter.get_finished_logs(),
        )
        _dump_report(scenario, report)

        _check_violations(scenario, report)
        scenario.validate(report)
        return report
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()
        logger_provider.shutdown()


def _check_violations(scenario: Scenario, report: LiveCheckReport) -> None:
    """Reconcile weaver violations against ``scenario.expected_violations``."""
    violations = report.violations
    expected = scenario.expected_violations

    unexpected = [
        v for v in violations if not any(ev.matches(v) for ev in expected)
    ]
    if unexpected:
        raise LiveCheckError(
            "Unexpected semconv violations (not declared in "
            "scenario.expected_violations):\n"
            + "\n".join(f"- {v}" for v in unexpected),
            report,
        )

    unmet = [
        ev for ev in expected if not any(ev.matches(v) for v in violations)
    ]
    if unmet:
        raise AssertionError(
            "Scenario declares expected_violations that weaver did not "
            "report — likely fixed upstream. Remove these entries:\n"
            + "\n".join(f"- {ev}" for ev in unmet)
        )
