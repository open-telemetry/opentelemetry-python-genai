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

import functools
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from jsonschema import Draft202012Validator

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


_CONTENT_SCHEMA_FILES: dict[str, str] = {
    "gen_ai.input.messages": "gen-ai-input-messages.json",
    "gen_ai.output.messages": "gen-ai-output-messages.json",
    "gen_ai.system_instructions": "gen-ai-system-instructions.json",
    "gen_ai.tool.definitions": "gen-ai-tool-definitions.json",
    "gen_ai.retrieval.documents": "gen-ai-retrieval-documents.json",
}


@functools.cache
def _get_content_validators() -> dict[str, Draft202012Validator]:
    import jsonschema

    from opentelemetry.test_util_genai._setup_weaver import (
        _provision_genai_root,
    )

    schema_dir = _provision_genai_root() / "model" / "gen-ai"
    validators: dict[str, Draft202012Validator] = {}
    for attr_name, filename in _CONTENT_SCHEMA_FILES.items():
        schema_path = schema_dir / filename
        if schema_path.is_file():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            validators[attr_name] = jsonschema.Draft202012Validator(schema)
    return validators


def _validate_content_attribute(
    attr_entry: dict[str, Any],
    name: str,
    val: Any,
) -> None:
    validators = _get_content_validators()
    validator = validators.get(name)
    if validator is None or val is None:
        return

    errors = sorted(validator.iter_errors(val), key=lambda e: str(e.path))
    live_check = attr_entry.setdefault("live_check_result", {})
    all_advice: list[dict[str, Any]] = [
        a
        for a in live_check.get("all_advice", [])
        if a.get("id") != "genai_content_schema"
    ]
    if errors:
        error_messages = [err.message for err in errors]
        signal_name = next(
            (a.get("signal_name") for a in all_advice if a.get("signal_name")),
            None,
        )
        signal_type = next(
            (a.get("signal_type") for a in all_advice if a.get("signal_type")),
            None,
        )
        finding: dict[str, Any] = {
            "id": "genai_content_schema",
            "level": "violation",
            "context": {
                "attribute": name,
                "errors": error_messages,
            },
            "message": (
                f"Attribute '{name}' value does not conform to the GenAI schema: "
                f"{error_messages}"
            ),
            "signal_name": signal_name,
            "signal_type": signal_type,
            "type": "PolicyFinding",
        }
        all_advice.append(finding)
        live_check["highest_advice_level"] = "violation"
    live_check["all_advice"] = all_advice


def _enrich_report_samples(
    report: LiveCheckReport,
    finished_spans: Sequence[ReadableSpan],
    finished_logs: Sequence[Any],
) -> None:
    # Weaver drops KvlistValue (maps) during OTLP-to-JSON conversion, resulting in
    # empty lists [] or None for structured content attributes (messages, tool definitions).
    # Re-hydrate them from the emitted spans/logs so scenario validate() checks succeed,
    # and validate the restored content against the semconv GenAI JSON schemas.
    report.__dict__.pop("violations", None)
    span_idx = 0
    log_idx = 0
    for entry in report.get("samples", []):
        if "span" in entry and span_idx < len(finished_spans):
            span = finished_spans[span_idx]
            span_idx += 1
            if span.attributes:
                for attr_entry in entry["span"].get("attributes", []):
                    name = attr_entry.get("name")
                    if name in span.attributes:
                        orig = span.attributes[name]
                        if isinstance(orig, (dict, list, tuple)):
                            val = _normalize_sample_value(orig)
                            attr_entry["value"] = val
                            if name in _CONTENT_SCHEMA_FILES:
                                _validate_content_attribute(
                                    attr_entry, name, val
                                )
        elif "log" in entry and log_idx < len(finished_logs):
            log = finished_logs[log_idx]
            log_idx += 1
            log_record = getattr(log, "log_record", log)
            attrs = getattr(log_record, "attributes", None)
            if attrs:
                for attr_entry in entry["log"].get("attributes", []):
                    name = attr_entry.get("name")
                    if name in attrs:
                        orig = attrs[name]
                        if isinstance(orig, (dict, list, tuple)):
                            val = _normalize_sample_value(orig)
                            attr_entry["value"] = val
                            if name in _CONTENT_SCHEMA_FILES:
                                _validate_content_attribute(
                                    attr_entry, name, val
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
