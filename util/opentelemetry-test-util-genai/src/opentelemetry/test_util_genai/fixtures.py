# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for GenAI instrumentation tests.

The intended import shape in a per-package ``tests/conftest.py`` is::

    from opentelemetry.test_util_genai.fixtures import *  # noqa: F401,F403

That single line picks up every fixture defined here. Replaces the
~70-line duplicated provider/exporter setup that every instrumentation
``tests/conftest.py`` carried.

Fixtures are function-scoped and yield the bare in-memory exporters /
providers — the per-instrumentation conftest is responsible for handing them
to the instrumentor's ``.instrument(tracer_provider=..., logger_provider=...,
meter_provider=...)`` call. Globals (``trace.set_tracer_provider`` and
friends) are deliberately **not** set so tests stay isolated and don't leak
across the session.

Parametrized fixtures
---------------------

``content_capture`` yields each ``ContentCapturingMode`` enum value in
``CONTENT_CAPTURE_MODES`` in turn (``NO_CONTENT`` and ``SPAN_ONLY``). It sets
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` for the duration of
the test and restores the previous value afterwards. ``SPAN_AND_EVENT`` and
``EVENT_ONLY`` coverage lives in targeted per-package tests rather than the
default matrix.

Conformance fixture
-------------------

``weaver_live_check`` yields a started ``WeaverLiveCheck`` for a single
conformance scenario. Consumed by ``tests/test_conformance.py`` via
``opentelemetry.test_util_genai.conformance.run_conformance``. Auto-skips
when the OTLP/gRPC exporter or the ``weaver`` binary aren't available —
local runs typically skip; CI installs ``weaver`` ahead of the
``*-conformance`` tox envs.
"""

from __future__ import annotations

import functools
import os
import shutil
import sys
import tarfile
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai._setup_weaver import (
    advice_data_glob,
    policies_dir,
    semconv_registry,
    weaver_config_file,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ContentCapturingMode

# ─── In-memory exporters and providers ──────────────────────────────────────


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    """Function-scoped in-memory span exporter."""
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


@pytest.fixture
def log_exporter() -> Iterator[InMemoryLogRecordExporter]:
    """Function-scoped in-memory log-record exporter."""
    exporter = InMemoryLogRecordExporter()
    yield exporter
    exporter.clear()


@pytest.fixture
def metric_reader() -> Iterator[InMemoryMetricReader]:
    """Function-scoped in-memory metric reader."""
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture
def tracer_provider(
    span_exporter: InMemorySpanExporter,
) -> Iterator[TracerProvider]:
    """``TracerProvider`` wired to ``span_exporter`` via ``SimpleSpanProcessor``.

    Hand this directly to ``instrumentor.instrument(tracer_provider=...)``;
    do NOT call ``trace.set_tracer_provider`` — keeping the global unset
    avoids cross-test leaks.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider
    provider.shutdown()


@pytest.fixture
def logger_provider(
    log_exporter: InMemoryLogRecordExporter,
) -> Iterator[LoggerProvider]:
    """``LoggerProvider`` wired to ``log_exporter`` via ``SimpleLogRecordProcessor``."""
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    yield provider
    provider.shutdown()


@pytest.fixture
def meter_provider(
    metric_reader: InMemoryMetricReader,
) -> Iterator[MeterProvider]:
    """``MeterProvider`` wired to ``metric_reader``."""
    provider = MeterProvider(metric_readers=[metric_reader])
    yield provider
    provider.shutdown()


# ─── Instrumentation scope guard ────────────────────────────────────────────


def _scope_problem(caller: str, kwargs: dict[str, Any]) -> str | None:
    name = kwargs.get("instrumentation_scope_name")
    if not name:
        return "built a TelemetryHandler without instrumentation_scope_name"
    if caller != name and not caller.startswith(f"{name}."):
        return f"reports the unrelated instrumentation scope {name!r}"
    expected_version = getattr(sys.modules.get(name), "__version__", None)
    if kwargs.get("instrumentation_scope_version") != expected_version:
        return (
            f"reports instrumentation scope version "
            f"{kwargs.get('instrumentation_scope_version')!r} instead of "
            f"{expected_version!r}"
        )
    return None


@pytest.fixture(autouse=True)
def _instrumentation_scope_guard() -> Iterator[None]:
    """Fail any test where instrumentation code builds a mis-scoped handler.

    ``TelemetryHandler`` falls back to the util's own module when
    ``instrumentation_scope_name`` is omitted, which makes the telemetry of
    every util-genai based instrumentation indistinguishable by
    ``otel.scope.name``. Only handlers constructed from
    ``opentelemetry.instrumentation.*`` are checked, so tests are free to build
    bare handlers of their own.
    """
    original_init = TelemetryHandler.__init__
    problems: set[str] = set()

    @functools.wraps(original_init)
    def guarded_init(
        self: TelemetryHandler, *args: Any, **kwargs: Any
    ) -> None:
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if caller.startswith("opentelemetry.instrumentation."):
            problem = _scope_problem(caller, kwargs)
            if problem is not None:
                problems.add(f"{caller} {problem}")
        original_init(self, *args, **kwargs)

    with patch.object(TelemetryHandler, "__init__", guarded_init):
        yield

    assert not problems, "; ".join(sorted(problems))


# ─── Content-capture parametrization ────────────────────────────────────────

# Default matrix every instrumentation exercises through `content_capture`.
# SPAN_AND_EVENT and EVENT_ONLY belong in targeted per-package tests rather
# than the default fan-out — they multiply the test count without buying
# coverage that a handful of explicit tests don't already give.
CONTENT_CAPTURE_MODES: tuple[ContentCapturingMode, ContentCapturingMode] = (
    ContentCapturingMode.NO_CONTENT,
    ContentCapturingMode.SPAN_ONLY,
)


@pytest.fixture(params=CONTENT_CAPTURE_MODES, ids=lambda m: m.name)
def content_capture(
    request: pytest.FixtureRequest,
) -> Iterator[ContentCapturingMode]:
    """Parametrized fixture yielding each content-capture mode in turn.

    Sets ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` for the test
    and restores the previous value afterwards.
    """
    mode: ContentCapturingMode = request.param
    previous = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
    )
    os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = mode.name
    try:
        yield mode
    finally:
        if previous is None:
            os.environ.pop(
                OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None
            )
        else:
            os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
                previous
            )


# ─── Weaver live-check ──────────────────────────────────────────────────────


@pytest.fixture
def weaver_live_check() -> Iterator[Any]:
    """Yield a started ``WeaverLiveCheck`` for one conformance scenario.

    Function-scoped so violations don't leak across scenarios. Auto-skips
    when the OTLP/gRPC exporter, the ``weaver`` binary, or the
    semantic-conventions registry can't be resolved.
    """
    try:
        import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
    except ImportError:
        pytest.skip("opentelemetry-exporter-otlp-proto-grpc not installed")

    if shutil.which("weaver") is None:
        pytest.skip(
            "weaver binary not on PATH — install it from "
            "https://github.com/open-telemetry/weaver/releases (CI installs "
            "it via the test.yml conformance setup step)"
        )

    # WeaverLiveCheck transitively imports the OTLP/gRPC exporter, so it
    # stays inside the function body — the probe above is what gates this.
    from opentelemetry.test.weaver_live_check import (
        WeaverLiveCheck,
    )

    try:
        policies = str(policies_dir())
        registry = str(semconv_registry())
        advice_data = advice_data_glob()
    except (OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
        pytest.skip(f"could not provision semantic-conventions: {exc}")

    with WeaverLiveCheck(
        registry=registry,
        policies_dir=policies,
        extra_args=[
            "--config",
            str(weaver_config_file()),
            "--advice-data",
            advice_data,
        ],
    ) as weaver:
        yield weaver
