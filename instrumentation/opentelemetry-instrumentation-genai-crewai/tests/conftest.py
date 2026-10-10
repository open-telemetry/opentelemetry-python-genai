# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for CrewAI instrumentation."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.instrumentor import instrument

if TYPE_CHECKING:
    from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


def pytest_configure(config: pytest.Config) -> None:
    # CrewAI reads these when it is imported, and older releases then install
    # a global TracerProvider. The instrumentation package imports crewai at
    # module level, so nothing in this conftest may import it before this
    # hook runs; the fixtures below import the instrumentor lazily.
    del config
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault(
        "CREWAI_STORAGE_DIR", tempfile.mkdtemp(prefix="crewai-test-storage-")
    )


def _instrumentor() -> CrewAIInstrumentor:
    from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor

    return CrewAIInstrumentor()


@pytest.fixture
def instrument_crewai(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[CrewAIInstrumentor]:
    """Instrument CrewAI with the shared test providers."""
    with instrument(
        _instrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_crewai_with_content(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[CrewAIInstrumentor]:
    """Instrument CrewAI and capture content on spans."""
    with instrument(
        _instrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
