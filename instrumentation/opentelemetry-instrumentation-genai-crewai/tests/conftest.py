# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for CrewAI instrumentation."""

import os
import tempfile
from collections.abc import Iterator

import pytest

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]

# CrewAI reads these at import time, so they must be set before any test
# module imports ``crewai``.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault(
    "CREWAI_STORAGE_DIR", tempfile.mkdtemp(prefix="crewai-test-storage-")
)


@pytest.fixture
def instrument_crewai(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[CrewAIInstrumentor]:
    """Instrument CrewAI with the shared test providers."""
    with instrument(
        CrewAIInstrumentor(),
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
        CrewAIInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
