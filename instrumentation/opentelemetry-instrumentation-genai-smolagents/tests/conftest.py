# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for smolagents instrumentation tests."""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument

# No VCR plugin: the instrumented model classes run inference in this process,
# so there is no HTTP traffic to record.
pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    with instrument(
        SmolagentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    with instrument(
        SmolagentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_event_only(tracer_provider, logger_provider, meter_provider):
    with instrument(
        SmolagentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="EVENT_ONLY",
        emit_event=True,
    ) as instrumentor:
        yield instrumentor
