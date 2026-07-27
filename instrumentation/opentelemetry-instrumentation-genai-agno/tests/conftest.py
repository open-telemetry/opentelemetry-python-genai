# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Agno instrumentation tests."""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


@pytest.fixture
def instrument_agno(tracer_provider, logger_provider, meter_provider):
    """Fixture to instrument Agno with test providers."""
    with instrument(
        AgnoInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def uninstrument_agno():
    """Fixture to ensure Agno is uninstrumented after test."""
    yield
    AgnoInstrumentor().uninstrument()
