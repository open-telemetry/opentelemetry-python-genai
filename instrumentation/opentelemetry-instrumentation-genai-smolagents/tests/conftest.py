# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for smolagents instrumentation tests."""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


@pytest.fixture
def instrument_smolagents(tracer_provider, logger_provider, meter_provider):
    with instrument(
        SmolagentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor
