# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for QwenPaw instrumentation tests."""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import importlib

import pytest

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
]


@pytest.fixture(name="runner_module")
def fixture_runner_module():
    try:
        return importlib.import_module("qwenpaw.app.runner.runner")
    except ImportError:
        pytest.skip("qwenpaw is not installed")


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument QwenPaw without content capture."""
    with instrument(
        QwenPawInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    """Instrument QwenPaw with ``SPAN_ONLY`` content capture."""
    with instrument(
        QwenPawInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
