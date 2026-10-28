# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for QwenPaw instrumentation tests."""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import importlib
from importlib.metadata import version

import pytest

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
]


@pytest.fixture(name="runner_module")
def fixture_runner_module():
    if int(version("qwenpaw").split(".")[0]) >= 2:
        pytest.skip("AgentRunner is the QwenPaw 1.x entry point")
    return importlib.import_module("qwenpaw.app.runner.runner")


@pytest.fixture(name="runtime_module")
def fixture_runtime_module():
    if int(version("qwenpaw").split(".")[0]) < 2:
        pytest.skip("Runtime is the QwenPaw 2.x entry point")
    return importlib.import_module("qwenpaw.runtime.runtime")


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
