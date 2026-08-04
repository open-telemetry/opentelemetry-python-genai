# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for QwenPaw instrumentation tests."""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import importlib

import pytest

from opentelemetry.instrumentation.genai.qwenpaw import (
    CoPawInstrumentor,
    QwenPawInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
]

# One instrumentor plugin per runtime distribution: `qwenpaw`, or `copaw`
# (QwenPaw's former name). Tests run against whichever is installed.
_RUNTIME_TARGETS = (
    ("qwenpaw.app.runner.runner", QwenPawInstrumentor),
    ("copaw.app.runner.runner", CoPawInstrumentor),
)


def _import_runtime_target():
    for module_name, instrumentor_cls in _RUNTIME_TARGETS:
        try:
            return importlib.import_module(module_name), instrumentor_cls
        except ImportError:
            continue
    pytest.skip("No supported QwenPaw runtime distribution is installed")


@pytest.fixture(name="runtime_target")
def fixture_runtime_target():
    return _import_runtime_target()


@pytest.fixture(name="runner_module")
def fixture_runner_module(runtime_target):
    return runtime_target[0]


@pytest.fixture(name="instrumentor_cls")
def fixture_instrumentor_cls(runtime_target):
    return runtime_target[1]


@pytest.fixture
def instrument_no_content(
    instrumentor_cls, tracer_provider, logger_provider, meter_provider
):
    """Instrument the installed runtime without content capture."""
    with instrument(
        instrumentor_cls(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(
    instrumentor_cls, tracer_provider, logger_provider, meter_provider
):
    """Instrument the installed runtime with ``SPAN_ONLY`` content capture."""
    with instrument(
        instrumentor_cls(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
