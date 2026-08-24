# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DSPy instrumentor lifecycle."""

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


def test_instrumentation_dependencies() -> None:
    assert DSPyInstrumentor().instrumentation_dependencies() == (
        "dspy >= 2.5.0, < 4",
    )


def test_instrument_uninstrument_cycle(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    instrumentor = DSPyInstrumentor()

    for _ in range(2):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        instrumentor.uninstrument()


def test_instrument_with_global_providers() -> None:
    instrumentor = DSPyInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()
