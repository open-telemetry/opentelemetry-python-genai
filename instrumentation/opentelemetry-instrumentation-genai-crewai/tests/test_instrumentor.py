# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CrewAI instrumentor lifecycle."""

import os

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


def test_instrumentation_dependencies() -> None:
    assert CrewAIInstrumentor().instrumentation_dependencies() == (
        "crewai >= 1.10.1",
    )


def test_instrument_uninstrument_cycle(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    instrumentor = CrewAIInstrumentor()

    for _ in range(2):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        instrumentor.uninstrument()


def test_instrument_with_global_providers() -> None:
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()


def test_native_telemetry_is_disabled_while_instrumented(
    monkeypatch,
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    instrumentor = CrewAIInstrumentor()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true"
    instrumentor.uninstrument()
    assert "CREWAI_DISABLE_TELEMETRY" not in os.environ


def test_explicit_native_telemetry_configuration_is_preserved(
    monkeypatch,
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "false")
    instrumentor = CrewAIInstrumentor()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "false"
    instrumentor.uninstrument()
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "false"
