# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CrewAI instrumentor lifecycle."""

from __future__ import annotations

import os

import pytest
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.crew_events import CrewKickoffStartedEvent

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


def test_instrumentation_dependencies() -> None:
    assert CrewAIInstrumentor().instrumentation_dependencies() == (
        "crewai >= 1.10.1, < 2",
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


@pytest.mark.parametrize("configured_value", [None, "false", "true"])
def test_instrumentation_preserves_crewai_telemetry_environment(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str | None,
) -> None:
    variable = "CREWAI_DISABLE_TELEMETRY"
    if configured_value is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, configured_value)

    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument()
    try:
        assert os.environ.get(variable) == configured_value
    finally:
        instrumentor.uninstrument()

    assert os.environ.get(variable) == configured_value


def test_uninstrument_preserves_existing_crewai_event_handler() -> None:
    observed_events: list[CrewKickoffStartedEvent] = []

    def existing_handler(
        source: object, event: CrewKickoffStartedEvent
    ) -> None:
        del source
        observed_events.append(event)

    crewai_event_bus.on(CrewKickoffStartedEvent)(existing_handler)
    instrumentor = CrewAIInstrumentor()
    try:
        instrumentor.instrument()
        instrumentor.uninstrument()

        event = CrewKickoffStartedEvent(
            crew_name="user crew",
            crew=None,
            inputs=None,
        )
        future = crewai_event_bus.emit(object(), event)
        if future is not None:
            future.result(timeout=5)
    finally:
        crewai_event_bus.off(CrewKickoffStartedEvent, existing_handler)
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

    assert observed_events == [event]
