# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry-point and instrument/uninstrument lifecycle tests.

The instrumentor patches nothing yet, so these cover the parts of its contract
that hold before any patching lands: the entry point resolves, the declared
dependency is reported, and repeated instrument/uninstrument cycles stay quiet.
"""

from __future__ import annotations

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.util._importlib_metadata import entry_points


def test_entrypoint_loads_instrumentor() -> None:
    (entrypoint,) = entry_points(
        group="opentelemetry_instrumentor", name="smolagents"
    )
    instrumentor = entrypoint.load()()
    assert isinstance(instrumentor, SmolagentsInstrumentor)


def test_instrumentation_dependencies() -> None:
    dependencies = SmolagentsInstrumentor().instrumentation_dependencies()
    assert "smolagents >= 1.24.0" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider, logger_provider, meter_provider
) -> None:
    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    assert instrumentor.is_instrumented_by_opentelemetry

    instrumentor.uninstrument()
    assert not instrumentor.is_instrumented_by_opentelemetry


def test_repeated_instrument_uninstrument(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # BaseInstrumentor returns a per-class singleton, so the lifecycle has to
    # survive being driven more than once.
    instrumentor = SmolagentsInstrumentor()
    for _ in range(2):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        assert instrumentor.is_instrumented_by_opentelemetry
        instrumentor.uninstrument()
        assert not instrumentor.is_instrumented_by_opentelemetry


def test_uninstrument_through_a_new_constructor_call(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # BaseInstrumentor.__new__ returns a per-class singleton but Python still
    # runs __init__ on every construction, so the documented
    # SmolagentsInstrumentor().instrument() / SmolagentsInstrumentor()
    # .uninstrument() form has to work on the live instance.
    SmolagentsInstrumentor().instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    SmolagentsInstrumentor().uninstrument()

    assert not SmolagentsInstrumentor().is_instrumented_by_opentelemetry


def test_uninstrument_without_instrument() -> None:
    # BaseInstrumentor.uninstrument() short-circuits, but _uninstrument() must
    # also be a no-op on its own: the rollback path in _instrument() calls it
    # after a partial patch.
    SmolagentsInstrumentor().uninstrument()
    SmolagentsInstrumentor()._uninstrument()


def test_instrument_with_no_providers() -> None:
    # Without providers the handler falls back to the globals; instrumenting
    # must not require a caller to pass them.
    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument()
    try:
        assert instrumentor.is_instrumented_by_opentelemetry
    finally:
        instrumentor.uninstrument()
