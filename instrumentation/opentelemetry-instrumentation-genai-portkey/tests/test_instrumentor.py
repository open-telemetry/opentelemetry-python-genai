# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PortkeyInstrumentor class."""

from __future__ import annotations

from opentelemetry.instrumentation.genai.portkey import (
    PortkeyInstrumentor,
)


def test_instrumentor_instantiation() -> None:
    """Test that the instrumentor can be instantiated."""
    instrumentor = PortkeyInstrumentor()
    assert instrumentor is not None
    assert isinstance(instrumentor, PortkeyInstrumentor)


def test_instrumentation_dependencies() -> None:
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = PortkeyInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies is not None
    assert len(dependencies) > 0
    assert "portkey-ai >= 1.0.0, < 3" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Test that instrument() and uninstrument() can be called multiple times."""
    instrumentor = PortkeyInstrumentor()

    # First instrumentation
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # First uninstrumentation
    instrumentor.uninstrument()

    # Second instrumentation (should work)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second uninstrumentation
    instrumentor.uninstrument()


def test_multiple_instrumentation_calls(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Test that multiple instrument() calls don't cause issues."""
    instrumentor = PortkeyInstrumentor()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    instrumentor.uninstrument()
