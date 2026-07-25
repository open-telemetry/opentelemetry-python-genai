# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgnoInstrumentor class."""

from __future__ import annotations

from opentelemetry.instrumentation.genai.agno import (
    AgnoInstrumentor,
)


def test_instrumentor_instantiation() -> None:
    """Test that the instrumentor can be instantiated."""
    instrumentor = AgnoInstrumentor()
    assert instrumentor is not None
    assert isinstance(instrumentor, AgnoInstrumentor)


def test_instrumentation_dependencies() -> None:
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = AgnoInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies is not None
    assert len(dependencies) > 0
    assert "agno >= 2.0.0" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Test that instrument() and uninstrument() can be called multiple times."""
    instrumentor = AgnoInstrumentor()

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
    instrumentor = AgnoInstrumentor()

    # First call
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second call (should be idempotent or handle gracefully)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Clean up
    instrumentor.uninstrument()


def test_uninstrument_without_instrument() -> None:
    """Test that uninstrument() can be called without prior instrument()."""
    instrumentor = AgnoInstrumentor()

    # This should not raise an error
    instrumentor.uninstrument()


def test_instrument_with_no_providers() -> None:
    """Test that instrument() works without explicit providers."""
    instrumentor = AgnoInstrumentor()

    # Should use global providers
    instrumentor.instrument()

    # Clean up
    instrumentor.uninstrument()


def test_instrumentor_has_required_attributes() -> None:
    """Test that the instrumentor has the required methods."""
    instrumentor = AgnoInstrumentor()

    assert hasattr(instrumentor, "instrument")
    assert hasattr(instrumentor, "uninstrument")
    assert hasattr(instrumentor, "instrumentation_dependencies")
    assert callable(instrumentor.instrument)
    assert callable(instrumentor.uninstrument)
    assert callable(instrumentor.instrumentation_dependencies)
