# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BedrockInstrumentor class."""

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor


def test_instrumentor_instantiation():
    """Test that the instrumentor can be instantiated."""
    instrumentor = BedrockInstrumentor()
    assert instrumentor is not None
    assert isinstance(instrumentor, BedrockInstrumentor)


def test_instrumentation_dependencies():
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = BedrockInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies is not None
    assert len(dependencies) > 0
    assert "botocore >= 1.35.0" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider, logger_provider, meter_provider
):
    """Test that instrument() and uninstrument() can be called."""
    instrumentor = BedrockInstrumentor()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    instrumentor.uninstrument()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    instrumentor.uninstrument()
