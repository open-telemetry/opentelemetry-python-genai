# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Cohere instrumentation tests."""
# pylint: disable=redefined-outer-name

import pytest

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


@pytest.fixture
def instrument_cohere(tracer_provider, logger_provider, meter_provider):
    """Fixture to instrument Cohere with test providers."""
    # pylint: disable=import-outside-toplevel
    from opentelemetry.instrumentation.genai.cohere import (
        CohereInstrumentor,
    )

    instrumentor = CohereInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture
def uninstrument_cohere():
    """Fixture to ensure Cohere is uninstrumented after test."""
    yield
    # pylint: disable=import-outside-toplevel
    from opentelemetry.instrumentation.genai.cohere import (
        CohereInstrumentor,
    )

    CohereInstrumentor().uninstrument()
