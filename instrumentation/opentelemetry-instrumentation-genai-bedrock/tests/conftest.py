# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for Amazon Bedrock instrumentation."""

from collections.abc import Iterator

import pytest

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


@pytest.fixture
def instrument_bedrock(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[BedrockInstrumentor]:
    """Instrument Amazon Bedrock with the shared test providers."""
    with instrument(
        BedrockInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor
