# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for CrewAI instrumentation."""

from collections.abc import Iterator

import pytest

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import (
    scrub_response_headers_overwrite,
)

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    """Configure VCR to remove account-identifying OpenAI values."""
    return {
        "filter_headers": [
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "openai-organization": "test_openai_org_id",
                "openai-project": "test_openai_project_id",
                "Set-Cookie": "test_set_cookie",
            }
        ),
    }


@pytest.fixture
def instrument_crewai(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[CrewAIInstrumentor]:
    """Instrument CrewAI with the shared test providers."""
    with instrument(
        CrewAIInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor
