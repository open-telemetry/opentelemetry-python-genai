# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Agno instrumentation tests."""

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(scope="module")
def vcr_config():
    """Scrub account-identifying headers for VCR cassettes."""
    from opentelemetry.test_util_genai.vcr import (
        scrub_response_headers_overwrite,
    )

    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
            ("x-goog-api-key", "test_google_genai_api_key"),
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
def instrument_agno(tracer_provider, logger_provider, meter_provider):
    """Fixture to instrument Agno with test providers."""
    with instrument(
        AgnoInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_agno_content_capture(
    tracer_provider, logger_provider, meter_provider
):
    """Fixture to instrument Agno with SPAN_ONLY content capture enabled."""
    with instrument(
        AgnoInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def uninstrument_agno():
    """Fixture to ensure Agno is uninstrumented after test."""
    yield
    AgnoInstrumentor().uninstrument()
