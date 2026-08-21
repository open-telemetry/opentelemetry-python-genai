# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Portkey AI instrumentation tests."""

from __future__ import annotations

import os

import pytest

from opentelemetry.instrumentation.genai.portkey import PortkeyInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(autouse=True)
def environment():
    if not os.getenv("PORTKEY_API_KEY"):
        os.environ["PORTKEY_API_KEY"] = "test_portkey_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    from opentelemetry.test_util_genai.vcr import (
        scrub_response_headers_overwrite,
    )

    return {
        "filter_headers": [
            ("x-portkey-api-key", "test_portkey_api_key"),
            ("authorization", "Bearer test_portkey_api_key"),
            ("cookie", "test_cookie"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "Set-Cookie": "test_set_cookie",
            }
        ),
    }


@pytest.fixture
def instrument_portkey(tracer_provider, logger_provider, meter_provider):
    """Fixture to instrument Portkey AI with test providers."""
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def uninstrument_portkey():
    """Fixture to ensure Portkey AI is uninstrumented after test."""
    yield
    PortkeyInstrumentor().uninstrument()
