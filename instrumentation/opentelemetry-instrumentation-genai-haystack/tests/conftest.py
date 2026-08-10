# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Haystack instrumentation tests."""
# pylint: disable=redefined-outer-name

import os

import pytest

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import scrub_response_headers

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(autouse=True)
def environment():
    """Set up environment variables for testing."""
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
    # Haystack pings deepset's telemetry endpoint on first Pipeline.run() /
    # component import unless disabled; not something this instrumentation
    # should be recording or waiting on network for in tests.
    os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording/replaying HTTP interactions."""
    return {
        "filter_headers": [
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers(
            ["openai-organization", "set-cookie"]
        ),
    }


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Haystack without content capture."""
    with instrument(
        HaystackInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Haystack with ``SPAN_ONLY`` content capture."""
    with instrument(
        HaystackInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_event_only(tracer_provider, logger_provider, meter_provider):
    """Instrument Haystack with ``EVENT_ONLY`` content capture."""
    with instrument(
        HaystackInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="EVENT_ONLY",
        emit_event=True,
    ) as instrumentor:
        yield instrumentor
