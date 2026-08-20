# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests configuration module."""

import os

import pytest
from groq import AsyncGroq, Groq

from opentelemetry.instrumentation.genai.groq import GroqInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(autouse=True)
def environment():
    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "test_groq_api_key"


@pytest.fixture
def groq_client():
    return Groq()


@pytest.fixture
def async_groq_client():
    return AsyncGroq()


@pytest.fixture(scope="module")
def vcr_config():
    from opentelemetry.test_util_genai.vcr import (
        scrub_response_headers_overwrite,
    )

    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_groq_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "Set-Cookie": "test_set_cookie",
            }
        ),
    }


@pytest.fixture(
    scope="function",
    params=[(True, "span_only")],
    name="content_mode",
)
def fixture_content_mode(request):
    # returns tuple: (latest_experimental_enabled: bool, content_mode: str)
    # we don't test (True, "event_only"), (True, "span_and_event") because it's util's
    # responsibility.
    return request.param


@pytest.fixture(scope="function")
def instrument_no_content(
    tracer_provider,
    logger_provider,
    meter_provider,
    content_mode,
):
    with instrument(
        GroqInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture(scope="function")
def instrument_with_content(
    tracer_provider, logger_provider, meter_provider, content_mode
):
    _, content_mode_value = content_mode
    with instrument(
        GroqInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture=content_mode_value,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture(scope="function")
def instrument_with_content_unsampled(
    span_exporter, logger_provider, meter_provider, content_mode
):
    _, content_mode_value = content_mode
    tracer_provider = TracerProvider(sampler=ALWAYS_OFF)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    with instrument(
        GroqInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture=content_mode_value,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture(scope="function")
def instrument_event_only(tracer_provider, logger_provider, meter_provider):
    with instrument(
        GroqInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="event_only",
    ) as instrumentor:
        yield instrumentor
