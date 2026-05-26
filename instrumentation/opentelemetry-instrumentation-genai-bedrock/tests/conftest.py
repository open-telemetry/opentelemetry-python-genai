# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for AWS Bedrock instrumentation tests."""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import os

import pytest

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import scrub_response_headers

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(autouse=True)
def environment():
    """Set up environment variables for testing."""
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key_id"
    if not os.getenv("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_access_key"
    if not os.getenv("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording/replaying HTTP interactions."""
    return {
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("x-amz-security-token", "REDACTED"),
            ("x-amz-date", "REDACTED"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers(
            ["x-amzn-requestid", "set-cookie"]
        ),
    }


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Bedrock without content capture (stable semconv mode)."""
    with instrument(
        BedrockInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        semconv="stable",
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Bedrock with ``SPAN_ONLY`` content capture (experimental semconv)."""
    with instrument(
        BedrockInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        semconv="gen_ai_latest_experimental",
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
