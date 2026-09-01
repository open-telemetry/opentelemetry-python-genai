# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for Amazon Bedrock instrumentation."""

import os
from collections.abc import Iterator

import boto3
import pytest

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import scrub_response_headers

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(autouse=True)
def environment() -> None:
    """Set up environment variables for testing."""
    if not (
        os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")
    ):
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "test_aws_access_key_id")
        os.environ.setdefault(
            "AWS_SECRET_ACCESS_KEY", "test_aws_secret_access_key"
        )
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def bedrock_client():
    """Create and return a Bedrock runtime client."""
    session = boto3.session.Session()
    kwargs = {
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }
    endpoint_url = os.getenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME") or os.getenv(
        "AWS_ENDPOINT_URL"
    )
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return session.client("bedrock-runtime", **kwargs)


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording/replaying HTTP interactions."""
    return {
        "filter_headers": [
            ("authorization", "Bearer test_aws_authorization"),
            ("x-amz-security-token", "test_aws_token"),
            ("x-api-key", "test_anthropic_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers(
            [
                "set-cookie",
                "x-amzn-requestid",
                "anthropic-organization-id",
                "anthropic-workspace-id",
            ]
        ),
    }


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


@pytest.fixture
def instrument_no_content(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[BedrockInstrumentor]:
    """Instrument Bedrock without content capture."""
    with instrument(
        BedrockInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[BedrockInstrumentor]:
    """Instrument Bedrock with SPAN_ONLY content capture."""
    with instrument(
        BedrockInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
