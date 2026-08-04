# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Qwen-Agent instrumentation tests."""
# pylint: disable=redefined-outer-name

import os

import pytest
from vcr.stubs import VCRHTTPResponse

# Set DASHSCOPE_API_KEY before any dashscope/qwen-agent imports:
# the dashscope SDK reads the environment variable at module import time.
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "test_dashscope_api_key"

from opentelemetry.instrumentation.genai.qwen_agent import (  # noqa: E402
    QwenAgentInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument  # noqa: E402
from opentelemetry.test_util_genai.vcr import (  # noqa: E402
    scrub_response_headers_overwrite,
)

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


# Newer urllib3 requires a ``version_string`` attribute on HTTP responses,
# but VCR.py's VCRHTTPResponse stub does not set it, causing AttributeError
# when the dashscope SDK replays SSE streaming responses.
if not hasattr(VCRHTTPResponse, "version_string"):
    VCRHTTPResponse.version_string = "HTTP/1.1"


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording/replaying HTTP interactions."""
    return {
        "filter_headers": [
            ("authorization", "Bearer test_dashscope_api_key"),
            ("x-dashscope-api-key", "test_dashscope_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "x-dashscope-request-id": "test_request_id",
                "Set-Cookie": "test_set_cookie",
            }
        ),
    }


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Qwen-Agent without content capture."""
    with instrument(
        QwenAgentInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Qwen-Agent with ``SPAN_ONLY`` content capture."""
    with instrument(
        QwenAgentInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
