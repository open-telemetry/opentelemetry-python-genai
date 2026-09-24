# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Anthropic instrumentation tests."""
# pylint: disable=redefined-outer-name

import json
import os

import pytest
import pytest_asyncio
from anthropic import Anthropic, AsyncAnthropic

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import scrub_response_headers

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


def multimodal_input_message():
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe these images."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "QUJD",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "QUJD",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "https://example.com/document.pdf",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "Document text",
                },
            },
            {
                "type": "document",
                "title": "Reference",
                "context": "Use the nested content.",
                "citations": {"enabled": True},
                "source": {
                    "type": "content",
                    "content": [
                        {"type": "text", "text": "Nested text"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/nested.png",
                            },
                        },
                    ],
                },
            },
        ],
    }


def assert_multimodal_input(span: ReadableSpan) -> None:
    value = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    assert value is not None
    assert isinstance(value, str)
    messages = json.loads(value)
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    parts = messages[0]["parts"]
    assert len(parts) == 8
    assert parts[0] == {
        "type": "text",
        "content": "Describe these images.",
    }
    assert parts[1] == {
        "type": "blob",
        "mime_type": "image/png",
        "modality": "image",
        "content": "QUJD",
    }
    assert parts[2] == {
        "type": "uri",
        "mime_type": None,
        "modality": "image",
        "uri": "https://example.com/image.png",
    }
    assert parts[3] == {
        "type": "blob",
        "mime_type": "application/pdf",
        "modality": "document",
        "content": "QUJD",
    }
    assert parts[4] == {
        "type": "uri",
        "mime_type": None,
        "modality": "document",
        "uri": "https://example.com/document.pdf",
    }
    assert parts[5] == {
        "type": "blob",
        "mime_type": "text/plain",
        "modality": "document",
        "content": "RG9jdW1lbnQgdGV4dA==",
    }
    assert parts[6] == {
        "type": "text",
        "content": "Nested text",
    }
    assert parts[7] == {
        "type": "uri",
        "mime_type": None,
        "modality": "image",
        "uri": "https://example.com/nested.png",
    }


@pytest.fixture(autouse=True)
def environment():
    """Set up environment variables for testing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_api_key"


@pytest.fixture
def anthropic_client():
    """Create and return an Anthropic client."""
    return Anthropic()


@pytest_asyncio.fixture
async def async_anthropic_client():
    """Create and return an async Anthropic client."""
    client = AsyncAnthropic()
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording/replaying HTTP interactions."""
    return {
        "filter_headers": [
            ("x-api-key", "test_anthropic_api_key"),
            ("authorization", "Bearer test_anthropic_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers(
            [
                "anthropic-organization-id",
                "anthropic-workspace-id",
                "set-cookie",
            ]
        ),
    }


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Anthropic without content capture."""
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Anthropic with ``SPAN_ONLY`` content capture (experimental semconv)."""
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_event_only(tracer_provider, logger_provider, meter_provider):
    """Instrument Anthropic with ``EVENT_ONLY`` content capture (experimental semconv)."""
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="EVENT_ONLY",
        emit_event=True,
    ) as instrumentor:
        yield instrumentor
