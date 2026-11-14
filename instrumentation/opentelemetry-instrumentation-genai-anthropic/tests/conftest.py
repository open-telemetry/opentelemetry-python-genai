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

# anthropic >= 2 has built-in tracing on the global tracer provider; keep it off so
# tests only see spans from this instrumentation.
os.environ["ANTHROPIC_OPEN_TELEMETRY"] = "false"


_REAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAARklEQVR42u3X"
    "QQ0AIAwAsSnZG4lInJxJwMRICGlyAvq9yF1PFUBAQEBAQBdAXWskICAgICAg"
    "ICAgICAgIOcKBAQEBPQd6ACUHHNEU5qggAAAAABJRU5ErkJggg=="
)
_REAL_PDF_B64 = (
    "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2Jq"
    "CjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2Jq"
    "CjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCAyMDAg"
    "MjAwXSAvQ29udGVudHMgNCAwIFIgL1Jlc291cmNlcyA8PCAvRm9udCA8PCAvRjEgNSAwIFIgPj4g"
    "Pj4gPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCA0NCA+PgpzdHJlYW0KQlQKL0YxIDEyIFRm"
    "CjcyIDEyMCBUZAooSGVsbG8gV29ybGQpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKNSAwIG9iago8"
    "PCAvVHlwZSAvRm9udCAvU3VidHlwZSAvVHlwZTEgL0Jhc2VGb250IC9IZWx2ZXRpY2EgPj4KZW5k"
    "b2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAwOSAwMDAwMCBuIAowMDAw"
    "MDAwMDU4IDAwMDAwIG4gCjAwMDAwMDAxMTUgMDAwMDAgbiAKMDAwMDAwMDI0MSAwMDAwMCBuIAow"
    "MDAwMDAwMzMzIDAwMDAwIG4gCnRyYWlsZXIKPDwgL1NpemUgNiAvUm9vdCAxIDAgUiA+PgpzdGFy"
    "dHhyZWYKNDAzCiUlRU9GCg=="
)
_IMAGE_URL = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
_NESTED_IMAGE_URL = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
_PDF_URL = (
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
)


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
                    "data": _REAL_PNG_B64,
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": _IMAGE_URL,
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": _REAL_PDF_B64,
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": _PDF_URL,
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
                "source": {
                    "type": "content",
                    "content": [
                        {"type": "text", "text": "Nested text"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": _NESTED_IMAGE_URL,
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
        "content": _REAL_PNG_B64,
    }
    assert parts[2] == {
        "type": "uri",
        "mime_type": None,
        "modality": "image",
        "uri": _IMAGE_URL,
    }
    assert parts[3] == {
        "type": "blob",
        "mime_type": "application/pdf",
        "modality": "document",
        "content": _REAL_PDF_B64,
    }
    assert parts[4] == {
        "type": "uri",
        "mime_type": None,
        "modality": "document",
        "uri": _PDF_URL,
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
        "uri": _NESTED_IMAGE_URL,
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
        # Older SDKs send `?beta=true` on beta endpoints and newer ones don't.
        "filter_query_parameters": ["beta"],
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
    ) as instrumentor:
        yield instrumentor
