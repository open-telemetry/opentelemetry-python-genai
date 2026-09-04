# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared test utilities for OpenAI instrumentation tests."""

import base64
import json
from types import SimpleNamespace
from typing import Any

import pytest

from opentelemetry.instrumentation.genai.openai.utils import (
    _content_to_parts,
    _prepare_input_messages,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.types import (
    BlobPart,
    InputMessage,
    TextPart,
    UriPart,
)

_REAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAARklEQVR42u3X"
    "QQ0AIAwAsSnZG4lInJxJwMRICGlyAvq9yF1PFUBAQEBAQBdAXWskICAgICAg"
    "ICAgICAgIOcKBAQEBPQd6ACUHHNEU5qggAAAAABJRU5ErkJggg=="
)
_REAL_PNG_BYTES = base64.b64decode(_REAL_PNG_B64)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
FETCH_RESPONSE_OPERATION_NAME = "fetch_response"
# TODO: use the semconv constants once these attributes are released in
# opentelemetry-semantic-conventions. Added to the GenAI semantic conventions
# in https://github.com/open-telemetry/semantic-conventions-genai/pull/353.
GEN_AI_REQUEST_STREAM_CURSOR = "gen_ai.request.stream_cursor"
GEN_AI_RESPONSE_STATUS = "gen_ai.response.status"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
    "gen_ai.usage.cache_creation.input_tokens"
)
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
USER_ONLY_PROMPT = [{"role": "user", "content": "Say this is a test"}]
USER_ONLY_EXPECTED_INPUT_MESSAGES = [
    {
        "role": "user",
        "parts": [
            {
                "type": "text",
                "content": USER_ONLY_PROMPT[0]["content"],
            }
        ],
    }
]
MULTIMODAL_PROMPT = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_REAL_PNG_B64}"},
            },
        ],
    }
]
MULTIMODAL_EXPECTED_INPUT_MESSAGES = [
    {
        "role": "user",
        "parts": [
            {"type": "text", "content": "Describe the image"},
            {
                "mime_type": "image/png",
                "modality": "image",
                "content": _REAL_PNG_B64,
                "type": "blob",
            },
        ],
    }
]
WEATHER_TOOL_PROMPT = [
    {"role": "system", "content": "You're a helpful assistant."},
    {
        "role": "user",
        "content": "What's the weather in Seattle and San Francisco today?",
    },
]
WEATHER_TOOL_EXPECTED_INPUT_MESSAGES = [
    {
        "role": "system",
        "parts": [
            {
                "type": "text",
                "content": WEATHER_TOOL_PROMPT[0]["content"],
            }
        ],
    },
    {
        "role": "user",
        "parts": [
            {
                "type": "text",
                "content": WEATHER_TOOL_PROMPT[1]["content"],
            }
        ],
    },
]


def test_chat_content_parts_capture_text_and_image_url():
    parts = _content_to_parts(
        [
            {"type": "text", "text": "Describe the image"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ]
    )

    assert parts == [
        TextPart(content="Describe the image"),
        UriPart(
            mime_type=None,
            modality="image",
            uri="https://example.com/image.png",
        ),
    ]


def test_chat_content_parts_capture_image_url_object():
    parts = _content_to_parts(
        [
            SimpleNamespace(
                type="image_url",
                image_url=SimpleNamespace(url="https://example.com/image.png"),
            )
        ]
    )

    assert parts == [
        UriPart(
            mime_type=None,
            modality="image",
            uri="https://example.com/image.png",
        )
    ]


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Plain text", [TextPart(content="Plain text")]),
        (
            ["First", "Second"],
            [TextPart(content="First"), TextPart(content="Second")],
        ),
        (
            [{"type": "input_text", "text": "Responses text"}],
            [TextPart(content="Responses text")],
        ),
        (None, []),
        (42, []),
        ({"type": "text", "text": "Not a content sequence"}, []),
    ],
)
def test_content_parts_support_text_forms_and_reject_invalid_content(
    content, expected
):
    assert _content_to_parts(content) == expected


def test_chat_content_parts_capture_data_url_as_blob():
    parts = _content_to_parts(
        [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_REAL_PNG_B64}"},
            }
        ]
    )

    assert parts == [
        BlobPart(
            mime_type="image/png",
            modality="image",
            content=_REAL_PNG_BYTES,
        )
    ]


def test_content_parts_preserve_order_around_image():
    parts = _content_to_parts(
        [
            {"type": "text", "text": "Before"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {"type": "text", "text": "After"},
        ]
    )

    assert parts == [
        TextPart(content="Before"),
        UriPart(
            mime_type=None,
            modality="image",
            uri="https://example.com/image.png",
        ),
        TextPart(content="After"),
    ]


@pytest.mark.parametrize(
    "image_part",
    [
        {"type": "image_url", "image_url": {}},
        {"type": "image_url", "image_url": {"url": 42}},
        {"type": "image_url", "file_id": "file-not-valid-for-chat"},
        {"type": "input_image"},
        {"type": "unsupported_image", "image_url": "https://example.com"},
    ],
)
def test_content_parts_ignore_invalid_or_unsupported_images(image_part):
    assert _content_to_parts([image_part]) == []


def test_chat_content_parts_drop_malformed_image_and_keep_text():
    parts = _content_to_parts(
        [
            {"type": "text", "text": "Keep this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,@@@@"},
            },
        ]
    )

    assert parts == [TextPart(content="Keep this")]


def test_prepare_input_messages_drops_messages_without_parts():
    messages = _prepare_input_messages(
        [
            {"role": "user", "content": []},
            {
                "role": "user",
                "content": [{"type": "unsupported", "value": "ignored"}],
            },
            {"role": "user", "content": "Keep this"},
        ]
    )

    assert messages == [
        InputMessage(
            role="user",
            parts=[TextPart(content="Keep this")],
        )
    ]


def _assert_optional_attribute(span, attribute_name, expected_value):
    """Helper to assert optional span attributes."""
    if expected_value is not None:
        assert expected_value == span.attributes[attribute_name]
    else:
        assert attribute_name not in span.attributes


def assert_all_attributes(
    span: ReadableSpan,
    request_model: str,
    latest_experimental_enabled: bool,
    response_id: str = None,
    response_model: str = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    operation_name: str = "chat",
    server_address: str = "api.openai.com",
    server_port: int = 443,
    request_service_tier: str | None = None,
    response_service_tier: str | None = None,
):
    assert span.name == f"{operation_name} {request_model}"
    assert (
        operation_name
        == span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
    )

    provider_name_attr_name = (
        "gen_ai.provider.name"
        if latest_experimental_enabled
        else GenAIAttributes.GEN_AI_SYSTEM
    )

    assert (
        GenAIAttributes.GenAiProviderNameValues.OPENAI.value
        == span.attributes[provider_name_attr_name]
    )
    assert (
        request_model == span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
    )

    _assert_optional_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model
    )
    _assert_optional_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_ID, response_id
    )
    _assert_optional_attribute(
        span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens
    )
    _assert_optional_attribute(
        span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens
    )

    assert server_address == span.attributes[ServerAttributes.SERVER_ADDRESS]
    if server_port != 443 and server_port > 0:
        assert server_port == span.attributes[ServerAttributes.SERVER_PORT]

    request_service_tier_attr_name = (
        OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER
        if latest_experimental_enabled
        else GenAIAttributes.GEN_AI_OPENAI_REQUEST_SERVICE_TIER
    )
    _assert_optional_attribute(
        span,
        request_service_tier_attr_name,
        request_service_tier,
    )

    response_service_tier_attr_name = (
        OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER
        if latest_experimental_enabled
        else GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER
    )
    _assert_optional_attribute(
        span,
        response_service_tier_attr_name,
        response_service_tier,
    )


def assert_fetch_response_attributes(
    span: ReadableSpan,
    *,
    response_id: str,
    response_model: str | None = None,
    response_status: str | None = None,
    finish_reasons: tuple | None = None,
    request_stream: bool | None = None,
    stream_cursor: str | None = None,
    response_service_tier: str | None = None,
    server_address: str = "api.openai.com",
):
    """Assert a ``gen_ai.fetch_response.client`` span matches the semconv.

    Fetching a stored response performs no inference, so the span must carry
    neither request-side attributes nor any token usage from the fetched
    response — those counts belong to the original generation.
    """
    # The response id is high cardinality, so it stays out of the span name.
    assert span.name == FETCH_RESPONSE_OPERATION_NAME
    assert span.kind is SpanKind.CLIENT
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == FETCH_RESPONSE_OPERATION_NAME
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response_id
    assert (
        span.attributes[OpenAIAttributes.OPENAI_API_TYPE]
        == OpenAIAttributes.OpenaiApiTypeValues.RESPONSES.value
    )
    assert span.attributes[ServerAttributes.SERVER_ADDRESS] == server_address

    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in span.attributes
    assert GenAIAttributes.GEN_AI_REQUEST_MODEL not in span.attributes
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes

    _assert_optional_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model
    )
    _assert_optional_attribute(span, GEN_AI_RESPONSE_STATUS, response_status)
    _assert_optional_attribute(
        span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
    )
    _assert_optional_attribute(
        span, GenAIAttributes.GEN_AI_REQUEST_STREAM, request_stream
    )
    _assert_optional_attribute(
        span, GEN_AI_REQUEST_STREAM_CURSOR, stream_cursor
    )
    _assert_optional_attribute(
        span,
        OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER,
        response_service_tier,
    )


def assert_log_parent(log, span):
    """Assert that the log record has the correct parent span context"""
    if span:
        assert log.log_record.trace_id == span.get_span_context().trace_id
        assert log.log_record.span_id == span.get_span_context().span_id
        assert (
            log.log_record.trace_flags == span.get_span_context().trace_flags
        )


def get_current_weather_tool_definition():
    return {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Boston, MA",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }


def get_responses_weather_tool_definition():
    return {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. Boston, MA",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    }


EXPECTED_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. Boston, MA",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    }
]


def remove_none_values(body):
    """Remove None values from a dictionary recursively"""
    result = {}
    for key, value in body.items():
        if value is None:
            continue
        if isinstance(value, dict):
            result[key] = remove_none_values(value)
        elif isinstance(value, list):
            result[key] = [
                remove_none_values(i) if isinstance(i, dict) else i
                for i in value
            ]
        else:
            result[key] = value
    return result


def assert_completion_attributes(
    span: ReadableSpan,
    request_model: str,
    response: Any,
    latest_experimental_enabled: bool,
    operation_name: str = "chat",
    server_address: str = "api.openai.com",
):
    return assert_all_attributes(
        span,
        request_model,
        latest_experimental_enabled,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        operation_name,
        server_address,
    )


def assert_messages_attribute(actual, expected):
    assert json.loads(actual) == expected


def format_simple_expected_output_message(
    content: str, finish_reason: str = "stop"
):
    return [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": content,
                }
            ],
            "finish_reason": finish_reason,
        }
    ]


def _get_usage_details(usage):
    return getattr(usage, "input_tokens_details", None) or getattr(
        usage, "prompt_tokens_details", None
    )


def assert_cache_attributes(span, usage):
    details = _get_usage_details(usage)
    assert details is not None

    cached_tokens = getattr(details, "cached_tokens", None)
    if cached_tokens is None:
        assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in span.attributes
    else:
        assert (
            span.attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
            == cached_tokens
        )

    cache_creation = getattr(details, "cache_creation_input_tokens", None)
    if cache_creation is None:
        assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS not in span.attributes
    else:
        assert (
            span.attributes[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS]
            == cache_creation
        )


def assert_message_in_logs(log, event_name, expected_content, parent_span):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes[GenAIAttributes.GEN_AI_SYSTEM]
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == remove_none_values(
            expected_content
        )
    assert_log_parent(log, parent_span)


def assert_embedding_attributes(
    span: ReadableSpan,
    request_model: str,
    latest_experimental_enabled: bool,
    response,
):
    """Assert that the span contains all required attributes for embeddings operation"""
    # Use the common assertion function
    assert_all_attributes(
        span,
        request_model,
        latest_experimental_enabled,
        response_id=None,  # Embeddings don't have a response ID
        response_model=response.model,
        input_tokens=response.usage.prompt_tokens,
        operation_name="embeddings",
        server_address="api.openai.com",
    )

    # Assert embeddings-specific attributes
    if (
        hasattr(span, "attributes")
        and "gen_ai.embeddings.dimension.count" in span.attributes
    ):
        # If dimensions were specified, verify that they match the actual dimensions
        assert span.attributes["gen_ai.embeddings.dimension.count"] == len(
            response.data[0].embedding
        )
