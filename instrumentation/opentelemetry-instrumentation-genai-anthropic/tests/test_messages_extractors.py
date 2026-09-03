# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Anthropic message parameter extraction."""

from io import BytesIO
from pathlib import Path

import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from anthropic.types import (
    ServerToolUseBlock,
    WebSearchToolResultBlock,
    WebSearchToolResultError,
)

from opentelemetry.instrumentation.genai.anthropic.messages_extractors import (
    extract_params,
    get_tool_definitions,
    get_input_messages,
)
from opentelemetry.instrumentation.genai.anthropic.utils import (
    _convert_dict_block_to_part,
    convert_content_to_parts,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    GenericPart,
    TextPart,
    UriPart,
    set_invocation_response_attributes,
)
from opentelemetry.instrumentation.genai.anthropic.utils import (
    _convert_content_block_to_part,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    ServerToolCallPart,
    ServerToolCallResponsePart,
)


def test_extract_params_reads_sampling_params_from_extra_body():
    params = extract_params(
        extra_body={"temperature": 0.7, "top_p": 0.9, "top_k": 40}
    )

    assert params.temperature == 0.7
    assert params.top_p == 0.9
    assert params.top_k == 40


def test_extract_params_prefers_named_sampling_params():
    params = extract_params(
        temperature=0.2,
        top_p=0.3,
        top_k=10,
        extra_body={"temperature": 0.7, "top_p": 0.9, "top_k": 40},
    )

    assert params.temperature == 0.2
    assert params.top_p == 0.3
    assert params.top_k == 10


def test_extract_params_ignores_non_numeric_extra_body_sampling_params():
    params = extract_params(
        extra_body={"temperature": "high", "top_p": True, "top_k": 2.5}
    )

    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None


def test_extract_params_ignores_non_mapping_extra_body():
    params = extract_params(extra_body="temperature=0.7")

    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None


def test_base64_image_source_converts_to_blob_part():
    part = _convert_dict_block_to_part(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "QUJD",
            },
        }
    )

    assert isinstance(part, BlobPart)
    assert part.content == b"ABC"
    assert part.mime_type == "image/png"
    assert part.modality == "image"


def test_url_image_source_converts_to_uri_part():
    part = _convert_dict_block_to_part(
        {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.png",
            },
        }
    )

    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/image.png"
    assert part.mime_type is None
    assert part.modality == "image"


def test_file_image_source_converts_to_file_part():
    part = _convert_dict_block_to_part(
        {
            "type": "image",
            "source": {"type": "file", "file_id": "file-image"},
        }
    )

    assert isinstance(part, FilePart)
    assert part.file_id == "file-image"
    assert part.mime_type is None
    assert part.modality == "image"


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "image/png", "data": "%%%"},
        {"type": "base64", "media_type": "image/png"},
        {"type": "url"},
        {"type": "url", "url": ""},
        {"type": "file"},
        {"type": "file", "file_id": ""},
        {"type": "unknown", "data": "QUJD"},
        None,
    ],
)
def test_invalid_image_source_is_ignored(source):
    assert (
        _convert_dict_block_to_part({"type": "image", "source": source})
        is None
    )


def test_mixed_text_and_image_parts_preserve_order():
    parts = convert_content_to_parts(
        [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png",
                },
            },
        ]
    )

    assert len(parts) == 2
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "Describe this image."
    assert isinstance(parts[1], UriPart)
    assert parts[1].uri == "https://example.com/image.png"


def test_image_only_input_message_is_preserved():
    messages = get_input_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "QUJD",
                        },
                    }
                ],
            }
        ]
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert len(messages[0].parts) == 1
    assert isinstance(messages[0].parts[0], BlobPart)


def test_base64_document_source_converts_to_blob_part():
    part = _convert_dict_block_to_part(
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "QUJD",
            },
        }
    )

    assert isinstance(part, BlobPart)
    assert part.content == b"ABC"
    assert part.mime_type == "application/pdf"
    assert part.modality == "document"


def test_url_document_source_converts_to_uri_part():
    part = _convert_dict_block_to_part(
        {
            "type": "document",
            "source": {
                "type": "url",
                "url": "https://example.com/document.pdf",
            },
        }
    )

    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/document.pdf"
    assert part.mime_type == "application/pdf"
    assert part.modality == "document"


def test_file_document_source_converts_to_file_part():
    part = _convert_dict_block_to_part(
        {
            "type": "document",
            "source": {"type": "file", "file_id": "file-document"},
        }
    )

    assert isinstance(part, FilePart)
    assert part.file_id == "file-document"
    assert part.mime_type is None
    assert part.modality == "document"


def test_plain_text_document_source_converts_to_blob_part():
    part = _convert_dict_block_to_part(
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": "Document text",
            },
        }
    )

    assert isinstance(part, BlobPart)
    assert part.content == b"Document text"
    assert part.mime_type == "text/plain"
    assert part.modality == "document"


def test_nested_content_document_source_preserves_part_order():
    parts = convert_content_to_parts(
        [
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
                                "url": "https://example.com/image.png",
                            },
                        },
                    ],
                },
            }
        ]
    )

    assert len(parts) == 1
    assert isinstance(parts[0], GenericPart)
    assert parts[0].type == "document"
    assert parts[0].value == {
        "parts": [
            {"content": "Nested text", "type": "text"},
            {
                "mime_type": None,
                "modality": "image",
                "uri": "https://example.com/image.png",
                "type": "uri",
            },
        ],
        "title": "Reference",
        "context": "Use the nested content.",
        "citations": {"enabled": True},
    }


@pytest.mark.parametrize(
    ("block_type", "media_type", "data", "input_type"),
    [
        ("image", "image/png", Path("private/image.png"), "path"),
        ("image", "image/png", BytesIO(b"image"), "stream"),
        (
            "document",
            "application/pdf",
            Path("private/document.pdf"),
            "path",
        ),
        ("document", "application/pdf", BytesIO(b"document"), "stream"),
    ],
)
def test_file_backed_base64_source_is_preserved_without_reading(
    block_type, media_type, data, input_type
):
    initial_position = data.tell() if isinstance(data, BytesIO) else None
    part = _convert_dict_block_to_part(
        {
            "type": block_type,
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }
    )

    assert isinstance(part, GenericPart)
    assert part.type == block_type
    assert part.value == {
        "source_type": "base64_file",
        "mime_type": media_type,
        "input_type": input_type,
    }
    if initial_position is not None:
        assert data.tell() == initial_position


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "application/pdf", "data": "%%%"},
        {"type": "url"},
        {"type": "text"},
        {"type": "content", "content": None},
        {"type": "file"},
        {"type": "file", "file_id": ""},
        {"type": "unknown"},
        None,
    ],
)
def test_invalid_document_source_is_ignored(source):
    assert (
        convert_content_to_parts([{"type": "document", "source": source}])
        == []
    )


def test_set_invocation_response_attributes_records_cache_tokens():
    invocation = MagicMock()
    message = SimpleNamespace(
        id="msg_123",
        model="claude-3-7-sonnet-20250219",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=15,
            cache_read_input_tokens=5,
        ),
    )
    set_invocation_response_attributes(
        invocation, message, capture_content=False
    )
    assert invocation.input_tokens == 30
    assert invocation.output_tokens == 20
    assert invocation.cache_write_input_tokens == 15
    assert invocation.cache_read_input_tokens == 5


def test_convert_server_tool_use_block():
    part = _convert_content_block_to_part(
        ServerToolUseBlock(
            id="srvtoolu_123",
            input={"query": "OpenTelemetry"},
            name="web_search",
            type="server_tool_use",
        )
    )

    assert isinstance(part, ServerToolCallPart)
    assert part.id == "srvtoolu_123"
    assert part.name == "web_search"
    assert part.server_tool_call == {
        "input": {"query": "OpenTelemetry"},
        "type": "web_search",
    }


def test_convert_server_tool_result_block():
    part = _convert_content_block_to_part(
        WebSearchToolResultBlock(
            content=WebSearchToolResultError(
                error_code="unavailable",
                type="web_search_tool_result_error",
            ),
            tool_use_id="srvtoolu_123",
            type="web_search_tool_result",
        )
    )

    assert isinstance(part, ServerToolCallResponsePart)
    assert part.id == "srvtoolu_123"
    assert part.server_tool_call_response == {
        "content": {
            "error_code": "unavailable",
            "type": "web_search_tool_result_error",
        },
        "type": "web_search",
    }


def test_convert_server_tool_dicts():
    call = _convert_content_block_to_part(
        {
            "type": "server_tool_use",
            "id": "srvtoolu_123",
            "name": "web_fetch",
            "input": {"url": "https://opentelemetry.io"},
        }
    )
    response = _convert_content_block_to_part(
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": "srvtoolu_123",
            "content": {
                "type": "web_fetch_result",
                "url": "https://opentelemetry.io",
            },
        }
    )

    assert isinstance(call, ServerToolCallPart)
    assert call.server_tool_call == {
        "input": {"url": "https://opentelemetry.io"},
        "type": "web_fetch",
    }
    assert isinstance(response, ServerToolCallResponsePart)
    assert response.server_tool_call_response == {
        "content": {
            "type": "web_fetch_result",
            "url": "https://opentelemetry.io",
        },
        "type": "web_fetch",
    }


def test_extract_params_keeps_tools():
    tools = [{"name": "get_weather", "input_schema": {"type": "object"}}]

    params = extract_params(tools=tools)

    assert params.tools is tools


def test_get_tool_definitions_maps_custom_tool_to_function():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    definitions = get_tool_definitions(
        [
            {
                "name": "get_weather",
                "description": "Get weather by city",
                "input_schema": schema,
            }
        ]
    )

    assert definitions == [
        FunctionToolDefinition(
            name="get_weather",
            description="Get weather by city",
            parameters=schema,
        )
    ]
    assert definitions[0].type == "function"


def test_get_tool_definitions_maps_explicit_custom_type_to_function():
    definitions = get_tool_definitions([{"type": "custom", "name": "noop"}])

    assert definitions == [
        FunctionToolDefinition(name="noop", description=None, parameters=None)
    ]


def test_get_tool_definitions_maps_server_tool_to_generic():
    definitions = get_tool_definitions(
        [
            {
                "name": "web_search",
                "type": "web_search_20250305",
                "max_uses": 3,
            }
        ]
    )

    assert definitions == [
        GenericToolDefinition(name="web_search", type="web_search_20250305")
    ]


def test_get_tool_definitions_falls_back_to_type_for_unnamed_toolset():
    definitions = get_tool_definitions(
        [{"type": "computer_toolset_20260801", "configs": []}]
    )

    assert definitions == [
        GenericToolDefinition(
            name="computer_toolset_20260801",
            type="computer_toolset_20260801",
        )
    ]


def test_get_tool_definitions_reads_tool_objects():
    class _Tool:
        name = "get_weather"
        description = "Get weather by city"
        input_schema = {"type": "object"}

    definitions = get_tool_definitions([_Tool()])

    assert definitions == [
        FunctionToolDefinition(
            name="get_weather",
            description="Get weather by city",
            parameters={"type": "object"},
        )
    ]


def test_get_tool_definitions_mixes_custom_and_server_tools():
    definitions = get_tool_definitions(
        [
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"name": "bash", "type": "bash_20250124"},
        ]
    )

    assert definitions == [
        FunctionToolDefinition(
            name="get_weather",
            description=None,
            parameters={"type": "object"},
        ),
        GenericToolDefinition(name="bash", type="bash_20250124"),
    ]


@pytest.mark.parametrize("tools", [None, []])
def test_get_tool_definitions_without_tools(tools):
    assert get_tool_definitions(tools) is None
