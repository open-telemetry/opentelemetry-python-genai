# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Anthropic message parameter extraction."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from anthropic.types import (
    ServerToolUseBlock,
    WebSearchToolResultBlock,
    WebSearchToolResultError,
)

from opentelemetry.instrumentation.genai.anthropic.messages_extractors import (
    extract_params,
    set_invocation_response_attributes,
)
from opentelemetry.instrumentation.genai.anthropic.utils import (
    _convert_content_block_to_part,
)
from opentelemetry.util.genai.types import (
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
