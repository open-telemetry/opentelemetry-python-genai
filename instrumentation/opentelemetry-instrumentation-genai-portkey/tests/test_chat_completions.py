# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Portkey AI chat completion instrumentation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from portkey_ai import Portkey

try:
    from portkey_ai import AsyncPortkey
except ImportError:
    AsyncPortkey = None  # type: ignore[assignment,misc]

from opentelemetry.instrumentation.genai.portkey import PortkeyInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode

_has_async_portkey = AsyncPortkey is not None


def _create_mock_chat_completion(
    id_val: str = "chatcmpl-123",
    model: str = "gpt-4o",
    content: str = "Hello from Portkey!",
    role: str = "assistant",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list[dict[str, object]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id_val,
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role=role,
                    content=content,
                    tool_calls=[
                        SimpleNamespace(
                            id=tc.get("id"),
                            type=tc.get("type"),
                            function=SimpleNamespace(
                                name=tc.get("function", {}).get("name"),  # type: ignore[union-attr]
                                arguments=tc.get("function", {}).get(  # type: ignore[union-attr]
                                    "arguments"
                                ),
                            ),
                        )
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _setup_mock_chat(client: Portkey, mock_resp: SimpleNamespace) -> None:
    raw_dict = {
        "id": mock_resp.id,
        "model": mock_resp.model,
        "choices": [
            {
                "index": c.index,
                "message": {
                    "role": c.message.role,
                    "content": c.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in c.message.tool_calls
                    ]
                    if getattr(c.message, "tool_calls", None)
                    else None,
                },
                "finish_reason": c.finish_reason,
            }
            for c in mock_resp.choices
        ],
        "usage": {
            "prompt_tokens": mock_resp.usage.prompt_tokens,
            "completion_tokens": mock_resp.usage.completion_tokens,
            "total_tokens": mock_resp.usage.total_tokens,
        },
    }
    if hasattr(client.chat.completions, "openai_client"):
        client.chat.completions.openai_client = MagicMock()
        mock_raw = MagicMock()
        mock_raw.text = json.dumps(raw_dict)
        mock_raw.headers = {}
        client.chat.completions.openai_client.with_raw_response.chat.completions.create.return_value = mock_raw
    client.chat.completions._post = MagicMock(return_value=mock_resp)


def _setup_async_mock_chat(
    client: AsyncPortkey, mock_resp: SimpleNamespace
) -> None:
    raw_dict = {
        "id": mock_resp.id,
        "model": mock_resp.model,
        "choices": [
            {
                "index": c.index,
                "message": {
                    "role": c.message.role,
                    "content": c.message.content,
                },
                "finish_reason": c.finish_reason,
            }
            for c in mock_resp.choices
        ],
        "usage": {
            "prompt_tokens": mock_resp.usage.prompt_tokens,
            "completion_tokens": mock_resp.usage.completion_tokens,
            "total_tokens": mock_resp.usage.total_tokens,
        },
    }
    if hasattr(client.chat.completions, "openai_client"):
        client.chat.completions.openai_client = MagicMock()
        mock_raw = MagicMock()
        mock_raw.text = json.dumps(raw_dict)
        mock_raw.headers = {}
        client.chat.completions.openai_client.with_raw_response.chat.completions.create = AsyncMock(
            return_value=mock_raw
        )
    client.chat.completions._post = AsyncMock(return_value=mock_resp)


def test_sync_chat_completions_basic(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(
            api_key="test_pk",
            provider="openai",
            base_url="https://api.portkey.ai/v1",
        )
        mock_resp = _create_mock_chat_completion()
        _setup_mock_chat(p, mock_resp)

        res = p.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}],
            model="gpt-4o",
            temperature=0.7,
            top_p=0.9,
            max_tokens=150,
            seed=42,
        )

        assert res.id == "chatcmpl-123"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chat gpt-4o"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == GenAIAttributes.GenAiOperationNameValues.CHAT.value
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
            == "gpt-4o"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
            == "openai"
        )
        assert (
            span.attributes.get(server_attributes.SERVER_ADDRESS)
            == "api.portkey.ai"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE)
            == 0.7
        )
        assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.9
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS)
            == 150
        )
        assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_SEED) == 42
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "chatcmpl-123"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
            == "gpt-4o"
        )
        assert span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ) == ("stop",)
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
            == 10
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 5
        )


@pytest.mark.skipif(
    not _has_async_portkey,
    reason="AsyncPortkey not available in this version of portkey-ai",
)
@pytest.mark.asyncio
async def test_async_chat_completions_basic(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        ap = AsyncPortkey(
            api_key="test_pk",
            provider="anthropic",
            base_url="https://custom.portkey.ai:8443/v1",
        )
        mock_resp = _create_mock_chat_completion(
            id_val="chatcmpl-async-1",
            model="claude-3-5-sonnet-20241022",
            content="Async response",
            prompt_tokens=20,
            completion_tokens=8,
        )
        _setup_async_mock_chat(ap, mock_resp)

        res = await ap.chat.completions.create(
            messages=[{"role": "user", "content": "Async prompt"}],
            model="claude-3-5-sonnet-20241022",
        )

        assert res.id == "chatcmpl-async-1"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chat claude-3-5-sonnet-20241022"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
            == "anthropic"
        )
        assert (
            span.attributes.get(server_attributes.SERVER_ADDRESS)
            == "custom.portkey.ai"
        )
        assert span.attributes.get(server_attributes.SERVER_PORT) == 8443
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
            == 20
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 8
        )


def test_sync_chat_completions_content_capture(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        p = Portkey(api_key="test_pk")
        mock_resp = _create_mock_chat_completion(
            content="Captured content reply"
        )
        _setup_mock_chat(p, mock_resp)

        p.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello there!"},
            ],
            model="gpt-4o",
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
            == "portkey"
        )
        input_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        )
        assert len(input_messages) == 2
        assert input_messages[0]["role"] == "system"
        assert (
            input_messages[0]["parts"][0]["content"]
            == "You are a helpful assistant."
        )
        assert input_messages[1]["role"] == "user"
        assert input_messages[1]["parts"][0]["content"] == "Hello there!"

        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert len(output_messages) == 1
        assert output_messages[0]["role"] == "assistant"
        assert (
            output_messages[0]["parts"][0]["content"]
            == "Captured content reply"
        )
        assert output_messages[0]["finish_reason"] == "stop"


def test_sync_chat_completions_tools(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        p = Portkey(api_key="test_pk", provider="openai")
        tool_call = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "San Francisco"}),
            },
        }
        mock_resp = _create_mock_chat_completion(
            content="",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )
        _setup_mock_chat(p, mock_resp)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        p.chat.completions.create(
            messages=[{"role": "user", "content": "Weather in SF?"}],
            model="gpt-4o",
            tools=tools,
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        tool_defs = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
        )
        assert len(tool_defs) == 1
        assert tool_defs[0]["name"] == "get_weather"
        assert tool_defs[0]["description"] == "Get current weather"

        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert len(output_messages) == 1
        parts = output_messages[0]["parts"]
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call"
        assert parts[0]["name"] == "get_weather"
        assert parts[0]["id"] == "call_abc123"
        assert parts[0]["arguments"] == {"location": "San Francisco"}


def test_sync_chat_completions_tools_default_no_content_capture(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk", provider="openai")
        mock_resp = _create_mock_chat_completion()
        _setup_mock_chat(p, mock_resp)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        p.chat.completions.create(
            messages=[{"role": "user", "content": "Weather in SF?"}],
            model="gpt-4o",
            tools=tools,
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        tool_defs = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
        )
        assert len(tool_defs) == 1
        assert tool_defs[0]["name"] == "get_weather"
        # Input messages should not be captured when content capture is off
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes


def test_sync_chat_completions_error_handling(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk", provider="openai")
        if hasattr(p.chat.completions, "openai_client"):
            p.chat.completions.openai_client = MagicMock()
            p.chat.completions.openai_client.with_raw_response.chat.completions.create.side_effect = RuntimeError(
                "Portkey connection error"
            )
        p.chat.completions._post = MagicMock(
            side_effect=RuntimeError("Portkey connection error")
        )

        with pytest.raises(RuntimeError, match="Portkey connection error"):
            p.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
            )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"
        )


@pytest.mark.skipif(
    not _has_async_portkey,
    reason="AsyncPortkey not available in this version of portkey-ai",
)
@pytest.mark.asyncio
async def test_async_chat_completions_error_handling(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        ap = AsyncPortkey(api_key="test_pk", provider="openai")
        if hasattr(ap.chat.completions, "openai_client"):
            ap.chat.completions.openai_client = MagicMock()
            ap.chat.completions.openai_client.with_raw_response.chat.completions.create = AsyncMock(
                side_effect=ValueError("Async validation error")
            )
        ap.chat.completions._post = AsyncMock(
            side_effect=ValueError("Async validation error")
        )

        with pytest.raises(ValueError, match="Async validation error"):
            await ap.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o",
            )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "ValueError"


def test_chat_completions_dict_and_multipart_content(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        p = Portkey(api_key="test_pk", provider="openai")
        mock_resp = _create_mock_chat_completion(
            content="Processed",
        )
        _setup_mock_chat(p, mock_resp)

        p.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": {"type": "text", "text": "single dict content"},
                },
                {
                    "role": "user",
                    "content": [
                        "text part",
                        {"type": "text", "text": "nested dict part"},
                    ],
                },
            ],
            model="gpt-4o",
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        input_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        )
        assert len(input_messages) == 2
        assert input_messages[0]["parts"] == [
            {"content": "single dict content", "type": "text"}
        ]
        assert input_messages[1]["parts"] == [
            {"content": "text part", "type": "text"},
            {"content": "nested dict part", "type": "text"},
        ]


def test_chat_completions_top_p_zero_and_max_tokens_zero(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk", provider="openai")
        mock_resp = _create_mock_chat_completion()
        _setup_mock_chat(p, mock_resp)

        p.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            top_p=0.0,
            max_tokens=0,
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.0
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 0
        )


def test_chat_completions_alternative_keys_p_and_max_completion_tokens(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk", provider="openai")
        mock_resp = _create_mock_chat_completion()
        _setup_mock_chat(p, mock_resp)

        p.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            p=0.0,
            max_completion_tokens=0,
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.0
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 0
        )


@pytest.mark.parametrize(
    ("provider_input", "expected_provider"),
    [
        ("azure-openai", "azure.ai.openai"),
        ("azure_openai", "azure.ai.openai"),
        ("bedrock", "aws.bedrock"),
        ("aws-bedrock", "aws.bedrock"),
        ("vertex-ai", "gcp.vertex_ai"),
        ("gemini", "gcp.gemini"),
        ("mistral", "mistral_ai"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("cohere", "cohere"),
        ("perplexity-ai", "perplexity"),
        ("deepseek", "deepseek"),
        ("custom_provider", "custom_provider"),
    ],
)
def test_chat_completions_provider_mapping(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
    provider_input,
    expected_provider,
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk", provider=provider_input)
        mock_resp = _create_mock_chat_completion()
        _setup_mock_chat(p, mock_resp)

        p.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
        )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
            == expected_provider
        )
