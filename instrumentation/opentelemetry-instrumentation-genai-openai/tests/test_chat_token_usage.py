# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from opentelemetry.instrumentation.genai.openai.chat_wrappers import (
    AsyncChatStreamWrapper,
    ChatStreamWrapper,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.invocation import InferenceInvocation


@pytest.fixture(autouse=True)
def fixture_vcr() -> Iterator[None]:
    yield


@pytest.fixture(
    params=["instrument_no_content", "instrument_with_content"],
    autouse=True,
)
def instrumentation(
    request: pytest.FixtureRequest, content_mode: tuple[bool, str]
) -> None:
    request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        pytest.param(
            {
                "prompt_tokens_details": {
                    "cached_tokens": 5,
                    "cache_write_tokens": 10,
                    "text_tokens": 70,
                    "image_tokens": 20,
                    "audio_tokens": 10,
                },
                "completion_tokens_details": {
                    "text_tokens": 18,
                    "audio_tokens": 2,
                },
            },
            id="all-details",
        ),
        pytest.param(
            {"prompt_tokens_details": {"audio_tokens": 10}},
            id="partial-details",
        ),
        pytest.param({}, id="absent-details"),
        pytest.param(
            {
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            },
            id="null-details",
        ),
        pytest.param(
            {
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "text_tokens": 0,
                    "image_tokens": 0,
                    "audio_tokens": 0,
                },
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                },
            },
            id="zero-details",
        ),
    ],
)
def usage(request: pytest.FixtureRequest) -> dict[str, Any]:
    return {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        **request.param,
    }


@pytest.fixture
def chat_completion(usage: dict[str, Any]) -> ChatCompletion:
    response: dict[str, Any] = {
        "id": "chatcmpl-usage",
        "created": 1,
        "model": "gpt-4",
        "object": "chat.completion",
        "usage": usage,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
    }
    return ChatCompletion(**response)


@pytest.fixture
def chat_chunks(chat_completion: ChatCompletion) -> list[ChatCompletionChunk]:
    common: dict[str, Any] = {
        "id": chat_completion.id,
        "created": chat_completion.created,
        "model": chat_completion.model,
        "object": "chat.completion.chunk",
    }
    return [
        ChatCompletionChunk(
            **common,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="hello"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            **common,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
        ),
        ChatCompletionChunk(**common, choices=[], usage=chat_completion.usage),
    ]


def assert_usage_is_buffered(invocation: InferenceInvocation) -> None:
    assert invocation.cache_write_input_tokens is None
    assert invocation.text_input_tokens is None
    assert invocation.image_input_tokens is None
    assert invocation.audio_input_tokens is None
    assert invocation.text_output_tokens is None
    assert invocation.audio_output_tokens is None


def assert_usage(
    span_exporter: InMemorySpanExporter, usage: dict[str, Any]
) -> None:
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 100
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    expected: dict[str, int] = {
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 100,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 20,
    }
    for field, suffix, mapping in (
        (
            "prompt_tokens_details",
            "input_tokens",
            {
                "cached_tokens": "cache_read",
                "cache_write_tokens": "cache_write",
                "text_tokens": "text",
                "image_tokens": "image",
                "audio_tokens": "audio",
            },
        ),
        (
            "completion_tokens_details",
            "output_tokens",
            {"text_tokens": "text", "audio_tokens": "audio"},
        ),
    ):
        details = usage.get(field)
        if isinstance(details, dict):
            for name, attribute in mapping.items():
                if value := details.get(name):
                    expected[f"gen_ai.usage.{attribute}.{suffix}"] = value
    actual = {
        key: value
        for key, value in span.attributes.items()
        if key.startswith("gen_ai.usage.")
    }
    print(json.dumps({"exported_usage": actual}, sort_keys=True))
    assert actual == expected
    assert all(type(value) is int for value in actual.values())


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_detailed_token_usage(
    openai_client: OpenAI,
    monkeypatch: pytest.MonkeyPatch,
    chat_completion: ChatCompletion,
    chat_chunks: list[ChatCompletionChunk],
    streaming: bool,
    usage: dict[str, Any],
    span_exporter: InMemorySpanExporter,
) -> None:
    result: ChatCompletion | MagicMock = chat_completion
    if streaming:
        result = MagicMock(spec=Stream)
        result.__iter__.return_value = chat_chunks
    monkeypatch.setattr(
        target=openai_client,
        name="request",
        value=Mock(return_value=result),
    )
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "hello"}],
        stream=streaming,
        **({"stream_options": {"include_usage": True}} if streaming else {}),
    )
    if streaming:
        assert isinstance(response, ChatStreamWrapper)
        received: list[ChatCompletionChunk] = []
        with response:
            for chunk in response:
                assert_usage_is_buffered(invocation=response._self_invocation)
                received.append(chunk)
        assert received == chat_chunks
    else:
        assert response == chat_completion

    assert_usage(span_exporter=span_exporter, usage=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_chat_detailed_token_usage(
    async_openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    chat_completion: ChatCompletion,
    chat_chunks: list[ChatCompletionChunk],
    streaming: bool,
    usage: dict[str, Any],
    span_exporter: InMemorySpanExporter,
) -> None:
    result: ChatCompletion | MagicMock = chat_completion
    if streaming:
        result = MagicMock(spec=AsyncStream)
        result.__aiter__.return_value = chat_chunks
    monkeypatch.setattr(
        target=async_openai_client,
        name="request",
        value=AsyncMock(return_value=result),
    )
    response = await async_openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "hello"}],
        stream=streaming,
        **({"stream_options": {"include_usage": True}} if streaming else {}),
    )
    if streaming:
        assert isinstance(response, AsyncChatStreamWrapper)
        received: list[ChatCompletionChunk] = []
        async with response:
            async for chunk in response:
                assert_usage_is_buffered(invocation=response._self_invocation)
                received.append(chunk)
        assert received == chat_chunks
    else:
        assert response == chat_completion

    assert_usage(span_exporter=span_exporter, usage=usage)
