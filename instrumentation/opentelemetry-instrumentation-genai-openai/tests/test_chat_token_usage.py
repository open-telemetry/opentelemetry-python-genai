# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
            (
                {
                    "prompt_tokens_details": {
                        "cached_tokens": 5,
                        "cache_write_tokens": 10,
                        "audio_tokens": 10,
                    },
                    "completion_tokens_details": {
                        "audio_tokens": 2,
                        "reasoning_tokens": 3,
                    },
                },
                {
                    GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 5,
                    "gen_ai.usage.cache_write.input_tokens": 10,
                    "gen_ai.usage.audio.input_tokens": 10,
                    "gen_ai.usage.audio.output_tokens": 2,
                    GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: 3,
                },
            ),
            id="all-details",
        ),
        pytest.param(
            (
                {"prompt_tokens_details": {"audio_tokens": 10}},
                {"gen_ai.usage.audio.input_tokens": 10},
            ),
            id="partial-details",
        ),
        pytest.param(
            (
                {"prompt_tokens_details": {"cache_write_tokens": 10}},
                {"gen_ai.usage.cache_write.input_tokens": 10},
            ),
            id="cache-write-only",
        ),
        pytest.param(({}, {}), id="absent-details"),
        pytest.param(
            (
                {
                    "prompt_tokens_details": {},
                    "completion_tokens_details": {},
                },
                {},
            ),
            id="empty-details",
        ),
        pytest.param(
            (
                {
                    "prompt_tokens_details": None,
                    "completion_tokens_details": None,
                },
                {},
            ),
            id="null-details",
        ),
        pytest.param(
            (
                {
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                        "audio_tokens": 0,
                    },
                    "completion_tokens_details": {
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                },
                {},
            ),
            id="zero-details",
        ),
        pytest.param(
            (
                {
                    "prompt_tokens_details": {
                        "cache_write_tokens": 10,
                        "audio_tokens": -3,
                    },
                    "completion_tokens_details": {
                        "audio_tokens": -2,
                    },
                },
                {
                    "gen_ai.usage.cache_write.input_tokens": 10,
                },
            ),
            id="negative-modalities",
        ),
        pytest.param(
            (
                {
                    "prompt_tokens_details": {
                        "text_tokens": 70,
                        "image_tokens": 20,
                        "audio_tokens": 10,
                    },
                    "completion_tokens_details": {
                        "text_tokens": 18,
                        "audio_tokens": 2,
                    },
                },
                {
                    "gen_ai.usage.audio.input_tokens": 10,
                    "gen_ai.usage.audio.output_tokens": 2,
                },
            ),
            id="unsupported-modalities-ignored",
        ),
    ],
)
def usage_case(
    request: pytest.FixtureRequest,
) -> tuple[dict[str, Any], dict[str, int]]:
    details, expected_details = request.param
    return (
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            **details,
        },
        {
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 100,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 20,
            **expected_details,
        },
    )


@pytest.fixture
def chat_completion(
    usage_case: tuple[dict[str, Any], dict[str, int]],
) -> ChatCompletion:
    usage, _ = usage_case
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
    assert invocation.input_tokens is None
    assert invocation.output_tokens is None
    assert invocation.cache_read_input_tokens is None
    assert invocation.cache_write_input_tokens is None
    assert invocation.text_input_tokens is None
    assert invocation.image_input_tokens is None
    assert invocation.audio_input_tokens is None
    assert invocation.text_output_tokens is None
    assert invocation.audio_output_tokens is None
    assert invocation.thinking_tokens is None


def assert_usage(
    span_exporter: InMemorySpanExporter, expected: dict[str, int]
) -> None:
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    actual = {
        key: value
        for key, value in span.attributes.items()
        if key.startswith("gen_ai.usage.")
    }
    assert actual == expected
    assert all(type(value) is int for value in actual.values())


@pytest.mark.parametrize(
    ("streaming", "drain_stream"),
    [(False, True), (True, True), (True, False)],
    ids=["nonstreaming", "stream-exhausted", "stream-closed"],
)
def test_chat_detailed_token_usage(
    openai_client: OpenAI,
    monkeypatch: pytest.MonkeyPatch,
    chat_completion: ChatCompletion,
    chat_chunks: list[ChatCompletionChunk],
    streaming: bool,
    drain_stream: bool,
    usage_case: tuple[dict[str, Any], dict[str, int]],
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
                assert not span_exporter.get_finished_spans()
                received.append(chunk)
                if not drain_stream and chunk.usage is not None:
                    break
        assert received == chat_chunks
    else:
        assert response == chat_completion

    _, expected = usage_case
    assert_usage(span_exporter=span_exporter, expected=expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("streaming", "drain_stream"),
    [(False, True), (True, True), (True, False)],
    ids=["nonstreaming", "stream-exhausted", "stream-closed"],
)
async def test_async_chat_detailed_token_usage(
    async_openai_client: AsyncOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    chat_completion: ChatCompletion,
    chat_chunks: list[ChatCompletionChunk],
    streaming: bool,
    drain_stream: bool,
    usage_case: tuple[dict[str, Any], dict[str, int]],
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
                assert not span_exporter.get_finished_spans()
                received.append(chunk)
                if not drain_stream and chunk.usage is not None:
                    break
        assert received == chat_chunks
    else:
        assert response == chat_completion

    _, expected = usage_case
    assert_usage(span_exporter=span_exporter, expected=expected)
