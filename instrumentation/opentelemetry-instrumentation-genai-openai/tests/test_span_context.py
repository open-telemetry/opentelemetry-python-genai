# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from openai import AsyncOpenAI, OpenAI

from opentelemetry import trace

from .test_responses import HAS_RESPONSES_API, RETRIEVE_RESPONSE_ID
from .test_utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    USER_ONLY_PROMPT,
)

try:
    import httpx as _http_lib
except ImportError:
    import httpx2 as _http_lib

requires_responses = pytest.mark.skipif(
    not HAS_RESPONSES_API, reason="Responses API requires a newer openai SDK"
)


def _no_current_span() -> bool:
    return not trace.get_current_span().get_span_context().is_valid


def _client() -> tuple[list[int], OpenAI]:
    seen: list[int] = []

    def hook(_request: Any) -> None:
        seen.append(trace.get_current_span().get_span_context().span_id)

    http_client = _http_lib.Client(event_hooks={"request": [hook]})
    return seen, OpenAI(http_client=http_client)


def _async_client() -> tuple[list[int], AsyncOpenAI]:
    seen: list[int] = []

    async def hook(_request: Any) -> None:
        seen.append(trace.get_current_span().get_span_context().span_id)

    http_client = _http_lib.AsyncClient(event_hooks={"request": [hook]})
    return seen, AsyncOpenAI(http_client=http_client)


def _span_id(span_exporter) -> int:
    (span,) = span_exporter.get_finished_spans()
    return span.context.span_id


def test_chat_span_is_current_only_during_the_request(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _client()
    with vcr.use_cassette("test_chat_completion_no_content.yaml"):
        client.chat.completions.create(
            model=DEFAULT_MODEL, messages=USER_ONLY_PROMPT
        )
        assert _no_current_span()

    assert seen == [_span_id(span_exporter)]


def test_chat_stream_span_is_not_current_between_reads(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _client()
    with vcr.use_cassette("test_chat_completion_streaming.yaml"):
        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=USER_ONLY_PROMPT,
            stream=True,
            stream_options={"include_usage": True},
        )
        assert _no_current_span()
        for _ in stream:
            assert _no_current_span()

    assert seen == [_span_id(span_exporter)]


@pytest.mark.asyncio
async def test_async_chat_stream_span_is_not_current_between_reads(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _async_client()
    with vcr.use_cassette("test_async_chat_completion_streaming.yaml"):
        stream = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=USER_ONLY_PROMPT,
            stream=True,
            stream_options={"include_usage": True},
        )
        assert _no_current_span()
        await asyncio.create_task(anext(stream))
        assert _no_current_span()
        async for _ in stream:
            assert _no_current_span()

    assert seen == [_span_id(span_exporter)]


def test_raw_response_stream_span_is_not_current_after_create(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _client()
    with vcr.use_cassette("test_chat_completion_streaming.yaml"):
        raw = client.chat.completions.with_raw_response.create(
            model=DEFAULT_MODEL,
            messages=USER_ONLY_PROMPT,
            stream=True,
            stream_options={"include_usage": True},
        )
        assert _no_current_span()
        for _ in raw.parse():
            assert _no_current_span()

    assert seen == [_span_id(span_exporter)]


def test_embeddings_span_is_current_only_during_the_request(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _client()
    with vcr.use_cassette("test_embeddings_no_content.yaml"):
        client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input="This is a test for embeddings",
        )
        assert _no_current_span()

    assert seen == [_span_id(span_exporter)]


@requires_responses
def test_responses_stream_span_is_not_current_between_reads(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _client()
    with vcr.use_cassette(
        "test_responses_create_streaming[content_mode0].yaml"
    ):
        stream = client.responses.create(
            model=DEFAULT_MODEL,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=True,
        )
        assert _no_current_span()
        for _ in stream:
            assert _no_current_span()

    assert seen == [_span_id(span_exporter)]


@requires_responses
def test_responses_stream_manager_request_runs_under_the_span(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _client()
    with vcr.use_cassette(
        "test_responses_stream_until_done[content_mode0].yaml"
    ):
        with client.responses.stream(
            model=DEFAULT_MODEL, input=USER_ONLY_PROMPT[0]["content"]
        ) as stream:
            assert _no_current_span()
            for _ in stream:
                assert _no_current_span()
        assert _no_current_span()

    assert seen == [_span_id(span_exporter)]


@requires_responses
def test_responses_retrieve_span_is_current_only_during_the_request(
    span_exporter, instrument_no_content, vcr
):
    seen, client = _client()
    with vcr.use_cassette("test_responses_retrieve_basic[content_mode0].yaml"):
        client.responses.retrieve(RETRIEVE_RESPONSE_ID)
        assert _no_current_span()

    assert seen == [_span_id(span_exporter)]
