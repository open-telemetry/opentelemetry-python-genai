# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Portkey AI streaming instrumentation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
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
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode

_has_async_portkey = AsyncPortkey is not None


class _MockSyncStream(Iterator[SimpleNamespace]):
    def __init__(
        self,
        chunks: list[SimpleNamespace],
        fail_after: int | None = None,
    ) -> None:
        self._chunks = chunks
        self._index = 0
        self._fail_after = fail_after
        self.closed = False

    def __iter__(self) -> _MockSyncStream:
        return self

    def __next__(self) -> SimpleNamespace:
        if self._fail_after is not None and self._index >= self._fail_after:
            raise ConnectionResetError("Stream connection reset")
        if self._index >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _MockSyncStream:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class _MockAsyncStream(AsyncIterator[SimpleNamespace]):
    def __init__(
        self,
        chunks: list[SimpleNamespace],
        fail_after: int | None = None,
    ) -> None:
        self._chunks = chunks
        self._index = 0
        self._fail_after = fail_after
        self.closed = False

    def __aiter__(self) -> _MockAsyncStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._fail_after is not None and self._index >= self._fail_after:
            raise ConnectionResetError("Async stream connection reset")
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> _MockAsyncStream:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


def _setup_mock_stream(client: Portkey, stream_obj: _MockSyncStream) -> None:
    if hasattr(client.chat.completions, "openai_client"):
        client.chat.completions.openai_client = MagicMock()
        client.chat.completions.openai_client.chat.completions.create.return_value = stream_obj
    client.chat.completions._post = MagicMock(return_value=stream_obj)


def _setup_async_mock_stream(
    client: AsyncPortkey, stream_obj: _MockAsyncStream
) -> None:
    if hasattr(client.chat.completions, "openai_client"):
        client.chat.completions.openai_client = MagicMock()
        client.chat.completions.openai_client.chat.completions.create = (
            AsyncMock(return_value=stream_obj)
        )
    client.chat.completions._post = AsyncMock(return_value=stream_obj)


def test_sync_chat_streaming_basic(
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

        chunk1 = SimpleNamespace(
            id="stream-chunk-1",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(role="assistant", content="Hello "),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            id="stream-chunk-1",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="world!"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=5, completion_tokens=2, total_tokens=7
            ),
        )
        stream_obj = _MockSyncStream([chunk1, chunk2])
        _setup_mock_stream(p, stream_obj)

        stream = p.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            stream=True,
        )

        assert len(span_exporter.get_finished_spans()) == 0

        chunks = list(stream)
        assert len(chunks) == 2

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chat gpt-4o"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "stream-chunk-1"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
            == "gpt-4o"
        )
        assert span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ) == ("stop",)
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 5
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 2
        )
        assert (
            GenAIAttributes.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK
            in span.attributes
        )

        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert len(output_messages) == 1
        assert output_messages[0]["parts"][0]["content"] == "Hello world!"


@pytest.mark.skipif(
    not _has_async_portkey,
    reason="AsyncPortkey not available in this version of portkey-ai",
)
@pytest.mark.asyncio
async def test_async_chat_streaming_basic(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        ap = AsyncPortkey(api_key="test_pk", provider="anthropic")

        chunk1 = SimpleNamespace(
            id="async-stream-1",
            model="claude-3-5-sonnet",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(role="assistant", content="Async "),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        chunk2 = SimpleNamespace(
            id="async-stream-1",
            model="claude-3-5-sonnet",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="streaming."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=4, total_tokens=14
            ),
        )
        stream_obj = _MockAsyncStream([chunk1, chunk2])
        _setup_async_mock_stream(ap, stream_obj)

        stream = await ap.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-5-sonnet",
            stream=True,
        )

        collected = []
        async for chunk in stream:
            collected.append(chunk)

        assert len(collected) == 2
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chat claude-3-5-sonnet"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "async-stream-1"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
            == 10
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 4
        )

        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert output_messages[0]["parts"][0]["content"] == "Async streaming."


def test_sync_streaming_caller_error_in_context_manager(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk")

        chunk1 = SimpleNamespace(
            id="stream-chunk-err",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="part1"),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        stream_obj = _MockSyncStream([chunk1])
        _setup_mock_stream(p, stream_obj)

        with pytest.raises(RuntimeError, match="User cancelled processing"):
            with p.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                stream=True,
            ) as stream:
                for _ in stream:
                    raise RuntimeError("User cancelled processing")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"
        )


def test_sync_streaming_midstream_sdk_error(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        p = Portkey(api_key="test_pk")

        chunk1 = SimpleNamespace(
            id="stream-chunk-err",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="part1"),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        stream_obj = _MockSyncStream([chunk1], fail_after=1)
        _setup_mock_stream(p, stream_obj)

        with pytest.raises(
            ConnectionResetError, match="Stream connection reset"
        ):
            stream = p.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                stream=True,
            )
            list(stream)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            span.attributes.get(ErrorAttributes.ERROR_TYPE)
            == "ConnectionResetError"
        )


@pytest.mark.skipif(
    not _has_async_portkey,
    reason="AsyncPortkey not available in this version of portkey-ai",
)
@pytest.mark.asyncio
async def test_async_streaming_caller_error_in_context_manager(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        ap = AsyncPortkey(api_key="test_pk")

        chunk1 = SimpleNamespace(
            id="async-stream-err",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="part1"),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        stream_obj = _MockAsyncStream([chunk1])
        _setup_async_mock_stream(ap, stream_obj)

        with pytest.raises(RuntimeError, match="Async user cancelled"):
            async with await ap.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                stream=True,
            ) as stream:
                async for _ in stream:
                    raise RuntimeError("Async user cancelled")

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
async def test_async_streaming_midstream_sdk_error(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        ap = AsyncPortkey(api_key="test_pk")

        chunk1 = SimpleNamespace(
            id="async-stream-err",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="part1"),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        stream_obj = _MockAsyncStream([chunk1], fail_after=1)
        _setup_async_mock_stream(ap, stream_obj)

        with pytest.raises(
            ConnectionResetError, match="Async stream connection reset"
        ):
            stream = await ap.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                stream=True,
            )
            async for _ in stream:
                pass

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            span.attributes.get(ErrorAttributes.ERROR_TYPE)
            == "ConnectionResetError"
        )


def test_sync_prompt_streaming(
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

        pchunk1 = SimpleNamespace(
            id="pchunk-1",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    text="Prompt ",
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        pchunk2 = SimpleNamespace(
            id="pchunk-1",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    text="stream.",
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=8, completion_tokens=3, total_tokens=11
            ),
        )
        stream_obj = _MockSyncStream([pchunk1, pchunk2])
        p.prompts.completions._post = MagicMock(return_value=stream_obj)

        stream = p.prompts.completions.create(
            prompt_id="pp-stream-test",
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) == 2

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chat"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_PROMPT_NAME)
            == "pp-stream-test"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "pchunk-1"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 8
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 3
        )

        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert output_messages[0]["parts"][0]["content"] == "Prompt stream."


def test_sync_streaming_tool_calls_multi_chunk(
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

        # Chunk 1: buffer initialized with call_id and function name
        chunk1 = SimpleNamespace(
            id="stream-tc-1",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_abc",
                                function=SimpleNamespace(
                                    name="get_weather",
                                    arguments='{"location":',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
        # Chunk 2: arguments appended to buffer
        chunk2 = SimpleNamespace(
            id="stream-tc-1",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(
                                    name=None, arguments=' "Paris"}'
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=15, completion_tokens=8),
        )
        stream_obj = _MockSyncStream([chunk1, chunk2])
        _setup_mock_stream(p, stream_obj)

        stream = p.chat.completions.create(
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"}
            ],
            model="gpt-4o",
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) == 2

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert len(output_messages) == 1
        part = output_messages[0]["parts"][0]
        assert part["type"] == "tool_call"
        assert part["name"] == "get_weather"
        assert part["id"] == "call_abc"
        assert part["arguments"] == {"location": "Paris"}
        assert output_messages[0]["finish_reason"] == "tool_calls"


def test_sync_streaming_no_finish_reason_defaults_to_stop(
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

        chunk1 = SimpleNamespace(
            id="stream-nofinish-1",
            model="gpt-4o",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="Hello!"),
                    finish_reason=None,
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )
        stream_obj = _MockSyncStream([chunk1])
        _setup_mock_stream(p, stream_obj)

        stream = p.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) == 1

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.UNSET
        output_messages = json.loads(
            span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        )
        assert output_messages[0]["finish_reason"] == "stop"


def test_sync_streaming_request_failure_records_span(
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
        p.chat.completions.openai_client = MagicMock()
        p.chat.completions.openai_client.chat.completions.create.side_effect = RuntimeError(
            "boom"
        )
        p.chat.completions._post = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            p.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                stream=True,
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
async def test_async_streaming_request_failure_records_span(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    with instrument(
        PortkeyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        ap = AsyncPortkey(api_key="test_pk", provider="openai")
        if hasattr(ap.chat.completions, "openai_client"):
            ap.chat.completions.openai_client = MagicMock()
            ap.chat.completions.openai_client.chat.completions.create = (
                AsyncMock(side_effect=RuntimeError("async boom"))
            )
        ap.chat.completions._post = AsyncMock(
            side_effect=RuntimeError("async boom")
        )

        with pytest.raises(RuntimeError, match="async boom"):
            await ap.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                stream=True,
            )

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert (
            span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"
        )
