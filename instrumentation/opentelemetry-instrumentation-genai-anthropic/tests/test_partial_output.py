# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

try:
    import httpx2 as _http_lib
except ImportError:
    import httpx as _http_lib
from anthropic import Anthropic, AsyncAnthropic

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.genai.anthropic.wrappers import (
    AsyncMessagesStreamWrapper,
    MessagesStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK,
)
from opentelemetry.util.genai.types import OutputMessage, TextPart


class _InterruptedResponse(
    _http_lib.SyncByteStream, _http_lib.AsyncByteStream
):
    def __init__(self, phase: str, error: BaseException) -> None:
        self.error = error
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_partial",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        ]
        event_count = {
            "before_message": 0,
            "before_content": 1,
            "after_content": 3,
        }[phase]
        self.chunks = [
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            for event in events[:event_count]
        ]

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks
        raise self.error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        raise self.error

    def respond(self, request: _http_lib.Request) -> _http_lib.Response:
        return _http_lib.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=self,
        )


@pytest.mark.parametrize("read_mode", ["create", "events", "text"])
@pytest.mark.parametrize("capture_content", [True, False])
@pytest.mark.parametrize(
    "phase", ["before_message", "before_content", "after_content"]
)
class TestPartialOutput:
    def _assert_telemetry(
        self,
        span_exporter,
        log_exporter,
        hook: Mock,
        capture_content: bool,
        phase: str,
        error: BaseException,
    ) -> None:
        spans = span_exporter.get_finished_spans()
        logs = log_exporter.get_finished_logs()
        assert len(spans) == 1
        assert len(logs) == 1
        span = spans[0]
        event = logs[0].log_record
        assert span.status.status_code == StatusCode.ERROR
        error_type = (
            "asyncio.exceptions.CancelledError"
            if isinstance(error, asyncio.CancelledError)
            else "ConnectionError"
        )
        assert span.attributes[ErrorAttributes.ERROR_TYPE] == error_type
        assert event.attributes[ErrorAttributes.ERROR_TYPE] == error_type
        assert (
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
            not in span.attributes
        )
        assert (
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
            not in event.attributes
        )

        expected_outputs = []
        has_output = capture_content and phase == "after_content"
        if has_output:
            expected_outputs = [
                OutputMessage(
                    role="assistant", parts=[TextPart(content="hello")]
                )
            ]
        if expected_outputs:
            expected = [asdict(message) for message in expected_outputs]
            assert (
                json.loads(
                    span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
                )
                == expected
            )
            assert event.attributes[
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
            ] == tuple(
                {**message, "parts": tuple(message["parts"])}
                for message in expected
            )
        else:
            assert (
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
            )
            assert (
                GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in event.attributes
            )

        if capture_content:
            hook.on_completion.assert_called_once()
            captured = hook.on_completion.call_args.kwargs
            assert captured["outputs"] == expected_outputs
            assert captured["inputs"][0].parts == [TextPart(content="prompt")]
            assert captured["system_instruction"] == [
                TextPart(content="instructions")
            ]
        else:
            hook.on_completion.assert_not_called()
            for attribute in (
                GenAIAttributes.GEN_AI_INPUT_MESSAGES,
                GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
            ):
                assert attribute not in span.attributes
                assert attribute not in event.attributes

    def test_sync_stream_failure_preserves_partial_output(
        self,
        read_mode,
        capture_content,
        phase,
        monkeypatch,
        tracer_provider,
        logger_provider,
        meter_provider,
        span_exporter,
        log_exporter,
    ) -> None:
        monkeypatch.delenv(
            OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK, raising=False
        )
        error = ConnectionError("stream interrupted")
        body = _InterruptedResponse(phase, error)
        hook = Mock()
        with (
            instrument(
                AnthropicInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture=(
                    "SPAN_AND_EVENT" if capture_content else "NO_CONTENT"
                ),
                emit_event=True,
                completion_hook=hook if capture_content else None,
            ),
            Anthropic(
                api_key="test-key",
                base_url="https://anthropic.test",
                max_retries=0,
                http_client=_http_lib.Client(
                    transport=_http_lib.MockTransport(body.respond)
                ),
            ) as client,
        ):
            kwargs = {
                "model": "claude-test",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "prompt"}],
                "system": "instructions",
            }
            manager = (
                client.messages.create(**kwargs, stream=True)
                if read_mode == "create"
                else client.messages.stream(**kwargs)
            )
            with pytest.raises(ConnectionError) as raised:
                with manager as stream:
                    iterator = (
                        stream.text_stream if read_mode == "text" else stream
                    )
                    for _ in iterator:
                        pass
            assert raised.value is error
            stream.close()

        self._assert_telemetry(
            span_exporter, log_exporter, hook, capture_content, phase, error
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_class", [ConnectionError, asyncio.CancelledError]
    )
    async def test_async_stream_failure_preserves_partial_output(
        self,
        read_mode,
        capture_content,
        phase,
        error_class,
        monkeypatch,
        tracer_provider,
        logger_provider,
        meter_provider,
        span_exporter,
        log_exporter,
    ) -> None:
        monkeypatch.delenv(
            OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK, raising=False
        )
        error = error_class("stream interrupted")
        body = _InterruptedResponse(phase, error)
        hook = Mock()
        with instrument(
            AnthropicInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture=(
                "SPAN_AND_EVENT" if capture_content else "NO_CONTENT"
            ),
            emit_event=True,
            completion_hook=hook if capture_content else None,
        ):
            async with AsyncAnthropic(
                api_key="test-key",
                base_url="https://anthropic.test",
                max_retries=0,
                http_client=_http_lib.AsyncClient(
                    transport=_http_lib.MockTransport(body.respond)
                ),
            ) as client:
                kwargs = {
                    "model": "claude-test",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "prompt"}],
                    "system": "instructions",
                }
                manager = (
                    await client.messages.create(**kwargs, stream=True)
                    if read_mode == "create"
                    else client.messages.stream(**kwargs)
                )
                with pytest.raises(error_class) as raised:
                    async with manager as stream:
                        iterator = (
                            stream.text_stream
                            if read_mode == "text"
                            else stream
                        )
                        async for _ in iterator:
                            pass
                assert raised.value is error
                await stream.close()

        self._assert_telemetry(
            span_exporter, log_exporter, hook, capture_content, phase, error
        )


@pytest.mark.parametrize(
    "wrapper_class", [MessagesStreamWrapper, AsyncMessagesStreamWrapper]
)
def test_partial_capture_error_preserves_original_error(
    wrapper_class, monkeypatch
) -> None:
    error = ConnectionError("stream interrupted")
    stream = _InterruptedResponse("after_content", error)
    stream.current_message_snapshot = SimpleNamespace(content=["hello"])
    invocation = Mock()
    wrapper = wrapper_class(stream, invocation, capture_content=True)
    extraction = Mock(side_effect=ValueError("invalid partial content"))
    monkeypatch.setattr(
        "opentelemetry.instrumentation.genai.anthropic.wrappers"
        "._set_response_attributes",
        extraction,
    )

    wrapper._on_stream_error(error)
    wrapper._on_stream_error(error)

    extraction.assert_called_once()
    invocation.fail.assert_called_once_with(error)
    invocation.stop.assert_not_called()
