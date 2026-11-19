# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Google GenAI Live API (realtime) instrumentation (PR #394)."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from google.genai import live as genai_live
from google.genai import types as genai_types
from google.genai.live import AsyncLive

from opentelemetry._logs import SeverityNumber
from opentelemetry.context import get_value
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai import (
    live as live_module,
)
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind, StatusCode, get_current_span
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    CLIENT_INFERENCE_CONTEXT_KEY,
    InferenceData,
)


class _FakeAsyncSession:
    """Fake google.genai.live.AsyncSession for deterministic unit tests."""

    def __init__(
        self,
        messages: list[genai_types.LiveServerMessage | BaseException],
        session_id: str | None = "test-live-session-123",
    ) -> None:
        self._messages = list(messages)
        self.session_id = session_id
        self.sent_realtime_inputs: list[dict[str, Any]] = []
        self.sent_client_contents: list[dict[str, Any]] = []
        self.sent_tool_responses: list[dict[str, Any]] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.sent_realtime_inputs.append(kwargs)

    async def send_client_content(self, **kwargs: Any) -> None:
        self.sent_client_contents.append(kwargs)

    async def send_tool_response(self, **kwargs: Any) -> None:
        self.sent_tool_responses.append(kwargs)

    async def send(self, **kwargs: Any) -> None:
        pass

    async def _receive(self) -> genai_types.LiveServerMessage:
        if not self._messages:
            return genai_types.LiveServerMessage()
        item = self._messages.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def receive(self) -> AsyncIterator[genai_types.LiveServerMessage]:
        while self._messages:
            msg = await self._receive()
            yield msg
            if msg.server_content and msg.server_content.turn_complete is True:
                break


_HAS_SERVER_VAD = hasattr(genai_types, "VoiceActivity") and hasattr(
    genai_types, "VoiceActivityType"
)


def _server_vad_messages() -> list[genai_types.LiveServerMessage]:
    if not _HAS_SERVER_VAD:
        return []
    return [
        genai_types.LiveServerMessage(
            voice_activity=genai_types.VoiceActivity(
                voice_activity_type=genai_types.VoiceActivityType.ACTIVITY_START
            )
        ),
        genai_types.LiveServerMessage(
            voice_activity=genai_types.VoiceActivity(
                voice_activity_type=genai_types.VoiceActivityType.ACTIVITY_END
            )
        ),
    ]


def _make_fake_connect(fake_session: _FakeAsyncSession):
    @contextlib.asynccontextmanager
    async def _fake_connect(
        self: Any,
        *,
        model: str,
        config: genai_types.LiveConnectConfigOrDict | None = None,
    ) -> AsyncIterator[_FakeAsyncSession]:
        yield fake_session

    return _fake_connect


@pytest.fixture
def instrumented_live(
    tracer_provider, logger_provider, meter_provider, monkeypatch
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        yield


@pytest.mark.asyncio
async def test_live_voice_turn_spans_events_and_metrics(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, "true")
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        *_server_vad_messages(),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="Say hello in three words."
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            thought=True,
                            text="Thinking of three word greeting",
                        ),
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x00\x01" * 1200,
                                mime_type="audio/pcm;rate=24000",
                            )
                        ),
                    ],
                ),
                output_transcription=genai_types.Transcription(
                    text="Hello there friend."
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                generation_complete=True,
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=38,
                response_token_count=41,
                total_token_count=79,
                prompt_tokens_details=[
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.AUDIO,
                        token_count=32,
                    ),
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.TEXT,
                        token_count=6,
                    ),
                ],
                response_tokens_details=[
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.AUDIO,
                        token_count=30,
                    ),
                    genai_types.ModalityTokenCount(
                        modality=genai_types.MediaModality.TEXT,
                        token_count=11,
                    ),
                ],
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-voice-1")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                system_instruction="Be concise.",
                max_output_tokens=256,
            )
            async with live.connect(
                model="models/gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                if not _HAS_SERVER_VAD:
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x00" * 800,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                if not _HAS_SERVER_VAD:
                    await session.send_realtime_input(
                        activity_end=genai_types.ActivityEnd()
                    )
                else:
                    await session.send_realtime_input(audio_stream_end=True)
                async for _ in session.receive():
                    pass

    # Verify spans: user_speech finishes before realtime_inference starts
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    speech_span, span = spans[0], spans[1]
    assert speech_span.name == "user_speech"
    assert speech_span.kind == SpanKind.INTERNAL
    assert speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"
    assert speech_span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"
    assert (
        speech_span.attributes["gen_ai.realtime_session.id"] == "sess-voice-1"
    )
    assert speech_span.end_time <= span.start_time

    assert (
        span.name
        == "realtime_inference gemini-2.5-flash-native-audio-preview-12-2025"
    )
    assert span.kind == SpanKind.CLIENT
    attrs = dict(span.attributes)
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "realtime_inference"
    assert attrs[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"
    assert (
        attrs[GenAI.GEN_AI_REQUEST_MODEL]
        == "gemini-2.5-flash-native-audio-preview-12-2025"
    )
    assert attrs["gen_ai.realtime_session.id"] == "sess-voice-1"
    assert attrs[GenAI.GEN_AI_OUTPUT_TYPE] == "speech"
    assert attrs[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 38
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 41
    assert attrs["gen_ai.usage.audio.input_tokens"] == 32
    assert attrs["gen_ai.usage.text.input_tokens"] == 6
    assert attrs["gen_ai.usage.audio.output_tokens"] == 30
    assert attrs["gen_ai.usage.text.output_tokens"] == 11
    assert (
        attrs[ServerAttributes.SERVER_ADDRESS]
        == "generativelanguage.googleapis.com"
    )
    assert attrs[ServerAttributes.SERVER_PORT] == 443

    input_msgs = attrs[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "blob"
    assert input_msgs[0]["parts"][0]["modality"] == "audio"
    assert input_msgs[0]["parts"][1] == {
        "type": "transcription",
        "content": "Say hello in three words.",
    }

    output_msgs = attrs[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["finish_reason"] == "stop"
    part_types = [p["type"] for p in output_msgs[0]["parts"]]
    assert part_types == ["reasoning", "blob", "transcription"]
    assert output_msgs[0]["parts"][2]["content"] == "Hello there friend."

    sys_instr = attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS]
    assert sys_instr == ({"type": "text", "content": "Be concise."},)

    # Verify events: only realtime_session.started and realtime_session.ended
    logs = log_exporter.get_finished_logs()
    event_names = [log.log_record.event_name for log in logs]
    assert event_names == [
        "gen_ai.client.realtime_session.started",
        "gen_ai.client.realtime_session.ended",
    ]

    started_log = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.started"
    )
    assert started_log.severity_number == SeverityNumber.INFO
    assert started_log.body is None
    assert (
        started_log.attributes["gen_ai.realtime_session.id"] == "sess-voice-1"
    )
    assert started_log.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"

    ended_log = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended_log.severity_number == SeverityNumber.INFO
    assert ended_log.body is None
    assert ended_log.attributes["gen_ai.realtime_session.id"] == "sess-voice-1"

    # Verify metrics (including per-modality token usage)
    metrics_data = metric_reader.get_metrics_data()
    metric_names = {}
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                metric_names[m.name] = m

    assert "gen_ai.client.operation.duration" in metric_names
    assert "gen_ai.client.token.usage" in metric_names
    token_dp = list(metric_names["gen_ai.client.token.usage"].data.data_points)
    assert len(token_dp) == 2
    token_totals = {
        dp.attributes.get(GenAI.GEN_AI_TOKEN_TYPE): dp.sum for dp in token_dp
    }
    assert token_totals == {"input": 38, "output": 41}


@pytest.mark.asyncio
async def test_live_gemini_2_5_two_generation_tool_call(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    gen1_messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="What is the weather in Paris?"
                )
            )
        ),
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(
                        id="call-weather-1",
                        name="get_weather",
                        args={"city": "Paris"},
                    )
                ]
            )
        ),
        # Trailing completion frame after tool_call must not start a phantom generation
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                generation_complete=True,
                turn_complete=True,
            )
        ),
    ]
    gen2_messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 500,
                                mime_type="audio/pcm;rate=24000",
                            )
                        )
                    ],
                ),
                output_transcription=genai_types.Transcription(
                    text="It is 18 degrees and sunny in Paris."
                ),
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=90,
                response_token_count=45,
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(
        gen1_messages + gen2_messages, session_id="sess-tool-25"
    )

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                tools=[
                    genai_types.Tool(
                        function_declarations=[
                            genai_types.FunctionDeclaration(
                                name="get_weather",
                                description="Get weather for a city",
                            )
                        ]
                    )
                ],
            )
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x00" * 400,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                # Receive gen 1 (input transcription + tool_call + trailing completion frame)
                msg1 = await session._receive()
                assert msg1.server_content is not None
                msg2 = await session._receive()
                assert msg2.tool_call is not None
                msg_trailing = await session._receive()
                assert msg_trailing.server_content is not None

                # Send tool response
                await session.send_tool_response(
                    function_responses=[
                        genai_types.FunctionResponse(
                            id="call-weather-1",
                            name="get_weather",
                            response={"temp_c": 18, "condition": "sunny"},
                        )
                    ]
                )

                # Receive gen 2 (spoken answer + turn_complete)
                msg3 = await session._receive()
                assert msg3.server_content is not None
                assert msg3.server_content.turn_complete is True

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    span1, span2 = spans[0], spans[1]
    assert span1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert span1.attributes[GenAI.GEN_AI_OUTPUT_TYPE] == "speech"
    s1_out = span1.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert s1_out[0]["finish_reason"] == "tool_call"
    assert s1_out[0]["parts"][0]["type"] == "tool_call"
    assert s1_out[0]["parts"][0]["name"] == "get_weather"

    assert span2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert span2.attributes[GenAI.GEN_AI_OUTPUT_TYPE] == "speech"
    s2_in = span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert s2_in[0]["role"] == "tool"
    assert s2_in[0]["parts"][0]["type"] == "tool_call_response"
    assert s2_in[0]["parts"][0]["id"] == "call-weather-1"


@pytest.mark.asyncio
async def test_live_gemini_3_single_generation_parallel_tool_calls_and_followup_turn(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="Compare weather in Tokyo and Osaka."
                )
            )
        ),
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(
                        id="call-tokyo-1",
                        name="get_weather",
                        args={"city": "Tokyo"},
                    ),
                    genai_types.FunctionCall(
                        id="call-osaka-2",
                        name="get_weather",
                        args={"city": "Osaka"},
                    ),
                ]
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 500,
                                mime_type="audio/pcm;rate=24000",
                            )
                        )
                    ],
                ),
                output_transcription=genai_types.Transcription(
                    text="Tokyo is 22C and Osaka is 24C."
                ),
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=110,
                response_token_count=55,
            ),
        ),
        # Second user turn after the parallel tool turn
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="You are welcome!")],
                ),
                turn_complete=True,
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-tool-31")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            async with live.connect(
                model="gemini-3.1-flash-live-preview",
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x00" * 400,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                await session._receive()
                await session._receive()
                # Send two separate tool responses for the two function calls
                await session.send_tool_response(
                    function_responses=[
                        genai_types.FunctionResponse(
                            id="call-tokyo-1",
                            name="get_weather",
                            response={"temp_c": 22},
                        )
                    ]
                )
                await session.send_tool_response(
                    function_responses=[
                        genai_types.FunctionResponse(
                            id="call-osaka-2",
                            name="get_weather",
                            response={"temp_c": 24},
                        )
                    ]
                )
                await session._receive()

                # Follow-up user turn
                await session.send_realtime_input(text="Thanks!")
                await session._receive()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]
    assert span1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    # Per PR #394, gen_ai.input.messages and gen_ai.output.messages
    # on a single-generation tool exchange omit intermediate tool call and
    # tool result events.
    in_msgs_1 = span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in_msgs_1) == 1
    assert in_msgs_1[0]["role"] == "user"

    out_msgs_1 = span1.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert len(out_msgs_1) == 1
    assert out_msgs_1[0]["finish_reason"] == "stop"

    # Second turn must only contain the new user message, not the second tool reply
    in_msgs_2 = span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in_msgs_2) == 1
    assert in_msgs_2[0]["role"] == "user"
    assert in_msgs_2[0]["parts"] == ({"type": "text", "content": "Thanks!"},)


@pytest.mark.asyncio
async def test_live_user_speech_and_interrupted_and_session_error(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, "true")
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 200,
                                mime_type="audio/pcm;rate=24000",
                            )
                        )
                    ],
                ),
                interrupted=True,
                turn_complete=True,
            )
        )
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-err-1")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            with pytest.raises(RuntimeError, match="websocket dropped"):
                async with live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025"
                ) as session:
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                    await session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=b"\x00\x00" * 100,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                    await session.send_realtime_input(
                        activity_end=genai_types.ActivityEnd()
                    )
                    await session._receive()
                    raise RuntimeError("websocket dropped")

    spans = span_exporter.get_finished_spans()
    span_by_op = {s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s for s in spans}
    assert "user_speech" in span_by_op
    assert "realtime_inference" in span_by_op

    speech_span = span_by_op["user_speech"]
    assert speech_span.name == "user_speech"
    assert speech_span.kind == SpanKind.INTERNAL
    assert speech_span.attributes["gen_ai.realtime_session.id"] == "sess-err-1"

    inf_span = span_by_op["realtime_inference"]
    assert inf_span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "interrupted",
    )

    logs = log_exporter.get_finished_logs()
    ended_log = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended_log.severity_number == SeverityNumber.WARN
    assert ended_log.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio
async def test_live_interleaved_events_speech_across_generations_and_error_spans(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, "true")
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(text="What is "),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="the weather?"
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                output_transcription=genai_types.Transcription(text="It is "),
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 100,
                                mime_type="audio/pcm",
                            )
                        )
                    ],
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                interrupted=True,
                turn_complete=True,
            ),
        ),
        # Turn 2: both user_speech and realtime_inference are active when receive raises
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                output_transcription=genai_types.Transcription(
                    text="Partial answer"
                ),
            )
        ),
        ConnectionError("mock connection failure"),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-interleaved")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
            )
            with pytest.raises(
                ConnectionError, match="mock connection failure"
            ):
                async with live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                    config=config,
                ) as session:
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                    # Drain Turn 1 (until turn_complete) while speech span remains open
                    async for _ in session.receive():
                        pass
                    # End Turn 1 speech after turn_complete
                    await session.send_realtime_input(
                        activity_end=genai_types.ActivityEnd()
                    )
                    # Turn 2 starts speech and generation, then receive raises ConnectionError
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                    async for _ in session.receive():
                        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4

    # First generation completed on interrupted turn_complete before activity_end arrived
    gen1_span = spans[0]
    assert (
        gen1_span.attributes[GenAI.GEN_AI_OPERATION_NAME]
        == "realtime_inference"
    )
    assert gen1_span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "interrupted",
    )
    gen1_in = gen1_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(gen1_in) == 1
    assert gen1_in[0]["role"] == "user"
    assert gen1_in[0]["parts"] == (
        {"type": "transcription", "content": "What is the weather?"},
    )

    # First speech span preserved across generation boundary and ended cleanly on activity_end
    speech1_span = spans[1]
    assert (
        speech1_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"
    )
    assert speech1_span.status.status_code == StatusCode.UNSET

    # Second speech span and second generation span both closed with error on receive failure
    error_spans = {
        s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s for s in spans[2:]
    }
    assert set(error_spans.keys()) == {"user_speech", "realtime_inference"}
    for err_span in error_spans.values():
        assert err_span.status.status_code == StatusCode.ERROR
        assert (
            err_span.attributes[ErrorAttributes.ERROR_TYPE]
            == "ConnectionError"
        )
    assert error_spans["realtime_inference"].attributes[
        GenAI.GEN_AI_RESPONSE_FINISH_REASONS
    ] == ("error",)


class _FakeWebSocket:
    """Deterministic fake websocket transport for public genai.Client replay tests."""

    def __init__(self, server_frames: list[dict[str, Any] | BaseException]):
        self._frames = list(server_frames)
        self.sent_frames: list[dict[str, Any]] = []

    async def send(self, payload: str | bytes) -> None:
        raw = (
            payload.decode("utf-8")
            if isinstance(payload, bytes)
            else str(payload)
        )
        self.sent_frames.append(json.loads(raw))

    async def recv(self, decode: bool = True) -> bytes | str:  # pylint: disable=unused-argument
        if not self._frames:
            return b"{}"
        item = self._frames.pop(0)
        if isinstance(item, BaseException):
            raise item
        encoded = json.dumps(item).encode("utf-8")
        return encoded if not decode else encoded.decode("utf-8")

    async def close(self) -> None:
        pass


def _make_fake_ws_connect(fake_ws: _FakeWebSocket):
    @contextlib.asynccontextmanager
    async def _ws_connect(
        uri: str,  # pylint: disable=unused-argument
        additional_headers: Any = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> AsyncIterator[_FakeWebSocket]:
        yield fake_ws

    return _ws_connect


@pytest.mark.asyncio
async def test_live_public_client_transport_replay_zero_settings_and_client_content(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, "true")
    server_frames: list[dict[str, Any] | BaseException] = [
        {"setupComplete": {}},
        {
            "voiceActivityDetectionSignal": {
                "vadSignalType": "VAD_SIGNAL_TYPE_SOS"
            }
        },
        {
            "voiceActivityDetectionSignal": {
                "vadSignalType": "VAD_SIGNAL_TYPE_EOS"
            }
        },
        {
            "serverContent": {
                "modelTurn": {
                    "role": "model",
                    "parts": [{"text": "Hello back!"}],
                },
                "turnComplete": True,
            },
            "usageMetadata": {
                "promptTokenCount": 12,
                "responseTokenCount": 4,
                "totalTokenCount": 16,
            },
        },
        # Trailing usage-only and completion-only frames must not start phantom generations
        {
            "usageMetadata": {
                "promptTokenCount": 12,
                "responseTokenCount": 4,
                "totalTokenCount": 16,
            }
        },
        {
            "serverContent": {
                "generationComplete": True,
                "turnComplete": True,
            }
        },
    ]
    fake_ws = _FakeWebSocket(server_frames)

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(
                api_key="test-key",
                vertexai=False,
                http_options=genai_types.HttpOptions(
                    base_url="https://custom-live.example.com:8443",
                    api_version="v1beta",
                ),
            )
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                temperature=0.0,
                top_p=0.0,
                top_k=0.0,
                seed=0,
                max_output_tokens=0,
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                if not _HAS_SERVER_VAD:
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                    await session.send_realtime_input(
                        activity_end=genai_types.ActivityEnd()
                    )
                # 1) Single Content object
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="First turn")],
                    ),
                    turn_complete=False,
                )
                # 2) Content dictionary
                await session.send_client_content(
                    turns={
                        "role": "user",
                        "parts": [{"text": "Second turn"}],
                    },
                    turn_complete=False,
                )
                # 3) List containing both Content and dict
                await session.send_client_content(
                    turns=[
                        genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text="Third turn")],
                        ),
                        {"role": "user", "parts": [{"text": "Fourth turn"}]},
                    ],
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass
                # Consume trailing usage-only and completion-only frames; should not create extra spans
                await session._receive()
                await session._receive()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    speech_span, span = spans[0], spans[1]
    assert speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"

    attrs = dict(span.attributes)
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "realtime_inference"
    assert attrs[GenAI.GEN_AI_OUTPUT_TYPE] == "text"
    assert attrs[ServerAttributes.SERVER_ADDRESS] == "custom-live.example.com"
    assert attrs[ServerAttributes.SERVER_PORT] == 8443
    assert attrs[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 0
    # Sampling parameters not in gen_ai.realtime_inference.client must not be set
    assert GenAI.GEN_AI_REQUEST_TEMPERATURE not in attrs
    assert GenAI.GEN_AI_REQUEST_TOP_P not in attrs
    assert GenAI.GEN_AI_REQUEST_TOP_K not in attrs
    assert GenAI.GEN_AI_REQUEST_SEED not in attrs

    in_msgs = attrs[GenAI.GEN_AI_INPUT_MESSAGES]
    texts = [m["parts"][0]["content"] for m in in_msgs]
    assert texts == ["First turn", "Second turn", "Third turn", "Fourth turn"]


@pytest.mark.asyncio
async def test_live_no_content_capture_skips_payload_buffering_and_session_http_options(
    span_exporter,
    log_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "NO_CONTENT"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, "true")
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        *_server_vad_messages(),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="Sensitive user speech"
                ),
                model_turn=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            thought=True,
                            text="Sensitive reasoning",
                        ),
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b"\x01\x02" * 400,
                                mime_type="audio/pcm",
                            )
                        ),
                    ],
                ),
                output_transcription=genai_types.Transcription(
                    text="Sensitive spoken response"
                ),
            ),
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=20,
                response_token_count=10,
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-no-content")

    created_trackers: list[live_module._LiveSessionTracker] = []
    orig_tracker_init = live_module._LiveSessionTracker.__init__

    def _spy_tracker_init(self, *args, **kwargs):
        orig_tracker_init(self, *args, **kwargs)
        created_trackers.append(self)

    with (
        patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)),
        patch.object(
            live_module._LiveSessionTracker, "__init__", _spy_tracker_init
        ),
        patch.object(
            live_module,
            "_extract_blob_bytes_and_mime",
            wraps=live_module._extract_blob_bytes_and_mime,
        ) as spy_extract_blob,
        patch.object(
            live_module,
            "to_input_messages",
            wraps=live_module.to_input_messages,
        ) as spy_to_input_messages,
        patch.object(
            live_module,
            "_to_part",
            wraps=live_module._to_part,
        ) as spy_to_part,
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            config = {
                "response_modalities": ["AUDIO"],
                "system_instruction": "Secret system instruction",
                "http_options": {
                    "base_url": "https://session-override.example.com:9443"
                },
            }
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                tracker = created_trackers[0]
                assert tracker.system_instruction == []
                assert tracker.tool_definitions == []

                if not _HAS_SERVER_VAD:
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x01" * 400,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                assert not tracker._pending_audio_chunks
                assert tracker._pending_input_messages == []

                if not _HAS_SERVER_VAD:
                    await session.send_realtime_input(
                        activity_end=genai_types.ActivityEnd()
                    )
                else:
                    await session.send_realtime_input(audio_stream_end=True)
                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": "Secret text"}]}
                )
                assert tracker._pending_input_messages == []

                # Receive VAD (if _HAS_SERVER_VAD) + content chunk (before turn_complete)
                if _HAS_SERVER_VAD:
                    await session._receive()
                    await session._receive()
                await session._receive()
                assert tracker._pending_input_transcript_deltas == []
                assert tracker._active_gen is not None
                assert tracker._active_gen.input_messages == []
                assert tracker._active_gen.input_transcript_deltas == []
                assert tracker._active_gen.output_transcript_deltas == []
                assert tracker._active_gen.output_thought_deltas == []
                assert tracker._active_gen.output_text_deltas == []
                assert tracker._active_gen.output_audio_chunks == []

                # Receive turn_complete
                await session._receive()

        spy_extract_blob.assert_not_called()
        spy_to_input_messages.assert_not_called()
        spy_to_part.assert_not_called()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    speech_span, inf_span = spans[0], spans[1]
    assert speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"

    attrs = dict(inf_span.attributes)
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "realtime_inference"
    assert (
        attrs[ServerAttributes.SERVER_ADDRESS]
        == "session-override.example.com"
    )
    assert attrs[ServerAttributes.SERVER_PORT] == 9443
    assert attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 20
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 10
    assert GenAI.GEN_AI_INPUT_MESSAGES not in attrs
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs
    assert GenAI.GEN_AI_SYSTEM_INSTRUCTIONS not in attrs
    assert GenAI.GEN_AI_TOOL_DEFINITIONS not in attrs


@pytest.mark.asyncio
async def test_live_barge_in_overlapping_speech_and_transcription(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        # Turn 1 input transcript + model starts speaking
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="Tell me about Paris."
                ),
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="Paris is the capital ")],
                ),
            )
        ),
        # While Turn 1 is still active, user barges in; server sends Turn 2's input_transcription
        # before Turn 1's interrupted/turn_complete frame arrives.
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="Stop, tell me about Rome instead."
                ),
            )
        ),
        # Turn 1 ends as interrupted
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                interrupted=True,
                turn_complete=True,
            )
        ),
        # Turn 2 model response
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="Rome is in Italy.")],
                ),
                turn_complete=True,
            )
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-barge-in")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Turn 1 user audio
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x01" * 100,
                        mime_type="audio/pcm",
                    ),
                )
                await session.send_realtime_input(audio_stream_end=True)
                # Receive Turn 1 model output start
                await session._receive()
                # User barges in with Turn 2 audio while Turn 1 is still active
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x02\x03" * 100,
                        mime_type="audio/pcm",
                    ),
                )
                await session.send_realtime_input(audio_stream_end=True)
                # Receive Turn 2 input_transcription (while Turn 1 active), Turn 1 interrupted, Turn 2 response
                await session._receive()
                await session._receive()
                await session._receive()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]

    in1 = span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in1) == 1
    assert in1[0]["parts"][1] == {
        "type": "transcription",
        "content": "Tell me about Paris.",
    }

    in2 = span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in2) == 1
    assert in2[0]["parts"][0]["type"] == "blob"
    assert in2[0]["parts"][1] == {
        "type": "transcription",
        "content": "Stop, tell me about Rome instead.",
    }


@pytest.mark.asyncio
async def test_live_generation_complete_before_turn_complete_preserves_end_timestamp(
    span_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="Done speaking.")],
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                generation_complete=True,
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                turn_complete=True,
            ),
            usage_metadata=genai_types.UsageMetadata(
                prompt_token_count=15,
                response_token_count=5,
            ),
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-gen-end")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session._receive()
                await session._receive()
                # Delay between generation_complete and turn_complete must not inflate duration
                await asyncio.sleep(0.06)
                await session._receive()

    (span,) = span_exporter.get_finished_spans()
    span_duration_s = (span.end_time - span.start_time) / 1e9
    assert span_duration_s < 0.05
    assert span.attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert span.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 5

    metrics_data = metric_reader.get_metrics_data()
    duration_dp = [
        dp
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "gen_ai.client.operation.duration"
        for dp in m.data.data_points
    ]
    assert len(duration_dp) == 1
    assert duration_dp[0].sum < 0.05


@pytest.mark.asyncio
async def test_live_session_closed_before_generation_completes_fails_span(
    span_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="Partial output...")],
                ),
            )
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-early-exit")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Receive first chunk, then exit session before completion frame
                await session._receive()

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "_OTHER"
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("error",)

    metrics_data = metric_reader.get_metrics_data()
    duration_dp = [
        dp
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "gen_ai.client.operation.duration"
        for dp in m.data.data_points
    ]
    assert len(duration_dp) == 1
    assert duration_dp[0].attributes[ErrorAttributes.ERROR_TYPE] == "_OTHER"


@pytest.mark.asyncio
async def test_live_caught_receive_error_marks_session_ended_event_as_error(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="Starting...")],
                ),
            )
        ),
        ConnectionError("terminal transport failure"),
    ]
    fake_session = _FakeAsyncSession(messages, session_id="sess-caught-err")

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                with pytest.raises(
                    ConnectionError, match="terminal transport failure"
                ):
                    async for _ in session.receive():
                        pass

    logs = log_exporter.get_finished_logs()
    ended_log = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended_log.severity_number == SeverityNumber.WARN
    assert (
        ended_log.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    )


@pytest.mark.asyncio
async def test_live_fallback_session_id_and_continuous_audio_coalescing(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    messages: list[genai_types.LiveServerMessage | BaseException] = [
        # Turn 1: multiple model output chunks while mic audio is streaming concurrently
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="First chunk ")],
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="second chunk ")],
                ),
            )
        ),
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="done.")],
                ),
                turn_complete=True,
            )
        ),
        # Turn 2: model responds to the continuous audio sent during Turn 1
        genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                input_transcription=genai_types.Transcription(
                    text="Follow up question"
                ),
                model_turn=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="Follow up answer.")],
                ),
                turn_complete=True,
            )
        ),
    ]
    fake_session = _FakeAsyncSession(messages, session_id=None)

    with patch.object(AsyncLive, "connect", _make_fake_connect(fake_session)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            live = AsyncLive(api_client=None)  # type: ignore[arg-type]
            async with live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Start Turn 1
                await session._receive()
                # Continuous mic audio arrives interleaved with Turn 1 server chunks
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x01\x02", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session._receive()
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x03\x04", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session._receive()  # Turn 1 completes
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x05\x06", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session._receive()  # Turn 2 completes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]

    session_id = span1.attributes.get("gen_ai.realtime_session.id")
    assert isinstance(session_id, str) and len(session_id) > 0
    assert span2.attributes.get("gen_ai.realtime_session.id") == session_id

    logs = log_exporter.get_finished_logs()
    for log in logs:
        assert (
            log.log_record.attributes.get("gen_ai.realtime_session.id")
            == session_id
        )

    # Turn 1 had no prior input messages; all mic audio sent during Turn 1 belongs
    # to Turn 2 as a single coalesced user InputMessage.
    assert GenAI.GEN_AI_INPUT_MESSAGES not in span1.attributes
    turn2_in = span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(turn2_in) == 1
    assert turn2_in[0]["role"] == "user"
    assert [p["type"] for p in turn2_in[0]["parts"]] == [
        "blob",
        "transcription",
    ]
    assert turn2_in[0]["parts"][0]["content"] == b"\x01\x02\x03\x04\x05\x06"
    assert turn2_in[0]["parts"][1]["content"] == "Follow up question"


@pytest.mark.asyncio
async def test_live_public_client_tool_exchange_2_5_and_3_1_and_callable_tools(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )

    def get_weather(city: str) -> dict[str, Any]:
        """Get weather for a city."""
        return {"city": city, "temp_c": 21}

    ws_25 = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "inputTranscription": {"text": "Weather in Paris?"},
                }
            },
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-paris-1",
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    ]
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "It is 21C in Paris."}],
                    },
                    "turnComplete": True,
                },
                "usageMetadata": {
                    "promptTokenCount": 40,
                    "responseTokenCount": 12,
                },
            },
        ]
    )
    ws_31 = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-tokyo-1",
                            "name": "get_weather",
                            "args": {"city": "Tokyo"},
                        },
                        {
                            "id": "call-osaka-2",
                            "name": "get_weather",
                            "args": {"city": "Osaka"},
                        },
                    ]
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Tokyo and Osaka are both 21C."}],
                    },
                    "turnComplete": True,
                },
                "usageMetadata": {
                    "promptTokenCount": 55,
                    "responseTokenCount": 16,
                },
            },
        ]
    )
    sockets = [ws_25, ws_31]

    @contextlib.asynccontextmanager
    async def _multi_ws_connect(
        uri: str,  # pylint: disable=unused-argument
        additional_headers: Any = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> AsyncIterator[_FakeWebSocket]:
        yield sockets.pop(0)

    with patch("google.genai.live.ws_connect", _multi_ws_connect):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            has_live_thinking = (
                "thinking_config" in genai_types.LiveConnectConfig.model_fields
                and hasattr(genai_types, "ThinkingLevel")
            )
            config_kwargs: dict[str, Any] = {
                "response_modalities": [genai_types.Modality.TEXT],
                "tools": [get_weather],
            }
            if has_live_thinking:
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_level=genai_types.ThinkingLevel.HIGH
                )
            config = genai_types.LiveConnectConfig(**config_kwargs)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x01" * 64,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                async for msg in session.receive():
                    if msg.tool_call:
                        await session.send_tool_response(
                            function_responses=[
                                genai_types.FunctionResponse(
                                    id="call-paris-1",
                                    name="get_weather",
                                    response={"temp_c": 21},
                                )
                            ]
                        )

            async with client.aio.live.connect(
                model="gemini-3.1-flash-live-preview",
                config=config,
            ) as session:
                await session.send(
                    input="Compare Tokyo and Osaka.", end_of_turn=True
                )
                async for msg in session.receive():
                    if msg.tool_call:
                        for fc in msg.tool_call.function_calls or []:
                            await session.send_tool_response(
                                function_responses=[
                                    genai_types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"temp_c": 21},
                                    )
                                ]
                            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    s25_1, s25_2, s31 = spans

    if has_live_thinking:
        assert s25_1.attributes["gen_ai.request.reasoning.level"] == "HIGH"
    assert s25_1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    tools_defs = s25_1.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    assert tools_defs[0]["name"] == "get_weather"

    assert s25_2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    s25_2_in = s25_2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert s25_2_in[0]["parts"][0]["type"] == "tool_call_response"

    if has_live_thinking:
        assert s31.attributes["gen_ai.request.reasoning.level"] == "HIGH"
    assert s31.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    s31_in = s31.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(s31_in) == 1
    assert s31_in[0]["parts"] == (
        {"type": "text", "content": "Compare Tokyo and Osaka."},
    )


@pytest.mark.asyncio
async def test_live_public_client_barge_in_interruption_and_receive_error(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "inputTranscription": {"text": "Tell me about Paris."},
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Paris is the capital "}],
                    },
                }
            },
            # User barges in: server sends Turn 2 inputTranscription before Turn 1 interrupted/turnComplete
            {
                "serverContent": {
                    "inputTranscription": {
                        "text": "Stop, tell me about Rome."
                    },
                }
            },
            {
                "serverContent": {
                    "interrupted": True,
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Rome is in Italy..."}],
                    },
                }
            },
            ConnectionError("transport lost mid-turn"),
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x01" * 32, mime_type="audio/pcm"
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)

                seen_chunks = 0
                async for _ in session.receive():
                    seen_chunks += 1
                    if seen_chunks == 1:
                        # Barge-in audio sent while Turn 1 is active
                        await session.send_realtime_input(
                            audio=genai_types.Blob(
                                data=b"\x02\x03" * 32, mime_type="audio/pcm"
                            )
                        )
                        await session.send_realtime_input(
                            audio_stream_end=True
                        )

                with pytest.raises(
                    ConnectionError, match="transport lost mid-turn"
                ):
                    async for _ in session.receive():
                        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]

    assert span1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "interrupted",
    )
    in1 = span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert in1[0]["parts"][1]["content"] == "Tell me about Paris."

    assert span2.status.status_code == StatusCode.ERROR
    assert span2.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    assert span2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("error",)
    in2 = span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert in2[0]["parts"][1]["content"] == "Stop, tell me about Rome."

    logs = log_exporter.get_finished_logs()
    ended = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended.severity_number == SeverityNumber.WARN
    assert ended.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio
async def test_live_transactional_sends_failed_requests_retries_and_session_send(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, "true")
    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            *[
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": f"Reply {idx}."}],
                        },
                        "turnComplete": True,
                    }
                }
                for idx in range(1, 6)
            ],
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # 1. Rejected send_realtime_input (multiple args) must not record input or start speech
                with pytest.raises(ValueError):
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart(),
                        text="Rejected text",
                    )
                # Retry with valid send_realtime_input
                await session.send_realtime_input(text="Valid realtime text")

                # 2. Rejected send_client_content (malformed dict in list) raises SDK error, not recorded
                with pytest.raises(Exception):
                    await session.send_client_content(
                        turns=[{"parts": "invalid-not-a-list"}]  # type: ignore[list-item]
                    )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Valid client content")],
                    )
                )

                # 3. Rejected send_tool_response (missing id on non-Vertex) raises ValueError, not recorded
                with pytest.raises(ValueError):
                    await session.send_tool_response(
                        function_responses=genai_types.FunctionResponse(
                            name="fn", response={"rejected": True}
                        )
                    )
                await session.send_tool_response(
                    function_responses=genai_types.FunctionResponse(
                        id="call-ok",
                        name="fn",
                        response={"accepted": True},
                    )
                )

                # 4. Rejected session.send(input=12345) followed by valid session.send calls
                with pytest.raises(ValueError):
                    await session.send(input=12345)  # type: ignore[arg-type]
                await session.send(input="Sent via send()", end_of_turn=True)
                await session.send(
                    input=genai_types.Blob(
                        data=b"\xaa\xbb", mime_type="image/png"
                    )
                )
                await session.send(
                    input=genai_types.FunctionResponse(
                        id="call-send-fn",
                        name="fn",
                        response={"via": "send"},
                    )
                )

                for _ in range(5):
                    async for _ in session.receive():
                        pass

                # 5. Transport failure during send_realtime_input(activity_end=...) fails open speech span
                await session.send_realtime_input(
                    activity_start=genai_types.ActivityStart()
                )
                with patch.object(
                    fake_ws,
                    "send",
                    side_effect=ConnectionError("ws send failed"),
                ):
                    with pytest.raises(
                        ConnectionError, match="ws send failed"
                    ):
                        await session.send_realtime_input(
                            activity_end=genai_types.ActivityEnd()
                        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 6
    inf_spans = spans[:5]
    speech_span = spans[5]

    assert inf_spans[0].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Valid realtime text"},),
        },
    )
    assert inf_spans[1].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Valid client content"},),
        },
    )
    in2 = inf_spans[2].attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in2) == 1
    assert in2[0]["parts"][0]["id"] == "call-ok"
    assert inf_spans[3].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Sent via send()"},),
        },
    )
    in4 = inf_spans[4].attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in4) == 2
    assert in4[0]["parts"][0]["modality"] == "image"
    assert in4[1]["parts"][0]["id"] == "call-send-fn"

    assert speech_span.name == "user_speech"
    assert speech_span.status.status_code == StatusCode.ERROR
    assert (
        speech_span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    )

    logs = log_exporter.get_finished_logs()
    ended = next(
        l.log_record
        for l in logs
        if l.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended.severity_number == SeverityNumber.WARN
    assert ended.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio
async def test_live_send_context_update_snapshot_and_missing_method_branch(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    def tool_v1(x: str) -> str:
        """Tool v1."""
        return x

    def tool_v2(y: str) -> str:
        """Tool v2."""
        return y

    has_context_update = hasattr(
        genai_types, "LiveClientContextUpdate"
    ) and hasattr(genai_live.AsyncSession, "send_context_update")

    if not has_context_update:
        # Oldest SDK branch: verify AsyncSession has no send_context_update and wrapping succeeds
        fake_ws = _FakeWebSocket([{"setupComplete": {}}])
        with patch(
            "google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)
        ):
            with instrument(
                GoogleGenAiSdkInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
            ):
                client = genai.Client(api_key="test-key", vertexai=False)
                async with client.aio.live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                ) as session:
                    assert not hasattr(session, "send_context_update")
        return

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 1 streaming..."}],
                    },
                }
            },
            {
                "serverContent": {
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 2 with updated context."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Initial system instruction",
                tools=[tool_v1],
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                # Start Turn 1 (first chunk)
                await session._receive()
                # Mid-generation context update: Turn 1 must keep Initial instruction & tool_v1,
                # while Turn 2 gets Updated instruction & tool_v2.
                await session.send_context_update(
                    system_instruction="Updated system instruction",
                    tools=[tool_v2],
                )
                # Failed context update (both args None raises ValueError) must not overwrite Turn 2's config
                with pytest.raises(ValueError):
                    await session.send_context_update()
                # Finish Turn 1
                await session._receive()
                # Run Turn 2
                await session._receive()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]

    sys1 = span1.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS]
    tools1 = span1.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    assert sys1 == ({"type": "text", "content": "Initial system instruction"},)
    assert [t["name"] for t in tools1] == ["tool_v1"]

    sys2 = span2.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS]
    tools2 = span2.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    assert sys2 == ({"type": "text", "content": "Updated system instruction"},)
    assert [t["name"] for t in tools2] == ["tool_v2"]

    # Also verify send_context_update with NO_CONTENT and EVENT_ONLY does not record instructions/tools
    span_exporter.clear()
    for mode in ("NO_CONTENT", "EVENT_ONLY"):
        monkeypatch.setenv(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, mode
        )
        ws_nc = _FakeWebSocket(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": "Silent"}],
                        },
                        "turnComplete": True,
                    }
                },
            ]
        )
        with patch(
            "google.genai.live.ws_connect", _make_fake_ws_connect(ws_nc)
        ):
            with instrument(
                GoogleGenAiSdkInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
            ):
                client = genai.Client(api_key="test-key", vertexai=False)
                async with client.aio.live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                ) as session:
                    await session.send_context_update(
                        system_instruction="Should not be captured",
                        tools=[tool_v1],
                    )
                    await session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=b"\x01\x02", mime_type="audio/pcm"
                        )
                    )
                    async for _ in session.receive():
                        pass

    nc_spans = span_exporter.get_finished_spans()
    assert len(nc_spans) == 2
    for s in nc_spans:
        assert GenAI.GEN_AI_INPUT_MESSAGES not in s.attributes
        assert GenAI.GEN_AI_OUTPUT_MESSAGES not in s.attributes
        assert GenAI.GEN_AI_SYSTEM_INSTRUCTIONS not in s.attributes
        assert GenAI.GEN_AI_TOOL_DEFINITIONS not in s.attributes


@pytest.mark.asyncio
async def test_live_in_progress_turn_complete_does_not_freeze_mark_end(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Step 1..."}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Step 2 done."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )
    completion_calls = [False, True]

    with (
        patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)),
        patch.object(
            live_module,
            "_is_interaction_complete",
            side_effect=lambda sc: (
                completion_calls.pop(0)
                if sc is not None and completion_calls
                else False
            ),
        ),
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Frame 1 has turnComplete=True, but _is_interaction_complete is False (in progress)
                await session._receive()
                await asyncio.sleep(0.05)
                # Frame 2 completes the interaction
                await session._receive()

    (span,) = span_exporter.get_finished_spans()
    span_duration_s = (span.end_time - span.start_time) / 1e9
    assert span_duration_s >= 0.04


@pytest.mark.asyncio
async def test_live_bounded_media_buffering_during_prolonged_silence(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    import base64

    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    out_chunks = [b"11111111", b"22222222", b"33333333"]
    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            *[
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/pcm;rate=24000",
                                        "data": base64.b64encode(chunk).decode(
                                            "ascii"
                                        ),
                                    }
                                }
                            ],
                        },
                        "turnComplete": idx == len(out_chunks) - 1,
                    }
                }
                for idx, chunk in enumerate(out_chunks)
            ],
        ]
    )

    with (
        patch.object(live_module, "_MAX_BUFFERED_MEDIA_BYTES", 16),
        patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)),
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Prolonged silence: repeated audio chunks and idle video
                # frames must not grow pending media past cap.
                for chunk in (b"AAAA", b"BBBB", b"CCCC", b"DDDD"):
                    await session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=chunk, mime_type="audio/pcm;rate=16000"
                        )
                    )
                await session.send_realtime_input(
                    video=genai_types.Blob(
                        data=b"VVVVVVVV", mime_type="image/jpeg"
                    )
                )
                # Final spoken utterance immediately preceding generation.
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"01234567", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"89abcdef", mime_type="audio/pcm;rate=16000"
                    )
                )
                async for _ in session.receive():
                    pass

    (span,) = span_exporter.get_finished_spans()
    in_msgs = span.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in_msgs) == 1
    in_blob = in_msgs[0]["parts"][0]
    assert in_blob["type"] == "blob"
    assert in_blob["modality"] == "audio"
    assert len(in_blob["content"]) <= 16
    assert in_blob["content"] == b"0123456789abcdef"

    out_msgs = span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert len(out_msgs) == 1
    out_blob = out_msgs[0]["parts"][0]
    assert out_blob["type"] == "blob"
    assert out_blob["modality"] == "audio"
    assert len(out_blob["content"]) == 16
    assert out_blob["content"] == b"1111111122222222"


@pytest.mark.asyncio
async def test_live_concurrent_receive_flushes_inflight_send_commit(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    frame_written = asyncio.Event()
    release_send = asyncio.Event()

    class _YieldingSendWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            await super().send(payload)
            frame = self.sent_frames[-1]
            if (
                "clientContent" in frame or "client_content" in frame
            ) and not frame_written.is_set():
                # Simulate writing the frame to the wire, then yielding while
                # draining while the concurrent receive loop processes Turn 1.
                frame_written.set()
                await asyncio.wait_for(release_send.wait(), timeout=5.0)

        async def recv(self, decode: bool = True) -> bytes | str:
            if len(self._frames) == 2:
                await asyncio.wait_for(frame_written.wait(), timeout=5.0)
            return await super().recv(decode=decode)

    fake_ws = _YieldingSendWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 1 answer"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 2 answer"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:

                async def _receive_turn() -> None:
                    async for _ in session.receive():
                        pass

                send_task = asyncio.create_task(
                    session.send_client_content(
                        turns=genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text="Turn 1 prompt")],
                        ),
                        turn_complete=True,
                    )
                )
                recv_task = asyncio.create_task(_receive_turn())
                # Turn 1 completes while send_task is still suspended inside ws.send.
                await recv_task
                release_send.set()
                await send_task

                # Turn 2 must only see Turn 2's prompt (no leak from Turn 1).
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 2 prompt")],
                    ),
                    turn_complete=True,
                )
                await _receive_turn()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]
    assert span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 1 prompt"},),
        },
    )
    assert span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 2 prompt"},),
        },
    )


@pytest.mark.asyncio
async def test_live_camelcase_dict_config_aliases(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    import inspect

    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    has_thinking_level = (
        "thinking_config" in genai_types.LiveConnectConfig.model_fields
        and hasattr(genai_types, "ThinkingConfig")
        and "thinking_level" in genai_types.ThinkingConfig.model_fields
    )
    supports_config_http_options = "does not support" not in inspect.getsource(
        genai_live.AsyncLive.connect
    )

    ws1 = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Camel reply 1"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )
    ws2 = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Camel reply 2"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )
    sockets = [ws1, ws2]

    @contextlib.asynccontextmanager
    async def _multi_ws_connect(
        uri: str,  # pylint: disable=unused-argument
        additional_headers: Any = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> AsyncIterator[_FakeWebSocket]:
        yield sockets.pop(0)

    config1: dict[str, Any] = {
        "responseModalities": ["AUDIO"],
        "maxOutputTokens": 128,
        "systemInstruction": "Be concise.",
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "lookup_camel",
                        "description": "Camel tool",
                    }
                ]
            }
        ],
    }
    if supports_config_http_options:
        config1["httpOptions"] = {
            "baseUrl": "https://custom-camel.example.com:8443"
        }
    if has_thinking_level:
        config1["thinkingConfig"] = {"thinkingLevel": "HIGH"}

    config2: dict[str, Any] = {
        "generationConfig": {
            "responseModalities": ["TEXT"],
            "maxOutputTokens": 64,
        }
    }

    with patch("google.genai.live.ws_connect", _multi_ws_connect):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client_kwargs: dict[str, Any] = {
                "api_key": "test-key",
                "vertexai": False,
            }
            if not supports_config_http_options:
                client_kwargs["http_options"] = {
                    "baseUrl": "https://custom-camel.example.com:8443"
                }
            client = genai.Client(**client_kwargs)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config1,  # type: ignore[arg-type]
            ) as session:
                async for _ in session.receive():
                    pass

            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config2,  # type: ignore[arg-type]
            ) as session:
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    attrs1 = dict(spans[0].attributes)
    assert (
        attrs1[GenAI.GEN_AI_OUTPUT_TYPE]
        == GenAI.GenAiOutputTypeValues.SPEECH.value
    )
    assert attrs1[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 128
    assert (
        attrs1[ServerAttributes.SERVER_ADDRESS] == "custom-camel.example.com"
    )
    assert attrs1[ServerAttributes.SERVER_PORT] == 8443
    assert attrs1[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "Be concise."},
    )
    assert [t["name"] for t in attrs1[GenAI.GEN_AI_TOOL_DEFINITIONS]] == [
        "lookup_camel"
    ]
    if has_thinking_level:
        assert attrs1["gen_ai.request.reasoning.level"] == "HIGH"

    attrs2 = dict(spans[1].attributes)
    assert (
        attrs2[GenAI.GEN_AI_OUTPUT_TYPE]
        == GenAI.GenAiOutputTypeValues.TEXT.value
    )
    assert attrs2[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 64


@pytest.mark.asyncio
async def test_live_nested_event_only_capture_policy(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "EVENT_ONLY"
    )

    def lookup_info(topic: str) -> str:
        """Lookup info."""
        return topic

    def _make_ws() -> _FakeWebSocket:
        return _FakeWebSocket(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": "First frame reply"}],
                        },
                        "turnComplete": True,
                    },
                    "usageMetadata": {
                        "promptTokenCount": 9,
                        "responseTokenCount": 4,
                    },
                },
            ]
        )

    sockets = [_make_ws(), _make_ws()]

    @contextlib.asynccontextmanager
    async def _multi_ws_connect(
        uri: str,  # pylint: disable=unused-argument
        additional_headers: Any = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> AsyncIterator[_FakeWebSocket]:
        yield sockets.pop(0)

    outer_handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    with patch("google.genai.live.ws_connect", _multi_ws_connect):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Nested system prompt",
                tools=[lookup_info],
            )

            # Case 1: outer inference opts into content capture under EVENT_ONLY.
            with outer_handler.inference(
                GenAI.GenAiProviderNameValues.GCP_GEMINI.value,
                request_model="outer-model",
            ):
                ctx_data = get_value(CLIENT_INFERENCE_CONTEXT_KEY)
                assert isinstance(ctx_data, InferenceData)
                ctx_data.capture_content = True
                async with client.aio.live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                    config=config,
                ) as session:
                    await session.send_client_content(
                        turns=genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text="Nested prompt")],
                        ),
                        turn_complete=True,
                    )
                    async for _ in session.receive():
                        pass

            # Case 2: outer inference leaves capture_content=False under EVENT_ONLY.
            with outer_handler.inference(
                GenAI.GenAiProviderNameValues.GCP_GEMINI.value,
                request_model="outer-model-disabled",
            ):
                async with client.aio.live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                    config=config,
                ) as session:
                    await session.send_client_content(
                        turns=genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text="Suppressed prompt")],
                        ),
                        turn_complete=True,
                    )
                    async for _ in session.receive():
                        pass

    # Both inner realtime_inference spans are suppressed by outer inference spans.
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(
        s.attributes[GenAI.GEN_AI_OPERATION_NAME] != "realtime_inference"
        for s in spans
    )

    detail_events = [
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name
        == "gen_ai.client.inference.operation.details"
    ]
    assert len(detail_events) == 2

    enabled_attrs = dict(detail_events[0].attributes)
    assert enabled_attrs[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Nested prompt"},),
        },
    )
    assert enabled_attrs[GenAI.GEN_AI_OUTPUT_MESSAGES] == (
        {
            "role": "assistant",
            "parts": ({"type": "text", "content": "First frame reply"},),
            "finish_reason": "stop",
        },
    )
    assert enabled_attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "Nested system prompt"},
    )
    assert [
        t["name"] for t in enabled_attrs[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["lookup_info"]

    disabled_attrs = dict(detail_events[1].attributes)
    assert GenAI.GEN_AI_INPUT_MESSAGES not in disabled_attrs
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in disabled_attrs
    assert GenAI.GEN_AI_SYSTEM_INSTRUCTIONS not in disabled_attrs
    assert GenAI.GEN_AI_TOOL_DEFINITIONS not in disabled_attrs


@pytest.mark.asyncio
async def test_live_send_transport_error_during_active_generation_fails_inference_immediately(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Partial chunk..."}],
                    },
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Start an active realtime_inference generation.
                await session._receive()
                # Terminal send failure must finalize the active generation
                # immediately with ConnectionError rather than waiting for close.
                with patch.object(
                    fake_ws,
                    "send",
                    side_effect=ConnectionError("send broken mid-turn"),
                ):
                    with pytest.raises(
                        ConnectionError, match="send broken mid-turn"
                    ):
                        await session.send_realtime_input(
                            audio=genai_types.Blob(
                                data=b"\x01\x02", mime_type="audio/pcm"
                            )
                        )
                # Delay before exiting session must not inflate span duration.
                await asyncio.sleep(0.06)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "realtime_inference"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("error",)
    span_duration_s = (span.end_time - span.start_time) / 1e9
    assert span_duration_s < 0.05

    ended = next(
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended.severity_number == SeverityNumber.WARN
    assert ended.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio
async def test_live_setup_handshake_failure_and_receive_cancellation(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    tracer = tracer_provider.get_tracer(__name__)

    @contextlib.asynccontextmanager
    async def _failing_ws_connect(
        uri: str,  # pylint: disable=unused-argument
        additional_headers: Any = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> AsyncIterator[_FakeWebSocket]:
        raise ConnectionError("dial refused")
        yield _FakeWebSocket([])  # pragma: no cover

    # Part 1: websocket connect & pre-setupComplete handshake failures emit no
    # spans or lifecycle events, propagate the original error, and keep context.
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        client = genai.Client(api_key="test-key", vertexai=False)
        with tracer.start_as_current_span("caller-setup-fail") as caller_span:
            with patch("google.genai.live.ws_connect", _failing_ws_connect):
                with pytest.raises(ConnectionError, match="dial refused"):
                    async with client.aio.live.connect(
                        model="gemini-2.5-flash-native-audio-preview-12-2025",
                    ):
                        pass
            assert get_current_span() is caller_span

            handshake_ws = _FakeWebSocket(
                [ConnectionError("handshake dropped before setupComplete")]
            )
            with patch(
                "google.genai.live.ws_connect",
                _make_fake_ws_connect(handshake_ws),
            ):
                with pytest.raises(
                    ConnectionError,
                    match="handshake dropped before setupComplete",
                ):
                    async with client.aio.live.connect(
                        model="gemini-2.5-flash-native-audio-preview-12-2025",
                    ):
                        pass
            assert get_current_span() is caller_span

    live_spans_after_setup = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name != "caller-setup-fail"
    ]
    assert live_spans_after_setup == []
    assert not log_exporter.get_finished_logs()
    span_exporter.clear()

    # Part 2: cancelling a receive task while both user_speech and
    # realtime_inference are active propagates CancelledError unmodified,
    # finalizes both spans once without exception events, emits ended WARN,
    # and restores caller context.
    recv_blocked = asyncio.Event()
    never_set = asyncio.Event()

    class _BlockingSecondRecvWebSocket(_FakeWebSocket):
        async def recv(self, decode: bool = True) -> bytes | str:
            if not self._frames:
                recv_blocked.set()
                await never_set.wait()
            return await super().recv(decode=decode)

    cancel_ws = _BlockingSecondRecvWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Speaking when cancelled..."}],
                    },
                }
            },
        ]
    )

    with patch(
        "google.genai.live.ws_connect", _make_fake_ws_connect(cancel_ws)
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            extra_env={OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true"},
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            with tracer.start_as_current_span("caller-cancel") as caller_span:
                context_after_cancel: Any = None

                async def _run_live_session() -> None:
                    nonlocal context_after_cancel
                    try:
                        async with client.aio.live.connect(
                            model="gemini-2.5-flash-native-audio-preview-12-2025",
                        ) as session:
                            await session.send_realtime_input(
                                activity_start=genai_types.ActivityStart()
                            )
                            async for _ in session.receive():
                                pass
                    finally:
                        context_after_cancel = get_current_span()

                task = asyncio.create_task(_run_live_session())
                await asyncio.wait_for(recv_blocked.wait(), timeout=5.0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

                assert context_after_cancel is caller_span
                assert get_current_span() is caller_span

    finished_live_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.name != "caller-cancel"
    ]
    assert len(finished_live_spans) == 2
    by_op = {
        s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s
        for s in finished_live_spans
    }
    assert set(by_op.keys()) == {"user_speech", "realtime_inference"}
    expected_cancel_error_type = (
        f"{asyncio.CancelledError.__module__}."
        f"{asyncio.CancelledError.__qualname__}"
    )
    for op_span in by_op.values():
        assert op_span.status.status_code == StatusCode.ERROR
        assert (
            op_span.attributes[ErrorAttributes.ERROR_TYPE]
            == expected_cancel_error_type
        )
        assert op_span.events == ()
    assert by_op["realtime_inference"].attributes[
        GenAI.GEN_AI_RESPONSE_FINISH_REASONS
    ] == ("error",)

    logs = [log.log_record for log in log_exporter.get_finished_logs()]
    assert [log.event_name for log in logs] == [
        "gen_ai.client.realtime_session.started",
        "gen_ai.client.realtime_session.ended",
    ]
    assert logs[0].severity_number == SeverityNumber.INFO
    assert logs[1].severity_number == SeverityNumber.WARN
    assert (
        logs[1].attributes[ErrorAttributes.ERROR_TYPE]
        == expected_cancel_error_type
    )


@pytest.mark.asyncio
async def test_live_vertex_output_tokens_include_thoughts(
    span_exporter,
    metric_reader,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai
    from google.auth.credentials import Credentials

    class _FakeCredentials(Credentials):
        def __init__(self) -> None:
            super().__init__()
            self.token = "fake-vertex-token"

        @property
        def valid(self) -> bool:
            return True

        def refresh(self, request: Any) -> None:  # pylint: disable=unused-argument
            pass

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Vertex answer with reasoning"}],
                    },
                    "turnComplete": True,
                },
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 18,
                    "thoughtsTokenCount": 4,
                    "totalTokenCount": 32,
                },
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(
                vertexai=True,
                project="test-proj",
                location="us-central1",
                credentials=_FakeCredentials(),
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                async for _ in session.receive():
                    pass

    (span,) = span_exporter.get_finished_spans()
    attrs = dict(span.attributes)
    assert (
        attrs[GenAI.GEN_AI_PROVIDER_NAME]
        == GenAI.GenAiProviderNameValues.GCP_VERTEX_AI.value
    )
    assert attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 22
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 4

    metrics_data = metric_reader.get_metrics_data()
    token_dps = [
        dp
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "gen_ai.client.token.usage"
        for dp in m.data.data_points
    ]
    by_token_type = {
        dp.attributes[GenAI.GEN_AI_TOKEN_TYPE]: dp.sum for dp in token_dps
    }
    assert by_token_type[GenAI.GenAiTokenTypeValues.INPUT.value] == 10
    assert by_token_type[GenAI.GenAiTokenTypeValues.OUTPUT.value] == 22


@pytest.mark.asyncio
async def test_live_prewrite_suspended_send_failure_not_committed_on_inbound_frame(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )
    has_context_update = hasattr(
        genai_types, "LiveClientContextUpdate"
    ) and hasattr(genai_live.AsyncSession, "send_context_update")

    # Part A: Turn 1 input is already confirmed. A second send (plus context
    # update when supported) suspends BEFORE writing to the socket while Turn 1's
    # response arrives and completes, and then fails with ConnectionError.
    # Neither Turn 1 nor the tracker may record the rejected input/config.
    expected_suspended = 2 if has_context_update else 1
    suspended_count = 0
    all_suspended = asyncio.Event()
    fail_suspended = asyncio.Event()

    class _PreWriteFailingWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            nonlocal suspended_count
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            frame = json.loads(raw)
            is_rejected = "Rejected prompt" in raw or "Rejected system" in raw
            if is_rejected:
                suspended_count += 1
                if suspended_count >= expected_suspended:
                    all_suspended.set()
                await asyncio.wait_for(fail_suspended.wait(), timeout=5.0)
                raise ConnectionError("pre-write transport error")
            self.sent_frames.append(frame)

    ws_a = _PreWriteFailingWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 1 complete"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(ws_a)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Initial system",
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Confirmed Turn 1")],
                    ),
                    turn_complete=True,
                )

                failing_tasks = [
                    asyncio.create_task(
                        session.send_client_content(
                            turns=genai_types.Content(
                                role="user",
                                parts=[
                                    genai_types.Part(text="Rejected prompt")
                                ],
                            ),
                            turn_complete=True,
                        )
                    )
                ]
                if has_context_update:
                    failing_tasks.append(
                        asyncio.create_task(
                            session.send_context_update(
                                system_instruction="Rejected system",
                            )
                        )
                    )

                await asyncio.wait_for(all_suspended.wait(), timeout=5.0)

                # Inbound Turn 1 arrives and completes while failing_tasks are
                # suspended pre-write; it must not treat inbound traffic as ack.
                async for _ in session.receive():
                    pass

                fail_suspended.set()
                for task in failing_tasks:
                    with pytest.raises(
                        ConnectionError, match="pre-write transport error"
                    ):
                        await task

    (span_a,) = span_exporter.get_finished_spans()
    assert span_a.status.status_code == StatusCode.UNSET
    assert span_a.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Confirmed Turn 1"},),
        },
    )
    assert span_a.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "Initial system"},
    )
    ended_a = next(
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended_a.severity_number == SeverityNumber.WARN
    assert ended_a.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"

    # Part B: While a generation is active, a pre-write suspended audio send
    # receives an unrelated inbound chunk and then fails; rejected audio must
    # not be committed to the active span or tracker.
    span_exporter.clear()
    log_exporter.clear()
    audio_suspended = asyncio.Event()
    fail_audio = asyncio.Event()

    class _PreWriteFailingAudioWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            frame = json.loads(raw)
            if "realtimeInput" in frame or "realtime_input" in frame:
                audio_suspended.set()
                await asyncio.wait_for(fail_audio.wait(), timeout=5.0)
                raise ConnectionError("audio pre-write failed")
            self.sent_frames.append(frame)

    ws_b = _PreWriteFailingAudioWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Active chunk 1"}],
                    },
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Active chunk 2"}],
                    },
                }
            },
        ]
    )
    created_trackers: list[live_module._LiveSessionTracker] = []
    orig_tracker_init = live_module._LiveSessionTracker.__init__

    def _capture_tracker(self, *args, **kwargs):
        orig_tracker_init(self, *args, **kwargs)
        created_trackers.append(self)

    with (
        patch("google.genai.live.ws_connect", _make_fake_ws_connect(ws_b)),
        patch.object(
            live_module._LiveSessionTracker, "__init__", _capture_tracker
        ),
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session._receive()
                audio_task = asyncio.create_task(
                    session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=b"REJECTED", mime_type="audio/pcm"
                        )
                    )
                )
                await asyncio.wait_for(audio_suspended.wait(), timeout=5.0)
                await session._receive()
                fail_audio.set()
                with pytest.raises(
                    ConnectionError, match="audio pre-write failed"
                ):
                    await audio_task

                tracker = created_trackers[0]
                assert not tracker._pending_audio_chunks
                assert tracker._pending_input_messages == []

    (span_b,) = span_exporter.get_finished_spans()
    assert span_b.status.status_code == StatusCode.ERROR
    assert span_b.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    assert GenAI.GEN_AI_INPUT_MESSAGES not in span_b.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["NO_CONTENT", "EVENT_ONLY"])
async def test_live_completion_hook_capture_and_nested_upload_disabled(
    mode: str,
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    import base64

    from google import genai

    from opentelemetry.util.genai.completion_hook import CompletionHook
    from opentelemetry.util.genai.types import (
        BlobPart,
        FunctionToolDefinition,
        Modality,
        TextPart,
        TranscriptionPart,
    )

    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, mode
    )

    class _RecordingHook(CompletionHook):
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def on_completion(
            self,
            *,
            inputs,
            outputs,
            system_instruction,
            tool_definitions=None,
            span=None,
            log_record=None,
        ) -> None:
            self.calls.append(
                {
                    "inputs": list(inputs),
                    "outputs": list(outputs),
                    "system_instruction": list(system_instruction),
                    "tool_definitions": (
                        list(tool_definitions)
                        if tool_definitions is not None
                        else None
                    ),
                    "span": span,
                    "log_record": log_record,
                }
            )

    def hook_tool(query: str) -> str:
        """Search documentation."""
        return query

    def _make_ws() -> _FakeWebSocket:
        return _FakeWebSocket(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "inputTranscription": {"text": "Hi Gemini"},
                        "modelTurn": {
                            "role": "model",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/pcm;rate=24000",
                                        "data": base64.b64encode(
                                            b"\x05\x06\x07\x08"
                                        ).decode("ascii"),
                                    }
                                }
                            ],
                        },
                        "outputTranscription": {"text": "Hello user"},
                        "turnComplete": True,
                    }
                },
            ]
        )

    # Case 1: Unsuppressed Live session with a recording completion hook in
    # NO_CONTENT / EVENT_ONLY forwards exact audio, transcriptions, instructions,
    # and tool definitions to the hook while keeping spans and lifecycle events
    # free of content payloads.
    hook = _RecordingHook()
    with patch(
        "google.genai.live.ws_connect", _make_fake_ws_connect(_make_ws())
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                system_instruction="Hook system prompt",
                tools=[hook_tool],
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x01\x02\x03\x04",
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                async for _ in session.receive():
                    pass

    assert len(hook.calls) == 1
    call = hook.calls[0]
    assert len(call["inputs"]) == 1
    assert call["inputs"][0].role == "user"
    assert call["inputs"][0].parts == [
        BlobPart(
            mime_type="audio/pcm;rate=16000",
            modality=Modality.AUDIO,
            content=b"\x01\x02\x03\x04",
        ),
        TranscriptionPart(content="Hi Gemini"),
    ]
    assert len(call["outputs"]) == 1
    assert call["outputs"][0].role == "assistant"
    assert call["outputs"][0].parts == [
        BlobPart(
            mime_type="audio/pcm;rate=24000",
            modality=Modality.AUDIO,
            content=b"\x05\x06\x07\x08",
        ),
        TranscriptionPart(content="Hello user"),
    ]
    assert call["system_instruction"] == [
        TextPart(content="Hook system prompt")
    ]
    assert call["tool_definitions"] is not None
    assert len(call["tool_definitions"]) == 1
    assert isinstance(call["tool_definitions"][0], FunctionToolDefinition)
    assert call["tool_definitions"][0].name == "hook_tool"
    assert call["log_record"] is None

    (span,) = span_exporter.get_finished_spans()
    for content_attr in (
        GenAI.GEN_AI_INPUT_MESSAGES,
        GenAI.GEN_AI_OUTPUT_MESSAGES,
        GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
        GenAI.GEN_AI_TOOL_DEFINITIONS,
    ):
        assert content_attr not in span.attributes

    logs = [log.log_record for log in log_exporter.get_finished_logs()]
    assert [log.event_name for log in logs] == [
        "gen_ai.client.realtime_session.started",
        "gen_ai.client.realtime_session.ended",
    ]
    for log_record in logs:
        for content_attr in (
            GenAI.GEN_AI_INPUT_MESSAGES,
            GenAI.GEN_AI_OUTPUT_MESSAGES,
            GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
            GenAI.GEN_AI_TOOL_DEFINITIONS,
        ):
            assert content_attr not in log_record.attributes

    # Case 2: Enclosing outer inference with uploads disabled (default
    # upload_completion=False, capture_content=False) must NOT invoke the hook
    # or capture content into the outer inference telemetry.
    span_exporter.clear()
    log_exporter.clear()
    nested_hook = _RecordingHook()
    outer_handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    with patch(
        "google.genai.live.ws_connect", _make_fake_ws_connect(_make_ws())
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=nested_hook,
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                system_instruction="Suppressed hook prompt",
                tools=[hook_tool],
            )
            with outer_handler.inference(
                GenAI.GenAiProviderNameValues.GCP_GEMINI.value,
                request_model="outer-no-upload",
            ):
                async with client.aio.live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                    config=config,
                ) as session:
                    await session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=b"\x01\x02\x03\x04",
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                    await session.send_realtime_input(audio_stream_end=True)
                    async for _ in session.receive():
                        pass

    assert nested_hook.calls == []
    nested_spans = span_exporter.get_finished_spans()
    assert len(nested_spans) == 1
    assert (
        nested_spans[0].attributes[GenAI.GEN_AI_OPERATION_NAME]
        != "realtime_inference"
    )
    for content_attr in (
        GenAI.GEN_AI_INPUT_MESSAGES,
        GenAI.GEN_AI_OUTPUT_MESSAGES,
        GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
        GenAI.GEN_AI_TOOL_DEFINITIONS,
    ):
        assert content_attr not in nested_spans[0].attributes
    for log in log_exporter.get_finished_logs():
        for content_attr in (
            GenAI.GEN_AI_INPUT_MESSAGES,
            GenAI.GEN_AI_OUTPUT_MESSAGES,
            GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
            GenAI.GEN_AI_TOOL_DEFINITIONS,
        ):
            assert content_attr not in log.log_record.attributes


@pytest.mark.asyncio
async def test_live_user_speech_disabled_by_default_and_opt_in(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    monkeypatch,
):
    from google import genai

    monkeypatch.delenv(
        OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, raising=False
    )

    def _make_vad_ws() -> _FakeWebSocket:
        return _FakeWebSocket(
            [
                {"setupComplete": {}},
                {
                    "voiceActivityDetectionSignal": {
                        "vadSignalType": "VAD_SIGNAL_TYPE_SOS"
                    }
                },
                {
                    "voiceActivityDetectionSignal": {
                        "vadSignalType": "VAD_SIGNAL_TYPE_EOS"
                    }
                },
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": "Speech reply"}],
                        },
                        "turnComplete": True,
                    }
                },
            ]
        )

    # Default configuration: neither client activity_start/activity_end nor
    # server VAD emits user_speech spans; realtime_inference still records
    # the flushed audio input.
    with patch(
        "google.genai.live.ws_connect", _make_fake_ws_connect(_make_vad_ws())
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_realtime_input(
                    activity_start=genai_types.ActivityStart()
                )
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x01\x02\x03\x04",
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(
                    activity_end=genai_types.ActivityEnd()
                )
                async for _ in session.receive():
                    pass

    default_spans = span_exporter.get_finished_spans()
    assert [
        s.attributes[GenAI.GEN_AI_OPERATION_NAME] for s in default_spans
    ] == ["realtime_inference"]
    assert default_spans[0].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"\x01\x02\x03\x04",
                },
            ),
        },
    )

    # Opt-in enabled via OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH=true:
    # user_speech spans are emitted for speech boundaries.
    span_exporter.clear()
    with patch(
        "google.genai.live.ws_connect", _make_fake_ws_connect(_make_vad_ws())
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
            extra_env={OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true"},
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_realtime_input(
                    activity_start=genai_types.ActivityStart()
                )
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x01\x02\x03\x04",
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await session.send_realtime_input(
                    activity_end=genai_types.ActivityEnd()
                )
                async for _ in session.receive():
                    pass

    optin_ops = [
        s.attributes[GenAI.GEN_AI_OPERATION_NAME]
        for s in span_exporter.get_finished_spans()
    ]
    assert "user_speech" in optin_ops
    assert optin_ops[-1] == "realtime_inference"


@pytest.mark.asyncio
@pytest.mark.parametrize("via_legacy_send", [False, True])
async def test_live_gemini_3_tool_response_draining_after_turn_complete_omitted_from_next_turn(
    via_legacy_send: bool,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    tool_resp_written = asyncio.Event()
    release_tool_resp_send = asyncio.Event()

    class _DrainingToolResponseWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            await super().send(payload)
            frame = self.sent_frames[-1]
            if (
                "toolResponse" in frame or "tool_response" in frame
            ) and not tool_resp_written.is_set():
                # Frame written to wire; yield while draining so concurrent
                # receive finishes Turn 1 and clears _active_gen first.
                tool_resp_written.set()
                await asyncio.wait_for(
                    release_tool_resp_send.wait(), timeout=5.0
                )

        async def recv(self, decode: bool = True) -> bytes | str:
            # Frame index 2 (2 remaining frames) is Turn 1's spoken answer +
            # turnComplete after the tool response reaches the server.
            if len(self._frames) == 2:
                await asyncio.wait_for(tool_resp_written.wait(), timeout=5.0)
            return await super().recv(decode=decode)

    fake_ws = _DrainingToolResponseWebSocket(
        [
            {"setupComplete": {}},
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-g3-drain-1",
                            "name": "get_weather",
                            "args": {"city": "Seattle"},
                        }
                    ]
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "It is rainy in Seattle."}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Sure, anything else?"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-3.1-flash-live-preview",
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Weather in Seattle?")],
                    ),
                    turn_complete=True,
                )

                fn_resp = genai_types.FunctionResponse(
                    id="call-g3-drain-1",
                    name="get_weather",
                    response={"weather": "rainy"},
                )

                # Receive Turn 1's tool call (sets awaiting_tool_response=True).
                msg1 = await session._receive()
                assert msg1.tool_call is not None

                # Stage tool response send; ws.send writes frame then suspends.
                if via_legacy_send:
                    send_task = asyncio.create_task(
                        session.send(input=fn_resp)
                    )
                else:
                    send_task = asyncio.create_task(
                        session.send_tool_response(
                            function_responses=[fn_resp]
                        )
                    )

                # Receive Turn 1's spoken answer + turnComplete BEFORE the tool
                # response send returns, clearing tracker._active_gen.
                msg2 = await session._receive()
                assert msg2.server_content is not None
                assert msg2.server_content.turn_complete is True

                # Now let the tool response send finish draining.
                release_tool_resp_send.set()
                await send_task

                # Turn 2 must NOT inherit Turn 1's intermediate tool response.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Follow-up question")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]
    assert span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Weather in Seattle?"},),
        },
    )
    assert span1.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES] == (
        {
            "role": "assistant",
            "parts": ({"type": "text", "content": "It is rainy in Seattle."},),
            "finish_reason": "stop",
        },
    )
    assert span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Follow-up question"},),
        },
    )


@pytest.mark.asyncio
async def test_live_deferred_sends_preserve_send_order_on_reverse_completion(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    # Part A: Two text sends in one turn where send 1 writes and pauses while
    # draining, send 2 finishes FIRST, and the server responds before send 1
    # finishes draining. Generation inputs must preserve send order [1, 2].
    send1_written = asyncio.Event()
    send2_written = asyncio.Event()
    release_send1 = asyncio.Event()

    class _ReverseOrderTextWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            await super().send(payload)
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            if "First part" in raw:
                send1_written.set()
                await asyncio.wait_for(release_send1.wait(), timeout=5.0)
            elif "Second part" in raw:
                send2_written.set()

        async def recv(self, decode: bool = True) -> bytes | str:
            if len(self._frames) == 1:
                await asyncio.wait_for(send2_written.wait(), timeout=5.0)
            return await super().recv(decode=decode)

    ws_text = _ReverseOrderTextWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Ordered text answer"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(ws_text)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                send1_task = asyncio.create_task(
                    session.send_client_content(
                        turns=genai_types.Content(
                            role="user",
                            parts=[genai_types.Part(text="First part")],
                        ),
                        turn_complete=False,
                    )
                )
                await asyncio.wait_for(send1_written.wait(), timeout=5.0)

                # Send 2 enters and finishes BEFORE send 1 finishes draining.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Second part")],
                    ),
                    turn_complete=True,
                )

                # Server completes Turn 1 while send 1 is still draining.
                async for _ in session.receive():
                    pass

                release_send1.set()
                await send1_task

    (text_span,) = span_exporter.get_finished_spans()
    assert text_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "First part"},),
        },
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Second part"},),
        },
    )

    # Part B: Two audio chunks in one turn completing in reverse order while
    # generation starts and completes -> coalesced audio must be AAAA + BBBB.
    span_exporter.clear()
    audio1_written = asyncio.Event()
    audio2_written = asyncio.Event()
    release_audio1 = asyncio.Event()
    audio_send_index = 0

    class _ReverseOrderAudioWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            nonlocal audio_send_index
            await super().send(payload)
            frame = self.sent_frames[-1]
            if "realtimeInput" in frame or "realtime_input" in frame:
                audio_send_index += 1
                if audio_send_index == 1:
                    audio1_written.set()
                    await asyncio.wait_for(release_audio1.wait(), timeout=5.0)
                elif audio_send_index == 2:
                    audio2_written.set()

        async def recv(self, decode: bool = True) -> bytes | str:
            if len(self._frames) == 1:
                await asyncio.wait_for(audio2_written.wait(), timeout=5.0)
            return await super().recv(decode=decode)

    ws_audio = _ReverseOrderAudioWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Ordered audio answer"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch(
        "google.genai.live.ws_connect", _make_fake_ws_connect(ws_audio)
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                audio1_task = asyncio.create_task(
                    session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=b"AAAA", mime_type="audio/pcm;rate=16000"
                        )
                    )
                )
                await asyncio.wait_for(audio1_written.wait(), timeout=5.0)

                # Chunk 2 finishes BEFORE chunk 1 finishes draining.
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"BBBB", mime_type="audio/pcm;rate=16000"
                    )
                )

                async for _ in session.receive():
                    pass

                release_audio1.set()
                await audio1_task

    (audio_span,) = span_exporter.get_finished_spans()
    assert audio_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"AAAABBBB",
                },
            ),
        },
    )


@pytest.mark.asyncio
async def test_live_prewrite_delayed_turn2_send_succeeds_for_turn2_not_turn1(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    has_context_update = hasattr(
        genai_types, "LiveClientContextUpdate"
    ) and hasattr(genai_live.AsyncSession, "send_context_update")

    expected_delayed = 2 if has_context_update else 1
    delayed_count = 0
    all_delayed = asyncio.Event()
    release_delayed = asyncio.Event()

    class _PreWriteDelayedTurn2WebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            nonlocal delayed_count
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            if "Turn 2 prompt" in raw or "Turn 2 system" in raw:
                delayed_count += 1
                if delayed_count >= expected_delayed:
                    all_delayed.set()
                # Pause BEFORE writing while Turn 1's response arrives, then
                # allow the send to succeed cleanly for Turn 2.
                await asyncio.wait_for(release_delayed.wait(), timeout=5.0)
            await super().send(payload)

    fake_ws = _PreWriteDelayedTurn2WebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 1 answer"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 2 answer"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Initial system",
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Confirmed Turn 1")],
                    ),
                    turn_complete=True,
                )

                # Start Turn 2 sends that pause pre-write before Turn 1 responds.
                delayed_tasks: list[asyncio.Task[Any]] = []
                if has_context_update:
                    delayed_tasks.append(
                        asyncio.create_task(
                            session.send_context_update(
                                system_instruction="Turn 2 system"
                            )
                        )
                    )
                delayed_tasks.append(
                    asyncio.create_task(
                        session.send_client_content(
                            turns=genai_types.Content(
                                role="user",
                                parts=[genai_types.Part(text="Turn 2 prompt")],
                            ),
                            turn_complete=True,
                        )
                    )
                )
                await asyncio.wait_for(all_delayed.wait(), timeout=5.0)

                # Turn 1 arrives and completes while Turn 2 sends are delayed.
                async for _ in session.receive():
                    pass

                # Now Turn 2's delayed sends succeed and Turn 2 completes.
                release_delayed.set()
                for task in delayed_tasks:
                    await task
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]

    assert span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Confirmed Turn 1"},),
        },
    )
    assert span1.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "Initial system"},
    )

    assert span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 2 prompt"},),
        },
    )
    expected_sys2 = (
        ({"type": "text", "content": "Turn 2 system"},)
        if has_context_update
        else ({"type": "text", "content": "Initial system"},)
    )
    assert span2.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == expected_sys2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "send_mode",
    ["modern_audio", "legacy_blob", "legacy_realtime_input"],
)
async def test_live_continuous_audio_streaming_preserves_current_turn_transcription_ownership(
    send_mode: str,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    has_vad_signal = (
        "voice_activity_detection_signal"
        in genai_types.LiveServerMessage.model_fields
    )

    frames: list[dict[str, Any]] = [
        {"setupComplete": {}},
        # Turn 1 starts with first inputTranscription delta + model output.
        {
            "serverContent": {
                "inputTranscription": {"text": "Hello "},
                "modelTurn": {
                    "role": "model",
                    "parts": [{"text": "Hi there, "}],
                },
            }
        },
        # Late inputTranscription delta for Turn 1 arrives AFTER continued
        # silent microphone chunks are streamed by the client.
        {
            "serverContent": {
                "inputTranscription": {"text": "world!"},
            }
        },
    ]
    if has_vad_signal:
        # Automatic VAD detects genuine new user speech (barge-in boundary).
        frames.append(
            {
                "voiceActivityDetectionSignal": {
                    "vadSignalType": "VAD_SIGNAL_TYPE_SOS"
                }
            }
        )
    frames.extend(
        [
            # Turn 2's transcription arrives after the genuine speech boundary
            # while Turn 1 is still awaiting interrupted/turnComplete.
            {
                "serverContent": {
                    "inputTranscription": {"text": "Next question"},
                }
            },
            {
                "serverContent": {
                    "interrupted": True,
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Sure, answering Turn 2."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    fake_ws = _FakeWebSocket(frames)

    async def _stream_audio(session: Any, data: bytes) -> None:
        blob = genai_types.Blob(data=data, mime_type="audio/pcm;rate=16000")
        if send_mode == "modern_audio":
            await session.send_realtime_input(audio=blob)
        elif send_mode == "legacy_blob":
            await session.send(input=blob)
        else:
            await session.send(
                input=genai_types.LiveClientRealtimeInput(media_chunks=[blob])
            )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Turn 1 initial audio (no audio_stream_end: automatic VAD).
                await _stream_audio(session, b"\x01\x02" * 8)

                # Frame 1: Turn 1 begins ("Hello ").
                await session._receive()

                # Client continues streaming silent microphone audio while
                # Turn 1 is active (no speech boundary).
                await _stream_audio(session, b"\x00\x00" * 8)

                # Frame 2: Late delta ("world!") must still belong to Turn 1.
                await session._receive()

                # Now a genuine speech boundary starts Turn 2: server VAD SOS
                # on latest SDK, or client activity_start on oldest SDK.
                if has_vad_signal:
                    await session._receive()
                else:
                    await session.send_realtime_input(
                        activity_start=genai_types.ActivityStart()
                    )
                await _stream_audio(session, b"\x03\x04" * 8)

                # Receive Turn 2 inputTranscription, Turn 1 interrupted, Turn 2.
                await session._receive()
                await session._receive()
                await session._receive()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]

    in1 = span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in1) == 1
    assert in1[0]["parts"][0] == {
        "type": "blob",
        "modality": "audio",
        "mime_type": "audio/pcm;rate=16000",
        "content": b"\x01\x02" * 8,
    }
    assert in1[0]["parts"][1] == {
        "type": "transcription",
        "content": "Hello world!",
    }

    in2 = span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert len(in2) == 1
    assert in2[0]["parts"][0]["type"] == "blob"
    assert in2[0]["parts"][1] == {
        "type": "transcription",
        "content": "Next question",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion_field", "expected_finish_reason", "fail_on"),
    [
        ("generationComplete", "stop", "recv"),
        ("generationComplete", "stop", "send"),
        ("interrupted", "interrupted", "recv"),
    ],
)
async def test_live_transport_error_after_generation_complete_or_interrupted_preserves_success(
    completion_field: str,
    expected_finish_reason: str,
    fail_on: str,
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    frames: list[dict[str, Any] | BaseException] = [
        {"setupComplete": {}},
        {
            "serverContent": {
                "modelTurn": {
                    "role": "model",
                    "parts": [{"text": "Completed response"}],
                },
                completion_field: True,
            }
        },
    ]
    if fail_on == "recv":
        frames.append(ConnectionError("transport dropped after completion"))

    fake_ws = _FakeWebSocket(frames)

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Receive completed frame (end timestamp frozen via mark_end).
                await session._receive()
                after_complete_ns = time.time_ns()
                await asyncio.sleep(0.05)
                before_error_ns = time.time_ns()

                if fail_on == "recv":
                    with pytest.raises(
                        ConnectionError,
                        match="transport dropped after completion",
                    ):
                        await session._receive()
                else:
                    with patch.object(
                        fake_ws,
                        "send",
                        side_effect=ConnectionError(
                            "transport dropped on later send"
                        ),
                    ):
                        with pytest.raises(
                            ConnectionError,
                            match="transport dropped on later send",
                        ):
                            await session.send_client_content(
                                turns=genai_types.Content(
                                    role="user",
                                    parts=[genai_types.Part(text="Next turn")],
                                ),
                                turn_complete=True,
                            )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        expected_finish_reason,
    )
    assert span.end_time <= after_complete_ns
    assert span.end_time < before_error_ns

    ended_log = next(
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended_log.severity_number == SeverityNumber.WARN
    assert (
        ended_log.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    )


@pytest.mark.asyncio
async def test_live_buffered_replies_keep_consecutive_turns_and_context_separate(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    has_context_update = hasattr(
        genai_types, "LiveClientContextUpdate"
    ) and hasattr(genai_live.AsyncSession, "send_context_update")

    def tool_v1(query: str) -> str:
        """Initial tool."""
        return query

    def tool_v2(query: str) -> str:
        """Updated tool."""
        return query

    frames: list[dict[str, Any]] = [
        {"setupComplete": {}},
        {
            "serverContent": {
                "modelTurn": {
                    "role": "model",
                    "parts": [{"text": "Turn 1 buffered reply"}],
                },
                "turnComplete": True,
            }
        },
        {
            "serverContent": {
                "modelTurn": {
                    "role": "model",
                    "parts": [{"text": "Turn 2 buffered reply"}],
                },
                "turnComplete": True,
            }
        },
    ]
    if has_context_update:
        frames.append(
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Turn 3 context-only reply"}],
                    },
                    "turnComplete": True,
                }
            }
        )

    fake_ws = _FakeWebSocket(frames)

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Initial system",
                tools=[tool_v1],
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                # Both turn-completing sends (and Turn 2's context update)
                # finish BEFORE either buffered reply is consumed.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 1 prompt")],
                    ),
                    turn_complete=True,
                )
                if has_context_update:
                    await session.send_context_update(
                        system_instruction="Turn 2 system",
                        tools=[tool_v2],
                    )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 2 prompt")],
                    ),
                    turn_complete=True,
                )
                if has_context_update:
                    # Context-only update queued after Turn 2 applies to Turn 3.
                    await session.send_context_update(
                        system_instruction="Turn 3 system",
                    )

                # Consume buffered replies one turn at a time.
                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass
                if has_context_update:
                    async for _ in session.receive():
                        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == (3 if has_context_update else 2)
    span1, span2 = spans[0], spans[1]

    assert span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 1 prompt"},),
        },
    )
    assert span1.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "Initial system"},
    )
    assert [
        t["name"] for t in span1.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["tool_v1"]

    assert span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 2 prompt"},),
        },
    )
    if has_context_update:
        assert span2.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
            {"type": "text", "content": "Turn 2 system"},
        )
        assert [
            t["name"] for t in span2.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
        ] == ["tool_v2"]

        span3 = spans[2]
        assert GenAI.GEN_AI_INPUT_MESSAGES not in span3.attributes
        assert span3.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
            {"type": "text", "content": "Turn 3 system"},
        )
        assert [
            t["name"] for t in span3.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
        ] == ["tool_v2"]
    else:
        assert span2.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
            {"type": "text", "content": "Initial system"},
        )


@pytest.mark.asyncio
async def test_live_late_context_commit_does_not_overwrite_newer_session_config(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    has_context_update = hasattr(
        genai_types, "LiveClientContextUpdate"
    ) and hasattr(genai_live.AsyncSession, "send_context_update")
    if not has_context_update:
        return

    def tool_init(query: str) -> str:
        """Initial tool."""
        return query

    def tool_a(query: str) -> str:
        """Tool A."""
        return query

    def tool_b(query: str) -> str:
        """Tool B."""
        return query

    update_a_written = asyncio.Event()
    release_update_a = asyncio.Event()

    class _DelayedContextUpdateWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            await super().send(payload)
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            if "System A" in raw:
                update_a_written.set()
                await asyncio.wait_for(release_update_a.wait(), timeout=5.0)

    fake_ws = _DelayedContextUpdateWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 1"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 2"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 3"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Initial system",
                tools=[tool_init],
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                # Turn 1: context update A writes and pauses while draining.
                update_a_task = asyncio.create_task(
                    session.send_context_update(
                        system_instruction="System A",
                        tools=[tool_a],
                    )
                )
                await asyncio.wait_for(update_a_written.wait(), timeout=5.0)
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 1 prompt")],
                    ),
                    turn_complete=True,
                )

                # Turn 1 completes while update A is still draining (deferred).
                async for _ in session.receive():
                    pass

                # Turn 2 applies newer context update B and completes BEFORE
                # update A finishes draining.
                await session.send_context_update(
                    system_instruction="System B",
                    tools=[tool_b],
                )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 2 prompt")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass

                # Now older update A finishes draining: it must update Turn 1
                # without rolling session-wide config back from B to A.
                release_update_a.set()
                await update_a_task

                # Turn 3 has no context update and must inherit B, not A.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 3 prompt")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    by_reply = {
        span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES][0]["parts"][0][
            "content"
        ]: span
        for span in spans
    }

    span1 = by_reply["Reply 1"]
    assert span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 1 prompt"},),
        },
    )
    assert span1.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "System A"},
    )
    assert [
        t["name"] for t in span1.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["tool_a"]

    span2 = by_reply["Reply 2"]
    assert span2.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "System B"},
    )
    assert [
        t["name"] for t in span2.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["tool_b"]

    span3 = by_reply["Reply 3"]
    assert span3.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 3 prompt"},),
        },
    )
    assert span3.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "System B"},
    )
    assert [
        t["name"] for t in span3.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["tool_b"]


@pytest.mark.asyncio
async def test_live_explicit_session_close_finalizes_once_including_close_errors(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    # Part A: Explicit session.close() finalizes active speech/inference spans
    # and emits realtime_session.ended immediately, before post-close cleanup
    # inside the connect context; repeated close() and context exit are no-ops.
    ws_ok = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Partial chunk"}],
                    },
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(ws_ok)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
            extra_env={OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true"},
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_realtime_input(
                    activity_start=genai_types.ActivityStart()
                )
                await session._receive()
                await session.close()
                # Finalization must happen immediately on close(), before
                # post-close cleanup and before exiting the connect context.
                assert len(span_exporter.get_finished_spans()) == 2
                assert (
                    sum(
                        1
                        for log in log_exporter.get_finished_logs()
                        if log.log_record.event_name
                        == "gen_ai.client.realtime_session.ended"
                    )
                    == 1
                )
                await session.close()
                # Slow cleanup after explicit close must not extend durations.
                await asyncio.sleep(0.06)

    spans_a = span_exporter.get_finished_spans()
    assert len(spans_a) == 2
    by_op_a = {s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s for s in spans_a}
    speech_a = by_op_a["user_speech"]
    inf_a = by_op_a["realtime_inference"]
    assert speech_a.status.status_code == StatusCode.UNSET
    assert (speech_a.end_time - speech_a.start_time) / 1e9 < 0.04
    assert inf_a.status.status_code == StatusCode.ERROR
    assert (inf_a.end_time - inf_a.start_time) / 1e9 < 0.04

    logs_a = [log.log_record for log in log_exporter.get_finished_logs()]
    started_a = [
        r
        for r in logs_a
        if r.event_name == "gen_ai.client.realtime_session.started"
    ]
    ended_a = [
        r
        for r in logs_a
        if r.event_name == "gen_ai.client.realtime_session.ended"
    ]
    assert len(started_a) == 1
    assert len(ended_a) == 1
    assert ended_a[0].severity_number == SeverityNumber.INFO
    assert ErrorAttributes.ERROR_TYPE not in ended_a[0].attributes
    started_ts_a = started_a[0].timestamp or started_a[0].observed_timestamp
    ended_ts_a = ended_a[0].timestamp or ended_a[0].observed_timestamp
    if started_ts_a is not None and ended_ts_a is not None:
        assert (ended_ts_a - started_ts_a) / 1e9 < 0.04

    # Part B: Explicit session.close() failure finalizes active operations with
    # the close error immediately and emits realtime_session.ended once with
    # WARN + error.type even when caught before slow post-close cleanup.
    span_exporter.clear()
    log_exporter.clear()

    class _FailingCloseWebSocket(_FakeWebSocket):
        async def close(self) -> None:
            raise ConnectionError("close handshake failed")

    ws_fail = _FailingCloseWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Incomplete before close error"}],
                    },
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(ws_fail)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
            extra_env={OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true"},
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_realtime_input(
                    activity_start=genai_types.ActivityStart()
                )
                await session._receive()
                with pytest.raises(
                    ConnectionError, match="close handshake failed"
                ):
                    await session.close()
                assert len(span_exporter.get_finished_spans()) == 2
                assert (
                    sum(
                        1
                        for log in log_exporter.get_finished_logs()
                        if log.log_record.event_name
                        == "gen_ai.client.realtime_session.ended"
                    )
                    == 1
                )
                await asyncio.sleep(0.06)

    spans_b = span_exporter.get_finished_spans()
    assert len(spans_b) == 2
    by_op_b = {s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s for s in spans_b}
    for s in (by_op_b["user_speech"], by_op_b["realtime_inference"]):
        assert s.status.status_code == StatusCode.ERROR
        assert s.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
        assert (s.end_time - s.start_time) / 1e9 < 0.04

    logs_b = [log.log_record for log in log_exporter.get_finished_logs()]
    started_b = [
        r
        for r in logs_b
        if r.event_name == "gen_ai.client.realtime_session.started"
    ]
    ended_b = [
        r
        for r in logs_b
        if r.event_name == "gen_ai.client.realtime_session.ended"
    ]
    assert len(started_b) == 1
    assert len(ended_b) == 1
    assert ended_b[0].severity_number == SeverityNumber.WARN
    assert (
        ended_b[0].attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    )
    started_ts_b = started_b[0].timestamp or started_b[0].observed_timestamp
    ended_ts_b = ended_b[0].timestamp or ended_b[0].observed_timestamp
    if started_ts_b is not None and ended_ts_b is not None:
        assert (ended_ts_b - started_ts_b) / 1e9 < 0.04

    # Part C: Explicit session.close() failure after generationComplete still
    # preserves generation success while recording the close error on the
    # session-ended event.
    span_exporter.clear()
    log_exporter.clear()
    ws_complete_then_fail_close = _FailingCloseWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Done before close error"}],
                    },
                    "generationComplete": True,
                }
            },
        ]
    )

    with patch(
        "google.genai.live.ws_connect",
        _make_fake_ws_connect(ws_complete_then_fail_close),
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session._receive()
                with pytest.raises(
                    ConnectionError, match="close handshake failed"
                ):
                    await session.close()

    (span_c,) = span_exporter.get_finished_spans()
    assert span_c.status.status_code == StatusCode.UNSET
    assert ErrorAttributes.ERROR_TYPE not in span_c.attributes
    assert span_c.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    ended_c = [
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.client.realtime_session.ended"
    ]
    assert len(ended_c) == 1
    assert ended_c[0].severity_number == SeverityNumber.WARN
    assert (
        ended_c[0].attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    )


@pytest.mark.asyncio
async def test_live_legacy_empty_send_completes_turn(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            *[
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": f"Reply {label}"}],
                        },
                        "turnComplete": True,
                    }
                }
                for label in ("A", "B", "C", "D")
            ],
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # SDK treats empty session.send() (and falsy input="") as
                # client_content(turn_complete=True) even when end_of_turn is
                # omitted/False.
                await session.send(input="A")
                await session.send()
                await session.send(input="B", end_of_turn=True)

                await session.send(input="C")
                await session.send(input="")
                await session.send(input="D", end_of_turn=True)

                for _ in range(4):
                    async for _ in session.receive():
                        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    for span, expected in zip(spans, ("A", "B", "C", "D")):
        assert span.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
            {
                "role": "user",
                "parts": ({"type": "text", "content": expected},),
            },
        )


@pytest.mark.asyncio
async def test_live_bound_generation_audio_drain_keeps_bounded_chunks_until_finalization(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    chunk1_written = asyncio.Event()
    release_chunks = asyncio.Event()
    audio_index = 0

    class _DrainingChunksWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            nonlocal audio_index
            await super().send(payload)
            frame = self.sent_frames[-1]
            if "realtimeInput" in frame or "realtime_input" in frame:
                audio_index += 1
                if audio_index == 1:
                    chunk1_written.set()
                    await asyncio.wait_for(release_chunks.wait(), timeout=5.0)

    fake_ws = _DrainingChunksWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Chunk 1 of reply"}],
                    },
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": " done"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    created_trackers: list[live_module._LiveSessionTracker] = []
    orig_tracker_init = live_module._LiveSessionTracker.__init__

    def _capture_tracker(self, *args, **kwargs):
        orig_tracker_init(self, *args, **kwargs)
        created_trackers.append(self)

    with (
        patch.object(live_module, "_MAX_BUFFERED_MEDIA_BYTES", 12),
        patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)),
        patch.object(
            live_module._LiveSessionTracker, "__init__", _capture_tracker
        ),
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                chunk1_task = asyncio.create_task(
                    session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=b"AAAA", mime_type="audio/pcm;rate=16000"
                        )
                    )
                )
                await asyncio.wait_for(chunk1_written.wait(), timeout=5.0)
                for payload in (b"BBBB", b"CCCC", b"DDDD"):
                    await session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=payload, mime_type="audio/pcm;rate=16000"
                        )
                    )

                # Start generation while chunk 1 is draining so the turn binds.
                await session._receive()
                release_chunks.set()
                await chunk1_task

                # Before finalization, audio remains in bounded chunks on the
                # active generation rather than concatenated into input_messages.
                active_gen = created_trackers[0]._active_gen
                assert active_gen is not None
                assert active_gen.input_messages == []
                assert list(active_gen.input_audio_chunks) == [
                    b"BBBB",
                    b"CCCC",
                    b"DDDD",
                ]
                assert active_gen.input_audio_bytes == 12

                # Finalize generation -> single join produces trailing 12 bytes.
                await session._receive()

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"BBBBCCCCDDDD",
                },
            ),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_manual_vad", [False, True])
async def test_live_budget_eviction_preserves_sealed_turn_boundaries_for_buffered_replies(
    use_manual_vad: bool,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 1"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 2"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with (
        patch.object(live_module, "_MAX_BUFFERED_MEDIA_BYTES", 16),
        patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)),
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                for payload in (b"1111111111111111", b"2222222222222222"):
                    if use_manual_vad:
                        await session.send_realtime_input(
                            activity_start=genai_types.ActivityStart()
                        )
                    await session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=payload, mime_type="audio/pcm;rate=16000"
                        )
                    )
                    if use_manual_vad:
                        await session.send_realtime_input(
                            activity_end=genai_types.ActivityEnd()
                        )
                    else:
                        await session.send_realtime_input(
                            audio_stream_end=True
                        )

                # Consume both buffered replies after Turn 1's payload was
                # fully evicted by Turn 2 under the 16-byte media cap.
                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span1, span2 = spans[0], spans[1]
    assert GenAI.GEN_AI_INPUT_MESSAGES not in span1.attributes
    assert span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"2222222222222222",
                },
            ),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("via_legacy_send", [False, True])
async def test_live_gemini_2_separate_tool_responses_grouped_in_single_turn(
    via_legacy_send: bool,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            # Gen 1: Gemini 2.5 requests two tools in one batch.
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-batch-1",
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        },
                        {
                            "id": "call-batch-2",
                            "name": "get_time",
                            "args": {"city": "Paris"},
                        },
                    ]
                }
            },
            # Gen 2: Model replies after receiving BOTH separate tool responses.
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "It is 20C and 3pm in Paris."}],
                    },
                    "turnComplete": True,
                }
            },
            # Gen 3: Follow-up user turn queued before Gen 2 reply was read.
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Follow-up reply."}],
                    },
                    "turnComplete": True,
                }
            },
            # Gen 4: Second tool batch where one call is cancelled by server.
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-cancel-keep",
                            "name": "lookup",
                            "args": {"key": "a"},
                        },
                        {
                            "id": "call-cancel-drop",
                            "name": "lookup",
                            "args": {"key": "b"},
                        },
                    ]
                }
            },
            {"toolCallCancellation": {"ids": ["call-cancel-drop"]}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "After cancellation reply."}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Final user reply."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(text="Weather and time in Paris?")
                        ],
                    ),
                    turn_complete=True,
                )
                msg1 = await session._receive()
                assert msg1.tool_call is not None

                # Answer the two tool calls in two SEPARATE send calls, then
                # queue a follow-up user turn before consuming either reply.
                resp1 = genai_types.FunctionResponse(
                    id="call-batch-1",
                    name="get_weather",
                    response={"temp_c": 20},
                )
                resp2 = genai_types.FunctionResponse(
                    id="call-batch-2",
                    name="get_time",
                    response={"time": "15:00"},
                )
                for resp in (resp1, resp2):
                    if via_legacy_send:
                        await session.send(input=resp)
                    else:
                        await session.send_tool_response(
                            function_responses=resp
                        )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Follow-up question")],
                    ),
                    turn_complete=True,
                )

                # Consume Gen 2 (tool answer) and Gen 3 (follow-up).
                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

                # Second batch: answer one call, server cancels the second.
                msg4 = await session._receive()
                assert msg4.tool_call is not None
                keep_resp = genai_types.FunctionResponse(
                    id="call-cancel-keep",
                    name="lookup",
                    response={"val": 1},
                )
                if via_legacy_send:
                    await session.send(input=keep_resp)
                else:
                    await session.send_tool_response(
                        function_responses=keep_resp
                    )
                cancel_msg = await session._receive()
                assert cancel_msg.tool_call_cancellation is not None
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="After cancel prompt")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 6
    gen1, gen2, gen3, gen4, gen5, gen6 = spans

    assert gen1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    gen2_inputs = gen2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert [m["parts"][0]["id"] for m in gen2_inputs] == [
        "call-batch-1",
        "call-batch-2",
    ]
    assert gen3.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Follow-up question"},),
        },
    )

    assert gen4.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    gen5_inputs = gen5.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert [m["parts"][0]["id"] for m in gen5_inputs] == ["call-cancel-keep"]
    assert gen6.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "After cancel prompt"},),
        },
    )


@pytest.mark.asyncio
async def test_live_deferred_media_commits_enforce_single_generation_byte_budget(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    first_written = asyncio.Event()
    release_first = asyncio.Event()
    media_send_count = 0

    class _DeferredMediaWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            nonlocal media_send_count
            await super().send(payload)
            frame = self.sent_frames[-1]
            if (
                "realtimeInput" in frame
                or "realtime_input" in frame
                or "clientContent" in frame
                or "client_content" in frame
            ):
                media_send_count += 1
                if media_send_count == 1:
                    first_written.set()
                    await asyncio.wait_for(release_first.wait(), timeout=5.0)

    fake_ws = _DeferredMediaWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Bounded deferred media reply"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with (
        patch.object(live_module, "_MAX_BUFFERED_MEDIA_BYTES", 16),
        patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)),
    ):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Send 1 pauses while draining; subsequent video, interleaved
                # audio, and image sends join the same turn and drain into the
                # bound generation after it starts and completes.
                first_task = asyncio.create_task(
                    session.send_realtime_input(
                        video=genai_types.Blob(
                            data=b"VVVVVVVV", mime_type="image/jpeg"
                        )
                    )
                )
                await asyncio.wait_for(first_written.wait(), timeout=5.0)
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"11111111", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(
                    video=genai_types.Blob(
                        data=b"WWWWWWWW", mime_type="image/jpeg"
                    )
                )
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"22222222", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                inline_data=genai_types.Blob(
                                    data=b"IIIIIIII", mime_type="image/png"
                                )
                            )
                        ],
                    ),
                    turn_complete=True,
                )

                # Generation starts and reaches turnComplete while sends drain.
                async for _ in session.receive():
                    pass

                release_first.set()
                await first_task

    (span,) = span_exporter.get_finished_spans()
    in_msgs = span.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    total_blob_bytes = sum(
        len(part["content"])
        for msg in in_msgs
        for part in msg["parts"]
        if part.get("type") == "blob"
    )
    assert total_blob_bytes <= 16
    assert in_msgs == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"22222222",
                },
            ),
        },
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "image",
                    "mime_type": "image/png",
                    "content": b"IIIIIIII",
                },
            ),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("via_legacy_send", [False, True])
async def test_live_gemini_2_tool_batch_rollback_and_retry_keeps_responses_in_single_turn(
    via_legacy_send: bool,
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    call_a_entered = asyncio.Event()
    fail_call_a = asyncio.Event()
    call_a_attempts = 0

    class _FailFirstToolAWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            nonlocal call_a_attempts
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            if "call-rb-a" in raw and call_a_attempts == 0:
                call_a_attempts += 1
                call_a_entered.set()
                await asyncio.wait_for(fail_call_a.wait(), timeout=5.0)
                raise ConnectionError("transient tool A send error")
            await super().send(payload)

    fake_ws = _FailFirstToolAWebSocket(
        [
            {"setupComplete": {}},
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-rb-a",
                            "name": "get_weather",
                            "args": {"city": "Tokyo"},
                        },
                        {
                            "id": "call-rb-b",
                            "name": "get_time",
                            "args": {"city": "Tokyo"},
                        },
                    ]
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Tokyo weather and time reply."}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Follow-up reply after retry."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    async def _send_tool_resp(
        session: Any, resp: genai_types.FunctionResponse
    ) -> None:
        if via_legacy_send:
            await session.send(input=resp)
        else:
            await session.send_tool_response(function_responses=resp)

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(text="Weather and time in Tokyo?")
                        ],
                    ),
                    turn_complete=True,
                )
                msg1 = await session._receive()
                assert msg1.tool_call is not None

                resp_a = genai_types.FunctionResponse(
                    id="call-rb-a",
                    name="get_weather",
                    response={"temp_c": 18},
                )
                resp_b = genai_types.FunctionResponse(
                    id="call-rb-b",
                    name="get_time",
                    response={"time": "09:00"},
                )

                # Start A (pauses inside ws.send), then B confirms and pops the
                # batch + seals the turn while A is still pending.
                task_a = asyncio.create_task(_send_tool_resp(session, resp_a))
                await asyncio.wait_for(call_a_entered.wait(), timeout=5.0)
                await _send_tool_resp(session, resp_b)

                # Now A fails; rollback must restore the batch AND unseal the
                # turn so retrying A joins the same turn as B.
                fail_call_a.set()
                with pytest.raises(
                    ConnectionError, match="transient tool A send error"
                ):
                    await task_a

                # Retry A -> completes batch and seals the single turn [B, A].
                await _send_tool_resp(session, resp_a)

                # Queue a follow-up turn before reading either buffered reply.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Follow-up after retry")],
                    ),
                    turn_complete=True,
                )

                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    gen1, gen2, gen3 = spans

    assert gen1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    gen2_inputs = gen2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert [m["parts"][0]["id"] for m in gen2_inputs] == [
        "call-rb-b",
        "call-rb-a",
    ]
    assert gen3.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Follow-up after retry"},),
        },
    )

    ended = next(
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended.severity_number == SeverityNumber.INFO
    assert ErrorAttributes.ERROR_TYPE not in ended.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("via_legacy_send", [False, True])
@pytest.mark.parametrize("send_b_first", [False, True])
async def test_live_gemini_2_cancelled_tool_call_while_send_suspended_does_not_reopen_turn(
    via_legacy_send: bool,
    send_b_first: bool,
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    call_a_entered = asyncio.Event()
    never_release = asyncio.Event()

    class _SuspendToolAWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            if "call-cancel-a" in raw:
                call_a_entered.set()
                await asyncio.wait_for(never_release.wait(), timeout=5.0)
            await super().send(payload)

    fake_ws = _SuspendToolAWebSocket(
        [
            {"setupComplete": {}},
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-cancel-a",
                            "name": "get_weather",
                            "args": {"city": "Seoul"},
                        },
                        {
                            "id": "call-cancel-b",
                            "name": "get_time",
                            "args": {"city": "Seoul"},
                        },
                    ]
                }
            },
            {"toolCallCancellation": {"ids": ["call-cancel-a"]}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Only Seoul time available."}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply to next prompt."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    async def _send_tool_resp(
        session: Any, resp: genai_types.FunctionResponse
    ) -> None:
        if via_legacy_send:
            await session.send(input=resp)
        else:
            await session.send_tool_response(function_responses=resp)

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(text="Weather and time in Seoul?")
                        ],
                    ),
                    turn_complete=True,
                )
                msg1 = await session._receive()
                assert msg1.tool_call is not None

                resp_a = genai_types.FunctionResponse(
                    id="call-cancel-a",
                    name="get_weather",
                    response={"temp_c": 12},
                )
                resp_b = genai_types.FunctionResponse(
                    id="call-cancel-b",
                    name="get_time",
                    response={"time": "10:00"},
                )

                if send_b_first:
                    await _send_tool_resp(session, resp_b)
                    task_a = asyncio.create_task(
                        _send_tool_resp(session, resp_a)
                    )
                    await asyncio.wait_for(call_a_entered.wait(), timeout=5.0)
                else:
                    task_a = asyncio.create_task(
                        _send_tool_resp(session, resp_a)
                    )
                    await asyncio.wait_for(call_a_entered.wait(), timeout=5.0)
                    await _send_tool_resp(session, resp_b)

                # Server cancels call A while its send task is suspended.
                cancel_msg = await session._receive()
                assert cancel_msg.tool_call_cancellation is not None

                # Cancelling task A must NOT restore cancelled call A or reopen
                # the batch's sealed turn containing confirmed response B.
                task_a.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task_a

                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Next user prompt")],
                    ),
                    turn_complete=True,
                )

                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    gen1, gen2, gen3 = spans

    assert gen1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    gen2_inputs = gen2.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert [m["parts"][0]["id"] for m in gen2_inputs] == ["call-cancel-b"]
    assert gen3.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Next user prompt"},),
        },
    )

    ended = next(
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended.severity_number == SeverityNumber.INFO
    assert ErrorAttributes.ERROR_TYPE not in ended.attributes


@pytest.mark.asyncio
async def test_live_pending_context_update_propagates_to_subsequent_generation_without_replacement(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    has_context_update = hasattr(
        genai_types, "LiveClientContextUpdate"
    ) and hasattr(genai_live.AsyncSession, "send_context_update")
    if not has_context_update:
        return

    def tool_init(query: str) -> str:
        """Initial tool."""
        return query

    def tool_a(query: str) -> str:
        """Tool A."""
        return query

    update_a_written = asyncio.Event()
    release_update_a = asyncio.Event()

    class _DrainingContextUpdateWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            await super().send(payload)
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            if "System A" in raw:
                update_a_written.set()
                await asyncio.wait_for(release_update_a.wait(), timeout=5.0)

    fake_ws = _DrainingContextUpdateWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 1"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 2"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply 3"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Initial system",
                tools=[tool_init],
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                # Turn 1: context update A writes before Turn 1 and stays
                # draining while Replies 1, 2, and 3 arrive and complete.
                update_a_task = asyncio.create_task(
                    session.send_context_update(
                        system_instruction="System A",
                        tools=[tool_a],
                    )
                )
                await asyncio.wait_for(update_a_written.wait(), timeout=5.0)
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 1 prompt")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass

                # Turn 2 has NO replacement context update; it must inherit
                # pending context update A once A finishes draining.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 2 prompt")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass

                # Turn 3 replaces ONLY system_instruction while still
                # inheriting tools from pending update A.
                await session.send_context_update(
                    system_instruction="System C",
                )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Turn 3 prompt")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass

                # All three generations wait on update A's confirmation.
                assert span_exporter.get_finished_spans() == ()

                release_update_a.set()
                await update_a_task

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    by_reply = {
        span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES][0]["parts"][0][
            "content"
        ]: span
        for span in spans
    }

    span1 = by_reply["Reply 1"]
    assert span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 1 prompt"},),
        },
    )
    assert span1.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "System A"},
    )
    assert [
        t["name"] for t in span1.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["tool_a"]

    span2 = by_reply["Reply 2"]
    assert span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 2 prompt"},),
        },
    )
    assert span2.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "System A"},
    )
    assert [
        t["name"] for t in span2.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["tool_a"]

    span3 = by_reply["Reply 3"]
    assert span3.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Turn 3 prompt"},),
        },
    )
    assert span3.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "System C"},
    )
    assert [
        t["name"] for t in span3.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS]
    ] == ["tool_a"]


@pytest.mark.asyncio
async def test_live_close_during_outstanding_receive_ignores_late_buffered_frames(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    has_vad_signal = (
        "voice_activity_detection_signal"
        in genai_types.LiveServerMessage.model_fields
    )

    started_spans: list[str] = []

    class _StartRecorder(SpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            started_spans.append(span.name)

    tracer_provider.add_span_processor(_StartRecorder())

    late_frames: list[dict[str, Any]] = [
        {
            "serverContent": {
                "modelTurn": {
                    "role": "model",
                    "parts": [{"text": "Late buffered model output"}],
                },
            }
        }
    ]
    if has_vad_signal:
        late_frames.append(
            {
                "voiceActivityDetectionSignal": {
                    "vadSignalType": "VAD_SIGNAL_TYPE_SOS"
                }
            }
        )

    for late_frame in late_frames:
        span_exporter.clear()
        log_exporter.clear()
        started_spans.clear()

        recv_entered = asyncio.Event()
        release_recv = asyncio.Event()
        recv_calls = 0

        class _BufferedAfterCloseWebSocket(_FakeWebSocket):
            async def recv(self, decode: bool = True) -> bytes | str:
                nonlocal recv_calls
                recv_calls += 1
                if recv_calls == 2:
                    recv_entered.set()
                    await asyncio.wait_for(release_recv.wait(), timeout=5.0)
                return await super().recv(decode=decode)

        fake_ws = _BufferedAfterCloseWebSocket(
            [{"setupComplete": {}}, late_frame]
        )

        with patch(
            "google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)
        ):
            with instrument(
                GoogleGenAiSdkInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_AND_EVENT",
                extra_env={
                    OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true"
                },
            ):
                client = genai.Client(api_key="test-key", vertexai=False)
                async with client.aio.live.connect(
                    model="gemini-2.5-flash-native-audio-preview-12-2025",
                ) as session:
                    recv_iter = session.receive()
                    recv_task = asyncio.create_task(recv_iter.__anext__())
                    await asyncio.wait_for(recv_entered.wait(), timeout=5.0)

                    # Another task closes the session while receive() is
                    # suspended waiting on the websocket.
                    await session.close()

                    # Release the buffered frame after finalization; it must
                    # still return to the caller without starting any span.
                    release_recv.set()
                    late_msg = await asyncio.wait_for(recv_task, timeout=5.0)
                    assert late_msg is not None
                    await recv_iter.aclose()

        assert started_spans == []
        assert span_exporter.get_finished_spans() == ()
        logs = [log.log_record for log in log_exporter.get_finished_logs()]
        assert [log.event_name for log in logs] == [
            "gen_ai.client.realtime_session.started",
            "gen_ai.client.realtime_session.ended",
        ]
        assert logs[1].severity_number == SeverityNumber.INFO
        assert ErrorAttributes.ERROR_TYPE not in logs[1].attributes


@pytest.mark.asyncio
async def test_live_multimodal_tool_response_preserves_inline_and_file_parts(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    if not hasattr(genai_types, "FunctionResponsePart"):
        return
    # google-genai < 3.0 raises TypeError in json.dumps for raw bytes inside
    # FunctionResponseBlob.data until encode_unserializable_types was added.
    supports_inline_bytes = "encode_unserializable_types" in inspect.getsource(
        genai_live.AsyncSession.send_tool_response
    )

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-mm-1",
                            "name": "capture_screen",
                            "args": {},
                        },
                        {
                            "id": "call-mm-2",
                            "name": "read_doc",
                            "args": {},
                        },
                    ]
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Analyzed image and document."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(text="Inspect screen and doc")
                        ],
                    ),
                    turn_complete=True,
                )
                msg1 = await session._receive()
                assert msg1.tool_call is not None

                call1_parts: list[genai_types.FunctionResponsePart] = []
                if supports_inline_bytes:
                    call1_parts.append(
                        genai_types.FunctionResponsePart(
                            inline_data=genai_types.FunctionResponseBlob(
                                mime_type="image/png",
                                data=b"\x89PNG\r\n",
                            )
                        )
                    )
                call1_parts.append(
                    genai_types.FunctionResponsePart(
                        file_data=genai_types.FunctionResponseFileData(
                            mime_type="image/jpeg",
                            file_uri="gs://bucket/frame.jpg",
                        )
                    )
                )

                await session.send_tool_response(
                    function_responses=[
                        genai_types.FunctionResponse(
                            id="call-mm-1",
                            name="capture_screen",
                            response={"status": "ok"},
                            parts=call1_parts,
                        ),
                        genai_types.FunctionResponse(
                            id="call-mm-2",
                            name="read_doc",
                            parts=[
                                genai_types.FunctionResponsePart(
                                    file_data=genai_types.FunctionResponseFileData(
                                        mime_type="application/pdf",
                                        file_uri="https://example.com/report.pdf",
                                    )
                                )
                            ],
                        ),
                    ]
                )
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    gen2_inputs = spans[1].attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    expected_parts: list[dict[str, Any]] = [
        {
            "type": "tool_call_response",
            "id": "call-mm-1",
            "response": {"status": "ok"},
        },
    ]
    if supports_inline_bytes:
        expected_parts.append(
            {
                "type": "blob",
                "modality": "image",
                "mime_type": "image/png",
                "content": b"\x89PNG\r\n",
            }
        )
    expected_parts.extend(
        [
            {
                "type": "uri",
                "modality": "image",
                "mime_type": "image/jpeg",
                "uri": "gs://bucket/frame.jpg",
            },
            {
                "type": "tool_call_response",
                "id": "call-mm-2",
                "response": None,
            },
            {
                "type": "uri",
                "modality": "document",
                "mime_type": "application/pdf",
                "uri": "https://example.com/report.pdf",
            },
        ]
    )
    assert gen2_inputs == ({"role": "tool", "parts": tuple(expected_parts)},)


@pytest.mark.asyncio
async def test_live_legacy_send_client_content_uses_payload_turn_complete_ignoring_end_of_turn(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply for Turn 1"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply for Turn 2"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Dict payload without turn_complete ignores outer end_of_turn=True.
                await session.send(
                    input={
                        "turns": [
                            {
                                "role": "user",
                                "parts": [{"text": "Dict part A"}],
                            }
                        ]
                    },
                    end_of_turn=True,
                )
                # Second dict sets turn_complete=True (ignoring end_of_turn=False).
                await session.send(
                    input={
                        "turns": [
                            {
                                "role": "user",
                                "parts": [{"text": "Dict part B"}],
                            }
                        ],
                        "turn_complete": True,
                    },
                    end_of_turn=False,
                )
                # LiveClientContent without turn_complete ignores end_of_turn=True.
                await session.send(
                    input=genai_types.LiveClientContent(
                        turns=[
                            genai_types.Content(
                                role="user",
                                parts=[genai_types.Part(text="Model part C")],
                            )
                        ]
                    ),
                    end_of_turn=True,
                )
                # Plain string honors end_of_turn=True and seals Turn 2.
                await session.send(input="Text part D", end_of_turn=True)

                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Dict part A"},),
        },
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Dict part B"},),
        },
    )
    assert spans[1].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Model part C"},),
        },
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Text part D"},),
        },
    )


@pytest.mark.asyncio
async def test_live_gemini_2_silent_progress_tool_response_preserves_turn_boundary(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            # Batch 1: non-blocking tool call receiving silent progress then final result.
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-nb-1",
                            "name": "long_job",
                            "args": {"job": "build"},
                        }
                    ]
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Build finished!"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Follow-up answer."}],
                    },
                    "turnComplete": True,
                }
            },
            # Batch 2: final response is also SILENT; next user prompt triggers reply.
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-nb-2",
                            "name": "log_metric",
                            "args": {"name": "latency"},
                        }
                    ]
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Summarized silent result."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Start build")],
                    ),
                    turn_complete=True,
                )
                msg1 = await session._receive()
                assert msg1.tool_call is not None

                # Silent progress update (will_continue=True) must NOT complete
                # the batch or seal the turn.
                await session.send_tool_response(
                    function_responses=genai_types.FunctionResponse(
                        id="call-nb-1",
                        name="long_job",
                        response={"progress": "50%"},
                        scheduling=genai_types.FunctionResponseScheduling.SILENT,
                        will_continue=True,
                    )
                )
                # Final result sent before reply is read joins the same turn
                # and seals it.
                await session.send_tool_response(
                    function_responses=genai_types.FunctionResponse(
                        id="call-nb-1",
                        name="long_job",
                        response={"result": "done"},
                        scheduling=genai_types.FunctionResponseScheduling.WHEN_IDLE,
                        will_continue=False,
                    )
                )
                # Follow-up user prompt starts a distinct turn.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Next question")],
                    ),
                    turn_complete=True,
                )

                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

                # Batch 2: final SILENT tool response does not trigger a reply
                # on its own, so the triggering user prompt joins the turn.
                msg4 = await session._receive()
                assert msg4.tool_call is not None
                await session.send_tool_response(
                    function_responses=genai_types.FunctionResponse(
                        id="call-nb-2",
                        name="log_metric",
                        response={"logged": True},
                        scheduling=genai_types.FunctionResponseScheduling.SILENT,
                        will_continue=False,
                    )
                )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(text="Summarize silent result")
                        ],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5
    gen1, gen2, gen3, gen4, gen5 = spans

    assert gen1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert gen2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "tool",
            "parts": (
                {
                    "type": "tool_call_response",
                    "id": "call-nb-1",
                    "response": {"progress": "50%"},
                },
            ),
        },
        {
            "role": "tool",
            "parts": (
                {
                    "type": "tool_call_response",
                    "id": "call-nb-1",
                    "response": {"result": "done"},
                },
            ),
        },
    )
    assert gen3.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Next question"},),
        },
    )
    assert gen4.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert gen5.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "tool",
            "parts": (
                {
                    "type": "tool_call_response",
                    "id": "call-nb-2",
                    "response": {"logged": True},
                },
            ),
        },
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Summarize silent result"},),
        },
    )


@pytest.mark.asyncio
async def test_live_interrupted_before_first_output_retires_queued_input_without_phantom_span(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            # Case A: Turn 1 interrupted before any model output.
            {
                "serverContent": {
                    "interrupted": True,
                    "turnComplete": True,
                }
            },
            # Turn 2 succeeds normally.
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Answer to Turn 2"}],
                    },
                    "turnComplete": True,
                }
            },
            # Case B: stray trailing completion frame while Turn 3 is queued.
            {
                "serverContent": {
                    "generationComplete": True,
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Answer to Turn 3"}],
                    },
                    "turnComplete": True,
                }
            },
            # Case C: audio + inputTranscription interrupted before output,
            # followed by a successful audio turn.
            {
                "serverContent": {
                    "inputTranscription": {"text": "Aborted audio speech"},
                }
            },
            {
                "serverContent": {
                    "interrupted": True,
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "inputTranscription": {"text": "Kept audio speech"},
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Answer to audio 2"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                # Case A: Queue Turn 1 and Turn 2 before reading Turn 1's
                # no-output interruption frame.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Interrupted prompt")],
                    ),
                    turn_complete=True,
                )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Successful prompt")],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

                # Case B: Queue Turn 3 before consuming a stray trailing
                # generationComplete+turnComplete frame.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                text="Turn 3 after stray completion"
                            )
                        ],
                    ),
                    turn_complete=True,
                )
                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

                # Case C: Audio + inputTranscription interrupted before output,
                # then followed by a successful audio turn.
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"AAAA", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                async for _ in session.receive():
                    pass

                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"BBBB", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    span1, span2, span3 = spans

    assert span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "Successful prompt"},),
        },
    )
    assert span1.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES] == (
        {
            "role": "assistant",
            "parts": ({"type": "text", "content": "Answer to Turn 2"},),
            "finish_reason": "stop",
        },
    )

    assert span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {"type": "text", "content": "Turn 3 after stray completion"},
            ),
        },
    )
    assert span2.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES] == (
        {
            "role": "assistant",
            "parts": ({"type": "text", "content": "Answer to Turn 3"},),
            "finish_reason": "stop",
        },
    )

    assert span3.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"BBBB",
                },
                {"type": "transcription", "content": "Kept audio speech"},
            ),
        },
    )
    assert span3.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES] == (
        {
            "role": "assistant",
            "parts": ({"type": "text", "content": "Answer to audio 2"},),
            "finish_reason": "stop",
        },
    )


@pytest.mark.asyncio
async def test_live_gemini_2_overlapping_tool_batches_match_by_id_out_of_order_and_rollback(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    b2a_entered = asyncio.Event()
    fail_b2a = asyncio.Event()
    b2a_attempts = 0

    class _FailFirstB2AWebSocket(_FakeWebSocket):
        async def send(self, payload: str | bytes) -> None:
            nonlocal b2a_attempts
            raw = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else str(payload)
            )
            if "call-b2a" in raw and b2a_attempts == 0:
                b2a_attempts += 1
                b2a_entered.set()
                await asyncio.wait_for(fail_b2a.wait(), timeout=5.0)
                raise ConnectionError("transient b2a send error")
            await super().send(payload)

    fake_ws = _FailFirstB2AWebSocket(
        [
            {"setupComplete": {}},
            # Batch 1: slow non-blocking tool call.
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-b1",
                            "name": "slow_task",
                            "args": {"job": "1"},
                        }
                    ]
                }
            },
            # Batch 2: overlapping two-call batch before Batch 1 finishes.
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call-b2a",
                            "name": "fast_a",
                            "args": {"x": 1},
                        },
                        {
                            "id": "call-b2b",
                            "name": "fast_b",
                            "args": {"x": 2},
                        },
                    ]
                }
            },
            # Reply to Batch 1 (completed first after Batch 2 rolled back).
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Batch 1 reply."}],
                    },
                    "turnComplete": True,
                }
            },
            # Reply to Batch 2 (completed after retrying call-b2a).
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Batch 2 reply."}],
                    },
                    "turnComplete": True,
                }
            },
            # Follow-up user prompt reply.
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Follow-up reply."}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(fake_ws)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Start slow task")],
                    ),
                    turn_complete=True,
                )
                msg1 = await session._receive()
                assert msg1.tool_call is not None

                # User sends another prompt before slow_task finishes,
                # producing an overlapping second tool-call batch.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Run fast tasks")],
                    ),
                    turn_complete=True,
                )
                msg2 = await session._receive()
                assert msg2.tool_call is not None

                resp_b1 = genai_types.FunctionResponse(
                    id="call-b1",
                    name="slow_task",
                    response={"status": "b1_done"},
                )
                resp_b2a = genai_types.FunctionResponse(
                    id="call-b2a",
                    name="fast_a",
                    response={"status": "b2a_done"},
                )
                resp_b2b = genai_types.FunctionResponse(
                    id="call-b2b",
                    name="fast_b",
                    response={"status": "b2b_done"},
                )

                # Respond to later Batch 2 out-of-order: b2a suspends, b2b
                # confirms and completes Batch 2 while Batch 1 stays pending.
                task_b2a = asyncio.create_task(
                    session.send_tool_response(function_responses=resp_b2a)
                )
                await asyncio.wait_for(b2a_entered.wait(), timeout=5.0)
                await session.send_tool_response(function_responses=resp_b2b)

                # Roll back b2a -> Batch 2 reopens and its turn unseals.
                fail_b2a.set()
                with pytest.raises(
                    ConnectionError, match="transient b2a send error"
                ):
                    await task_b2a

                # Complete Batch 1 while Batch 2 is still incomplete; Batch 1's
                # sealed turn must stay separate from Batch 2's partial turn.
                await session.send_tool_response(function_responses=resp_b1)

                # Retry b2a -> joins Batch 2's turn [b2b, b2a] and seals it.
                await session.send_tool_response(function_responses=resp_b2a)

                # Queue a follow-up user turn before reading buffered replies.
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="After both batches")],
                    ),
                    turn_complete=True,
                )

                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 5
    gen1, gen2, gen3, gen4, gen5 = spans

    assert gen1.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert gen2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert [
        m["parts"][0]["id"]
        for m in gen3.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    ] == ["call-b1"]
    assert [
        m["parts"][0]["id"]
        for m in gen4.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    ] == ["call-b2b", "call-b2a"]
    assert gen5.attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": ({"type": "text", "content": "After both batches"},),
        },
    )

    ended = next(
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name == "gen_ai.client.realtime_session.ended"
    )
    assert ended.severity_number == SeverityNumber.INFO
    assert ErrorAttributes.ERROR_TYPE not in ended.attributes


@pytest.mark.asyncio
async def test_live_two_queued_audio_turns_preserve_distinct_transcripts_in_order(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    from google import genai

    has_vad_signal = (
        "voice_activity_detection_signal"
        in genai_types.LiveServerMessage.model_fields
    )

    # Part A: Two audio turns queued before reading buffered transcripts/replies.
    ws_a = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "inputTranscription": {"text": "First audio question"},
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "First answer"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "inputTranscription": {"text": "Second audio question"},
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Second answer"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(ws_a)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"1111", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"2222", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)

                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans_a = span_exporter.get_finished_spans()
    assert len(spans_a) == 2
    assert spans_a[0].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"1111",
                },
                {"type": "transcription", "content": "First audio question"},
            ),
        },
    )
    assert spans_a[1].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"2222",
                },
                {"type": "transcription", "content": "Second audio question"},
            ),
        },
    )

    if not has_vad_signal:
        return

    # Part B: Both transcripts (separated by server VAD SOS/EOS) arrive before
    # either modelTurn reply.
    span_exporter.clear()
    ws_b = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "voiceActivityDetectionSignal": {
                    "vadSignalType": "VAD_SIGNAL_TYPE_SOS"
                }
            },
            {
                "serverContent": {
                    "inputTranscription": {"text": "VAD turn 1"},
                }
            },
            {
                "voiceActivityDetectionSignal": {
                    "vadSignalType": "VAD_SIGNAL_TYPE_EOS"
                }
            },
            {
                "voiceActivityDetectionSignal": {
                    "vadSignalType": "VAD_SIGNAL_TYPE_SOS"
                }
            },
            {
                "serverContent": {
                    "inputTranscription": {"text": "VAD turn 2"},
                }
            },
            {
                "voiceActivityDetectionSignal": {
                    "vadSignalType": "VAD_SIGNAL_TYPE_EOS"
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "VAD reply 1"}],
                    },
                    "turnComplete": True,
                }
            },
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "VAD reply 2"}],
                    },
                    "turnComplete": True,
                }
            },
        ]
    )

    with patch("google.genai.live.ws_connect", _make_fake_ws_connect(ws_b)):
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_AND_EVENT",
        ):
            client = genai.Client(api_key="test-key", vertexai=False)
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"3333", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"4444", mime_type="audio/pcm;rate=16000"
                    )
                )
                await session.send_realtime_input(audio_stream_end=True)

                async for _ in session.receive():
                    pass
                async for _ in session.receive():
                    pass

    spans_b = span_exporter.get_finished_spans()
    assert len(spans_b) == 2
    assert spans_b[0].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"3333",
                },
                {"type": "transcription", "content": "VAD turn 1"},
            ),
        },
    )
    assert spans_b[1].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": "audio/pcm;rate=16000",
                    "content": b"4444",
                },
                {"type": "transcription", "content": "VAD turn 2"},
            ),
        },
    )
