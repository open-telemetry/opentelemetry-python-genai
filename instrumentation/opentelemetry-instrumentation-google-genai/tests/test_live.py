# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Google GenAI Live API (realtime) instrumentation (PR #394)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from google.genai import live as genai_live
from google.genai import types as genai_types
from google.genai.live import AsyncLive

from opentelemetry._logs import SeverityNumber
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai import (
    live as live_module,
)
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
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
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

    input_msgs = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "blob"
    assert input_msgs[0]["parts"][0]["modality"] == "audio"
    assert input_msgs[0]["parts"][1] == {
        "type": "transcription",
        "content": "Say hello in three words.",
    }

    output_msgs = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["finish_reason"] == "stop"
    part_types = [p["type"] for p in output_msgs[0]["parts"]]
    assert part_types == ["reasoning", "blob", "transcription"]
    assert output_msgs[0]["parts"][2]["content"] == "Hello there friend."

    sys_instr = json.loads(attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS])
    assert sys_instr == [{"type": "text", "content": "Be concise."}]

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
    s1_out = json.loads(span1.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert s1_out[0]["finish_reason"] == "tool_call"
    assert s1_out[0]["parts"][0]["type"] == "tool_call"
    assert s1_out[0]["parts"][0]["name"] == "get_weather"

    assert span2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert span2.attributes[GenAI.GEN_AI_OUTPUT_TYPE] == "speech"
    s2_in = json.loads(span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
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
    in_msgs_1 = json.loads(span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(in_msgs_1) == 1
    assert in_msgs_1[0]["role"] == "user"

    out_msgs_1 = json.loads(span1.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert len(out_msgs_1) == 1
    assert out_msgs_1[0]["finish_reason"] == "stop"

    # Second turn must only contain the new user message, not the second tool reply
    in_msgs_2 = json.loads(span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(in_msgs_2) == 1
    assert in_msgs_2[0]["role"] == "user"
    assert in_msgs_2[0]["parts"] == [{"type": "text", "content": "Thanks!"}]


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
    gen1_in = json.loads(gen1_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(gen1_in) == 1
    assert gen1_in[0]["role"] == "user"
    assert gen1_in[0]["parts"] == [
        {"type": "transcription", "content": "What is the weather?"}
    ]

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

    in_msgs = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
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
                assert tracker._pending_audio_chunks == []
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

    in1 = json.loads(span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(in1) == 1
    assert in1[0]["parts"][1] == {
        "type": "transcription",
        "content": "Tell me about Paris.",
    }

    in2 = json.loads(span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
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
    turn2_in = json.loads(span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(turn2_in) == 1
    assert turn2_in[0]["role"] == "user"
    assert [p["type"] for p in turn2_in[0]["parts"]] == [
        "blob",
        "transcription",
    ]
    assert turn2_in[0]["parts"][0]["content"] == "AQIDBAUG"
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
            client = genai.Client(api_key="test-key")
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
    tools_defs = json.loads(s25_1.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS])
    assert tools_defs[0]["name"] == "get_weather"

    assert s25_2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    s25_2_in = json.loads(s25_2.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert s25_2_in[0]["parts"][0]["type"] == "tool_call_response"

    if has_live_thinking:
        assert s31.attributes["gen_ai.request.reasoning.level"] == "HIGH"
    assert s31.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    s31_in = json.loads(s31.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(s31_in) == 1
    assert s31_in[0]["parts"] == [
        {"type": "text", "content": "Compare Tokyo and Osaka."}
    ]


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
            client = genai.Client(api_key="test-key")
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
    in1 = json.loads(span1.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert in1[0]["parts"][1]["content"] == "Tell me about Paris."

    assert span2.status.status_code == StatusCode.ERROR
    assert span2.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    assert span2.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("error",)
    in2 = json.loads(span2.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
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
    fake_ws = _FakeWebSocket(
        [
            {"setupComplete": {}},
            {
                "serverContent": {
                    "modelTurn": {
                        "role": "model",
                        "parts": [{"text": "Reply after retries."}],
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
            client = genai.Client(api_key="test-key")
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
    assert len(spans) == 2
    inf_span, speech_span = spans[0], spans[1]

    in_msgs = json.loads(inf_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(in_msgs) == 6
    assert in_msgs[0]["parts"] == [
        {"type": "text", "content": "Valid realtime text"}
    ]
    assert in_msgs[1]["parts"] == [
        {"type": "text", "content": "Valid client content"}
    ]
    assert in_msgs[2]["parts"][0]["id"] == "call-ok"
    assert in_msgs[3]["parts"] == [
        {"type": "text", "content": "Sent via send()"}
    ]
    assert in_msgs[4]["parts"][0]["modality"] == "image"
    assert in_msgs[5]["parts"][0]["id"] == "call-send-fn"

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
                client = genai.Client(api_key="test-key")
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
            client = genai.Client(api_key="test-key")
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

    sys1 = json.loads(span1.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS])
    tools1 = json.loads(span1.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS])
    assert sys1 == [{"type": "text", "content": "Initial system instruction"}]
    assert [t["name"] for t in tools1] == ["tool_v1"]

    sys2 = json.loads(span2.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS])
    tools2 = json.loads(span2.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS])
    assert sys2 == [{"type": "text", "content": "Updated system instruction"}]
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
                client = genai.Client(api_key="test-key")
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
            client = genai.Client(api_key="test-key")
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
