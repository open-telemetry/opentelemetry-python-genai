# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Google GenAI Live API (realtime_inference and user_speech)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

from google.genai import Client
from google.genai import types as genai_types

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _ConformanceWebSocket:
    def __init__(self, server_frames: list[dict[str, Any]]) -> None:
        self._frames = list(server_frames)

    async def send(self, payload: str | bytes) -> None:
        pass

    async def recv(self, decode: bool = True) -> bytes | str:  # pylint: disable=unused-argument
        if not self._frames:
            return b"{}"
        encoded = json.dumps(self._frames.pop(0)).encode("utf-8")
        return encoded if not decode else encoded.decode("utf-8")

    async def close(self) -> None:
        pass


def _make_ws_connect(ws: _ConformanceWebSocket):
    @contextlib.asynccontextmanager
    async def _ws_connect(
        uri: str,  # pylint: disable=unused-argument
        additional_headers: Any = None,  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> AsyncIterator[_ConformanceWebSocket]:
        yield ws

    return _ws_connect


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_types(messages_json: str | None) -> list[str]:
    messages = json.loads(messages_json) if messages_json else []
    return [part["type"] for message in messages for part in message["parts"]]


def _blob_modalities(messages_json: str | None) -> list[str]:
    messages = json.loads(messages_json) if messages_json else []
    return [
        part["modality"]
        for message in messages
        for part in message["parts"]
        if part["type"] == "blob"
    ]


def _get_weather(city: str) -> dict[str, Any]:
    """Get the current weather for a city."""
    return {"city": city, "temp_c": 20}


_COMMON_LIVE_VIOLATIONS = (
    ExpectedViolation(
        advice_id="genai_operation_name_unknown",
        message_substring="realtime_inference",
    ),
    ExpectedViolation(
        advice_id="missing_event",
        message_substring="gen_ai.client.realtime_session.started",
    ),
    ExpectedViolation(
        advice_id="missing_event",
        message_substring="gen_ai.client.realtime_session.ended",
    ),
    ExpectedViolation(
        advice_id="missing_attribute",
        message_substring="gen_ai.realtime_session.id",
    ),
)


class LiveScenario(Scenario):
    expected_spans = {
        "user_speech": 1,
        "realtime_inference": 1,
    }
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_operation_name_unknown",
            message_substring="user_speech",
        ),
        *_COMMON_LIVE_VIOLATIONS,
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,  # pylint: disable=unused-argument
    ) -> None:
        server_frames: list[dict[str, Any]] = [
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
                    "inputTranscription": {"text": "Hello Gemini"},
                    "modelTurn": {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": "Formulating greeting"},
                            {
                                "inlineData": {
                                    "data": "AAE=",
                                    "mimeType": "audio/pcm;rate=24000",
                                }
                            },
                            {"text": "Hello there!"},
                        ],
                    },
                    "outputTranscription": {"text": "Hello there!"},
                    "generationComplete": True,
                    "turnComplete": True,
                },
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "responseTokenCount": 18,
                    "cachedContentTokenCount": 10,
                    "thoughtsTokenCount": 4,
                    "totalTokenCount": 48,
                    "promptTokensDetails": [
                        {"modality": "AUDIO", "tokenCount": 24},
                        {"modality": "TEXT", "tokenCount": 6},
                    ],
                    "responseTokensDetails": [
                        {"modality": "AUDIO", "tokenCount": 12},
                        {"modality": "TEXT", "tokenCount": 6},
                    ],
                    "cacheTokensDetails": [
                        {"modality": "AUDIO", "tokenCount": 6},
                        {"modality": "TEXT", "tokenCount": 4},
                    ],
                },
            },
        ]
        ws = _ConformanceWebSocket(server_frames)

        async def _drive() -> None:
            client = Client(
                api_key="test_google_genai_api_key", vertexai=False
            )
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                system_instruction="Be concise.",
                max_output_tokens=128,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_level=genai_types.ThinkingLevel.HIGH
                ),
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=b"\x00\x01" * 64,
                        mime_type="audio/pcm;rate=16000",
                    ),
                )
                await session.send_realtime_input(audio_stream_end=True)
                async for _ in session.receive():
                    pass

        with patch("google.genai.live.ws_connect", _make_ws_connect(ws)):
            with instrument(
                GoogleGenAiSdkInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_AND_EVENT",
            ):
                asyncio.run(_drive())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)

        inf_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "realtime_inference"
        ]
        assert len(inf_spans) == 1
        inf_span = inf_spans[0]
        assert _attr(inf_span, "gen_ai.output.type") == "speech"
        assert _attr(inf_span, "gen_ai.request.max_tokens") == 128
        assert _attr(inf_span, "gen_ai.request.reasoning.level") == "HIGH"
        assert _attr(inf_span, "gen_ai.realtime_session.id")
        assert _attr(inf_span, "gen_ai.usage.input_tokens") == 30
        assert _attr(inf_span, "gen_ai.usage.output_tokens") == 18
        assert _attr(inf_span, "gen_ai.usage.cache_read.input_tokens") == 10
        assert _attr(inf_span, "gen_ai.usage.reasoning.output_tokens") == 4
        assert _attr(inf_span, "gen_ai.usage.audio.input_tokens") == 24
        assert _attr(inf_span, "gen_ai.usage.text.input_tokens") == 6
        assert _attr(inf_span, "gen_ai.usage.audio.output_tokens") == 12
        assert _attr(inf_span, "gen_ai.usage.text.output_tokens") == 6
        assert (
            _attr(inf_span, "gen_ai.usage.audio.cache_read.input_tokens") == 6
        )
        assert (
            _attr(inf_span, "gen_ai.usage.text.cache_read.input_tokens") == 4
        )
        assert _part_types(_attr(inf_span, "gen_ai.input.messages")) == [
            "blob",
            "transcription",
        ]
        assert _blob_modalities(_attr(inf_span, "gen_ai.input.messages")) == [
            "audio"
        ]
        assert _part_types(_attr(inf_span, "gen_ai.output.messages")) == [
            "reasoning",
            "blob",
            "text",
            "transcription",
        ]

        event_names = {
            entry["log"]["event_name"]
            for entry in report["samples"]
            if "log" in entry and "event_name" in entry["log"]
        }
        assert event_names == {
            "gen_ai.client.realtime_session.started",
            "gen_ai.client.realtime_session.ended",
        }


class LiveMultimodalAndToolsScenario(Scenario):
    expected_spans = {
        "realtime_inference": 3,
    }
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _COMMON_LIVE_VIOLATIONS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,  # pylint: disable=unused-argument
    ) -> None:
        # Session 1: Gemini 2.5 (two-generation tool exchange + image/video/text input)
        ws_25 = _ConformanceWebSocket(
            [
                {"setupComplete": {}},
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": "call-1",
                                "name": "_get_weather",
                                "args": {"city": "Tokyo"},
                            }
                        ]
                    },
                    "usageMetadata": {
                        "promptTokenCount": 40,
                        "responseTokenCount": 8,
                    },
                },
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": "It is 20C in Tokyo."}],
                        },
                        "turnComplete": True,
                    },
                    "usageMetadata": {
                        "promptTokenCount": 52,
                        "responseTokenCount": 10,
                    },
                },
            ]
        )
        # Session 2: Gemini 3.1 (single-generation tool exchange)
        ws_31 = _ConformanceWebSocket(
            [
                {"setupComplete": {}},
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": "call-2",
                                "name": "_get_weather",
                                "args": {"city": "Paris"},
                            }
                        ]
                    }
                },
                {
                    "serverContent": {
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": "It is 18C in Paris."}],
                        },
                        "turnComplete": True,
                    },
                    "usageMetadata": {
                        "promptTokenCount": 35,
                        "responseTokenCount": 14,
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
        ) -> AsyncIterator[_ConformanceWebSocket]:
            yield sockets.pop(0)

        async def _drive() -> None:
            client = Client(
                api_key="test_google_genai_api_key", vertexai=False
            )
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                tools=[_get_weather],
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send_realtime_input(
                    media=genai_types.Blob(
                        data=b"\xff\xd8\xff\xe0",
                        mime_type="image/jpeg",
                    )
                )
                await session.send_realtime_input(
                    media=genai_types.Blob(
                        data=b"\x00\x00\x00\x18",
                        mime_type="video/mp4",
                    )
                )
                await session.send_client_content(
                    turns=genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text="Weather in Tokyo?")],
                    ),
                    turn_complete=True,
                )
                msg = await session._receive()
                assert msg.tool_call is not None
                await session.send_tool_response(
                    function_responses=[
                        genai_types.FunctionResponse(
                            id="call-1",
                            name="_get_weather",
                            response={"temp_c": 20},
                        )
                    ]
                )
                async for _ in session.receive():
                    pass

            async with client.aio.live.connect(
                model="gemini-3.1-flash-live-preview",
                config=config,
            ) as session:
                await session.send(input="Weather in Paris?", end_of_turn=True)
                async for chunk in session.receive():
                    if chunk.tool_call:
                        await session.send_tool_response(
                            function_responses=[
                                genai_types.FunctionResponse(
                                    id="call-2",
                                    name="_get_weather",
                                    response={"temp_c": 18},
                                )
                            ]
                        )

        with patch("google.genai.live.ws_connect", _multi_ws_connect):
            with instrument(
                GoogleGenAiSdkInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_AND_EVENT",
            ):
                asyncio.run(_drive())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)

        inf_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "realtime_inference"
        ]
        assert len(inf_spans) == 3
        gen25_1, gen25_2, gen31 = inf_spans

        assert _attr(gen25_1, "gen_ai.output.type") == "text"
        assert _attr(gen25_1, "gen_ai.response.finish_reasons") == [
            "tool_call"
        ]
        assert _part_types(_attr(gen25_1, "gen_ai.input.messages")) == [
            "blob",
            "blob",
            "text",
        ]
        assert _blob_modalities(_attr(gen25_1, "gen_ai.input.messages")) == [
            "image",
            "video",
        ]
        assert _part_types(_attr(gen25_1, "gen_ai.output.messages")) == [
            "tool_call"
        ]
        tools_json = json.loads(_attr(gen25_1, "gen_ai.tool.definitions"))
        assert tools_json[0]["name"] == "_get_weather"

        assert _attr(gen25_2, "gen_ai.response.finish_reasons") == ["stop"]
        assert _part_types(_attr(gen25_2, "gen_ai.input.messages")) == [
            "tool_call_response"
        ]
        assert _part_types(_attr(gen25_2, "gen_ai.output.messages")) == [
            "text"
        ]

        assert _attr(gen31, "gen_ai.response.finish_reasons") == ["stop"]
        assert _part_types(_attr(gen31, "gen_ai.input.messages")) == ["text"]
        assert _part_types(_attr(gen31, "gen_ai.output.messages")) == ["text"]


class LiveNoContentScenario(Scenario):
    expected_spans = {
        "realtime_inference": 1,
    }
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _COMMON_LIVE_VIOLATIONS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,  # pylint: disable=unused-argument
    ) -> None:
        ws = _ConformanceWebSocket(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "inputTranscription": {"text": "Secret prompt"},
                        "modelTurn": {
                            "role": "model",
                            "parts": [{"text": "Secret answer"}],
                        },
                        "turnComplete": True,
                    },
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "responseTokenCount": 5,
                    },
                },
            ]
        )

        async def _drive() -> None:
            client = Client(
                api_key="test_google_genai_api_key", vertexai=False
            )
            config = genai_types.LiveConnectConfig(
                response_modalities=[genai_types.Modality.TEXT],
                system_instruction="Secret instruction",
                tools=[_get_weather],
            )
            async with client.aio.live.connect(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=config,
            ) as session:
                await session.send(input="Secret prompt", end_of_turn=True)
                async for _ in session.receive():
                    pass

        with patch("google.genai.live.ws_connect", _make_ws_connect(ws)):
            with instrument(
                GoogleGenAiSdkInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="NO_CONTENT",
            ):
                asyncio.run(_drive())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)

        inf_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "realtime_inference"
        ]
        assert len(inf_spans) == 1
        inf_span = inf_spans[0]
        assert _attr(inf_span, "gen_ai.input.messages") is None
        assert _attr(inf_span, "gen_ai.output.messages") is None
        assert _attr(inf_span, "gen_ai.system_instructions") is None
        assert _attr(inf_span, "gen_ai.tool.definitions") is None
