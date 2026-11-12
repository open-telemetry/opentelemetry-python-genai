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
                            {
                                "inlineData": {
                                    "data": "AAE=",
                                    "mimeType": "audio/pcm;rate=24000",
                                }
                            }
                        ],
                    },
                    "outputTranscription": {"text": "Hello there!"},
                    "generationComplete": True,
                    "turnComplete": True,
                },
                "usageMetadata": {
                    "promptTokenCount": 24,
                    "responseTokenCount": 12,
                    "totalTokenCount": 36,
                    "promptTokensDetails": [
                        {"modality": "AUDIO", "tokenCount": 24}
                    ],
                    "responseTokensDetails": [
                        {"modality": "AUDIO", "tokenCount": 12}
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
        assert _attr(inf_span, "gen_ai.realtime_session.id")
        assert _part_types(_attr(inf_span, "gen_ai.input.messages")) == [
            "blob",
            "transcription",
        ]
        assert _part_types(_attr(inf_span, "gen_ai.output.messages")) == [
            "blob",
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
