# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai interactions with inline and URI media inputs."""

from __future__ import annotations

import base64
import io
import os
import struct
import wave
import zlib
from typing import Any

from google.genai import Client
from google.genai.interactions import (
    AudioContent,
    DocumentContent,
    ImageContent,
    TextContent,
    UserInputStep,
    VideoContent,
)

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


def _make_minimal_png() -> bytes:
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + chunk
            + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def _make_minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 400)
    return buf.getvalue()


_PROMPT = "Reply with OK if you received all attachments."
_DOCUMENT_URI = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/salary_table.pdf"
_INLINE_DOCUMENT = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 200 200]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 44>>stream\nBT /F1 12 Tf 20 100 Td (City: Paris) Tj ET\nendstream\nendobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000232 00000 n \n0000000297 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n391\n%%EOF\n"
)
_INLINE_DOCUMENT_B64 = base64.b64encode(_INLINE_DOCUMENT).decode("ascii")

_IMAGE_URI = "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/drawing1.png"
_INLINE_IMAGE = _make_minimal_png()
_INLINE_IMAGE_B64 = base64.b64encode(_INLINE_IMAGE).decode("ascii")

_AUDIO_URI = "https://storage.googleapis.com/cloud-samples-data/generative-ai/audio/hello_gemini_are_you_there.wav"
_INLINE_AUDIO = _make_minimal_wav()
_INLINE_AUDIO_B64 = base64.b64encode(_INLINE_AUDIO).decode("ascii")

_VIDEO_URI = "https://storage.googleapis.com/cloud-samples-data/generative-ai/video/cartwheel.mp4"
# Minimal 16x16 H.264 MP4 generated with ffmpeg (no metadata/SEI).
_INLINE_VIDEO_B64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMGbW9vdgAAAGxtdmhkAAAAAAAAAA"
    "AAAAAAAAAD6AAAB9AAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA"
    "AAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAlV0cmFrAAAAXHRraGQAAA"
    "ADAAAAAAAAAAAAAAABAAAAAAAAB9AAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAB"
    "AAAAAAAAAAAAAAAAAABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAfQAA"
    "AAAAABAAAAAAHNbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAAAgABVxAAAAAAALWhkbHIA"
    "AAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABeG1pbmYAAAAUdm1oZAAAAA"
    "EAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAThzdGJsAAAA"
    "uHN0c2QAAAAAAAAAAQAAAKhhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAABAAEABIAAAASA"
    "AAAAAAAAABDExhdmMgbGlieDI2NAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAALmF2Y0MBQsAK"
    "/+EAFmdCwArZHsBEAAADAAQAAAMACDxImSABAAVoy4PLIAAAABBwYXNwAAAAAQAAAAEAAAAUYn"
    "RydAAAAAAAAACIAAAAAAAAABhzdHRzAAAAAAAAAAEAAAACAABAAAAAABRzdHNzAAAAAAAAAAEA"
    "AAABAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAACAAAAAQAAABxzdHN6AAAAAAAAAAAAAAACAAAAGA"
    "AAAAoAAAAUc3RjbwAAAAAAAAABAAADNgAAAD11ZHRhAAAANW1ldGEAAAAAAAAAIWhkbHIAAAAA"
    "AAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAACGlsc3QAAAAIZnJlZQAAACptZGF0AAAAFGWIhAW8Ri"
    "gAD9jHAAGwaOAAI0OAAAAABkGaOAr6gA=="
)
_INLINE_VIDEO = base64.b64decode(_INLINE_VIDEO_B64)


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


class InteractionsMediaInputScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or "test_google_genai_api_key"
        )
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("interactions_media_input_conformance.yaml"):
                client = Client(api_key=api_key, vertexai=False)
                client.interactions.create(
                    model="gemini-2.5-flash",
                    input=[
                        UserInputStep(
                            type="user_input",
                            content=[
                                TextContent(type="text", text=_PROMPT),
                                DocumentContent(
                                    type="document",
                                    mime_type="application/pdf",
                                    uri=_DOCUMENT_URI,
                                ),
                                DocumentContent(
                                    type="document",
                                    mime_type="application/pdf",
                                    data=_INLINE_DOCUMENT_B64,
                                ),
                                ImageContent(
                                    type="image",
                                    mime_type="image/png",
                                    uri=_IMAGE_URI,
                                ),
                                ImageContent(
                                    type="image",
                                    mime_type="image/png",
                                    data=_INLINE_IMAGE_B64,
                                ),
                                AudioContent(
                                    type="audio",
                                    mime_type="audio/wav",
                                    uri=_AUDIO_URI,
                                ),
                                AudioContent(
                                    type="audio",
                                    mime_type="audio/wav",
                                    data=_INLINE_AUDIO_B64,
                                ),
                                VideoContent(
                                    type="video",
                                    mime_type="video/mp4",
                                    uri=_VIDEO_URI,
                                ),
                                VideoContent(
                                    type="video",
                                    mime_type="video/mp4",
                                    data=_INLINE_VIDEO_B64,
                                ),
                            ],
                        )
                    ],
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert len(spans) == 1
        assert _attr(spans[0], "gen_ai.input.messages") == [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": _PROMPT},
                    {
                        "type": "uri",
                        "modality": "document",
                        "mime_type": "application/pdf",
                        "uri": _DOCUMENT_URI,
                    },
                    {
                        "type": "blob",
                        "modality": "document",
                        "mime_type": "application/pdf",
                        "content": _INLINE_DOCUMENT,
                    },
                    {
                        "type": "uri",
                        "modality": "image",
                        "mime_type": "image/png",
                        "uri": _IMAGE_URI,
                    },
                    {
                        "type": "blob",
                        "modality": "image",
                        "mime_type": "image/png",
                        "content": _INLINE_IMAGE,
                    },
                    {
                        "type": "uri",
                        "modality": "audio",
                        "mime_type": "audio/wav",
                        "uri": _AUDIO_URI,
                    },
                    {
                        "type": "blob",
                        "modality": "audio",
                        "mime_type": "audio/wav",
                        "content": _INLINE_AUDIO,
                    },
                    {
                        "type": "uri",
                        "modality": "video",
                        "mime_type": "video/mp4",
                        "uri": _VIDEO_URI,
                    },
                    {
                        "type": "blob",
                        "modality": "video",
                        "mime_type": "video/mp4",
                        "content": _INLINE_VIDEO,
                    },
                ],
            }
        ]
