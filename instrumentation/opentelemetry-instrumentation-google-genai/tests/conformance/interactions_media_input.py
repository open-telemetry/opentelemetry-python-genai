# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: google-genai interactions with inline and URI media inputs."""

from __future__ import annotations

import base64
import os
from typing import Any, ClassVar

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
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 200 200]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 44>>stream\nBT /F1 12 Tf 20 100 Td (City: Paris) Tj ET\nendstream\nendobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000232 00000 n \n0000000297 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n391\n%%EOF\n"
)
_MINIMAL_PDF_B64 = base64.b64encode(_MINIMAL_PDF).decode("ascii")
_DOCUMENT_PROMPT = (
    "What city is named in this PDF? Reply with the city name only."
)

_MEDIA_PROMPT = "Describe these attached media inputs."
_DOCUMENT_URI = "https://example.com/document.pdf"
_INLINE_DOCUMENT = b"%PDF-1.4\ninline document"
_INLINE_DOCUMENT_B64 = base64.b64encode(_INLINE_DOCUMENT).decode("ascii")
_IMAGE_URI = "https://example.com/image.png"
_INLINE_IMAGE = b"\x89PNG\r\n\x1a\ninline image"
_INLINE_IMAGE_B64 = base64.b64encode(_INLINE_IMAGE).decode("ascii")
_AUDIO_URI = "https://example.com/audio.wav"
_INLINE_AUDIO = b"RIFFinline audio"
_INLINE_AUDIO_B64 = base64.b64encode(_INLINE_AUDIO).decode("ascii")
_VIDEO_URI = "https://example.com/video.mp4"
_INLINE_VIDEO = b"\x00\x00\x00\x18ftypinline video"
_INLINE_VIDEO_B64 = base64.b64encode(_INLINE_VIDEO).decode("ascii")


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


class _InteractionsMediaScenario(Scenario):
    expected_spans = {"interactions.create": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_operation_name_unknown",
            message_substring="interactions.create",
        ),
    )
    cassette_name: ClassVar[str]

    def _input_content(self) -> list[Any]:
        raise NotImplementedError

    def _expected_input_parts(self) -> list[dict[str, Any]]:
        raise NotImplementedError

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
            with vcr.use_cassette(self.cassette_name):
                client = Client(api_key=api_key, vertexai=False)
                client.interactions.create(
                    model="gemini-2.5-flash",
                    input=[
                        UserInputStep(
                            type="user_input",
                            content=self._input_content(),
                        )
                    ],
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "interactions.create"
        ]
        assert len(spans) == 1
        assert _attr(spans[0], "gen_ai.input.messages") == [
            {
                "role": "user",
                "parts": self._expected_input_parts(),
            }
        ]


class InteractionsDocumentInputScenario(_InteractionsMediaScenario):
    cassette_name = "interactions_document_input_conformance.yaml"

    def _input_content(self) -> list[Any]:
        return [
            TextContent(type="text", text=_DOCUMENT_PROMPT),
            DocumentContent(
                type="document",
                mime_type="application/pdf",
                data=_MINIMAL_PDF_B64,
            ),
        ]

    def _expected_input_parts(self) -> list[dict[str, Any]]:
        return [
            {"type": "text", "content": _DOCUMENT_PROMPT},
            {
                "type": "blob",
                "modality": "document",
                "mime_type": "application/pdf",
                "content": _MINIMAL_PDF,
            },
        ]


class InteractionsMediaInputScenario(_InteractionsMediaScenario):
    cassette_name = "interactions_media_input_conformance.yaml"

    def _input_content(self) -> list[Any]:
        return [
            TextContent(type="text", text=_MEDIA_PROMPT),
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
        ]

    def _expected_input_parts(self) -> list[dict[str, Any]]:
        return [
            {"type": "text", "content": _MEDIA_PROMPT},
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
        ]
