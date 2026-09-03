# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: openai-v2 chat completion with multimodal input.

Sends a content-part array carrying text, an external image URL and an inline
base64 image, and asserts each lands on the input message as the matching
semconv part (``text`` / ``uri`` / ``blob``).
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

IMAGE_URL = "https://opentelemetry.io/img/logos/opentelemetry-logo-nav.png"
# 1x1 transparent PNG, sent inline as a base64 data URL.
IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


class MultimodalScenario(Scenario):
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
        with instrument(
            OpenAIInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("multimodal_conformance.yaml"):
                OpenAI().chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe each image in one short sentence.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_URL},
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_DATA_URL},
                                },
                            ],
                        }
                    ],
                    model="gpt-4o-mini",
                    stream=False,
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)

        # Weaver validates the shape of each part; assert the media parts
        # actually round-tripped onto the input message with their modality.
        chat_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert chat_spans, "no chat span emitted"

        input_parts = {
            (part_type, modality)
            for span in chat_spans
            for part_type, modality in _part_fields(
                _attr(span, "gen_ai.input.messages")
            )
        }
        assert ("text", None) in input_parts, (
            f"expected a text part on the input message, saw {input_parts}"
        )
        assert ("uri", "image") in input_parts, (
            f"expected an image uri part on the input message, saw {input_parts}"
        )
        assert ("blob", "image") in input_parts, (
            f"expected an image blob part on the input message, saw {input_parts}"
        )


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_fields(messages_json: str | None) -> list[tuple[str, str | None]]:
    # gen_ai.input.messages is a JSON string of
    # [{"role": ..., "parts": [{"type": ..., ...}]}]; keep modality so
    # image/audio/document are distinguishable on blob/uri/file parts.
    messages = json.loads(messages_json) if messages_json else []
    return [
        (part["type"], part.get("modality"))
        for message in messages
        for part in message["parts"]
    ]
