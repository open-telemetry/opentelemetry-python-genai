# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios for OpenAI image inputs."""

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

_REAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAARklEQVR42u3X"
    "QQ0AIAwAsSnZG4lInJxJwMRICGlyAvq9yF1PFUBAQEBAQBdAXWskICAgICAg"
    "ICAgICAgIOcKBAQEBPQd6ACUHHNEU5qggAAAAABJRU5ErkJggg=="
)


class ChatCompletionsMultimodalScenario(Scenario):
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
            with vcr.use_cassette(
                "chat_completions_multimodal_conformance.yaml"
            ):
                OpenAI().chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What is in this image?",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            "data:image/png;base64,"
                                            f"{_REAL_PNG_B64}"
                                        )
                                    },
                                },
                            ],
                        }
                    ],
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        _assert_input_image_blob(report)


class ResponsesMultimodalScenario(Scenario):
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
            with vcr.use_cassette("responses_multimodal_conformance.yaml"):
                OpenAI().responses.create(
                    model="gpt-4.1",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "What is in this image?",
                                },
                                {
                                    "type": "input_image",
                                    "image_url": (
                                        "data:image/png;base64,"
                                        f"{_REAL_PNG_B64}"
                                    ),
                                },
                            ],
                        }
                    ],
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        _assert_input_image_blob(report)


def _assert_input_image_blob(report: LiveCheckReport) -> None:
    chat_spans = [
        entry["span"]
        for entry in report["samples"]
        if "span" in entry
        and _attr(entry["span"], "gen_ai.operation.name") == "chat"
    ]
    assert chat_spans, "no chat span emitted"

    input_parts = {
        (part["type"], part.get("modality"))
        for span in chat_spans
        for message in _messages(_attr(span, "gen_ai.input.messages"))
        for part in message["parts"]
    }
    assert ("blob", "image") in input_parts, (
        f"expected an image blob part on an input message, saw {input_parts}"
    )


def _attr(span: dict[str, Any], name: str) -> Any:
    for attribute in span["attributes"]:
        if attribute["name"] == name:
            return attribute["value"]
    return None


def _messages(value: str | None) -> list[dict[str, Any]]:
    return json.loads(value) if value else []
