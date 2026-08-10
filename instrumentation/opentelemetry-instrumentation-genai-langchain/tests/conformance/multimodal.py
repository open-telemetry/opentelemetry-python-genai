# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: langchain multimodal chat.

Exercises an inline base64 image content block through both ChatOpenAI (an
``image_url`` data URI) and ChatAnthropic (an ``image`` block with a
``source.base64`` payload), asserting the bytes round-trip onto the input
message as an image ``Blob`` part (``type == "blob"``,
``modality == "image"``).
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

# A tiny valid PNG, base64-encoded. Pinned to the recorded cassettes' requests.
_REAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAARklEQVR42u3X"
    "QQ0AIAwAsSnZG4lInJxJwMRICGlyAvq9yF1PFUBAQEBAQBdAXWskICAgICAg"
    "ICAgICAgIOcKBAQEBPQd6ACUHHNEU5qggAAAAABJRU5ErkJggg=="
)


class OpenAIMultimodalScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # langchain can't populate server.address on chat spans.
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
        ),
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        key_override = (
            {}
            if os.getenv("OPENAI_API_KEY")
            else {"OPENAI_API_KEY": "test_openai_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                LangChainInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1,
                    max_tokens=100,
                )
                messages = [
                    HumanMessage(
                        content=[
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
                        ]
                    ),
                ]
                with vcr.use_cassette(
                    "test_chat_openai_multimodal_image_llm_call.yaml"
                ):
                    llm.invoke(messages)

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        _assert_input_image_blob(report)


class AnthropicMultimodalScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # langchain can't populate server.address on chat spans.
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
        ),
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        key_override = (
            {}
            if os.getenv("ANTHROPIC_API_KEY")
            else {"ANTHROPIC_API_KEY": "test_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                LangChainInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                llm = ChatAnthropic(
                    model="claude-sonnet-4-5",
                    temperature=0.1,
                    max_tokens=1024,
                )
                messages = [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _REAL_PNG_B64,
                                },
                            },
                        ]
                    ),
                ]
                with vcr.use_cassette(
                    "test_chat_anthropic_multimodal_image_llm_call.yaml"
                ):
                    llm.invoke(messages)

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        _assert_input_image_blob(report)


def _assert_input_image_blob(report: LiveCheckReport) -> None:
    # Lib-specific: weaver validates each part's *shape*, but not that an
    # inline image actually round-tripped. Assert the base64 image landed
    # on an input message as an image blob part.
    chat_spans = [
        entry["span"]
        for entry in report["samples"]
        if "span" in entry
        and _attr(entry["span"], "gen_ai.operation.name") == "chat"
    ]
    assert chat_spans, "no chat span emitted"

    input_parts = {
        (t, m)
        for span in chat_spans
        for t, m in _part_fields(_attr(span, "gen_ai.input.messages"))
    }
    assert ("blob", "image") in input_parts, (
        f"expected an image blob part on an input message, saw {input_parts}"
    )


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_fields(messages_json: str | None) -> list[tuple[str, str | None]]:
    # gen_ai.{input,output}.messages is a JSON string of
    # [{"role": ..., "parts": [{"type": ..., "modality": ...}]}]. Keep
    # modality so image/audio/video are distinguishable on blob parts.
    messages = json.loads(messages_json) if messages_json else []
    return [
        (part["type"], part.get("modality"))
        for message in messages
        for part in message["parts"]
    ]
