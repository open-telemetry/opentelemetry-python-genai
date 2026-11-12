# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: model chat inference for Agno."""

from __future__ import annotations

import json
from typing import Any

from agno.media import Audio, File, Image
from agno.models.message import Message
from agno.models.response import ModelResponse
from tests.mock_model import MockModel

try:
    from agno.models.message import MessageMetrics
except ImportError:
    from agno.models.message import Metrics as MessageMetrics

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

_MISSING_SERVER_ADDRESS = (
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring="server.address",
    ),
)


def _chat_spans(report: LiveCheckReport) -> list[dict[str, Any]]:
    return [
        entry["span"]
        for entry in report["samples"]
        if "span" in entry
        and _attr(entry["span"], "gen_ai.operation.name") == "chat"
    ]


def _attr(span: dict[str, Any], name: str) -> Any:
    for attribute in span["attributes"]:
        if attribute["name"] == name:
            return attribute["value"]
    return None


def _message_part_types(messages_json: str | None) -> set[str]:
    if not messages_json:
        return set()
    messages = json.loads(messages_json)
    return {
        part["type"]
        for message in messages
        for part in message.get("parts", [])
        if "type" in part
    }


class _ConformanceChatModel(MockModel):
    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content="Conformance model response",
            provider_data={
                "model": "gpt-4o-2024-08-06",
                "id": "chatcmpl-conformance",
                "finish_reason": "stop",
            },
            response_usage=MessageMetrics(
                input_tokens=12,
                output_tokens=6,
            ),
        )

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield ModelResponse(content="Conformance ")
        yield ModelResponse(
            content="stream response",
            provider_data={
                "model": "gpt-4o-2024-08-06",
                "id": "chatcmpl-stream-conformance",
                "finish_reason": "stop",
            },
            response_usage=MessageMetrics(
                input_tokens=14,
                output_tokens=8,
            ),
        )


class _ConformanceToolAndReasoningModel(MockModel):
    def __init__(self, **kwargs: Any) -> None:
        self._turn = 0
        super().__init__(**kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self._turn += 1
        if self._turn == 1:
            return ModelResponse(
                content=None,
                reasoning_content="Need to call get_weather.",
                tool_calls=[
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
                provider_data={
                    "model": "gpt-4o-2024-08-06",
                    "id": "chatcmpl-tools-1",
                    "finish_reason": "tool_calls",
                },
                response_usage=MessageMetrics(
                    input_tokens=16,
                    output_tokens=8,
                ),
            )
        return ModelResponse(
            content="The weather in Paris is sunny.",
            reasoning_content="Checking tool output for Paris weather.",
            provider_data={
                "model": "gpt-4o-2024-08-06",
                "id": "chatcmpl-tools-2",
                "finish_reason": "stop",
            },
            response_usage=MessageMetrics(
                input_tokens=24,
                output_tokens=10,
            ),
        )


class _ConformanceMultimodalModel(MockModel):
    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content="Described media attachments.",
            images=[
                Image(
                    url="https://example.com/generated.png",
                    mime_type="image/png",
                )
            ],
            provider_data={
                "model": "gpt-4o-2024-08-06",
                "id": "chatcmpl-multimodal-conformance",
                "finish_reason": "stop",
            },
            response_usage=MessageMetrics(
                input_tokens=30,
                output_tokens=12,
            ),
        )


class ChatScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _MISSING_SERVER_ADDRESS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = _ConformanceChatModel(id="gpt-4o", provider="OpenAI")
            model.response(
                messages=[
                    Message(role="user", content="hello chat conformance")
                ]
            )


class ChatStreamingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    )
    expected_violations = _MISSING_SERVER_ADDRESS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = _ConformanceChatModel(id="gpt-4o", provider="OpenAI")
            list(
                model.response_stream(
                    messages=[
                        Message(
                            role="user",
                            content="hello streaming chat conformance",
                        )
                    ]
                )
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert spans, "expected a chat span"
        assert _attr(spans[0], "gen_ai.request.stream") is True


class ChatToolCallingScenario(Scenario):
    expected_spans = {"chat": 2, "execute_tool": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _MISSING_SERVER_ADDRESS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        from agno.tools.function import Function

        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Sunny in {city}"

        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = _ConformanceToolAndReasoningModel(
                id="gpt-4o", provider="OpenAI"
            )
            model.response(
                messages=[
                    Message(
                        role="user", content="What is the weather in Paris?"
                    ),
                ],
                tools=[Function.from_callable(get_weather)],
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 2, f"expected 2 chat spans, saw {len(spans)}"
        assert any(_attr(s, "gen_ai.tool.definitions") for s in spans), (
            "expected gen_ai.tool.definitions on the chat span"
        )
        all_types: set[str] = set()
        for span in spans:
            all_types |= _message_part_types(
                _attr(span, "gen_ai.input.messages")
            )
            all_types |= _message_part_types(
                _attr(span, "gen_ai.output.messages")
            )
        for expected_part in ("tool_call", "tool_call_response", "reasoning"):
            assert expected_part in all_types, (
                f"expected {expected_part!r} part on chat messages, saw {all_types}"
            )


class ChatMultimodalScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _MISSING_SERVER_ADDRESS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = _ConformanceMultimodalModel(id="gpt-4o", provider="OpenAI")
            model.response(
                messages=[
                    Message(
                        role="user",
                        content="Describe these media items",
                        images=[
                            Image(
                                url="https://example.com/photo.jpg",
                                format="jpg",
                            )
                        ],
                        audio=[
                            Audio(content=b"\x00\x01\x02\x03", format="wav")
                        ],
                        files=[
                            File(
                                id="file-conformance-123",
                                mime_type="application/pdf",
                                external={"id": "file-conformance-123"},
                            )
                        ],
                    )
                ]
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert spans, "expected a chat span"
        span = spans[0]
        input_types = _message_part_types(_attr(span, "gen_ai.input.messages"))
        output_types = _message_part_types(
            _attr(span, "gen_ai.output.messages")
        )
        all_types = input_types | output_types
        for expected_part in ("blob", "uri", "file"):
            assert expected_part in all_types, (
                f"expected {expected_part!r} part on chat messages, saw {all_types}"
            )
