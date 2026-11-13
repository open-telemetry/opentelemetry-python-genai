# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: model chat inference for Agno."""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai
from agno.media import Audio, File, Image
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.tools.function import Function

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

# Agno's OpenAIChat adapter leaves base_url unset by default and does not copy
# response.model or choices[0].finish_reason onto ModelResponse.
_OPENAI_CHAT_GAPS = tuple(
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring=attribute,
    )
    for attribute in (
        "server.address",
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
    )
)

# OpenRouter sets a default base_url (populating server.address), while
# inheriting OpenAIChat's response parsing that omits response.model and
# finish_reasons on ModelResponse.
_OPENROUTER_CHAT_GAPS = tuple(
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring=attribute,
    )
    for attribute in (
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
    )
)

_REAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAARklEQVR42u3X"
    "QQ0AIAwAsSnZG4lInJxJwMRICGlyAvq9yF1PFUBAQEBAQBdAXWskICAgICAg"
    "ICAgICAgIOcKBAQEBPQd6ACUHHNEU5qggAAAAABJRU5ErkJggg=="
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


def _message_part_type_and_modality(
    messages_json: str | None,
) -> set[tuple[str, str | None]]:
    if not messages_json:
        return set()
    messages = json.loads(messages_json)
    return {
        (part["type"], part.get("modality"))
        for message in messages
        for part in message.get("parts", [])
        if "type" in part
    }


def _mock_openai_client(
    handler: Any,
) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
        ),
    )


class ChatScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _OPENAI_CHAT_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        payload = {
            "id": "chatcmpl-conformance",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Conformance model response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 6,
                "total_tokens": 18,
            },
        }
        client = _mock_openai_client(
            lambda _request: httpx.Response(200, json=payload)
        )
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = OpenAIChat(id="gpt-4o", client=client)
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
    expected_violations = _OPENAI_CHAT_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        sse_lines = [
            'data: {"id":"chatcmpl-stream-conformance","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"role":"assistant","content":"Conformance "},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-stream-conformance","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"content":"stream response"},"finish_reason":"stop"}]}',
            'data: {"id":"chatcmpl-stream-conformance","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[],"usage":{"prompt_tokens":14,"completion_tokens":8,"total_tokens":22}}',
            "data: [DONE]",
        ]
        sse_body = ("\n\n".join(sse_lines) + "\n\n").encode("utf-8")
        client = _mock_openai_client(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body,
            )
        )
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = OpenAIChat(id="gpt-4o", client=client)
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
    expected_violations = _OPENAI_CHAT_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Sunny in {city}"

        turn = 0

        def _handle_request(_request: httpx.Request) -> httpx.Response:
            nonlocal turn
            turn += 1
            if turn == 1:
                return httpx.Response(
                    200,
                    json={
                        "id": "chatcmpl-tools-1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "gpt-4o-2024-08-06",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "reasoning_content": "Need to call get_weather.",
                                    "tool_calls": [
                                        {
                                            "id": "call_weather_1",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city": "Paris"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 16,
                            "completion_tokens": 8,
                            "total_tokens": 24,
                        },
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tools-2",
                    "object": "chat.completion",
                    "created": 1700000001,
                    "model": "gpt-4o-2024-08-06",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "The weather in Paris is sunny.",
                                "reasoning_content": "Checking tool output for Paris weather.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 24,
                        "completion_tokens": 10,
                        "total_tokens": 34,
                    },
                },
            )

        client = _mock_openai_client(_handle_request)
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = OpenAIChat(id="gpt-4o", client=client)
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
        input_types: set[str] = set()
        output_types: set[str] = set()
        for span in spans:
            input_types |= _message_part_types(
                _attr(span, "gen_ai.input.messages")
            )
            output_types |= _message_part_types(
                _attr(span, "gen_ai.output.messages")
            )
        for expected_output_part in ("tool_call", "reasoning"):
            assert expected_output_part in output_types, (
                f"expected {expected_output_part!r} part on output messages, saw {output_types}"
            )
        assert "tool_call_response" in input_types, (
            f"expected 'tool_call_response' part on input messages, saw {input_types}"
        )


class ChatMultimodalScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _OPENROUTER_CHAT_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        payload = {
            "id": "chatcmpl-multimodal-conformance",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Described media attachments.",
                        "images": [
                            {
                                "image_url": {
                                    "url": (
                                        "data:image/png;base64,"
                                        f"{_REAL_PNG_B64}"
                                    )
                                }
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "total_tokens": 42,
            },
        }
        client = _mock_openai_client(
            lambda _request: httpx.Response(200, json=payload)
        )
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = OpenRouter(id="gpt-4o", client=client)
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
        input_parts = _message_part_type_and_modality(
            _attr(span, "gen_ai.input.messages")
        )
        output_parts = _message_part_type_and_modality(
            _attr(span, "gen_ai.output.messages")
        )
        for expected_input in (
            ("uri", "image"),
            ("blob", "audio"),
            ("file", "document"),
        ):
            assert expected_input in input_parts, (
                f"expected {expected_input!r} on input messages, saw {input_parts}"
            )
        assert ("blob", "image") in output_parts, (
            f"expected ('blob', 'image') on output messages, saw {output_parts}"
        )
