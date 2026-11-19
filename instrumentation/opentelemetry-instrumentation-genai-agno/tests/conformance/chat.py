# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: model chat inference for Agno."""

from __future__ import annotations

from typing import Any

import httpx
import openai
from agno.media import Audio, File, Image, Video
from agno.models.google import Gemini
from agno.models.message import Message
from agno.models.moonshot import MoonShot
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.tools.function import Function
from google import genai
from google.genai.types import HttpOptions

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

# Agno's OpenAIChat adapter does not copy response.model or
# choices[0].finish_reason onto ModelResponse.
_OPENAI_CHAT_GAPS = tuple(
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring=attribute,
    )
    for attribute in (
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

# Agno's Gemini adapter does not copy response.response_id, response.model_version,
# or finish_reason onto ModelResponse.provider_data.
_GEMINI_CHAT_GAPS = tuple(
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring=attribute,
    )
    for attribute in (
        "gen_ai.response.id",
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


def _assert_attrs_exact(
    span: dict[str, Any],
    expected: dict[str, tuple[Any, type[Any]]],
    *,
    stop_sequences: list[str] | None = None,
) -> None:
    for name, (expected_value, expected_type) in expected.items():
        value = _attr(span, name)
        assert type(value) is expected_type, (
            f"expected {name} to have type {expected_type.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )
        assert value == expected_value, (
            f"expected {name} == {expected_value!r}, got {value!r}"
        )
    if stop_sequences is not None:
        actual_stops = _attr(span, "gen_ai.request.stop_sequences")
        assert type(actual_stops) is list and actual_stops == stop_sequences
        assert all(type(item) is str for item in actual_stops)


def _message_part_types(messages: Any) -> set[str]:
    if not messages:
        return set()
    return {
        part["type"]
        for message in messages
        for part in message.get("parts", [])
        if "type" in part
    }


def _message_part_type_and_modality(
    messages: Any,
) -> set[tuple[str, str | None]]:
    if not messages:
        return set()
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
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35,
                "prompt_tokens_details": {
                    "cached_tokens": 8,
                    "cache_write_tokens": 3,
                    "audio_tokens": 4,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 5,
                    "audio_tokens": 6,
                },
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
            model = OpenAIChat(
                id="gpt-4o",
                client=client,
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
                frequency_penalty=0.5,
                presence_penalty=0.2,
                seed=42,
                stop=["END"],
            )
            model.response(
                messages=[
                    Message(role="user", content="hello chat conformance")
                ]
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 1, f"expected 1 chat span, saw {len(spans)}"
        _assert_attrs_exact(
            spans[0],
            {
                "gen_ai.response.id": ("chatcmpl-conformance", str),
                "gen_ai.request.temperature": (0.7, float),
                "gen_ai.request.top_p": (0.9, float),
                "gen_ai.request.max_tokens": (256, int),
                "gen_ai.request.frequency_penalty": (0.5, float),
                "gen_ai.request.presence_penalty": (0.2, float),
                "gen_ai.request.seed": (42, int),
                "gen_ai.usage.input_tokens": (20, int),
                "gen_ai.usage.output_tokens": (15, int),
                "gen_ai.usage.cache_read.input_tokens": (8, int),
                "gen_ai.usage.cache_write.input_tokens": (3, int),
                "gen_ai.usage.reasoning.output_tokens": (5, int),
                "gen_ai.usage.audio.input_tokens": (4, int),
                "gen_ai.usage.audio.output_tokens": (6, int),
            },
            stop_sequences=["END"],
        )


class ChatGeminiScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = _GEMINI_CHAT_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        payload = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Gemini conformance response"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 30,
                "cachedContentTokenCount": 10,
                "candidatesTokenCount": 12,
                "thoughtsTokenCount": 8,
                "totalTokenCount": 50,
            },
        }
        http_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=payload)
            )
        )
        client = genai.Client(
            api_key="test-key",
            http_options=HttpOptions(httpx_client=http_client),
        )
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = Gemini(
                id="gemini-2.5-flash",
                client=client,
                temperature=0.4,
                top_p=0.85,
                top_k=40,
                max_output_tokens=128,
                stop_sequences=["HALT"],
            )
            model.response(
                messages=[
                    Message(role="user", content="hello gemini conformance")
                ]
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 1, f"expected 1 chat span, saw {len(spans)}"
        _assert_attrs_exact(
            spans[0],
            {
                "gen_ai.provider.name": ("gcp.gemini", str),
                "gen_ai.request.temperature": (0.4, float),
                "gen_ai.request.top_p": (0.85, float),
                "gen_ai.request.top_k": (40, int),
                "gen_ai.request.max_tokens": (128, int),
                "gen_ai.usage.input_tokens": (30, int),
                "gen_ai.usage.output_tokens": (20, int),
                "gen_ai.usage.cache_read.input_tokens": (10, int),
                "gen_ai.usage.reasoning.output_tokens": (8, int),
            },
            stop_sequences=["HALT"],
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
            'data: {"id":"chatcmpl-stream-conformance","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[],"usage":{"prompt_tokens":14,"completion_tokens":8,"total_tokens":22,"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":2,"audio_tokens":1},"completion_tokens_details":{"reasoning_tokens":3,"audio_tokens":2}}}',
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
            model = OpenAIChat(
                id="gpt-4o",
                client=client,
                temperature=0.5,
                top_p=0.8,
                max_tokens=128,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                seed=7,
                stop=["STOP"],
            )
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
        assert len(spans) == 1, f"expected 1 chat span, saw {len(spans)}"
        _assert_attrs_exact(
            spans[0],
            {
                "gen_ai.request.stream": (True, bool),
                "gen_ai.response.id": ("chatcmpl-stream-conformance", str),
                "gen_ai.request.temperature": (0.5, float),
                "gen_ai.request.top_p": (0.8, float),
                "gen_ai.request.max_tokens": (128, int),
                "gen_ai.request.frequency_penalty": (0.1, float),
                "gen_ai.request.presence_penalty": (0.2, float),
                "gen_ai.request.seed": (7, int),
                "gen_ai.usage.input_tokens": (14, int),
                "gen_ai.usage.output_tokens": (8, int),
                "gen_ai.usage.cache_read.input_tokens": (4, int),
                "gen_ai.usage.cache_write.input_tokens": (2, int),
                "gen_ai.usage.reasoning.output_tokens": (3, int),
                "gen_ai.usage.audio.input_tokens": (1, int),
                "gen_ai.usage.audio.output_tokens": (2, int),
            },
            stop_sequences=["STOP"],
        )


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
    expected_spans = {"chat": 2}
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
            video_model = MoonShot(id="kimi-k2.5", client=client)
            video_model.response(
                messages=[
                    Message(
                        role="user",
                        content="Describe this video",
                        videos=[
                            Video(
                                id="ms://video-conformance-123",
                                url="https://example.com/clip.mp4",
                                format="mp4",
                            )
                        ],
                    )
                ]
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 2, f"expected 2 chat spans, saw {len(spans)}"
        input_parts: set[tuple[str, str | None]] = set()
        output_parts: set[tuple[str, str | None]] = set()
        for span in spans:
            input_parts |= _message_part_type_and_modality(
                _attr(span, "gen_ai.input.messages")
            )
            output_parts |= _message_part_type_and_modality(
                _attr(span, "gen_ai.output.messages")
            )
        for expected_input in (
            ("uri", "image"),
            ("blob", "audio"),
            ("file", "document"),
            ("uri", "video"),
        ):
            assert expected_input in input_parts, (
                f"expected {expected_input!r} on input messages, saw {input_parts}"
            )
        assert ("blob", "image") in output_parts, (
            f"expected ('blob', 'image') on output messages, saw {output_parts}"
        )
