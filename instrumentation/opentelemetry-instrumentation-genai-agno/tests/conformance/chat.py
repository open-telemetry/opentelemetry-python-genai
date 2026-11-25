# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: model chat inference for Agno."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
import openai
from agno.media import Audio, File, Image, Video
from agno.models.google import Gemini
from agno.models.message import Message
from agno.models.moonshot import MoonShot
from agno.models.openai import OpenAIChat
from agno.models.openai.responses import OpenAIResponses
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
    expected_spans = {"chat": 4}
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
        chat_prompt = [
            Message(
                role="user",
                content="A farmer has 17 sheep and all but 9 run away. How many sheep remain? Explain your reasoning.",
            )
        ]
        responses_prompt = [Message(role="user", content="Say this is a test")]

        async def _run_async_chat() -> None:
            async_chat = OpenAIChat(
                id="gpt-5.1",
                async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
                temperature=0.5,
                top_p=0.8,
                max_tokens=128,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                seed=7,
                stop=["STOP"],
            )
            async for _ in async_chat.aresponse_stream(messages=chat_prompt):
                pass

        async def _run_async_responses() -> None:
            async_responses = OpenAIResponses(
                id="gpt-4o-mini",
                async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
            )
            async for _ in async_responses.aresponse_stream(
                messages=responses_prompt
            ):
                pass

        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("openai_chat_streaming_reasoning.yaml"):
                chat_model = OpenAIChat(
                    id="gpt-5.1",
                    client=openai.OpenAI(api_key="test_openai_api_key"),
                    temperature=0.5,
                    top_p=0.8,
                    max_tokens=128,
                    frequency_penalty=0.1,
                    presence_penalty=0.2,
                    seed=7,
                    stop=["STOP"],
                )
                list(chat_model.response_stream(messages=chat_prompt))
            with vcr.use_cassette("openai_chat_streaming_reasoning.yaml"):
                asyncio.run(_run_async_chat())
            with vcr.use_cassette("openai_responses_streaming.yaml"):
                responses_model = OpenAIResponses(
                    id="gpt-4o-mini",
                    client=openai.OpenAI(api_key="test_openai_api_key"),
                )
                list(
                    responses_model.response_stream(messages=responses_prompt)
                )
            with vcr.use_cassette("openai_responses_streaming.yaml"):
                asyncio.run(_run_async_responses())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 4, f"expected 4 chat spans, saw {len(spans)}"
        by_model: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            model_name = str(_attr(span, "gen_ai.request.model"))
            by_model.setdefault(model_name, []).append(span)
        assert len(by_model["gpt-5.1"]) == 2
        assert len(by_model["gpt-4o-mini"]) == 2
        for span in by_model["gpt-5.1"]:
            _assert_attrs_exact(
                span,
                {
                    "gen_ai.request.stream": (True, bool),
                    "gen_ai.response.id": (
                        "chatcmpl-EMxk4dQaiUamioadk6bK1Y462yRui",
                        str,
                    ),
                    "gen_ai.request.temperature": (0.5, float),
                    "gen_ai.request.top_p": (0.8, float),
                    "gen_ai.request.max_tokens": (128, int),
                    "gen_ai.request.frequency_penalty": (0.1, float),
                    "gen_ai.request.presence_penalty": (0.2, float),
                    "gen_ai.request.seed": (7, int),
                    "gen_ai.usage.input_tokens": (29, int),
                    "gen_ai.usage.output_tokens": (92, int),
                    "gen_ai.usage.reasoning.output_tokens": (10, int),
                    "server.address": ("api.openai.com", str),
                },
                stop_sequences=["STOP"],
            )
        for span in by_model["gpt-4o-mini"]:
            _assert_attrs_exact(
                span,
                {
                    "gen_ai.request.stream": (True, bool),
                    "gen_ai.response.id": (
                        "resp_048385de343c399f006a2e1ac3c6e881a0b3e7d86be0a3aa51",
                        str,
                    ),
                    "gen_ai.usage.input_tokens": (22, int),
                    "gen_ai.usage.output_tokens": (6, int),
                    "server.address": ("api.openai.com", str),
                },
            )


class ChatGeminiStreamingScenario(Scenario):
    expected_spans = {"chat": 2}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
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
        prompt = [
            Message(
                role="user",
                content="Write a short 3-sentence summary of why OpenTelemetry is useful.",
            )
        ]

        async def _run_async_gemini_stream() -> None:
            async_model = Gemini(
                id="gemini-2.5-flash",
                client=genai.Client(
                    api_key="test_google_genai_api_key",
                    http_options=HttpOptions(api_version="v1beta"),
                ),
                temperature=0.2,
                max_output_tokens=1024,
            )
            async for _ in async_model.aresponse_stream(messages=prompt):
                pass

        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("gemini_stream_generate_content.yaml"):
                sync_model = Gemini(
                    id="gemini-2.5-flash",
                    client=genai.Client(
                        api_key="test_google_genai_api_key",
                        http_options=HttpOptions(api_version="v1beta"),
                    ),
                    temperature=0.2,
                    max_output_tokens=1024,
                )
                list(sync_model.response_stream(messages=prompt))
            with vcr.use_cassette("gemini_stream_generate_content.yaml"):
                asyncio.run(_run_async_gemini_stream())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 2, f"expected 2 chat spans, saw {len(spans)}"
        for span in spans:
            _assert_attrs_exact(
                span,
                {
                    "gen_ai.provider.name": ("gcp.gemini", str),
                    "gen_ai.request.stream": (True, bool),
                    "gen_ai.request.temperature": (0.2, float),
                    "gen_ai.request.max_tokens": (1024, int),
                    "gen_ai.usage.input_tokens": (16, int),
                    "gen_ai.usage.output_tokens": (870, int),
                    "gen_ai.usage.reasoning.output_tokens": (800, int),
                    "server.address": (
                        "generativelanguage.googleapis.com",
                        str,
                    ),
                },
            )


class ChatToolCallingScenario(Scenario):
    expected_spans = {"chat": 2, "execute_tool": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.execute_tool.duration",
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
                            ),
                            Image(
                                content=base64.b64decode(_REAL_PNG_B64),
                                format="png",
                            ),
                        ],
                        audio=[
                            Audio(content=b"\x00\x01\x02\x03", format="wav")
                        ],
                        files=[
                            File(
                                id="file-conformance-123",
                                mime_type="application/pdf",
                                external={"id": "file-conformance-123"},
                            ),
                            File(
                                content=b"%PDF-1.4\nconformance",
                                mime_type="application/pdf",
                                filename="report.pdf",
                            ),
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
                            ),
                            Video(
                                content=b"\x00\x00\x00\x18ftypmp42",
                                format="mp4",
                            ),
                        ],
                    )
                ]
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 2, f"expected 2 chat spans, saw {len(spans)}"
        provider_models = {
            (
                _attr(span, "gen_ai.provider.name"),
                _attr(span, "gen_ai.request.model"),
            )
            for span in spans
        }
        assert provider_models == {
            ("openrouter", "gpt-4o"),
            ("moonshot", "kimi-k2.5"),
        }, f"unexpected provider/model pairs: {provider_models}"
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
            ("blob", "image"),
            ("blob", "audio"),
            ("file", "document"),
            ("blob", "document"),
            ("uri", "video"),
            ("blob", "video"),
        ):
            assert expected_input in input_parts, (
                f"expected {expected_input!r} on input messages, saw {input_parts}"
            )
        input_blobs_by_modality: dict[str, list[Any]] = {}
        for span in spans:
            for message in _attr(span, "gen_ai.input.messages") or ():
                for part in message.get("parts", ()):
                    if part.get("type") == "blob":
                        input_blobs_by_modality.setdefault(
                            part.get("modality"), []
                        ).append(part.get("content"))
        assert base64.b64decode(_REAL_PNG_B64) in input_blobs_by_modality.get(
            "image", []
        )
        assert b"%PDF-1.4\nconformance" in input_blobs_by_modality.get(
            "document", []
        )
        assert b"\x00\x00\x00\x18ftypmp42" in input_blobs_by_modality.get(
            "video", []
        )
        assert ("blob", "image") in output_parts, (
            f"expected ('blob', 'image') on output messages, saw {output_parts}"
        )


# Agno's OpenAIChat._parse_provider_response_delta only populates provider_data["id"]
# when choice_delta.content is not None, so audio-only streams omit response_id.
_OPENAI_AUDIO_STREAM_GAPS = _OPENAI_CHAT_GAPS + (
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring="gen_ai.response.id",
    ),
)


class ChatAudioOutputScenario(Scenario):
    expected_spans = {"chat": 2}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    )
    expected_violations = _OPENAI_AUDIO_STREAM_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        non_stream_payload = {
            "id": "chatcmpl-audio-conformance",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-audio-preview",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "audio": {
                            "id": "audio_conf_1",
                            "data": "AAE=",
                            "expires_at": 1700003600,
                            "transcript": "Hello audio",
                        },
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 18,
                "total_tokens": 30,
                "completion_tokens_details": {"audio_tokens": 14},
            },
        }
        sse_lines = [
            'data: {"id":"chatcmpl-audio-stream-conformance","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-audio-preview","choices":[{"index":0,"delta":{"role":"assistant","audio":{"id":"audio_conf_s1","data":"AA","transcript":"Hello "}},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-audio-stream-conformance","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-audio-preview","choices":[{"index":0,"delta":{"audio":{"data":"E=","transcript":"audio"}},"finish_reason":"stop"}]}',
            'data: {"id":"chatcmpl-audio-stream-conformance","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-audio-preview","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":16,"total_tokens":26,"completion_tokens_details":{"audio_tokens":12}}}',
            "data: [DONE]",
        ]
        sse_body = ("\n\n".join(sse_lines) + "\n\n").encode("utf-8")

        sync_client = _mock_openai_client(
            lambda _request: httpx.Response(200, json=non_stream_payload)
        )
        stream_client = _mock_openai_client(
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
            model = OpenAIChat(id="gpt-4o-audio-preview", client=sync_client)
            model.response(
                messages=[Message(role="user", content="Speak conformance")]
            )
            stream_model = OpenAIChat(
                id="gpt-4o-audio-preview", client=stream_client
            )
            list(
                stream_model.response_stream(
                    messages=[
                        Message(
                            role="user", content="Speak stream conformance"
                        )
                    ]
                )
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 2, f"expected 2 chat spans, saw {len(spans)}"
        non_stream_spans = [
            s for s in spans if _attr(s, "gen_ai.request.stream") is None
        ]
        stream_spans = [
            s for s in spans if _attr(s, "gen_ai.request.stream") is True
        ]
        assert len(non_stream_spans) == 1
        assert len(stream_spans) == 1
        assert (
            _attr(non_stream_spans[0], "gen_ai.response.id")
            == "chatcmpl-audio-conformance"
        )
        for span in (non_stream_spans[0], stream_spans[0]):
            output_messages = _attr(span, "gen_ai.output.messages")
            output_parts = _message_part_type_and_modality(output_messages)
            assert ("blob", "audio") in output_parts, (
                f"expected ('blob', 'audio') on output messages, saw {output_parts}"
            )
            audio_blobs = [
                part
                for message in output_messages
                for part in message.get("parts", [])
                if part.get("type") == "blob"
                and part.get("modality") == "audio"
            ]
            assert len(audio_blobs) == 1
            assert audio_blobs[0]["content"] == b"\x00\x01"


class ChatOpenAIResponsesScenario(Scenario):
    expected_spans = {"chat": 4, "execute_tool": 4}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.execute_tool.duration",
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
        def get_current_weather(location: str) -> str:
            """Get the current weather in a given location."""
            if "Seattle" in location:
                return '{"location":"Seattle, WA","temperature":58,"unit":"F","conditions":"rain"}'
            return '{"location":"San Francisco, CA","temperature":65,"unit":"F","conditions":"fog"}'

        prompt_text = "What's the weather in Seattle and San Francisco today?"

        async def _run_async() -> None:
            async_model = OpenAIResponses(
                id="gpt-4o-mini",
                async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
            )
            await async_model.aresponse(
                messages=[Message(role="user", content=prompt_text)],
                tools=[Function.from_callable(get_current_weather)],
            )

        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("openai_responses_tool_calling.yaml"):
                sync_model = OpenAIResponses(
                    id="gpt-4o-mini",
                    client=openai.OpenAI(api_key="test_openai_api_key"),
                )
                sync_model.response(
                    messages=[Message(role="user", content=prompt_text)],
                    tools=[Function.from_callable(get_current_weather)],
                )
            with vcr.use_cassette("openai_responses_tool_calling.yaml"):
                asyncio.run(_run_async())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _chat_spans(report)
        assert len(spans) == 4, f"expected 4 chat spans, saw {len(spans)}"
        turn1_id = "resp_06846aae2b482d8e006a9b1773c4ec87d08182937f685f261f"
        turn2_id = "resp_06846aae2b482d8e006a9b1775645c87d08005ec0fb2110764"
        expected_call_ids = [
            "call_krAHY11uDHrSPJIjNZLF7RHU",
            "call_fr3gmaw1ffbKMC3N64V6Iux4",
        ]
        turn1_spans = [
            s for s in spans if _attr(s, "gen_ai.response.id") == turn1_id
        ]
        turn2_spans = [
            s for s in spans if _attr(s, "gen_ai.response.id") == turn2_id
        ]
        assert len(turn1_spans) == 2
        assert len(turn2_spans) == 2
        for span in turn1_spans:
            assert _attr(span, "gen_ai.request.stream") is None
            _assert_attrs_exact(
                span,
                {
                    "gen_ai.provider.name": ("openai", str),
                    "gen_ai.request.model": ("gpt-4o-mini", str),
                    "gen_ai.response.id": (turn1_id, str),
                    "gen_ai.usage.input_tokens": (64, int),
                    "gen_ai.usage.output_tokens": (52, int),
                    "server.address": ("api.openai.com", str),
                },
            )
            assert _attr(span, "gen_ai.tool.definitions")
            call_ids = [
                part["id"]
                for msg in (_attr(span, "gen_ai.output.messages") or ())
                for part in msg.get("parts", ())
                if part.get("type") == "tool_call"
            ]
            assert call_ids == expected_call_ids
        for span in turn2_spans:
            assert _attr(span, "gen_ai.request.stream") is None
            _assert_attrs_exact(
                span,
                {
                    "gen_ai.provider.name": ("openai", str),
                    "gen_ai.request.model": ("gpt-4o-mini", str),
                    "gen_ai.response.id": (turn2_id, str),
                    "gen_ai.usage.input_tokens": (138, int),
                    "gen_ai.usage.output_tokens": (60, int),
                    "server.address": ("api.openai.com", str),
                },
            )
            result_ids = [
                part["id"]
                for msg in (_attr(span, "gen_ai.input.messages") or ())
                for part in msg.get("parts", ())
                if part.get("type") == "tool_call_response"
            ]
            assert result_ids == expected_call_ids
