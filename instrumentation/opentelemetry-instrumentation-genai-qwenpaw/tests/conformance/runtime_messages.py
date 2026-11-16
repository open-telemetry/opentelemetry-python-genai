# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive rich AgentScope events through the 2.x runtime's envelope encoder."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from tests.harness import patched_runtime_agent, runtime_request

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class RuntimeMessagesScenario(Scenario):
    expected_spans = {"invoke_agent": 1}
    expected_metrics = ("gen_ai.invoke_agent.duration",)

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        (span,) = [
            sample["span"] for sample in report["samples"] if "span" in sample
        ]
        attributes = {
            attribute["name"]: attribute["value"]
            for attribute in span["attributes"]
        }
        (input_message,) = json.loads(attributes[GenAI.GEN_AI_INPUT_MESSAGES])
        assert input_message["role"] == "user"
        input_parts = input_message["parts"]
        assert [
            (part["type"], part.get("modality")) for part in input_parts
        ] == [
            ("text", None),
            ("uri", "image"),
            ("uri", "video"),
            ("uri", "document"),
            ("uri", "audio"),
            ("blob", "image"),
            ("blob", "video"),
            ("blob", "document"),
            ("blob", "audio"),
        ]
        assert input_parts[0]["content"] == "Describe the weather."
        assert [part["uri"] for part in input_parts[1:5]] == [
            "https://example.invalid/image.png",
            "https://example.invalid/video.mp4",
            "https://example.invalid/file.pdf",
            "https://example.invalid/audio.wav",
        ]
        assert [
            (part["mime_type"], part["content"]) for part in input_parts[5:]
        ] == [
            ("image/png", "YWJj"),
            ("video/mp4", "YWJj"),
            ("application/pdf", "YWJj"),
            ("audio/wav", "YWJj"),
        ]
        output_parts = [
            part
            for message in json.loads(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
            for part in message["parts"]
        ]
        assert [
            (part["type"], part.get("modality")) for part in output_parts
        ] == [
            ("reasoning", None),
            ("tool_call", None),
            ("tool_call_response", None),
            ("text", None),
            ("blob", "image"),
            ("blob", "video"),
            ("blob", "audio"),
        ]
        assert output_parts[:4] == [
            {"type": "reasoning", "content": "Thinking."},
            {
                "type": "tool_call",
                "name": "weather",
                "id": "call-1",
                "arguments": {"city": "Paris"},
            },
            {
                "type": "tool_call_response",
                "id": "call-1",
                "response": "sunny",
            },
            {"type": "text", "content": "It is sunny."},
        ]
        assert [
            (part["mime_type"], part["content"]) for part in output_parts[4:]
        ] == [
            ("image/png", "YWJj"),
            ("video/mp4", "YWJj"),
            ("audio/wav", "YWJj"),
        ]
        assert GenAI.GEN_AI_RESPONSE_FINISH_REASONS not in attributes

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        del vcr
        with instrument(
            QwenPawInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture="SPAN_ONLY",
        ):
            asyncio.run(_drive_events())


async def _drive_events() -> None:
    from agentscope.event import EventType
    from qwenpaw.schemas import (
        AudioContent,
        FileContent,
        ImageContent,
        VideoContent,
    )

    async def reply_stream(*, inputs: Any) -> Any:
        del inputs
        events = [
            (EventType.THINKING_BLOCK_START, {}),
            (EventType.THINKING_BLOCK_DELTA, {"delta": "Thinking."}),
            (EventType.THINKING_BLOCK_END, {}),
            (EventType.TOOL_CALL_START, {"tool_call_name": "weather"}),
            (EventType.TOOL_CALL_DELTA, {"delta": '{"city":"Paris"}'}),
            (EventType.TOOL_CALL_END, {}),
            (EventType.TOOL_RESULT_START, {"tool_call_name": "weather"}),
            (EventType.TOOL_RESULT_TEXT_DELTA, {"delta": "sunny"}),
            (EventType.TOOL_RESULT_END, {}),
            (EventType.TEXT_BLOCK_START, {}),
            (EventType.TEXT_BLOCK_DELTA, {"delta": "It is sunny."}),
            (EventType.TEXT_BLOCK_END, {}),
        ]
        for kind, extra in events:
            yield SimpleNamespace(
                type=kind,
                block_id="block-1",
                tool_call_id="call-1",
                metadata={},
                **extra,
            )
        for mime_type in ("image/png", "video/mp4", "audio/wav"):
            for kind in (
                EventType.DATA_BLOCK_START,
                EventType.DATA_BLOCK_DELTA,
                EventType.DATA_BLOCK_END,
            ):
                yield SimpleNamespace(
                    type=kind,
                    block_id=mime_type,
                    media_type=mime_type,
                    data="YWJj",
                    metadata={},
                )

    request = runtime_request("Describe the weather.")
    request.input[0].content.extend(
        [
            ImageContent(image_url="https://example.invalid/image.png"),
            VideoContent(video_url="https://example.invalid/video.mp4"),
            FileContent(file_url="https://example.invalid/file.pdf"),
            AudioContent(data="https://example.invalid/audio.wav"),
            ImageContent(image_url="data:image/png;base64,YWJj"),
            VideoContent(video_url="data:video/mp4;base64,YWJj"),
            FileContent(file_url="data:application/pdf;base64,YWJj"),
            AudioContent(data="YWJj", format="wav"),
        ]
    )
    with patched_runtime_agent(reply_stream) as runtime:
        async for _ in runtime.run(request):
            pass
