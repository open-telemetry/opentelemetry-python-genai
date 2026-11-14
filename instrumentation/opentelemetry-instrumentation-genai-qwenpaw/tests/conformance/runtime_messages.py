# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive rich AgentScope events through the 2.x runtime's envelope encoder."""

from __future__ import annotations

import asyncio
import importlib
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest import mock

from tests.harness import make_runtime, runtime_request

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class RuntimeMessagesScenario(Scenario):
    expected_spans = {"invoke_agent": 1}
    expected_metrics = ("gen_ai.invoke_agent.duration",)

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
    from qwenpaw.schemas import FileContent, ImageContent, VideoContent

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

    async def close() -> None:
        pass

    agent = SimpleNamespace(reply_stream=reply_stream, close=close)
    runtime_module = importlib.import_module("qwenpaw.runtime.runtime")
    runtime = make_runtime()
    runtime.workspace.plugins.modes = []
    request = runtime_request("Describe the weather.")
    request.input[0].content.extend(
        [
            ImageContent(image_url="https://example.invalid/image.png"),
            VideoContent(video_url="https://example.invalid/video.mp4"),
            FileContent(file_url="https://example.invalid/file.pdf"),
        ]
    )
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                runtime_module.AgentBuilder,
                "build",
                mock.AsyncMock(return_value=agent),
            )
        )
        if hasattr(runtime_module, "record_agent_activity"):
            stack.enter_context(
                mock.patch.object(
                    runtime_module, "record_agent_activity", mock.AsyncMock()
                )
            )
        async for _ in runtime.run(request):
            pass
