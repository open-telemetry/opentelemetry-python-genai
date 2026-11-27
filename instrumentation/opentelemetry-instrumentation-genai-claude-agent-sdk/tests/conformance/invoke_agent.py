# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: one agent turn that calls a built-in tool.

The SDK drives a CLI subprocess, so the stream is replayed through a fake
``Transport`` with no CLI and no network.
"""

from __future__ import annotations

from typing import Any

import anyio
import claude_agent_sdk
from claude_agent_sdk.types import ClaudeAgentOptions

from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from .._replay import (
    ReplayTransport,
    _message,
    assistant,
    result,
    tool_result,
)

STREAM = [
    _message(session_id="conformance", model="claude-sonnet-4-5"),
    assistant(
        [
            {
                "type": "thinking",
                "thinking": "I should read the file first.",
                "signature": "sig-1",
            },
            {"type": "text", "text": "Reading the file."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Read",
                "input": {"file_path": "/tmp/a.txt"},
            },
        ],
        usage=(12, 30),
        session_id="conformance",
        stop_reason="tool_use",
    ),
    tool_result("toolu_1", session_id="conformance"),
    assistant(
        [{"type": "text", "text": "Done: hello"}],
        usage=(8, 8),
        session_id="conformance",
        stop_reason="end_turn",
    ),
    result(
        usage=(20, 38),
        session_id="conformance",
        stop_reason="end_turn",
    ),
]


async def _run_turn() -> None:
    options = ClaudeAgentOptions(model="claude-sonnet-4-5")
    async for _ in claude_agent_sdk.query(
        prompt="Read /tmp/a.txt and summarise it",
        options=options,
        transport=ReplayTransport(STREAM),
    ):
        pass


class InvokeAgentScenario(Scenario):
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
        with instrument(
            ClaudeAgentSDKInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            anyio.run(_run_turn)

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        agent = next(
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and entry["span"]["name"].startswith("invoke_agent")
        )
        attributes = {a["name"]: a["value"] for a in agent["attributes"]}
        part_types = {
            part["type"]
            for message in attributes["gen_ai.output.messages"]
            for part in message["parts"]
        }
        assert {
            "text",
            "reasoning",
            "tool_call",
            "tool_call_response",
        } <= part_types
