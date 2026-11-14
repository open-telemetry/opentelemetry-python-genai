# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: one agent turn that calls a built-in tool.

The Claude Agent SDK drives a CLI subprocess, so the message stream is replayed
through a fake ``Transport``: no CLI, no network. The turn covers the two
operations this instrumentation emits -- ``invoke_agent`` for the run and
``execute_tool`` for the built-in tool call observed in the stream.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import claude_agent_sdk
from claude_agent_sdk._internal.transport import Transport
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

STREAM = [
    {
        "type": "system",
        "subtype": "init",
        "session_id": "conformance",
        "tools": ["Read"],
        "model": "claude-sonnet-4-5",
    },
    {
        "type": "assistant",
        "session_id": "conformance",
        "message": {
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "text", "text": "Reading the file."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a.txt"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 30},
        },
    },
    {
        "type": "user",
        "session_id": "conformance",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "hello",
                }
            ],
        },
    },
    {
        "type": "assistant",
        "session_id": "conformance",
        "message": {
            "id": "msg_2",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Done: hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 8},
        },
    },
    {
        "type": "result",
        "subtype": "success",
        "session_id": "conformance",
        "duration_ms": 12,
        "duration_api_ms": 10,
        "is_error": False,
        "num_turns": 1,
        "result": "Done: hello",
        "total_cost_usd": 0.0,
        "usage": {"input_tokens": 20, "output_tokens": 38},
    },
]


class ReplayTransport(Transport):
    """Replays a fixed message stream; answers control requests in place."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.pending: list[dict[str, Any]] = []

    async def connect(self) -> None:
        return None

    async def write(self, data: str) -> None:
        payload = json.loads(data)
        if payload.get("type") == "control_request":
            self.pending.append(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": payload.get("request_id"),
                        "response": {"ok": True},
                    },
                }
            )

    def read_messages(self):
        async def gen():
            for message in self.messages:
                while self.pending:
                    yield self.pending.pop(0)
                yield message
            while self.pending:
                yield self.pending.pop(0)

        return gen()

    async def close(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    async def end_input(self) -> None:
        return None


async def _run_turn() -> None:
    options = ClaudeAgentOptions(model="claude-sonnet-4-5")
    async for _ in claude_agent_sdk.query(
        prompt="Read /tmp/a.txt and summarise it",
        options=options,
        transport=ReplayTransport(STREAM),
    ):
        pass


class InvokeAgentScenario(Scenario):
    expected_spans = {"invoke_agent": 1, "execute_tool": 1}
    expected_metrics = (
        "gen_ai.invoke_agent.duration",
        "gen_ai.execute_tool.duration",
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
            ClaudeAgentSDKInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            anyio.run(_run_turn)

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        agent_names = {
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.agent.name"
        }
        assert agent_names == {"claude-agent-sdk"}, (
            "The agent name must identify the instrumented SDK; "
            f"saw {sorted(agent_names)}"
        )
