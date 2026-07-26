# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: one QwenPaw turn traced as ``invoke_agent``.

This package emits exactly one semconv operation — ``invoke_agent`` for
each ``AgentRunner.query_handler`` turn. QwenPaw delegates model (LLM) and
tool execution to AgentScope, so no ``chat`` or ``execute_tool`` spans are
produced by this package and no scenario covers them — those operations
belong to the AgentScope instrumentation. The runner's command path is
stubbed with a canned assistant reply (the package's unit tests mock the
same seam; no HTTP is involved and no cassette is needed).
"""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

from tests.harness import (
    fake_run_command_path,
    make_request,
    patched_command_path,
    user_command_msgs,
)

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InvokeAgentScenario(Scenario):
    expected_spans = {"invoke_agent": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        del vcr  # the command path is stubbed; no HTTP is involved
        runner_module = importlib.import_module("qwenpaw.app.runner.runner")
        with instrument(
            QwenPawInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with patched_command_path(
                runner_module, fake_run_command_path("Conformance reply.")
            ):
                asyncio.run(_drive_one_turn(runner_module))

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        # The base validate() already asserts exactly one invoke_agent span.
        (span,) = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "invoke_agent"
        ]
        assert _attr(span, "gen_ai.agent.id") == "conformance-agent"
        assert _attr(span, "gen_ai.conversation.id") == "sess-conformance"

        # The conformance env installs qwenpaw, whose runner always exposes a
        # display name (config value or the built-in fallback).
        agent_name = _attr(span, "gen_ai.agent.name")
        assert isinstance(agent_name, str) and agent_name, (
            f"expected a non-empty gen_ai.agent.name, saw {agent_name!r}"
        )

        input_parts = _part_types(_attr(span, "gen_ai.input.messages"))
        output_parts = _part_types(_attr(span, "gen_ai.output.messages"))
        assert "text" in input_parts, (
            f"expected a text part on an input message, saw {input_parts}"
        )
        assert "text" in output_parts, (
            f"expected a text part on an output message, saw {output_parts}"
        )


async def _drive_one_turn(runner_module: Any) -> None:
    runner = runner_module.AgentRunner(agent_id="conformance-agent")
    async for _ in runner.query_handler(
        user_command_msgs(), make_request(session_id="sess-conformance")
    ):
        pass


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_types(messages_json: str | None) -> list[str]:
    messages = json.loads(messages_json) if messages_json else []
    return [
        part["type"] for message in messages for part in message["parts"]
    ]
