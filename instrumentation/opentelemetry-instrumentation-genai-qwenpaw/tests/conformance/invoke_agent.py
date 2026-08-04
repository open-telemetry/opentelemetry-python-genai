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
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InvokeAgentScenario(Scenario):
    # The specific attribute values (agent id/name, conversation id, message
    # content) are covered by unit tests; conformance only validates the
    # telemetry shape against the semconv registry.
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


async def _drive_one_turn(runner_module: Any) -> None:
    runner = runner_module.AgentRunner(agent_id="conformance-agent")
    async for _ in runner.query_handler(
        user_command_msgs(), make_request(session_id="sess-conformance")
    ):
        pass
