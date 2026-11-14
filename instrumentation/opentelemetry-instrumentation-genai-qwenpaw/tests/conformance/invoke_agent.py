# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: one QwenPaw turn traced as ``invoke_agent``.

This package emits exactly one semconv operation — ``invoke_agent`` for
each runner (1.x) or runtime (2.x) turn. QwenPaw delegates model (LLM) and
tool execution to AgentScope, so no ``chat`` or ``execute_tool`` spans are
produced by this package and no scenario covers them — those operations
belong to the AgentScope instrumentation. A registered slash command drives the real 2.x runtime;
the 1.x command path uses a canned reply. Neither requires HTTP or a cassette.
"""

from __future__ import annotations

import asyncio
import importlib
from importlib.metadata import version
from typing import Any

from tests.harness import (
    fake_run_command_path,
    make_request,
    make_runtime,
    patched_command_path,
    runtime_request,
    user_command_msgs,
)

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


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
        del vcr
        with instrument(
            QwenPawInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            if int(version("qwenpaw").split(".")[0]) >= 2:
                asyncio.run(_drive_runtime())
                return
            runner_module = importlib.import_module(
                "qwenpaw.app.runner.runner"
            )
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


async def _drive_runtime() -> None:
    runtime = make_runtime("Conformance reply.")
    async for _ in runtime.run(runtime_request(session_id="sess-conformance")):
        pass
