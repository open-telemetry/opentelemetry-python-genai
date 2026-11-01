# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate errors from the 1.x runner and 2.x runtime command paths."""

from __future__ import annotations

import asyncio
import importlib
from importlib.metadata import version
from typing import Any

from tests.harness import (
    failing_run_command_path,
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
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class RuntimeErrorScenario(Scenario):
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
            asyncio.run(_drive_error())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        (span,) = [
            entry["span"] for entry in report["samples"] if "span" in entry
        ]
        attributes = {
            attr["name"]: attr["value"] for attr in span["attributes"]
        }
        assert attributes[ERROR_TYPE] == "ConnectionError"


async def _drive_error() -> None:
    if int(version("qwenpaw").split(".")[0]) < 2:
        await _drive_runner_error()
        return

    from qwenpaw.runtime.slash_command_registry import CommandSpec

    error = ConnectionError("conformance failure")

    async def fail(ctx: Any, args: str) -> None:
        del ctx, args
        raise error

    runtime = make_runtime()
    runtime.workspace.plugins.slash_command_registry.register(
        CommandSpec(name="fail", handler=fail)
    )
    try:
        async for _ in runtime.run(runtime_request("/fail")):
            pass
    except ConnectionError as caught:
        assert caught is error
    else:
        raise AssertionError("Runtime did not re-raise the command error")


async def _drive_runner_error() -> None:
    runner_module = importlib.import_module("qwenpaw.app.runner.runner")
    error = ConnectionError("conformance failure")
    runner = runner_module.AgentRunner(agent_id="conformance-agent")
    with patched_command_path(runner_module, failing_run_command_path(error)):
        try:
            async for _ in runner.query_handler(
                user_command_msgs(), make_request()
            ):
                pass
        except ConnectionError as caught:
            assert caught is error
        else:
            raise AssertionError("Runner did not re-raise the command error")
