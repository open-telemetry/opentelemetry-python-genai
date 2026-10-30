# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: background workflow run for Agno."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("agno.workflow.workflow")

from agno.workflow.workflow import Workflow

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class WorkflowBackgroundScenario(Scenario):
    expected_spans = {"invoke_workflow": 1}
    expected_metrics = ("gen_ai.invoke_workflow.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        async def step(step_input: Any) -> str:
            return "done"

        async def _run() -> None:
            workflow = Workflow(
                name="test-conformance-bg-workflow", steps=[step]
            )
            await workflow.arun(
                "hello background",
                background=True,
                user_id="user-1",
                session_id="session-bg",
            )
            current = asyncio.current_task()
            await asyncio.gather(
                *[t for t in asyncio.all_tasks() if t is not current]
            )

        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            asyncio.run(_run())
