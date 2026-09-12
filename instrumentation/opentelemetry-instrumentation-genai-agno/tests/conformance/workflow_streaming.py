# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: streaming workflow run for Agno."""

from __future__ import annotations

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


class WorkflowStreamingScenario(Scenario):
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
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):

            def step1(step_input: Any) -> str:
                return "step1 stream output"

            workflow = Workflow(
                name="test-conformance-workflow-streaming",
                steps=[step1],
                session_id="session-workflow-streaming",
            )
            stream = workflow.run(
                "hello workflow streaming conformance", stream=True
            )
            for _ in stream:
                pass
