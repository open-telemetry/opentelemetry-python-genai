# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic agent run for Agno."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import patch

from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.tools.function import Function, FunctionCall
from tests.mock_model import MockModel

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


def _agent_span_attributes(report: LiveCheckReport) -> dict[str, Any]:
    """The attributes weaver saw on the run's ``invoke_agent`` span, by name."""
    for entry in report["samples"]:
        span = entry.get("span")
        if span is None:
            continue
        attributes: dict[str, Any] = {
            attribute["name"]: attribute["value"]
            for attribute in span.get("attributes", [])
        }
        if attributes.get("gen_ai.operation.name") == "invoke_agent":
            return attributes
    raise AssertionError("weaver reported no invoke_agent span")


class UsageScenario(Scenario):
    """A scenario whose span must carry the normalized integer usage counts.

    The base validator only counts spans and checks metric names, so a run
    that lost its usage attributes, or folded the cached prompt tokens
    wrongly, still passes it. These assertions read the span that weaver saw.
    """

    expected_usage: ClassVar[dict[str, int]] = {
        "gen_ai.usage.input_tokens": 1012,
        "gen_ai.usage.output_tokens": 8,
    }

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        attributes = _agent_span_attributes(report)
        for name, expected in self.expected_usage.items():
            assert name in attributes, (
                f"{name} missing from the invoke_agent span; saw "
                f"{sorted(attributes)}"
            )
            value = attributes[name]
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{name} must be an integer count, weaver saw {value!r} "
                f"({type(value).__name__})"
            )
            assert value == expected, f"{name} is {value}, expected {expected}"


class AgentScenario(Scenario):
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
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):

            def sample_tool(x: int) -> int:
                """Double a number."""
                return x * 2

            func = Function.from_callable(sample_tool)
            func_call = FunctionCall(
                function=func,
                arguments={"x": 5},
                call_id="call-conformance",
            )
            func_call.execute()

            agent = Agent(
                name="test-conformance-agent",
                model=MockModel(id="mock-model"),
                session_id="session-conformance",
                tools=[sample_tool],
            )
            mock_output = ModelResponse(content="Conformance Hello back!")
            with (
                patch.object(Agent, "run", wraps=agent.run),
                patch(
                    "agno.models.base.Model.response", return_value=mock_output
                ),
            ):
                agent.run("hello conformance world")


class AgentUsageScenario(UsageScenario):
    """Conformance scenario: an agent run that reports token usage."""

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
        from agno.agent import RunOutput

        from opentelemetry.instrumentation.genai.agno.utils import (
            _run_metrics_cls,
        )

        metrics_cls = _run_metrics_cls()
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            agent = Agent(
                name="test-conformance-usage-agent",
                model=MockModel(id="mock-model"),
                session_id="session-conformance-usage",
            )
            run_output = RunOutput(
                content="Conformance usage back!",
                model_provider="Anthropic",
                metrics=metrics_cls(
                    input_tokens=12,
                    output_tokens=8,
                    cache_read_tokens=1000,
                ),
            )
            with patch(
                "agno.agent._run.run_dispatch", return_value=run_output
            ):
                agent.run("hello conformance usage")


class AgentEventStreamScenario(UsageScenario):
    """Conformance scenario: an agent event stream that reports token usage."""

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
        from agno.agent import RunCompletedEvent, RunStartedEvent

        from opentelemetry.instrumentation.genai.agno.utils import (
            _run_metrics_cls,
        )

        metrics_cls = _run_metrics_cls()

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            yield RunStartedEvent(model_provider="Anthropic")
            yield RunCompletedEvent(
                metrics=metrics_cls(
                    input_tokens=12,
                    output_tokens=8,
                    cache_read_tokens=1000,
                )
            )

        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            agent = Agent(
                name="test-conformance-event-stream-agent",
                model=MockModel(id="mock-model"),
                session_id="session-conformance-stream",
            )
            with patch("agno.agent._run.run_dispatch", side_effect=dispatch):
                for _ in agent.run("hello conformance stream", stream=True):
                    pass
