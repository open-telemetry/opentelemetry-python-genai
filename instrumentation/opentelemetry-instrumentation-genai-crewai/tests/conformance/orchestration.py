# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


def _emit(event: BaseEvent) -> None:
    future = crewai_event_bus.emit(object(), event)
    if future is not None:
        future.result(timeout=5)


class OrchestrationScenario(Scenario):
    expected_spans = {
        "invoke_workflow": 1,
        "invoke_agent": 1,
    }
    expected_metrics = ("gen_ai.client.operation.duration",)

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
            CrewAIInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture="SPAN_ONLY",
        ):
            crew = CrewKickoffStartedEvent(
                crew_name="research crew",
                crew=None,
                inputs={"topic": "OpenTelemetry"},
            )
            _emit(crew)

            agent = SimpleNamespace(
                id="agent-1", role="Researcher", goal="Research"
            )
            agent_started = AgentExecutionStartedEvent.model_construct(
                event_id="agent-start",
                started_event_id=None,
                agent=agent,
                task=object(),
                tools=None,
                task_prompt="Research OpenTelemetry",
            )
            _emit(agent_started)

            _emit(
                AgentExecutionCompletedEvent.model_construct(
                    event_id="agent-complete",
                    started_event_id=agent_started.event_id,
                    agent=agent,
                    task=object(),
                    output="report",
                )
            )
            _emit(
                CrewKickoffCompletedEvent(
                    crew_name="research crew",
                    crew=None,
                    output="report",
                    started_event_id=crew.event_id,
                )
            )
