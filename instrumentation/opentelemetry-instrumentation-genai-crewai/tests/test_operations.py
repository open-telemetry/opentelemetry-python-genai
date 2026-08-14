# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.llm_events import LLMCallStartedEvent

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import SpanKind, StatusCode


def test_crew_kickoff_emits_workflow_span(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    listener = instrument_crewai._event_listener
    started = CrewKickoffStartedEvent(
        crew_name="research crew", crew=None, inputs={"topic": "otel"}
    )
    listener._on_crew_started(object(), started)
    listener._on_crew_completed(
        object(),
        CrewKickoffCompletedEvent(
            crew_name="research crew",
            crew=None,
            output="report",
            started_event_id=started.event_id,
        ),
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "invoke_workflow research crew"
    assert span.kind == SpanKind.INTERNAL
    assert span.attributes is not None
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_workflow"
    assert span.attributes[GenAI.GEN_AI_WORKFLOW_NAME] == "research crew"


def test_agent_execution_emits_agent_span(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    listener = instrument_crewai._event_listener
    agent = SimpleNamespace(id="agent-1", role="Researcher", goal="Research")
    started = AgentExecutionStartedEvent.model_construct(
        event_id="agent-start",
        started_event_id=None,
        agent=agent,
        task=object(),
        tools=None,
        task_prompt="Find OpenTelemetry news",
    )
    listener._on_agent_started(object(), started)
    completed = AgentExecutionCompletedEvent.model_construct(
        event_id="agent-complete",
        started_event_id=started.event_id,
        agent=agent,
        task=object(),
        output="Findings",
    )
    listener._on_agent_completed(object(), completed)

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "invoke_agent Researcher"
    assert span.kind == SpanKind.INTERNAL
    assert span.attributes is not None
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert span.attributes[GenAI.GEN_AI_AGENT_NAME] == "Researcher"
    assert span.attributes[GenAI.GEN_AI_AGENT_ID] == "agent-1"


def test_llm_events_do_not_emit_inference_spans(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    registered_event_types = {
        event_type
        for event_type, _ in instrument_crewai._event_listener._handlers
    }
    assert LLMCallStartedEvent not in registered_event_types
    assert not span_exporter.get_finished_spans()


def test_failed_operations_record_errors(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    listener = instrument_crewai._event_listener

    crew_started = CrewKickoffStartedEvent(
        crew_name="crew", crew=None, inputs=None
    )
    listener._on_crew_started(object(), crew_started)
    listener._on_crew_failed(
        object(),
        CrewKickoffFailedEvent(
            crew_name="crew",
            crew=None,
            error="crew failed",
            started_event_id=crew_started.event_id,
        ),
    )

    agent = SimpleNamespace(id="agent-1", role="Researcher", goal="Research")
    agent_started = AgentExecutionStartedEvent.model_construct(
        event_id="agent-error-start",
        started_event_id=None,
        agent=agent,
        task=object(),
        tools=None,
        task_prompt="Research",
    )
    listener._on_agent_started(object(), agent_started)
    listener._on_agent_failed(
        object(),
        AgentExecutionErrorEvent.model_construct(
            event_id="agent-error",
            started_event_id=agent_started.event_id,
            agent=agent,
            task=object(),
            error="agent failed",
        ),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert error_attributes.ERROR_TYPE in span.attributes
