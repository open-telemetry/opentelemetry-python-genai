# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from crewai import LLM, Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
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
from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.genai.crewai.event_listener import (
    _input_messages,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import SpanKind, StatusCode


def test_crew_inputs_are_preserved_as_separate_messages() -> None:
    messages = _input_messages({"topic": "otel", "limit": 3})

    assert [message.role for message in messages] == ["user", "user"]
    assert [
        part.content for message in messages for part in message.parts
    ] == [
        '{"topic": "otel"}',
        '{"limit": 3}',
    ]


@pytest.mark.vcr()
def test_successful_kickoff_emits_completed_spans(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    vcr: Any,
) -> None:
    """Exercise the public CrewAI API against a replayed provider response."""
    key_override = (
        {}
        if os.getenv("OPENAI_API_KEY")
        else {"OPENAI_API_KEY": "test_openai_api_key"}
    )
    with (
        mock.patch.dict(
            os.environ,
            {**key_override, "CREWAI_DISABLE_TELEMETRY": "true"},
        ),
        mock.patch(
            "crewai.utilities.task_output_storage_handler."
            "KickoffTaskOutputsSQLiteStorage",
            return_value=mock.Mock(spec=KickoffTaskOutputsSQLiteStorage),
        ),
    ):
        llm = LLM(model="openai/gpt-4o-mini")
        researcher = Agent(
            role="Researcher",
            goal="Answer the question accurately",
            backstory="A concise research assistant",
            llm=llm,
        )
        task = Task(
            description="Say this is a test.",
            expected_output="A short confirmation.",
            agent=researcher,
        )
        with vcr.use_cassette("successful_kickoff.yaml"):
            Crew(agents=[researcher], tasks=[task]).kickoff()
        assert crewai_event_bus.flush()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    agent_span = next(
        span
        for span in spans
        if span.attributes
        and span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_agent"
    )
    workflow_span = next(
        span
        for span in spans
        if span.attributes
        and span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_workflow"
    )

    assert agent_span.status.status_code == StatusCode.UNSET
    assert workflow_span.status.status_code == StatusCode.UNSET
    assert agent_span.end_time is not None
    assert workflow_span.end_time is not None


@pytest.mark.vcr()
def test_failed_kickoff_records_error_message_on_all_spans(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    vcr: Any,
) -> None:
    """Verify public CrewAI failures retain the provider error message."""
    key_override = (
        {}
        if os.getenv("OPENAI_API_KEY")
        else {"OPENAI_API_KEY": "test_openai_api_key"}
    )
    with (
        mock.patch.dict(
            os.environ,
            {**key_override, "CREWAI_DISABLE_TELEMETRY": "true"},
        ),
        mock.patch(
            "crewai.utilities.task_output_storage_handler."
            "KickoffTaskOutputsSQLiteStorage",
            return_value=mock.Mock(spec=KickoffTaskOutputsSQLiteStorage),
        ),
    ):
        llm = LLM(model="openai/gpt-4o-mini")
        researcher = Agent(
            role="Researcher",
            goal="Answer the question accurately",
            backstory="A concise research assistant",
            llm=llm,
            max_retry_limit=0,
        )
        task = Task(
            description="Say this is a test.",
            expected_output="A short confirmation.",
            agent=researcher,
        )
        with (
            vcr.use_cassette("failed_kickoff.yaml"),
            pytest.raises(Exception, match="CrewAI VCR test"),
        ):
            Crew(agents=[researcher], tasks=[task]).kickoff()
        assert crewai_event_bus.flush()

    spans = span_exporter.get_finished_spans()
    assert spans
    assert {
        span.attributes[GenAI.GEN_AI_OPERATION_NAME]
        for span in spans
        if span.attributes is not None
    } == {"invoke_agent", "invoke_workflow"}
    for span in spans:
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description is not None
        assert "Invalid API key for CrewAI VCR test" in span.status.description
        assert span.attributes is not None
        assert error_attributes.ERROR_TYPE in span.attributes
        assert span.end_time is not None


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
    assert {span.status.description for span in spans} == {
        "agent failed",
        "crew failed",
    }
    for span in spans:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert span.attributes[error_attributes.ERROR_TYPE] == "str"
