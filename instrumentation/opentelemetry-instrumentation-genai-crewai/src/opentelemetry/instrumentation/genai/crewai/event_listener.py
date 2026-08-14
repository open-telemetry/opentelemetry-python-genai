# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate CrewAI orchestration events into GenAI telemetry."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus
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

from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    GenAIInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text

EventHandler = Callable[[object, BaseEvent], None]


def _completion_key(event: BaseEvent) -> str | None:
    return event.started_event_id


def _error(value: object, fallback: str) -> BaseException:
    return (
        value
        if isinstance(value, BaseException)
        else RuntimeError(str(value or fallback))
    )


def _input_message(value: object) -> InputMessage | None:
    if value is None:
        return None
    if isinstance(value, str):
        content = value
    else:
        try:
            content = json.dumps(value, default=str)
        except (TypeError, ValueError):
            content = str(value)
    return InputMessage(role="user", parts=[Text(content=content)])


def _output_message(value: object) -> OutputMessage | None:
    if value is None:
        return None
    content = value if isinstance(value, str) else str(value)
    return OutputMessage(
        role="assistant",
        parts=[Text(content=content)],
        finish_reason="",
    )


class CrewAIEventListener(BaseEventListener):
    """Listen for CrewAI-owned workflow and agent operations."""

    def __init__(self, telemetry_handler: TelemetryHandler) -> None:
        self._telemetry_handler = telemetry_handler
        self._invocations: dict[str, GenAIInvocation] = {}
        self._handlers: list[tuple[type[BaseEvent], EventHandler]] = []
        self._event_bus: CrewAIEventsBus | None = None
        self._lock = threading.RLock()
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        self._event_bus = crewai_event_bus
        self._register(CrewKickoffStartedEvent, self._on_crew_started)
        self._register(CrewKickoffCompletedEvent, self._on_crew_completed)
        self._register(CrewKickoffFailedEvent, self._on_crew_failed)
        self._register(AgentExecutionStartedEvent, self._on_agent_started)
        self._register(AgentExecutionCompletedEvent, self._on_agent_completed)
        self._register(AgentExecutionErrorEvent, self._on_agent_failed)

    def _register(
        self,
        event_type: type[BaseEvent],
        handler: EventHandler,
    ) -> None:
        if self._event_bus is None:
            return
        registered = self._event_bus.on(event_type)(handler)
        self._handlers.append((event_type, registered))

    def _remember(self, event: BaseEvent, invocation: GenAIInvocation) -> None:
        with self._lock:
            previous = self._invocations.setdefault(event.event_id, invocation)
        if previous is not invocation:
            invocation.stop()

    def _pop(self, event: BaseEvent) -> GenAIInvocation | None:
        key = _completion_key(event)
        if key is None:
            return None
        with self._lock:
            return self._invocations.pop(key, None)

    def _on_crew_started(
        self, source: object, event: CrewKickoffStartedEvent
    ) -> None:
        del source
        if not is_instrumentation_enabled():
            return
        invocation = self._telemetry_handler.workflow(event.crew_name)
        message = _input_message(event.inputs)
        if message is not None:
            invocation.input_messages = [message]
        self._remember(event, invocation)

    def _on_crew_completed(
        self, source: object, event: CrewKickoffCompletedEvent
    ) -> None:
        del source
        invocation = self._pop(event)
        if not isinstance(invocation, WorkflowInvocation):
            return
        message = _output_message(event.output)
        if message is not None:
            invocation.output_messages = [message]
        invocation.stop()

    def _on_crew_failed(
        self, source: object, event: CrewKickoffFailedEvent
    ) -> None:
        del source
        invocation = self._pop(event)
        if isinstance(invocation, WorkflowInvocation):
            invocation.fail(_error(event.error, "CrewAI crew kickoff failed"))

    def _on_agent_started(
        self, source: object, event: AgentExecutionStartedEvent
    ) -> None:
        del source
        if not is_instrumentation_enabled():
            return
        agent = event.agent
        invocation = self._telemetry_handler.invoke_local_agent(
            agent_name=agent.role
        )
        invocation.agent_id = str(agent.id)
        invocation.agent_description = agent.goal
        invocation.input_messages = [
            InputMessage(role="user", parts=[Text(content=event.task_prompt)])
        ]
        self._remember(event, invocation)

    def _on_agent_completed(
        self, source: object, event: AgentExecutionCompletedEvent
    ) -> None:
        del source
        invocation = self._pop(event)
        if not isinstance(invocation, AgentInvocation):
            return
        message = _output_message(event.output)
        if message is not None:
            invocation.output_messages = [message]
        invocation.stop()

    def _on_agent_failed(
        self, source: object, event: AgentExecutionErrorEvent
    ) -> None:
        del source
        invocation = self._pop(event)
        if isinstance(invocation, AgentInvocation):
            invocation.fail(
                _error(event.error, "CrewAI agent execution failed")
            )

    def shutdown(self) -> None:
        if self._event_bus is not None:
            for event_type, handler in self._handlers:
                self._event_bus.off(event_type, handler)
        with self._lock:
            invocations = list(self._invocations.values())
            self._invocations.clear()
        for invocation in invocations:
            invocation.stop()
        self._handlers.clear()
        self._event_bus = None
