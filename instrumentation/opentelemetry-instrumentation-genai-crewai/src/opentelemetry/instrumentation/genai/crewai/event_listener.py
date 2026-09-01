# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate CrewAI orchestration events into GenAI telemetry."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import TypeVar

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
from opentelemetry.util.genai.types import Error, InputMessage, OutputMessage, Text

EventT = TypeVar("EventT", bound=BaseEvent)
RegisteredHandler = Callable[..., object]
CompletionEvent = (
    CrewKickoffCompletedEvent
    | CrewKickoffFailedEvent
    | AgentExecutionCompletedEvent
    | AgentExecutionErrorEvent
)


def _completion_key(event: BaseEvent) -> str | None:
    return event.started_event_id


def _error(value: object) -> Error:
    if isinstance(value, BaseException):
        return Error.from_exception(value)
    return Error(type=type(value).__name__, message=str(value))


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
        self._pending_completions: dict[str, CompletionEvent] = {}
        self._handlers: list[tuple[type[BaseEvent], RegisteredHandler]] = []
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
        event_type: type[EventT],
        handler: Callable[[object, EventT], None],
    ) -> None:
        if self._event_bus is None:
            return
        registered = self._event_bus.on(event_type)(handler)
        self._handlers.append((event_type, registered))

    def _remember(self, event: BaseEvent, invocation: GenAIInvocation) -> None:
        with self._lock:
            previous = self._invocations.setdefault(event.event_id, invocation)
            pending = self._pending_completions.pop(event.event_id, None)
            if pending is None:
                candidates = [
                    completion_key
                    for completion_key, completion in (
                        self._pending_completions.items()
                    )
                    if self._matches(invocation, completion)
                ]
                if len(candidates) == 1:
                    pending = self._pending_completions.pop(candidates[0])
        if previous is not invocation:
            invocation.stop()
        elif pending is not None:
            with self._lock:
                self._invocations.pop(event.event_id, None)
            self._finish(invocation, pending)

    def _pop(self, event: CompletionEvent) -> GenAIInvocation | None:
        key = _completion_key(event)
        if key is None:
            return None
        with self._lock:
            invocation = self._invocations.pop(key, None)
            if invocation is None:
                # CrewAI 1.10 can report a task event ID for an agent or crew
                # failure. Fall back only when the operation match is unique.
                expected_type: type[GenAIInvocation] = (
                    WorkflowInvocation
                    if isinstance(
                        event,
                        (CrewKickoffCompletedEvent, CrewKickoffFailedEvent),
                    )
                    else AgentInvocation
                )
                candidates = [
                    candidate_key
                    for candidate_key, candidate in self._invocations.items()
                    if isinstance(candidate, expected_type)
                ]
                if len(candidates) == 1:
                    invocation = self._invocations.pop(candidates[0])
            if invocation is None:
                self._pending_completions[key] = event
            return invocation

    @staticmethod
    def _matches(
        invocation: GenAIInvocation,
        event: CompletionEvent,
    ) -> bool:
        return (
            isinstance(invocation, WorkflowInvocation)
            and isinstance(
                event, (CrewKickoffCompletedEvent, CrewKickoffFailedEvent)
            )
        ) or (
            isinstance(invocation, AgentInvocation)
            and isinstance(
                event,
                (AgentExecutionCompletedEvent, AgentExecutionErrorEvent),
            )
        )

    @staticmethod
    def _finish(
        invocation: GenAIInvocation,
        event: CompletionEvent,
    ) -> None:
        if isinstance(event, CrewKickoffCompletedEvent) and isinstance(
            invocation, WorkflowInvocation
        ):
            message = _output_message(event.output)
            if message is not None:
                invocation.output_messages = [message]
            invocation.stop()
        elif isinstance(event, CrewKickoffFailedEvent) and isinstance(
            invocation, WorkflowInvocation
        ):
            invocation.fail(_error(event.error))
        elif isinstance(event, AgentExecutionCompletedEvent) and isinstance(
            invocation, AgentInvocation
        ):
            message = _output_message(event.output)
            if message is not None:
                invocation.output_messages = [message]
            invocation.stop()
        elif isinstance(event, AgentExecutionErrorEvent) and isinstance(
            invocation, AgentInvocation
        ):
            invocation.fail(_error(event.error))

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
        if invocation is not None:
            self._finish(invocation, event)

    def _on_crew_failed(
        self, source: object, event: CrewKickoffFailedEvent
    ) -> None:
        del source
        invocation = self._pop(event)
        if invocation is not None:
            self._finish(invocation, event)

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
        if invocation is not None:
            self._finish(invocation, event)

    def _on_agent_failed(
        self, source: object, event: AgentExecutionErrorEvent
    ) -> None:
        del source
        invocation = self._pop(event)
        if invocation is not None:
            self._finish(invocation, event)

    def shutdown(self) -> None:
        if self._event_bus is not None:
            for event_type, handler in self._handlers:
                self._event_bus.off(event_type, handler)
        with self._lock:
            invocations = list(self._invocations.values())
            self._invocations.clear()
            self._pending_completions.clear()
        for invocation in invocations:
            invocation.stop()
        self._handlers.clear()
        self._event_bus = None
