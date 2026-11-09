# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream wrappers for Agno instrumentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from opentelemetry.instrumentation.genai.agno.utils import (
    _get_property_value,
)
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    LocalAgentInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import OutputMessage, TextPart
from opentelemetry.util.genai.utils import content_to_text

_AGENT_COMPLETED_EVENTS = frozenset(
    {
        "RunCompleted",
        "TeamRunCompleted",
        "RunCompletedEvent",
        "TeamRunCompletedEvent",
    }
)
_AGENT_COMPLETED_TYPES = frozenset(
    {
        "RunOutput",
        "TeamRunOutput",
        "RunCompletedEvent",
        "TeamRunCompletedEvent",
        "RunCompleted",
        "TeamRunCompleted",
    }
)
_AGENT_ERROR_NAMES = frozenset(
    {"RunError", "RunErrorEvent", "TeamRunError", "TeamRunErrorEvent"}
)
_AGENT_CONTENT_EVENTS = frozenset(
    {
        "RunContent",
        "IntermediateRunContent",
        "RunContentCompleted",
        "TeamRunContent",
        "TeamRunIntermediateContent",
        "TeamRunContentCompleted",
    }
)
_AGENT_CONTENT_TYPES = frozenset(
    {
        "RunContentEvent",
        "IntermediateRunContentEvent",
        "RunContentCompletedEvent",
    }
)
_WORKFLOW_COMPLETED_EVENTS = frozenset(
    {"WorkflowCompleted", "WorkflowCompletedEvent"}
)
_WORKFLOW_COMPLETED_TYPES = frozenset(
    {"WorkflowRunOutput", "WorkflowCompletedEvent", "WorkflowCompleted"}
)
_WORKFLOW_CONTENT_EVENTS = frozenset(
    {
        "StepOutput",
        "StepOutputEvent",
        "StepCompleted",
        "StepCompletedEvent",
        "RunContent",
        "RunContentEvent",
    }
)


def _chunk_content(chunk: Any) -> Any:
    if isinstance(chunk, str):
        return chunk
    return _get_property_value(chunk, "content")


def _set_conversation_id(
    invocation: AgentInvocation | WorkflowInvocation, chunk: Any
) -> None:
    session_id = getattr(chunk, "session_id", None)
    if session_id and not invocation.conversation_id:
        invocation.conversation_id = str(session_id)


class _StreamOutput:
    """Output and finalization shared by the agent and workflow stream wrappers."""

    _self_agent_invocation: AgentInvocation | WorkflowInvocation
    _self_output_parts: list[str]

    if TYPE_CHECKING:
        _self_exhausted: bool

    def _start(self, invocation: AgentInvocation | WorkflowInvocation) -> None:
        self._self_agent_invocation = invocation
        self._self_output_parts = []
        invocation.suspend()

    def _add_output(self, content: object) -> None:
        if not self._self_agent_invocation.should_capture_content:
            return
        text = content_to_text(content)
        if text is not None:
            self._self_output_parts.append(text)

    def _set_output(self, content: object) -> None:
        # Content that cannot be converted keeps the text captured so far.
        if not self._self_agent_invocation.should_capture_content:
            return
        text = content_to_text(content)
        if text is not None:
            self._self_output_parts = [text]

    def _apply_output(self) -> None:
        text = "".join(self._self_output_parts)
        if text:
            self._self_agent_invocation.output_messages = [
                OutputMessage(role="assistant", parts=[TextPart(content=text)])
            ]

    def _on_stream_end(self) -> None:
        self._apply_output()
        self._self_agent_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._apply_output()
        self._self_agent_invocation.fail(error)


class _AgentStreamMixin(_StreamOutput):
    _self_completed: bool
    _self_finish_reason: str | None

    def _init_agent_state(self, invocation: LocalAgentInvocation) -> None:
        self._start(invocation)
        self._self_completed = False
        self._self_finish_reason = None

    def _process_chunk(self, chunk: Any) -> None:
        invocation = self._self_agent_invocation
        _set_conversation_id(invocation, chunk)

        metrics = getattr(chunk, "metrics", None)
        if metrics is not None and isinstance(invocation, AgentInvocation):
            if getattr(metrics, "input_tokens", None) is not None:
                invocation.input_tokens = metrics.input_tokens
            if getattr(metrics, "output_tokens", None) is not None:
                invocation.output_tokens = metrics.output_tokens

        event_name = str(getattr(chunk, "event", ""))
        chunk_type = type(chunk).__name__
        if (
            event_name in _AGENT_ERROR_NAMES
            or chunk_type in _AGENT_ERROR_NAMES
        ):
            self._self_finish_reason = "error"

        if (
            event_name in _AGENT_COMPLETED_EVENTS
            or chunk_type in _AGENT_COMPLETED_TYPES
        ):
            if self._self_finish_reason is None:
                self._self_finish_reason = "stop"
            content = _chunk_content(chunk)
            if content is not None:
                self._set_output(content)
                self._self_completed = True
        elif not self._self_completed and (
            not event_name
            or event_name in _AGENT_CONTENT_EVENTS
            or chunk_type in _AGENT_CONTENT_TYPES
        ):
            content = _chunk_content(chunk)
            if content is not None:
                self._add_output(content)

    def _set_finish_reason(self, reason: str) -> None:
        if isinstance(self._self_agent_invocation, AgentInvocation):
            self._self_agent_invocation.finish_reasons = [reason]

    def _on_stream_end(self) -> None:
        # A run the caller closed before it completed has no finish reason.
        if self._self_finish_reason is not None:
            self._set_finish_reason(self._self_finish_reason)
        elif self._self_completed or self._self_exhausted:
            self._set_finish_reason("stop")
        super()._on_stream_end()

    def _on_stream_error(self, error: BaseException) -> None:
        self._set_finish_reason("error")
        super()._on_stream_error(error)


class AgnoAgentStreamWrapper(_AgentStreamMixin, SyncStreamWrapper[Any]):
    """Synchronous stream wrapper for Agno Agent and Team runs."""

    def __init__(self, stream: Any, invocation: LocalAgentInvocation) -> None:
        super().__init__(stream, invocation)
        self._init_agent_state(invocation)


class AsyncAgnoAgentStreamWrapper(_AgentStreamMixin, AsyncStreamWrapper[Any]):
    """Asynchronous stream wrapper for Agno Agent and Team runs."""

    def __init__(self, stream: Any, invocation: LocalAgentInvocation) -> None:
        super().__init__(stream, invocation)
        self._init_agent_state(invocation)


class _WorkflowStreamMixin(_StreamOutput):
    _self_completed: bool
    _self_on_close: Callable[[], None] | None

    def _init_workflow_state(
        self,
        invocation: WorkflowInvocation,
        on_close: Callable[[], None] | None,
    ) -> None:
        self._start(invocation)
        self._self_completed = False
        self._self_on_close = on_close

    def _process_chunk(self, chunk: Any) -> None:
        _set_conversation_id(self._self_agent_invocation, chunk)

        event_name = str(getattr(chunk, "event", ""))
        chunk_type = type(chunk).__name__
        if (
            event_name in _WORKFLOW_COMPLETED_EVENTS
            or chunk_type in _WORKFLOW_COMPLETED_TYPES
        ):
            content = _chunk_content(chunk)
            if content is not None:
                self._set_output(content)
                self._self_completed = True
            return
        if self._self_completed:
            return

        # Each step's output supersedes the previous one; the workflow's
        # output is the last step's.
        step_output = getattr(chunk, "step_output", None)
        if step_output is not None:
            content = _get_property_value(step_output, "content")
        elif not event_name or event_name in _WORKFLOW_CONTENT_EVENTS:
            content = _chunk_content(chunk)
        else:
            return
        if content is not None:
            self._set_output(content)

    def _release(self) -> None:
        on_close, self._self_on_close = self._self_on_close, None
        if on_close is not None:
            on_close()

    def _on_stream_end(self) -> None:
        try:
            super()._on_stream_end()
        finally:
            self._release()

    def _on_stream_error(self, error: BaseException) -> None:
        try:
            super()._on_stream_error(error)
        finally:
            self._release()


class AgnoWorkflowStreamWrapper(_WorkflowStreamMixin, SyncStreamWrapper[Any]):
    """Synchronous stream wrapper for Agno Workflow runs."""

    def __init__(
        self,
        stream: Any,
        invocation: WorkflowInvocation,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(stream, invocation)
        self._init_workflow_state(invocation, on_close)


class AsyncAgnoWorkflowStreamWrapper(
    _WorkflowStreamMixin, AsyncStreamWrapper[Any]
):
    """Asynchronous stream wrapper for Agno Workflow runs."""

    def __init__(
        self,
        stream: Any,
        invocation: WorkflowInvocation,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(stream, invocation)
        self._init_workflow_state(invocation, on_close)
