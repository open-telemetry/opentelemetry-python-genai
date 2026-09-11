# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream wrappers for Agno instrumentation."""

from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.genai.agno.utils import (
    _get_property_value,
    format_content,
)
from opentelemetry.util.genai.invocation import (
    LocalAgentInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
    OutputMessage,
    TextPart,
)


def _extract_chunk_content(chunk: Any) -> str | None:
    if chunk is None:
        return None
    if isinstance(chunk, str):
        return chunk
    content = _get_property_value(chunk, "content")
    if content is not None:
        return format_content(content)
    return None


class _AgentStreamMixin:
    _self_agent_invocation: LocalAgentInvocation
    _self_capture_content: bool
    _self_content_parts: list[str]
    _self_completed_content: str | None
    _self_finish_reason: str

    def _process_chunk(self, chunk: Any) -> None:
        session_id = getattr(chunk, "session_id", None)
        if session_id and not self._self_agent_invocation.conversation_id:
            self._self_agent_invocation.conversation_id = str(session_id)

        metrics = getattr(chunk, "metrics", None)
        if metrics is not None:
            if getattr(metrics, "input_tokens", None) is not None:
                self._self_agent_invocation.input_tokens = metrics.input_tokens
            if getattr(metrics, "output_tokens", None) is not None:
                self._self_agent_invocation.output_tokens = (
                    metrics.output_tokens
                )

        event_name = str(getattr(chunk, "event", ""))
        chunk_type = type(chunk).__name__
        is_completed = event_name in (
            "RunCompleted",
            "TeamRunCompleted",
            "RunCompletedEvent",
            "TeamRunCompletedEvent",
        ) or chunk_type in (
            "RunOutput",
            "TeamRunOutput",
            "RunCompletedEvent",
            "TeamRunCompletedEvent",
            "RunCompleted",
            "TeamRunCompleted",
        )
        is_error = event_name in (
            "RunError",
            "RunErrorEvent",
            "TeamRunError",
            "TeamRunErrorEvent",
        ) or chunk_type in (
            "RunErrorEvent",
            "TeamRunErrorEvent",
            "RunError",
            "TeamRunError",
        )
        if is_error:
            self._self_finish_reason = "error"

        if is_completed:
            content = _extract_chunk_content(chunk)
            if content is not None:
                self._self_completed_content = content
        elif self._self_capture_content:
            is_content_chunk = (
                not event_name
                or event_name
                in (
                    "RunContent",
                    "IntermediateRunContent",
                    "RunContentCompleted",
                    "TeamRunContent",
                    "TeamRunIntermediateContent",
                    "TeamRunContentCompleted",
                )
                or chunk_type
                in (
                    "RunContentEvent",
                    "IntermediateRunContentEvent",
                    "RunContentCompletedEvent",
                )
            )
            if is_content_chunk:
                content = _extract_chunk_content(chunk)
                if content is not None:
                    self._self_content_parts.append(content)

    def _finalize(self, error: BaseException | None = None) -> None:
        if self._self_capture_content:
            if self._self_completed_content is not None:
                final_content = self._self_completed_content
            else:
                final_content = "".join(self._self_content_parts)

            if final_content:
                finish_reason = (
                    "error" if error is not None else self._self_finish_reason
                )
                self._self_agent_invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[TextPart(content=final_content)],
                        finish_reason=finish_reason,
                    )
                ]

        if error is not None:
            self._self_agent_invocation.fail(error)
        else:
            self._self_agent_invocation.stop()

    def _on_stream_end(self) -> None:
        self._finalize()

    def _on_stream_error(self, error: BaseException) -> None:
        self._finalize(error)


class AgnoAgentStreamWrapper(_AgentStreamMixin, SyncStreamWrapper[Any]):
    """Synchronous stream wrapper for Agno Agent and Team runs."""

    def __init__(
        self,
        stream: Any,
        invocation: LocalAgentInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        self._self_agent_invocation = invocation
        self._self_capture_content = capture_content
        self._self_content_parts = []
        self._self_completed_content = None
        self._self_finish_reason = "stop"


class AsyncAgnoAgentStreamWrapper(_AgentStreamMixin, AsyncStreamWrapper[Any]):
    """Asynchronous stream wrapper for Agno Agent and Team runs."""

    def __init__(
        self,
        stream: Any,
        invocation: LocalAgentInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        self._self_agent_invocation = invocation
        self._self_capture_content = capture_content
        self._self_content_parts = []
        self._self_completed_content = None
        self._self_finish_reason = "stop"


class _WorkflowStreamMixin:
    _self_workflow_invocation: WorkflowInvocation
    _self_capture_content: bool
    _self_content_parts: list[str]
    _self_completed_content: str | None
    _self_finish_reason: str

    def _process_chunk(self, chunk: Any) -> None:
        session_id = getattr(chunk, "session_id", None)
        if session_id and not self._self_workflow_invocation.conversation_id:
            self._self_workflow_invocation.conversation_id = str(session_id)

        event_name = str(getattr(chunk, "event", ""))
        chunk_type = type(chunk).__name__
        is_completed = event_name in (
            "WorkflowCompleted",
            "WorkflowCompletedEvent",
        ) or chunk_type in (
            "WorkflowRunOutput",
            "WorkflowCompletedEvent",
            "WorkflowCompleted",
        )
        is_error = event_name in (
            "WorkflowError",
            "WorkflowErrorEvent",
        ) or chunk_type in ("WorkflowErrorEvent", "WorkflowError")
        if is_error:
            self._self_finish_reason = "error"

        if is_completed:
            content = _extract_chunk_content(chunk)
            if content is not None:
                self._self_completed_content = content
        elif (
            hasattr(chunk, "step_output")
            and getattr(chunk, "step_output") is not None
        ):
            step_output = getattr(chunk, "step_output")
            step_content = _get_property_value(step_output, "content")
            if step_content is not None and self._self_capture_content:
                self._self_content_parts.append(format_content(step_content))
        elif self._self_capture_content:
            is_content_chunk = not event_name or event_name in (
                "StepOutput",
                "StepOutputEvent",
                "StepCompleted",
                "StepCompletedEvent",
                "RunContent",
                "RunContentEvent",
            )
            if is_content_chunk:
                content = _extract_chunk_content(chunk)
                if content is not None:
                    self._self_content_parts.append(content)

    def _finalize(self, error: BaseException | None = None) -> None:
        if self._self_capture_content:
            if self._self_completed_content is not None:
                final_content = self._self_completed_content
            elif self._self_content_parts:
                final_content = self._self_content_parts[-1]
            else:
                final_content = ""

            if final_content:
                finish_reason = (
                    "error" if error is not None else self._self_finish_reason
                )
                self._self_workflow_invocation.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[TextPart(content=final_content)],
                        finish_reason=finish_reason,
                    )
                ]

        if error is not None:
            self._self_workflow_invocation.fail(error)
        else:
            self._self_workflow_invocation.stop()

    def _on_stream_end(self) -> None:
        self._finalize()

    def _on_stream_error(self, error: BaseException) -> None:
        self._finalize(error)


class AgnoWorkflowStreamWrapper(_WorkflowStreamMixin, SyncStreamWrapper[Any]):
    """Synchronous stream wrapper for Agno Workflow runs."""

    def __init__(
        self,
        stream: Any,
        invocation: WorkflowInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        self._self_workflow_invocation = invocation
        self._self_capture_content = capture_content
        self._self_content_parts = []
        self._self_completed_content = None
        self._self_finish_reason = "stop"


class AsyncAgnoWorkflowStreamWrapper(
    _WorkflowStreamMixin, AsyncStreamWrapper[Any]
):
    """Asynchronous stream wrapper for Agno Workflow runs."""

    def __init__(
        self,
        stream: Any,
        invocation: WorkflowInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        self._self_workflow_invocation = invocation
        self._self_capture_content = capture_content
        self._self_content_parts = []
        self._self_completed_content = None
        self._self_finish_reason = "stop"
