# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream wrappers for Agno instrumentation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agno.agent import RunOutput
    from agno.media import Image, Video
    from agno.models.base import MessageData, Model
    from agno.models.message import Message
    from agno.models.response import ModelResponse
    from agno.run.workflow import WorkflowRunOutput
    from agno.team import TeamRunOutput

    AgnoRunOutput = RunOutput | TeamRunOutput | WorkflowRunOutput

from opentelemetry.instrumentation.genai.agno.utils import (
    _get_property_value,
    extract_model_finish_reasons,
    format_content,
    format_model_output_message,
    has_model_output_content,
    has_model_token_usage,
    has_stream_data_content,
    populate_provider_data,
    record_model_token_usage,
)
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
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
    _self_on_close: Callable[[], None] | None

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
        try:
            if self._self_capture_content:
                if self._self_completed_content is not None:
                    final_content = self._self_completed_content
                elif self._self_content_parts:
                    final_content = self._self_content_parts[-1]
                else:
                    final_content = ""

                if final_content:
                    finish_reason = (
                        "error"
                        if error is not None
                        else self._self_finish_reason
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
        finally:
            if (
                hasattr(self, "_self_on_close")
                and self._self_on_close is not None
            ):
                try:
                    self._self_on_close()
                finally:
                    self._self_on_close = None

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
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(stream)
        self._self_workflow_invocation = invocation
        self._self_capture_content = capture_content
        self._self_content_parts = []
        self._self_completed_content = None
        self._self_finish_reason = "stop"
        self._self_on_close = on_close


class AsyncAgnoWorkflowStreamWrapper(
    _WorkflowStreamMixin, AsyncStreamWrapper[Any]
):
    """Asynchronous stream wrapper for Agno Workflow runs."""

    def __init__(
        self,
        stream: Any,
        invocation: WorkflowInvocation,
        capture_content: bool,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(stream)
        self._self_workflow_invocation = invocation
        self._self_capture_content = capture_content
        self._self_content_parts = []
        self._self_completed_content = None
        self._self_finish_reason = "stop"
        self._self_on_close = on_close


class _ModelStreamMixin:
    _self_invocation: InferenceInvocation | None
    _self_start_invocation: Callable[[], InferenceInvocation] | None
    _self_assistant_message: Message | None
    _self_stream_data: MessageData | None
    _self_model: Model | None
    _self_stream_images: list[Image]
    _self_stream_videos: list[Video]

    def _ensure_invocation(self) -> InferenceInvocation:
        if self._self_invocation is None:
            assert self._self_start_invocation is not None
            invocation = self._self_start_invocation()
            setattr(invocation, "_request_stream", True)
            invocation.suspend()
            self._self_invocation = invocation
        return self._self_invocation

    def _process_chunk(self, chunk: ModelResponse) -> None:
        invocation = self._ensure_invocation()
        populate_provider_data(invocation, chunk.provider_data)
        if invocation.should_capture_content:
            if chunk.images:
                self._self_stream_images.extend(chunk.images)
            if chunk.videos:
                self._self_stream_videos.extend(chunk.videos)

    def _finalize_telemetry(self, error: BaseException | None = None) -> None:
        invocation = self._ensure_invocation()
        if self._self_assistant_message is not None:
            populate_provider_data(
                invocation,
                getattr(self._self_assistant_message, "provider_data", None),
            )
        if self._self_stream_data is not None:
            populate_provider_data(
                invocation,
                self._self_stream_data.response_provider_data,
            )

        metrics = getattr(self._self_assistant_message, "metrics", None)
        if has_model_token_usage(metrics):
            record_model_token_usage(
                invocation, metrics, model=self._self_model
            )
        elif self._self_stream_data is not None:
            sd_metrics = getattr(
                self._self_stream_data, "response_metrics", None
            )
            if has_model_token_usage(sd_metrics):
                record_model_token_usage(
                    invocation, sd_metrics, model=self._self_model
                )

        finish_reasons = (
            ["error"]
            if error is not None
            else extract_model_finish_reasons(
                self._self_assistant_message, self._self_stream_data
            )
        )
        if finish_reasons:
            invocation.finish_reasons = finish_reasons

        if invocation.should_capture_content:
            if self._self_assistant_message is not None and (
                has_model_output_content(self._self_assistant_message)
                or has_stream_data_content(self._self_stream_data)
                or bool(self._self_stream_images or self._self_stream_videos)
            ):
                invocation.output_messages = [
                    format_model_output_message(
                        self._self_assistant_message,
                        stream_data=self._self_stream_data,
                        model=self._self_model,
                        stream_images=self._self_stream_images,
                        stream_videos=self._self_stream_videos,
                    )
                ]

        if error is not None:
            invocation.fail(error)
        else:
            invocation.stop()

    def _on_stream_end(self) -> None:
        self._finalize_telemetry()

    def _on_stream_error(self, error: BaseException) -> None:
        self._finalize_telemetry(error)


class AgnoModelStreamWrapper(_ModelStreamMixin, SyncStreamWrapper[Any]):
    """Stream wrapper for synchronous Agno model responses."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation | None = None,
        assistant_message: Message | None = None,
        stream_data: MessageData | None = None,
        model: Model | None = None,
        *,
        start_invocation: Callable[[], InferenceInvocation] | None = None,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        if invocation is not None:
            invocation.suspend()
        self._self_start_invocation = start_invocation
        self._self_assistant_message = assistant_message
        self._self_stream_data = stream_data
        self._self_model = model
        self._self_stream_images = []
        self._self_stream_videos = []

    def _advance(self, read: Callable[[], Any]) -> Any:
        self._ensure_invocation()
        return super()._advance(read)


class AsyncAgnoModelStreamWrapper(_ModelStreamMixin, AsyncStreamWrapper[Any]):
    """Stream wrapper for asynchronous Agno model responses."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation | None = None,
        assistant_message: Message | None = None,
        stream_data: MessageData | None = None,
        model: Model | None = None,
        *,
        start_invocation: Callable[[], InferenceInvocation] | None = None,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        if invocation is not None:
            invocation.suspend()
        self._self_start_invocation = start_invocation
        self._self_assistant_message = assistant_message
        self._self_stream_data = stream_data
        self._self_model = model
        self._self_stream_images = []
        self._self_stream_videos = []

    async def _advance(self, read: Callable[[], Awaitable[Any]]) -> Any:
        self._ensure_invocation()
        return await super()._advance(read)
