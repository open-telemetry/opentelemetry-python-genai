# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream wrappers for Agno instrumentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agno.agent import RunOutput
    from agno.models.base import MessageData
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
    safe_int,
)
from opentelemetry.semconv._incubating.attributes.user_attributes import (
    USER_ID,
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
    Role,
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
        if (
            session_id is not None
            and not self._self_agent_invocation.conversation_id
        ):
            self._self_agent_invocation.conversation_id = str(session_id)

        user_id = getattr(chunk, "user_id", None)
        if (
            user_id is not None
            and USER_ID not in self._self_agent_invocation.attributes
        ):
            self._self_agent_invocation.attributes[USER_ID] = str(user_id)

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
    _self_on_close: Callable[[], Any] | None

    def _process_chunk(self, chunk: Any) -> None:
        session_id = getattr(chunk, "session_id", None)
        if (
            session_id is not None
            and not self._self_workflow_invocation.conversation_id
        ):
            self._self_workflow_invocation.conversation_id = str(session_id)

        user_id = getattr(chunk, "user_id", None)
        if (
            user_id is not None
            and USER_ID not in self._self_workflow_invocation.attributes
        ):
            self._self_workflow_invocation.attributes[USER_ID] = str(user_id)

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

        if self._self_on_close is not None:
            self._self_on_close()

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
        *,
        on_close: Callable[[], Any] | None = None,
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
        *,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(stream)
        self._self_workflow_invocation = invocation
        self._self_capture_content = capture_content
        self._self_content_parts = []
        self._self_completed_content = None
        self._self_finish_reason = "stop"
        self._self_on_close = on_close


class _ModelStreamMixin:
    _self_invocation: InferenceInvocation
    _self_assistant_message: Message | None
    _self_stream_data: MessageData | None
    _self_accumulated_chunks: list[str]

    def _process_chunk(self, chunk: ModelResponse) -> None:
        if self._self_invocation.should_capture_content:
            content = getattr(chunk, "content", None)
            if content is not None:
                formatted = format_content(content)
                if formatted:
                    self._self_accumulated_chunks.append(formatted)

        provider_data = getattr(chunk, "provider_data", None)
        if isinstance(provider_data, dict):
            provider_dict = cast(dict[str, Any], provider_data)
            if (
                "model" in provider_dict
                and not self._self_invocation.response_model_name
            ):
                self._self_invocation.response_model_name = str(
                    cast(object, provider_dict["model"])
                )
            if "id" in provider_dict and not self._self_invocation.response_id:
                self._self_invocation.response_id = str(
                    cast(object, provider_dict["id"])
                )

        usage = getattr(chunk, "response_usage", None)
        if usage is not None:
            self._record_metrics(usage)
        else:
            in_tok = safe_int(getattr(chunk, "input_tokens", None))
            out_tok = safe_int(getattr(chunk, "output_tokens", None))
            if (
                in_tok is not None
                and self._self_invocation.input_tokens is None
            ):
                self._self_invocation.input_tokens = in_tok
            if (
                out_tok is not None
                and self._self_invocation.output_tokens is None
            ):
                self._self_invocation.output_tokens = out_tok

    def _record_metrics(self, metrics: Any) -> None:
        if metrics is None:
            return
        if (
            tok := safe_int(getattr(metrics, "input_tokens", None))
        ) is not None:
            self._self_invocation.input_tokens = tok
        if (
            tok := safe_int(getattr(metrics, "output_tokens", None))
        ) is not None:
            self._self_invocation.output_tokens = tok
        if (
            tok := safe_int(getattr(metrics, "cache_read_tokens", None))
        ) is not None:
            self._self_invocation.cache_read_input_tokens = tok
        if (
            tok := safe_int(getattr(metrics, "cache_write_tokens", None))
        ) is not None:
            self._self_invocation.cache_write_input_tokens = tok
        if (
            tok := safe_int(getattr(metrics, "reasoning_tokens", None))
        ) is not None:
            self._self_invocation.thinking_tokens = tok

    def _finalize_telemetry(self, error: BaseException | None = None) -> None:
        metrics = getattr(self._self_assistant_message, "metrics", None)
        if metrics is not None:
            self._record_metrics(metrics)
        elif self._self_stream_data is not None:
            sd_metrics = getattr(
                self._self_stream_data, "response_metrics", None
            )
            if sd_metrics is not None:
                self._record_metrics(sd_metrics)

        provider_data = getattr(
            self._self_assistant_message, "provider_data", None
        )
        if not provider_data and self._self_stream_data is not None:
            provider_data = getattr(
                self._self_stream_data, "provider_data", None
            )
        if isinstance(provider_data, dict):
            provider_dict = cast(dict[str, Any], provider_data)
            if (
                "model" in provider_dict
                and not self._self_invocation.response_model_name
            ):
                self._self_invocation.response_model_name = str(
                    cast(object, provider_dict["model"])
                )
            if "id" in provider_dict and not self._self_invocation.response_id:
                self._self_invocation.response_id = str(
                    cast(object, provider_dict["id"])
                )

        finish_reasons = (
            ["error"]
            if error is not None
            else extract_model_finish_reasons(
                self._self_assistant_message, self._self_stream_data
            )
        )
        self._self_invocation.finish_reasons = finish_reasons

        if self._self_invocation.should_capture_content:
            has_stream_data_content = (
                self._self_stream_data is not None
                and bool(
                    self._self_stream_data.response_content
                    or self._self_stream_data.response_reasoning_content
                    or self._self_stream_data.response_tool_calls
                )
            )
            if self._self_assistant_message is not None and (
                has_model_output_content(self._self_assistant_message)
                or has_stream_data_content
            ):
                self._self_invocation.output_messages = [
                    format_model_output_message(
                        self._self_assistant_message,
                        finish_reason=finish_reasons[0]
                        if finish_reasons
                        else "stop",
                        stream_data=self._self_stream_data,
                    )
                ]
            elif self._self_accumulated_chunks:
                text = "".join(self._self_accumulated_chunks)
                self._self_invocation.output_messages = [
                    OutputMessage(
                        role=Role.ASSISTANT.value,
                        parts=[TextPart(content=text)],
                        finish_reason=finish_reasons[0]
                        if finish_reasons
                        else "stop",
                    )
                ]

        if error is not None:
            self._self_invocation.fail(error)
        else:
            self._self_invocation.stop()

    def _on_stream_end(self) -> None:
        self._finalize_telemetry()

    def _on_stream_error(self, error: BaseException) -> None:
        self._finalize_telemetry(error)


class AgnoModelStreamWrapper(_ModelStreamMixin, SyncStreamWrapper[Any]):
    """Stream wrapper for synchronous Agno model responses."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        assistant_message: Message | None = None,
        stream_data: MessageData | None = None,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_assistant_message = assistant_message
        self._self_stream_data = stream_data
        self._self_accumulated_chunks = []


class AsyncAgnoModelStreamWrapper(_ModelStreamMixin, AsyncStreamWrapper[Any]):
    """Stream wrapper for asynchronous Agno model responses."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        assistant_message: Message | None = None,
        stream_data: MessageData | None = None,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_assistant_message = assistant_message
        self._self_stream_data = stream_data
        self._self_accumulated_chunks = []
