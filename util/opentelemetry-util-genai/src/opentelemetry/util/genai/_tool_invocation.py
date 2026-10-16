# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import timeit
from dataclasses import dataclass, field

from opentelemetry._logs import Logger
from opentelemetry.context import Context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
    _ContextData,
    _json_dumps_unless_str,
)
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.utils import ContentCapturingMode
from opentelemetry.util.types import AnyValue, AttributeValue


@dataclass
class ToolData(_ContextData):
    """Typed data container for a tool invocation."""

    tool_name: str | None = None
    tool_type: str | None = None
    agent_name: str | None = None
    tool_call_id: str | None = None
    tool_description: str | None = None
    tool_call_arguments: AnyValue | None = None
    tool_call_result: AnyValue | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


class ToolInvocation(GenAIInvocation):
    """Represents a tool call invocation for execute_tool span tracking.

    Not used as a message part — use ToolCallRequestPart for that purpose.

    Use handler.tool(name) rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md#execute-tool-span

    Semantic convention attributes for execute_tool spans:
    - gen_ai.operation.name: "execute_tool" (Required)
    - gen_ai.tool.name: Name of the tool (Recommended)
    - gen_ai.agent.name: Human-readable name of the agent executing the tool
      (Conditionally Required "When applicable")
    - gen_ai.tool.call.id: Tool call identifier (Recommended if available)
    - gen_ai.tool.type: Type classification - "function", "extension", or "datastore" (Recommended if available)
    - gen_ai.tool.description: Tool description (Recommended if available)
    - gen_ai.tool.call.arguments: Parameters passed to tool (Opt-In, may contain sensitive data)
    - gen_ai.tool.call.result: Result returned by tool (Opt-In, may contain sensitive data)
    - error.type: Error type if operation failed (Conditionally Required)
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        name: str,
        *,
        tool_type: str | None = None,
        agent_name: str | None = None,
        tool_call_id: str | None = None,
        tool_description: str | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        data: ToolData | None = None,
    ) -> None:
        """Use handler.tool(name) instead of calling this directly.

        .. deprecated:: 1.2b0
            Passing ``tool_call_id`` or ``tool_description`` to the constructor
            is deprecated. Set ``invocation.tool_call_id`` and
            ``invocation.tool_description`` on the returned invocation instead.
        """
        _operation_name = GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_TOOL_NAME, name),
                (GenAI.GEN_AI_TOOL_TYPE, tool_type),
            )
            if v is not None
        }
        if data is None:
            data = ToolData(
                tool_name=name,
                tool_type=tool_type,
                agent_name=agent_name,
                tool_call_id=tool_call_id,
                tool_description=tool_description,
            )
        self.data: ToolData = data
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {name}" if name else _operation_name,
            span_kind=SpanKind.INTERNAL,
            start_attributes=start_attributes,
            context=context,
            _attach_to_context=_attach_to_context,
            content_capturing_mode=content_capturing_mode,
            attributes=self.data.attributes,
            metric_attributes=self.data.metric_attributes,
        )
        self._name: str = name
        self._tool_type: str | None = tool_type
        self._agent_name: str | None = agent_name
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes

    @property
    def name(self) -> str | None:
        return self.data.tool_name

    @property
    def tool_type(self) -> str | None:
        return self.data.tool_type

    @property
    def agent_name(self) -> str | None:
        return self.data.agent_name

    @property
    def tool_call_id(self) -> str | None:
        return self.data.tool_call_id

    @tool_call_id.setter
    def tool_call_id(self, value: str | None) -> None:
        self.data.tool_call_id = value

    @property
    def tool_description(self) -> str | None:
        return self.data.tool_description

    @tool_description.setter
    def tool_description(self, value: str | None) -> None:
        self.data.tool_description = value

    @property
    def arguments(self) -> AnyValue | None:
        return self.data.tool_call_arguments

    @arguments.setter
    def arguments(self, value: AnyValue | None) -> None:
        self.data.tool_call_arguments = value

    @property
    def tool_result(self) -> AnyValue | None:
        return self.data.tool_call_result

    @tool_result.setter
    def tool_result(self, value: AnyValue | None) -> None:
        self.data.tool_call_result = value

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_TOOL_NAME: self.data.tool_name or self._name,
        }
        if self.data.tool_type is not None:
            attrs[GenAI.GEN_AI_TOOL_TYPE] = self.data.tool_type
        if self.data.agent_name is not None:
            attrs[GenAI.GEN_AI_AGENT_NAME] = self.data.agent_name
        attrs.update(self.metric_attributes)
        return attrs

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        capture_content_on_span = self._should_capture_content_on_span
        optional_attrs = (
            (GenAI.GEN_AI_TOOL_TYPE, self.data.tool_type),
            (GenAI.GEN_AI_TOOL_CALL_ID, self.data.tool_call_id),
            (GenAI.GEN_AI_TOOL_DESCRIPTION, self.data.tool_description),
            (GenAI.GEN_AI_AGENT_NAME, self.data.agent_name),
            (
                GenAI.GEN_AI_TOOL_CALL_ARGUMENTS,
                _json_dumps_unless_str(self.arguments)
                if capture_content_on_span
                else None,
            ),
            (
                GenAI.GEN_AI_TOOL_CALL_RESULT,
                _json_dumps_unless_str(self.tool_result)
                if capture_content_on_span
                else None,
            ),
        )
        attributes: dict[str, AttributeValue] = {
            k: v for k, v in optional_attrs if v is not None
        }
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._record_metrics()

    def _record_metrics(self) -> None:
        duration_seconds = max(
            timeit.default_timer() - self._monotonic_start_s,
            0.0,
        )
        self._instruments.execute_tool_duration.record(
            duration_seconds,
            attributes=self._get_metric_attributes(),
            context=self._span_context,
        )

    def _on_stream_chunk(self, chunk_at: float) -> None:
        """A streamed tool result only scopes context; the conventions define no chunk timing metrics for tools."""
