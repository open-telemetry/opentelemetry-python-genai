# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry._logs import Logger
from opentelemetry.context import Context
from opentelemetry.metrics import Meter
from opentelemetry.trace import Tracer
from opentelemetry.util.genai._attribute import _Attribute
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.semconv.gen_ai import GenAiOperationName
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    ExecuteToolInternalOperation,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
)
from opentelemetry.util.types import AnyValue


class ToolInvocation(GenAIInvocation, ExecuteToolInternalOperation):
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

    _name = _Attribute[str]("tool_name")
    tool_result = _Attribute[AnyValue | None]("tool_call_result")
    arguments = _Attribute[AnyValue | None]("tool_call_arguments")

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger,
        name: str,
        *,
        tool_type: str | None = None,
        agent_name: str | None = None,
        tool_call_id: str | None = None,
        tool_description: str | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
    ) -> None:
        """Use handler.tool(name) instead of calling this directly.

        .. deprecated:: 1.2b0
            Passing ``tool_call_id`` or ``tool_description`` to the constructor
            is deprecated. Set ``invocation.tool_call_id`` and
            ``invocation.tool_description`` on the returned invocation instead.
        """
        _operation_name = GenAiOperationName.EXECUTE_TOOL.value
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        ExecuteToolInternalOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            operation_name=_operation_name,
            tool_name=name,
            tool_type=tool_type,
            agent_name=agent_name,
            tool_call_id=tool_call_id,
            tool_description=tool_description,
            content_capturing_mode=mode,
        )
        GenAIInvocation.__init__(self)
        self.start(context=context)

    @property
    def should_capture_content_on_span(self) -> bool:
        """Returns whether content capture is enabled on spans.

        .. deprecated:: 1.2b0
            Use :attr:`should_capture_content` instead.
        """
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
