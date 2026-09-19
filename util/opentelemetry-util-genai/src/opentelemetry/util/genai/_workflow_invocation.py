# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry._logs import Logger
from opentelemetry.metrics import Meter
from opentelemetry.trace import Tracer
from opentelemetry.util.genai._attribute import _Attribute
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.semconv.gen_ai import GenAiOperationName
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    InvokeWorkflowInternalOperation,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
)


class WorkflowInvocation(GenAIInvocation, InvokeWorkflowInternalOperation):
    """
    Represents a predetermined sequence of operations (e.g. agent, LLM, tool,
    and retrieval invocations). A workflow groups multiple operations together,
    accepting input(s) and producing final output(s).

    Use handler.workflow(name) rather than constructing this directly.
    """

    _name = _Attribute[str | None]("workflow_name")

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger,
        completion_hook: CompletionHook,
        name: str | None,
        *,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        """Use handler.workflow(name) rather than calling this directly."""
        _operation_name = GenAiOperationName.INVOKE_WORKFLOW.value
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        InvokeWorkflowInternalOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            completion_hook=completion_hook,
            operation_name=_operation_name,
            workflow_name=name,
            content_capturing_mode=mode,
        )
        GenAIInvocation.__init__(self)
        if self.input_messages is None:
            self.input_messages = []
        if self.output_messages is None:
            self.output_messages = []
        self.start()
