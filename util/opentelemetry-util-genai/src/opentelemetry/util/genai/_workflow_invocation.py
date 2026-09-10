# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import timeit
from dataclasses import asdict

from opentelemetry._logs import Logger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
)
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
)
from opentelemetry.util.types import AttributeValue


class WorkflowInvocation(GenAIInvocation):
    """
    Represents a predetermined sequence of operations (e.g. agent, LLM, tool,
    and retrieval invocations). A workflow groups multiple operations together,
    accepting input(s) and producing final output(s).

    Use handler.workflow(name) rather than constructing this directly.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        name: str | None,
        *,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        """Use handler.workflow(name) rather than calling this directly."""
        _operation_name = GenAI.GenAiOperationNameValues.INVOKE_WORKFLOW.value
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {name}" if name else _operation_name,
            span_kind=SpanKind.INTERNAL,
            content_capturing_mode=content_capturing_mode,
        )
        self._name: str | None = name
        self.conversation_id: str | None = None
        self.input_messages: list[InputMessage] = []
        self.output_messages: list[OutputMessage] = []
        self._start(self._get_start_attributes())

    def _get_start_attributes(self) -> dict[str, AttributeValue]:
        """Return sampling-relevant attributes available at span creation time."""
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
        }
        if self._name is not None:
            attrs[GenAI.GEN_AI_WORKFLOW_NAME] = self._name
        return attrs

    def _get_messages_for_span(self) -> dict[str, AttributeValue]:
        if not self._should_capture_content_on_span:
            return {}
        optional_attrs = (
            (
                GenAI.GEN_AI_INPUT_MESSAGES,
                gen_ai_json_dumps([asdict(m) for m in self.input_messages])
                if self.input_messages
                else None,
            ),
            (
                GenAI.GEN_AI_OUTPUT_MESSAGES,
                gen_ai_json_dumps([asdict(m) for m in self.output_messages])
                if self.output_messages
                else None,
            ),
        )
        return {
            key: value for key, value in optional_attrs if value is not None
        }

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {}
        if self._name is not None:
            attrs[GenAI.GEN_AI_WORKFLOW_NAME] = self._name
        attrs.update(self.metric_attributes)
        return attrs

    def _apply_finish(self, error: Error | None = None) -> None:
        attributes: dict[str, AttributeValue] = self._get_messages_for_span()
        if self.conversation_id is not None:
            attributes[GenAI.GEN_AI_CONVERSATION_ID] = self.conversation_id
        if error is not None:
            self._apply_error_attributes(error)
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._call_completion_hook(
            inputs=self.input_messages,
            outputs=self.output_messages,
        )
        self._record_metrics()

    def _record_metrics(self) -> None:
        duration_seconds = max(
            timeit.default_timer() - self._monotonic_start_s,
            0.0,
        )
        self._instruments.invoke_workflow_duration.record(
            duration_seconds,
            attributes=self._get_metric_attributes(),
            context=self._span_context,
        )
