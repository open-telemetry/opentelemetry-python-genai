# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry._logs import Logger
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._invocation import Error, GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.types import AttributeValue

# semconv-genai#427 (unreleased)
# TODO: switch to the semconv module once #427 merges.
_GEN_AI_OPERATION_NAME_RUN_GUARDRAIL = "run_guardrail"
# semconv-genai#427 (unreleased)
# TODO: switch to the semconv module once #427 merges.
_GEN_AI_GUARDRAIL_COMPONENT_NAME = "gen_ai.guardrail.component.name"
# semconv-genai#427 (unreleased)
# TODO: switch to the semconv module once #427 merges.
_GEN_AI_GUARDRAIL_VERDICT_TYPE = "gen_ai.guardrail.verdict.type"
# semconv-genai#427 (unreleased)
# TODO: switch to the semconv module once #427 merges.
_GEN_AI_GUARDRAIL_TARGET_TYPE = "gen_ai.guardrail.target.type"


class GuardrailInvocation(GenAIInvocation):
    """Represents a guardrail invocation for run_guardrail span tracking.

    Use handler.guardrail(name) rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions-genai/pull/427

    Semantic convention attributes for run_guardrail spans:
    - gen_ai.operation.name: "run_guardrail"
    - gen_ai.guardrail.component.name: Name of the guardrail
    - gen_ai.provider.name: Name of the GenAI provider
    - gen_ai.guardrail.verdict.type: "deny" if triggered, otherwise "allow"
    - gen_ai.guardrail.target.type: "input" or "output". Required by
      semconv-genai#427 and must be supplied by instrumentations that know the
      guardrail direction.
    """

    def __init__(
        self,
        tracer: Tracer,
        metrics_recorder: InvocationMetricsRecorder,
        logger: Logger,
        completion_hook: CompletionHook,
        name: str,
        *,
        provider: str,
        target_type: str | None = None,
    ) -> None:
        """Use handler.guardrail(name) instead of calling this directly."""
        operation_name = _GEN_AI_OPERATION_NAME_RUN_GUARDRAIL
        super().__init__(
            tracer,
            metrics_recorder,
            logger,
            completion_hook,
            operation_name=operation_name,
            span_name=f"{operation_name} {name}" if name else operation_name,
            span_kind=SpanKind.INTERNAL,
        )
        self._name: str = name
        self._provider: str = provider
        self.triggered: bool = False
        self.target_type: str | None = target_type
        self._start(self._get_start_attributes())

    def _get_start_attributes(self) -> dict[str, AttributeValue]:
        """Return sampling-relevant attributes available at span creation time."""
        return {
            GEN_AI_OPERATION_NAME: self._operation_name,
            _GEN_AI_GUARDRAIL_COMPONENT_NAME: self._name,
            GEN_AI_PROVIDER_NAME: self._provider,
        }

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {
            GEN_AI_OPERATION_NAME: self._operation_name,
            GEN_AI_PROVIDER_NAME: self._provider,
        }
        attrs.update(self.metric_attributes)
        return attrs

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        optional_attrs = (
            (_GEN_AI_GUARDRAIL_TARGET_TYPE, self.target_type),
        )
        attributes: dict[str, AttributeValue] = {
            _GEN_AI_GUARDRAIL_VERDICT_TYPE: (
                "deny" if self.triggered else "allow"
            ),
            **{key: value for key, value in optional_attrs if value is not None},
        }
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._metrics_recorder.record(self)
