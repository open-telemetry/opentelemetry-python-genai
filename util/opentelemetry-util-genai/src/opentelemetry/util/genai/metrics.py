# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers for emitting GenAI metrics from LLM invocations."""

from __future__ import annotations

import timeit
from typing import TYPE_CHECKING

from opentelemetry.context import Context
from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.instruments import (
    create_duration_histogram,
    create_time_per_output_chunk_histogram,
    create_time_to_first_chunk_histogram,
    create_token_histogram,
    create_workflow_duration_histogram,
)
from opentelemetry.util.types import Attributes

from ._invocation import GenAIInvocation

if TYPE_CHECKING:
    from ._workflow_invocation import WorkflowInvocation


class InvocationMetricsRecorder:
    """Records duration, token usage, and streaming timing histograms for GenAI invocations."""

    def __init__(self, meter: Meter):
        self._duration_histogram: Histogram = create_duration_histogram(meter)
        self._workflow_duration_histogram: Histogram = (
            create_workflow_duration_histogram(meter)
        )
        self._token_histogram: Histogram = create_token_histogram(meter)
        self._time_to_first_chunk_histogram: Histogram = (
            create_time_to_first_chunk_histogram(meter)
        )
        self._time_per_output_chunk_histogram: Histogram = (
            create_time_per_output_chunk_histogram(meter)
        )

    def record(self, invocation: GenAIInvocation) -> None:
        """Record duration and token metrics for an invocation if possible."""
        attributes = invocation._get_metric_attributes()
        token_counts = invocation._get_metric_token_counts()

        duration_seconds = max(
            timeit.default_timer() - invocation._monotonic_start_s,
            0.0,
        )
        self._duration_histogram.record(
            duration_seconds,
            attributes=attributes,
            context=invocation._span_context,
        )

        for token_type, token_count in token_counts.items():
            self._token_histogram.record(
                token_count,
                attributes=attributes | {GenAI.GEN_AI_TOKEN_TYPE: token_type},
                context=invocation._span_context,
            )

    def record_workflow(self, invocation: WorkflowInvocation) -> None:
        """Record duration metric for a workflow invocation."""
        attributes = invocation._get_metric_attributes()
        duration_seconds = max(
            timeit.default_timer() - invocation._monotonic_start_s,
            0.0,
        )
        self._workflow_duration_histogram.record(
            duration_seconds,
            attributes=attributes,
            context=invocation._span_context,
        )

    def record_time_to_first_chunk(
        self,
        ttfc_seconds: float,
        *,
        attributes: Attributes = None,
        context: Context | None = None,
    ) -> None:
        """Record the streaming time-to-first-chunk."""
        self._time_to_first_chunk_histogram.record(
            ttfc_seconds,
            attributes=attributes,
            context=context,
        )

    def record_time_per_chunk(
        self,
        gap_seconds: float,
        *,
        attributes: Attributes = None,
        context: Context | None = None,
    ) -> None:
        """Record one streaming inter-chunk gap."""
        self._time_per_output_chunk_histogram.record(
            gap_seconds,
            attributes=attributes,
            context=context,
        )


__all__ = ["InvocationMetricsRecorder"]
