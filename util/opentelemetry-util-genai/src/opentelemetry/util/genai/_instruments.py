# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Final

from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS: Final = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]

_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS: Final = [
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
]

_GEN_AI_EXECUTE_TOOL_DURATION_BUCKETS: Final = (
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS
)

_GEN_AI_INVOKE_WORKFLOW_DURATION_BUCKETS: Final = [
    1,
    5,
    10,
    30,
    60,
    120,
    300,
    600,
    1800,
    3600,
    7200,
]

_GEN_AI_EXECUTE_TOOL_DURATION: Final = "gen_ai.execute_tool.duration"
_GEN_AI_INVOKE_WORKFLOW_DURATION: Final = "gen_ai.invoke_workflow.duration"


class _Instruments:
    """Pre-created metric instruments for GenAI telemetry."""

    def __init__(self, meter: Meter) -> None:
        self.operation_duration: Histogram = meter.create_histogram(
            name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
            description="Duration of GenAI client operation",
            unit="s",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        self.token_usage: Histogram = meter.create_histogram(
            name=gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE,
            description="Number of input and output tokens used by GenAI clients",
            unit="{token}",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
        )
        self.time_to_first_chunk: Histogram = meter.create_histogram(
            name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK,
            description="Time to receive the first chunk, measured from when the client issues the generation request to when the first chunk is received in the response stream.",
            unit="s",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        self.time_per_output_chunk: Histogram = meter.create_histogram(
            name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK,
            description="Time per output chunk, recorded for each chunk received after the first one, measured as the time elapsed from the end of the previous chunk to the end of the current chunk.",
            unit="s",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )
        self.execute_tool_duration: Histogram = meter.create_histogram(
            name=_GEN_AI_EXECUTE_TOOL_DURATION,
            description="Measures the duration of a tool execution.",
            unit="s",
            explicit_bucket_boundaries_advisory=_GEN_AI_EXECUTE_TOOL_DURATION_BUCKETS,
        )
        self.invoke_workflow_duration: Histogram = meter.create_histogram(
            name=_GEN_AI_INVOKE_WORKFLOW_DURATION,
            description="Measures the duration of a workflow execution.",
            unit="s",
            explicit_bucket_boundaries_advisory=_GEN_AI_INVOKE_WORKFLOW_DURATION_BUCKETS,
        )


__all__ = ["_Instruments"]
