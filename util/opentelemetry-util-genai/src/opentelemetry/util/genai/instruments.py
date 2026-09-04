# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.metrics import Histogram, Meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS = [
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

_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS = [
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


def create_duration_histogram(meter: Meter) -> Histogram:
    return meter.create_histogram(
        name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
        description="Duration of GenAI client operation",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    )


def create_workflow_duration_histogram(meter: Meter) -> Histogram:
    return meter.create_histogram(
        name="gen_ai.invoke_workflow.duration",
        description="Measures the duration of a workflow execution.",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    )


def create_token_histogram(meter: Meter) -> Histogram:
    return meter.create_histogram(
        name=gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE,
        description="Number of input and output tokens used by GenAI clients",
        unit="{token}",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
    )


def create_time_to_first_chunk_histogram(meter: Meter) -> Histogram:
    return meter.create_histogram(
        name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK,
        description="Time to receive the first chunk, measured from when the client issues the generation request to when the first chunk is received in the response stream.",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    )


def create_time_per_output_chunk_histogram(meter: Meter) -> Histogram:
    return meter.create_histogram(
        name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK,
        description="Time per output chunk, recorded for each chunk received after the first one, measured as the time elapsed from the end of the previous chunk to the end of the current chunk.",
        unit="s",
        explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    )
