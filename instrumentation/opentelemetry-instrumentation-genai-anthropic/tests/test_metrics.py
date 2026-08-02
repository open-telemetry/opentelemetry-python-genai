# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Metric-recording tests for the Anthropic instrumentation."""

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

MODEL = "claude-sonnet-4-20250514"
MESSAGES = [{"role": "user", "content": "Say hello in one word."}]


def _metrics_by_name(metric_reader):
    by_name = {}
    for rm in metric_reader.get_metrics_data().resource_metrics:
        for scope in rm.scope_metrics:
            for metric in scope.metrics:
                by_name[metric.name] = metric
    return by_name


def _assert_streaming_timing_metrics(metric_reader):
    """Assert the streaming timing metrics are emitted through the real
    Anthropic stream wrapper path.

    Regression coverage for the ``invocation=invocation`` wiring in
    ``wrappers.py``: dropping it would keep every span/attribute test green but
    silently stop emitting TTFC and per-output-chunk metrics for Anthropic
    streaming.
    """
    metrics = _metrics_by_name(metric_reader)

    ttfc = metrics.get(
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK
    )
    assert ttfc is not None
    ttfc_point = ttfc.data.data_points[0]
    assert ttfc_point.count == 1
    assert ttfc_point.sum >= 0
    assert (
        ttfc_point.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert ttfc_point.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == MODEL

    per_chunk = metrics.get(
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK
    )
    assert per_chunk is not None
    per_chunk_point = per_chunk.data.data_points[0]
    # One record per inter-chunk gap; the streaming cassette has several events.
    assert per_chunk_point.count >= 1
    assert per_chunk_point.sum >= 0


def test_sync_messages_streaming_timing_metrics(
    metric_reader, anthropic_client, instrument_with_content, vcr
):
    with vcr.use_cassette("test_sync_messages_create_streaming.yaml"):
        with anthropic_client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=MESSAGES,
            stream=True,
        ) as stream:
            for _ in stream:
                pass

    _assert_streaming_timing_metrics(metric_reader)


@pytest.mark.asyncio()
async def test_async_messages_streaming_timing_metrics(
    metric_reader, async_anthropic_client, instrument_with_content, vcr
):
    with vcr.use_cassette("test_async_messages_create_streaming.yaml"):
        async with await async_anthropic_client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=MESSAGES,
            stream=True,
        ) as stream:
            async for _ in stream:
                pass

    _assert_streaming_timing_metrics(metric_reader)
