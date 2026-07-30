# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.test_utils import DEFAULT_MODEL, USER_ONLY_PROMPT

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.util.genai.utils import is_experimental_mode

_DURATION_BUCKETS = (
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
)
_TOKEN_USAGE_BUCKETS = (
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
)


def assert_all_metric_attributes(data_point, latest_experimental_enabled):
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in data_point.attributes
    assert (
        data_point.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )

    provider_name_attr_name = (
        "gen_ai.provider.name"
        if latest_experimental_enabled
        else GenAIAttributes.GEN_AI_SYSTEM
    )
    assert provider_name_attr_name in data_point.attributes
    assert (
        data_point.attributes[provider_name_attr_name]
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )
    assert GenAIAttributes.GEN_AI_REQUEST_MODEL in data_point.attributes
    assert (
        data_point.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4o-mini"
    )
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in data_point.attributes
    assert (
        data_point.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4o-mini-2024-07-18"
    )

    system_fingerprint_attr_key = (
        OpenAIAttributes.OPENAI_RESPONSE_SYSTEM_FINGERPRINT
        if latest_experimental_enabled
        else GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SYSTEM_FINGERPRINT
    )
    response_service_tier_attr_key = (
        OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER
        if latest_experimental_enabled
        else GenAIAttributes.GEN_AI_OPENAI_RESPONSE_SERVICE_TIER
    )
    request_service_tier_attr_key = (
        OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER
        if latest_experimental_enabled
        else GenAIAttributes.GEN_AI_OPENAI_REQUEST_SERVICE_TIER
    )

    assert system_fingerprint_attr_key in data_point.attributes
    assert (
        data_point.attributes[system_fingerprint_attr_key] == "fp_0ba0d124f1"
    )
    assert request_service_tier_attr_key not in data_point.attributes
    assert response_service_tier_attr_key in data_point.attributes
    assert data_point.attributes[response_service_tier_attr_key] == "default"
    assert (
        data_point.attributes[ServerAttributes.SERVER_ADDRESS]
        == "api.openai.com"
    )


def test_chat_completion_metrics(
    metric_reader, openai_client, instrument_with_content, vcr
):
    latest_experimental_enabled = is_experimental_mode()
    with vcr.use_cassette("test_chat_completion_metrics.yaml"):
        openai_client.chat.completions.create(
            messages=USER_ONLY_PROMPT, model=DEFAULT_MODEL, stream=False
        )

    metrics = metric_reader.get_metrics_data().resource_metrics
    assert len(metrics) == 1

    metric_data = metrics[0].scope_metrics[0].metrics
    assert len(metric_data) == 2

    duration_metric = next(
        (
            m
            for m in metric_data
            if m.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
        ),
        None,
    )
    assert duration_metric is not None

    duration_point = duration_metric.data.data_points[0]
    assert duration_point.sum > 0
    assert_all_metric_attributes(duration_point, latest_experimental_enabled)
    assert duration_point.explicit_bounds == _DURATION_BUCKETS

    token_usage_metric = next(
        (
            m
            for m in metric_data
            if m.name == gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE
        ),
        None,
    )
    assert token_usage_metric is not None

    input_token_usage = next(
        (
            d
            for d in token_usage_metric.data.data_points
            if d.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]
            == GenAIAttributes.GenAiTokenTypeValues.INPUT.value
        ),
        None,
    )
    assert input_token_usage is not None
    assert input_token_usage.sum == 12

    assert input_token_usage.explicit_bounds == _TOKEN_USAGE_BUCKETS
    assert input_token_usage.bucket_counts[2] == 1
    assert_all_metric_attributes(
        input_token_usage, latest_experimental_enabled
    )

    output_token_usage = next(
        (
            d
            for d in token_usage_metric.data.data_points
            if d.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]
            == GenAIAttributes.GenAiTokenTypeValues.COMPLETION.value
        ),
        None,
    )
    assert output_token_usage is not None
    assert output_token_usage.sum == 5
    # assert against buckets [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]
    assert output_token_usage.bucket_counts[2] == 1
    assert_all_metric_attributes(
        output_token_usage, latest_experimental_enabled
    )


@pytest.mark.asyncio()
async def test_async_chat_completion_metrics(
    metric_reader, async_openai_client, instrument_with_content, vcr
):
    latest_experimental_enabled = is_experimental_mode()
    with vcr.use_cassette("test_async_chat_completion_metrics.yaml"):
        await async_openai_client.chat.completions.create(
            messages=USER_ONLY_PROMPT, model=DEFAULT_MODEL, stream=False
        )

    metrics = metric_reader.get_metrics_data().resource_metrics
    assert len(metrics) == 1

    metric_data = metrics[0].scope_metrics[0].metrics
    assert len(metric_data) == 2

    duration_metric = next(
        (
            m
            for m in metric_data
            if m.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
        ),
        None,
    )
    assert duration_metric is not None
    assert duration_metric.data.data_points[0].sum > 0
    assert_all_metric_attributes(
        duration_metric.data.data_points[0], latest_experimental_enabled
    )

    token_usage_metric = next(
        (
            m
            for m in metric_data
            if m.name == gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE
        ),
        None,
    )
    assert token_usage_metric is not None

    input_token_usage = next(
        (
            d
            for d in token_usage_metric.data.data_points
            if d.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]
            == GenAIAttributes.GenAiTokenTypeValues.INPUT.value
        ),
        None,
    )

    assert input_token_usage is not None
    assert input_token_usage.sum == 12
    assert_all_metric_attributes(
        input_token_usage, latest_experimental_enabled
    )

    output_token_usage = next(
        (
            d
            for d in token_usage_metric.data.data_points
            if d.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]
            == GenAIAttributes.GenAiTokenTypeValues.COMPLETION.value
        ),
        None,
    )

    assert output_token_usage is not None
    assert output_token_usage.sum == 12
    assert_all_metric_attributes(
        output_token_usage, latest_experimental_enabled
    )


def _metrics_by_name(metric_reader):
    resource_metrics = metric_reader.get_metrics_data().resource_metrics
    by_name = {}
    for rm in resource_metrics:
        for scope in rm.scope_metrics:
            for metric in scope.metrics:
                by_name[metric.name] = metric
    return by_name


def _assert_streaming_timing_metrics(metric_reader):
    """Assert the streaming timing metrics are emitted through the real
    ChatStreamWrapper path.

    Regression coverage for the ``invocation=invocation`` wiring in
    chat_wrappers.py: dropping it would still pass every util-layer test but
    would silently stop emitting TTFC and per-output-chunk metrics for OpenAI
    streaming.
    """
    metrics = _metrics_by_name(metric_reader)

    ttfc = metrics.get(
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK
    )
    assert ttfc is not None
    ttfc_point = ttfc.data.data_points[0]
    # Exactly one time-to-first-chunk record per stream.
    assert ttfc_point.count == 1
    assert ttfc_point.sum >= 0
    assert (
        ttfc_point.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    # response.model is set eagerly during streaming, so it lands on the
    # first-chunk metric too.
    assert (
        ttfc_point.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-0613"
    )

    per_chunk = metrics.get(
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK
    )
    assert per_chunk is not None
    per_chunk_point = per_chunk.data.data_points[0]
    # One record per inter-chunk gap; the streaming cassette has several
    # chunks, so at least one gap is recorded.
    assert per_chunk_point.count >= 1
    assert per_chunk_point.sum >= 0
    assert (
        per_chunk_point.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-0613"
    )


def test_chat_completion_streaming_timing_metrics(
    metric_reader, openai_client, instrument_with_content, vcr
):
    if not is_experimental_mode():
        pytest.skip("streaming timing metrics require experimental semconv")
    kwargs = {
        "model": "gpt-4",
        "messages": USER_ONLY_PROMPT,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with vcr.use_cassette("test_chat_completion_streaming.yaml"):
        response = openai_client.chat.completions.create(**kwargs)
        for _ in response:
            pass

    _assert_streaming_timing_metrics(metric_reader)


@pytest.mark.asyncio()
async def test_async_chat_completion_streaming_timing_metrics(
    metric_reader, async_openai_client, instrument_with_content, vcr
):
    if not is_experimental_mode():
        pytest.skip("streaming timing metrics require experimental semconv")
    kwargs = {
        "model": "gpt-4",
        "messages": USER_ONLY_PROMPT,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with vcr.use_cassette("test_async_chat_completion_streaming.yaml"):
        response = await async_openai_client.chat.completions.create(**kwargs)
        async for _ in response:
            pass

    _assert_streaming_timing_metrics(metric_reader)
