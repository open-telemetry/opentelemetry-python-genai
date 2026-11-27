# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from openai import APIConnectionError, NotFoundError, OpenAI

from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
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

from .test_utils import (
    COMPLETION_MODEL,
    COMPLETION_PROMPT,
    TEXT_COMPLETION_OPERATION_NAME,
    USER_ONLY_EXPECTED_INPUT_MESSAGES,
    assert_all_attributes,
    assert_inference_event_without_content,
    assert_messages_attribute,
    format_text_output_messages,
)


def _assert_system_fingerprint(span, system_fingerprint):
    if system_fingerprint:
        assert (
            span.attributes[
                OpenAIAttributes.OPENAI_RESPONSE_SYSTEM_FINGERPRINT
            ]
            == system_fingerprint
        )
    else:
        assert (
            OpenAIAttributes.OPENAI_RESPONSE_SYSTEM_FINGERPRINT
            not in span.attributes
        )


def _duration_point(metric_reader):
    metrics = metric_reader.get_metrics_data().resource_metrics
    assert len(metrics) == 1
    duration_metric = next(
        m
        for m in metrics[0].scope_metrics[0].metrics
        if m.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    )
    (point,) = duration_metric.data.data_points
    return point


def test_completion_with_content(
    span_exporter, metric_reader, openai_client, instrument_with_content, vcr
):
    with vcr.use_cassette("test_completion_with_content.yaml"):
        response = openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
        )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        COMPLETION_MODEL,
        True,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 16
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        response.choices[0].finish_reason,
    )
    _assert_system_fingerprint(span, response.system_fingerprint)
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        USER_ONLY_EXPECTED_INPUT_MESSAGES,
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_text_output_messages(response.choices[0].text),
    )

    point = _duration_point(metric_reader)
    assert (
        point.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == TEXT_COMPLETION_OPERATION_NAME
    )


def test_completion_no_content(
    span_exporter, log_exporter, openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_completion_with_content.yaml"):
        response = openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
        )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        COMPLETION_MODEL,
        True,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert_inference_event_without_content(log_exporter.get_finished_logs())


def test_completion_404(
    span_exporter, metric_reader, openai_client, instrument_no_content, vcr
):
    llm_model_value = "this-model-does-not-exist"

    with vcr.use_cassette("test_completion_404.yaml"):
        with pytest.raises(NotFoundError):
            openai_client.completions.create(
                model=llm_model_value,
                prompt=COMPLETION_PROMPT,
            )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        llm_model_value,
        True,
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert (
        span.attributes[ErrorAttributes.ERROR_TYPE] == "openai.NotFoundError"
    )

    point = _duration_point(metric_reader)
    assert point.sum > 0
    assert (
        point.attributes[ErrorAttributes.ERROR_TYPE] == "openai.NotFoundError"
    )


def test_completion_bad_endpoint(span_exporter, instrument_no_content, vcr):
    client = OpenAI(base_url="http://localhost:4242")

    with vcr.use_cassette("test_completion_bad_endpoint.yaml"):
        with pytest.raises(APIConnectionError):
            client.completions.create(
                model=COMPLETION_MODEL,
                prompt=COMPLETION_PROMPT,
                timeout=0.1,
            )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        COMPLETION_MODEL,
        True,
        server_address="localhost",
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert span.attributes[ServerAttributes.SERVER_PORT] == 4242
    assert (
        span.attributes[ErrorAttributes.ERROR_TYPE]
        == "openai.APIConnectionError"
    )


def test_completion_streaming(
    span_exporter, openai_client, instrument_with_content, vcr
):
    text = ""
    finish_reason = None
    last_chunk = None

    with vcr.use_cassette("test_completion_streaming.yaml"):
        response = openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            for choice in chunk.choices:
                text += choice.text
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            last_chunk = chunk

    assert last_chunk.usage is not None
    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        COMPLETION_MODEL,
        True,
        last_chunk.id,
        last_chunk.model,
        last_chunk.usage.prompt_tokens,
        last_chunk.usage.completion_tokens,
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        finish_reason,
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        USER_ONLY_EXPECTED_INPUT_MESSAGES,
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_text_output_messages(text),
    )


def test_completion_streaming_no_content(
    span_exporter, openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_completion_streaming.yaml"):
        response = openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        for _ in response:
            pass

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == TEXT_COMPLETION_OPERATION_NAME
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_completion_streaming_user_exception_propagates(
    span_exporter, openai_client, instrument_with_content, vcr
):
    with vcr.use_cassette("test_completion_streaming.yaml"):
        response = openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        with pytest.raises(RuntimeError, match="user failure"):
            with response:
                for _ in response:
                    raise RuntimeError("user failure")

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


def test_completion_streaming_not_complete(
    span_exporter, openai_client, instrument_with_content, vcr
):
    text = ""

    with vcr.use_cassette("test_completion_streaming.yaml"):
        response = openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            text += chunk.choices[0].text
            break
        response.close()

    (span,) = span_exporter.get_finished_spans()
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_text_output_messages(text),
    )
