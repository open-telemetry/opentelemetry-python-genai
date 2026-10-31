# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.stub import Stubber

from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)


def test_custom_hook_is_called(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    bedrock_client,
    vcr,
):
    """A hook passed to instrument() is called after converse."""
    hook = MagicMock()
    instrumentor = BedrockInstrumentor()
    _OpenTelemetrySemanticConventionStability._initialized = False
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        completion_hook=hook,
    )

    try:
        with vcr.use_cassette("test_converse_with_content.yaml"):
            bedrock_client.converse(
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "Say this is a test"}],
                    }
                ],
                modelId="amazon.nova-micro-v1:0",
            )
    finally:
        instrumentor.uninstrument()

    hook.on_completion.assert_called_once()
    kwargs = hook.on_completion.call_args.kwargs
    assert kwargs["inputs"]
    assert kwargs["outputs"]
    assert kwargs["span"] is not None

    # Content goes to the hook only - not to span or log record attributes
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span_attrs = spans[0].attributes or {}
    assert "gen_ai.input.messages" not in span_attrs
    assert "gen_ai.output.messages" not in span_attrs
    (log,) = log_exporter.get_finished_logs()
    log_attrs = log.log_record.attributes or {}
    assert "gen_ai.input.messages" not in log_attrs
    assert "gen_ai.output.messages" not in log_attrs


@pytest.mark.parametrize(
    ("content_capture", "capture_on_span", "capture_on_event"),
    [
        ("NO_CONTENT", False, False),
        ("SPAN_ONLY", True, False),
        ("EVENT_ONLY", False, True),
        ("SPAN_AND_EVENT", True, True),
    ],
)
def test_prompt_variables_follow_content_capture_with_hook(
    content_capture,
    capture_on_span,
    capture_on_event,
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    bedrock_client,
) -> None:
    hook = MagicMock()
    params = {
        "modelId": "amazon.nova-micro-v1:0",
        "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        "promptVariables": {"user_name": {"text": "Alice"}},
    }
    response = {
        "output": {
            "message": {"role": "assistant", "content": [{"text": "Hi"}]}
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }
    with Stubber(bedrock_client) as stubber:
        stubber.add_response("converse", response, params)
        with instrument(
            BedrockInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture=content_capture,
            emit_event=True,
            completion_hook=hook,
        ):
            bedrock_client.converse(**params)

    hook.on_completion.assert_called_once()
    assert hook.on_completion.call_args.kwargs["inputs"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    if capture_on_span:
        assert attrs["gen_ai.prompt.variable.user_name"] == "Alice"
        assert "gen_ai.input.messages" in attrs
    else:
        assert "gen_ai.prompt.variable.user_name" not in attrs
        assert "gen_ai.input.messages" not in attrs

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    event_attrs = logs[0].log_record.attributes or {}
    if capture_on_event:
        assert event_attrs["gen_ai.prompt.variable.user_name"] == "Alice"
        assert "gen_ai.input.messages" in event_attrs
    else:
        assert "gen_ai.prompt.variable.user_name" not in event_attrs
        assert "gen_ai.input.messages" not in event_attrs


def test_stream_prompt_variables_not_recorded_with_hook_and_no_content(
    span_exporter,
    log_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    bedrock_client,
) -> None:
    hook = MagicMock()
    params = {
        "modelId": "amazon.nova-micro-v1:0",
        "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        "promptVariables": {"user_name": {"text": "Alice"}},
    }
    events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Hi"},
            }
        },
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    with Stubber(bedrock_client) as stubber:
        # botocore cannot validate an EventStream iterator against the output shape.
        stubber._validate_response = lambda *args, **kwargs: None
        stubber.add_response(
            "converse_stream", {"stream": iter(events)}, params
        )
        with instrument(
            BedrockInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="NO_CONTENT",
            emit_event=True,
            completion_hook=hook,
        ):
            response = bedrock_client.converse_stream(**params)
            assert list(response["stream"]) == events

    hook.on_completion.assert_called_once()
    assert hook.on_completion.call_args.kwargs["inputs"]
    assert hook.on_completion.call_args.kwargs["outputs"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span_attrs = spans[0].attributes or {}
    assert "gen_ai.prompt.variable.user_name" not in span_attrs
    assert "gen_ai.input.messages" not in span_attrs
    assert "gen_ai.output.messages" not in span_attrs

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    event_attrs = logs[0].log_record.attributes or {}
    assert "gen_ai.prompt.variable.user_name" not in event_attrs
    assert "gen_ai.input.messages" not in event_attrs
    assert "gen_ai.output.messages" not in event_attrs


@patch.dict(
    os.environ,
    {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "span_only"},
)
def test_default_hook_loaded_from_env(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    bedrock_client,
    vcr,
):
    """When no hook kwarg is given, load_completion_hook() provides the default."""
    default_hook = MagicMock()
    instrumentor = BedrockInstrumentor()
    with patch(
        "opentelemetry.instrumentation.genai.bedrock.load_completion_hook",
        return_value=default_hook,
    ):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )

    try:
        with vcr.use_cassette("test_converse_with_content.yaml"):
            bedrock_client.converse(
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "Say this is a test"}],
                    }
                ],
                modelId="amazon.nova-micro-v1:0",
            )
    finally:
        instrumentor.uninstrument()

    default_hook.on_completion.assert_called_once()
    kwargs = default_hook.on_completion.call_args.kwargs
    assert kwargs["inputs"]
    assert kwargs["outputs"]
    assert kwargs["span"] is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span_attrs = spans[0].attributes or {}
    assert "gen_ai.input.messages" in span_attrs
    assert "gen_ai.output.messages" in span_attrs
