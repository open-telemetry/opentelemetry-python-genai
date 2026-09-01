# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
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

    # Content goes to the hook only — not to span attributes or log records
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span_attrs = spans[0].attributes or {}
    assert "gen_ai.input.messages" not in span_attrs
    assert "gen_ai.output.messages" not in span_attrs
    assert log_exporter.get_finished_logs() == ()


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
