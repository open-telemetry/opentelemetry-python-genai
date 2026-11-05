# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import ollama

from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

from .conftest import CHAT_MODEL, HOST

MESSAGES = [{"role": "user", "content": "Say hello in three words."}]


def test_completion_hook_passed_to_instrument_is_called(
    tracer_provider, logger_provider, meter_provider, span_exporter, vcr
):
    hook = MagicMock()
    with instrument(
        OllamaInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
        completion_hook=hook,
    ):
        with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
            response = ollama.Client(host=HOST).chat(
                model=CHAT_MODEL, messages=MESSAGES
            )

    hook.on_completion.assert_called_once()
    kwargs = hook.on_completion.call_args.kwargs
    assert kwargs["span"] is not None
    # The hook gets the content even when it is kept off the span.
    assert kwargs["inputs"][0].parts[0].content == MESSAGES[0]["content"]
    assert kwargs["outputs"][0].parts[0].content == response.message.content
    (span,) = span_exporter.get_finished_spans()
    assert "gen_ai.input.messages" not in span.attributes


def test_completion_hook_defaults_to_load_completion_hook(
    tracer_provider, logger_provider, meter_provider, vcr
):
    # load_completion_hook resolves OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK.
    default_hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.ollama.load_completion_hook",
            return_value=default_hook,
        ) as load_hook,
        instrument(
            OllamaInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ),
    ):
        with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
            ollama.Client(host=HOST).chat(model=CHAT_MODEL, messages=MESSAGES)

    load_hook.assert_called_once()
    default_hook.on_completion.assert_called_once()
    assert default_hook.on_completion.call_args.kwargs["span"] is not None
