# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from unittest.mock import MagicMock, patch, sentinel

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util.genai.types import ContentCapturingMode


def test_completion_hook_forwarded_to_handler(
    tracer_provider, logger_provider, meter_provider
):
    """A hook passed to instrument() reaches the TelemetryHandler."""
    hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.langchain.TelemetryHandler"
        ) as get_handler,
        instrument(
            LangChainInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ),
    ):
        assert get_handler.call_args.kwargs["completion_hook"] is hook


def test_completion_hook_defaults_to_load_completion_hook(
    tracer_provider, logger_provider, meter_provider
):
    """Without an explicit hook, the one from load_completion_hook() is used."""
    with (
        patch(
            "opentelemetry.instrumentation.genai.langchain.TelemetryHandler"
        ) as get_handler,
        patch(
            "opentelemetry.instrumentation.genai.langchain.load_completion_hook",
            return_value=sentinel.default_hook,
        ) as load_hook,
        instrument(
            LangChainInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ),
    ):
        load_hook.assert_called_once()
        assert (
            get_handler.call_args.kwargs["completion_hook"]
            is sentinel.default_hook
        )


def test_content_capturing_mode_forwarded_to_handler(
    tracer_provider, logger_provider, meter_provider
):
    """An explicit content mode passed to instrument() reaches the handler."""
    instrumentor = LangChainInstrumentor()
    with patch(
        "opentelemetry.instrumentation.genai.langchain.TelemetryHandler"
    ) as get_handler:
        try:
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capturing_mode=ContentCapturingMode.SPAN_AND_EVENT,
            )
            assert (
                get_handler.call_args.kwargs["content_capturing_mode"]
                is ContentCapturingMode.SPAN_AND_EVENT
            )
        finally:
            instrumentor.uninstrument()
