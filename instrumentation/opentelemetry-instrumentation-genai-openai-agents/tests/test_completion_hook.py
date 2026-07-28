# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from unittest.mock import MagicMock, patch, sentinel

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument


def test_completion_hook_forwarded_to_handler(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """A hook passed to instrument() reaches get_telemetry_handler()."""
    hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.openai_agents.get_telemetry_handler"
        ) as get_handler,
        instrument(
            OpenAIAgentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ),
    ):
        assert get_handler.call_args.kwargs["completion_hook"] is hook


def test_completion_hook_defaults_to_load_completion_hook(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Without an explicit hook, the one from load_completion_hook() is used."""
    with (
        patch(
            "opentelemetry.instrumentation.genai.openai_agents.get_telemetry_handler"
        ) as get_handler,
        patch(
            "opentelemetry.instrumentation.genai.openai_agents.load_completion_hook",
            return_value=sentinel.default_hook,
        ) as load_hook,
        instrument(
            OpenAIAgentsInstrumentor(),
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
