# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation.genai.portkey import (
    PortkeyInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument


def test_completion_hook_forwarded_to_handler(
    tracer_provider, logger_provider, meter_provider
):
    """A hook passed to instrument() reaches the TelemetryHandler."""
    hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.portkey.TelemetryHandler"
        ) as handler_cls,
        instrument(
            PortkeyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ),
    ):
        assert handler_cls.call_args.kwargs["completion_hook"] is hook


def test_completion_hook_defaults_to_load_completion_hook(
    tracer_provider, logger_provider, meter_provider
):
    """Without an explicit hook, the one from load_completion_hook() is used."""
    sentinel_hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.portkey.load_completion_hook",
            return_value=sentinel_hook,
        ),
        patch(
            "opentelemetry.instrumentation.genai.portkey.TelemetryHandler"
        ) as handler_cls,
        instrument(
            PortkeyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ),
    ):
        assert handler_cls.call_args.kwargs["completion_hook"] is sentinel_hook
