# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from unittest.mock import MagicMock, patch, sentinel

from anthropic import Anthropic
from anthropic.resources.messages import Messages
from anthropic.types import Message, TextBlock, Usage

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument


def _fake_message(model: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(text="hello", type="text")],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=2),
    )


def test_completion_hook_forwarded_to_handler(
    tracer_provider, logger_provider, meter_provider
):
    """A hook passed to instrument() reaches the TelemetryHandler."""
    hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.anthropic.TelemetryHandler"
        ) as handler_cls,
        instrument(
            AnthropicInstrumentor(),
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
    with (
        patch(
            "opentelemetry.instrumentation.genai.anthropic.TelemetryHandler"
        ) as handler_cls,
        patch(
            "opentelemetry.instrumentation.genai.anthropic.load_completion_hook",
            return_value=sentinel.default_hook,
        ) as load_hook,
        instrument(
            AnthropicInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ),
    ):
        load_hook.assert_called_once()
        assert (
            handler_cls.call_args.kwargs["completion_hook"]
            is sentinel.default_hook
        )


def test_completion_hook_invoked(
    monkeypatch, tracer_provider, logger_provider, meter_provider
):
    """The hook's on_completion is called after a chat completion."""
    model = "claude-test"
    monkeypatch.setattr(
        Messages,
        "create",
        lambda self, **kwargs: _fake_message(kwargs["model"]),
        raising=False,
    )
    hook = MagicMock()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        completion_hook=hook,
    ):
        Anthropic().messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        )

    hook.on_completion.assert_called_once()
    assert hook.on_completion.call_args.kwargs["span"] is not None
