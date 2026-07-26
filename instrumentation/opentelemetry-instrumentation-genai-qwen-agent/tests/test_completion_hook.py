# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from unittest.mock import MagicMock, patch, sentinel

from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import Message

from opentelemetry.instrumentation.genai.qwen_agent import (
    QwenAgentInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _StubChatModel(BaseChatModel):
    """A minimal BaseChatModel subclass for tests without network access."""

    def __init__(self, model="qwen-max", model_type="qwen_dashscope"):
        super().__init__({"model": model, "model_type": model_type})
        # Disable raw_api mode which requires stream-only and an API key.
        self.use_raw_api = False

    def _chat_no_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_with_functions(self, messages, functions, **kwargs):
        raise NotImplementedError


def test_completion_hook_forwarded_to_handler(
    tracer_provider, logger_provider, meter_provider
):
    """A hook passed to instrument() reaches the TelemetryHandler."""
    hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.qwen_agent.TelemetryHandler"
        ) as handler_cls,
        instrument(
            QwenAgentInstrumentor(),
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
            "opentelemetry.instrumentation.genai.qwen_agent.TelemetryHandler"
        ) as handler_cls,
        patch(
            "opentelemetry.instrumentation.genai.qwen_agent.load_completion_hook",
            return_value=sentinel.default_hook,
        ) as load_hook,
        instrument(
            QwenAgentInstrumentor(),
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
    tracer_provider, logger_provider, meter_provider
):
    """The hook's on_completion is called after a chat completion."""
    hook = MagicMock()
    fake_response = [Message(role="assistant", content="Hello!")]
    with (
        instrument(
            QwenAgentInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ),
        patch.object(
            _StubChatModel, "_chat_no_stream", return_value=fake_response
        ),
    ):
        _StubChatModel().chat(
            messages=[Message(role="user", content="Hello")],
            stream=False,
        )

    hook.on_completion.assert_called_once()
    assert hook.on_completion.call_args.kwargs["span"] is not None
