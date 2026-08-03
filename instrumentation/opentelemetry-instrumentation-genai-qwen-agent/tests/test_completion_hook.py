# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from unittest.mock import MagicMock, patch, sentinel

from qwen_agent.agent import Agent
from qwen_agent.llm.schema import Message

from opentelemetry.instrumentation.genai.qwen_agent import (
    QwenAgentInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _StubAgent(Agent):
    """A minimal Agent subclass that skips the heavy Agent.__init__."""

    @classmethod
    def create(cls):
        obj = cls.__new__(cls)
        obj.name = "HookBot"
        obj.description = ""
        obj.system_message = ""
        obj.llm = None
        obj.function_map = {}
        obj.extra_generate_cfg = {}
        return obj

    def _run(self, messages, **kwargs):
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
    """The hook's on_completion is called after an agent run completes."""
    hook = MagicMock()

    def fake_run(messages, **kwargs):
        yield [Message(role="assistant", content="Hello!")]

    with (
        instrument(
            QwenAgentInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ),
        patch.object(_StubAgent, "_run", side_effect=fake_run),
    ):
        list(_StubAgent.create().run([Message(role="user", content="Hello")]))

    hook.on_completion.assert_called_once()
    assert hook.on_completion.call_args.kwargs["span"] is not None
