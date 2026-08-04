# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the completion hook is wired into the telemetry handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, sentinel

import pytest

from opentelemetry.test_util_genai.instrumentor import instrument

from .harness import (
    fake_run_command_path,
    make_request,
    patched_command_path,
    user_command_msgs,
)

pytest.importorskip("agentscope.message")


def test_completion_hook_forwarded_to_handler(
    instrumentor_cls, tracer_provider, logger_provider, meter_provider
):
    """A hook passed to instrument() reaches the TelemetryHandler."""
    hook = MagicMock()
    with (
        patch(
            "opentelemetry.instrumentation.genai.qwenpaw.TelemetryHandler"
        ) as handler_cls,
        instrument(
            instrumentor_cls(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ),
    ):
        assert handler_cls.call_args.kwargs["completion_hook"] is hook


def test_completion_hook_defaults_to_load_completion_hook(
    instrumentor_cls, tracer_provider, logger_provider, meter_provider
):
    """Without an explicit hook, the one from load_completion_hook() is used."""
    with (
        patch(
            "opentelemetry.instrumentation.genai.qwenpaw.TelemetryHandler"
        ) as handler_cls,
        patch(
            "opentelemetry.instrumentation.genai.qwenpaw.load_completion_hook",
            return_value=sentinel.default_hook,
        ) as load_hook,
        instrument(
            instrumentor_cls(),
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


@pytest.mark.asyncio
async def test_completion_hook_invoked(
    instrumentor_cls,
    runner_module,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    """The hook's on_completion is called once the turn's stream is drained."""
    hook = MagicMock()
    with instrument(
        instrumentor_cls(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
        completion_hook=hook,
    ):
        runner = runner_module.AgentRunner(agent_id="entry-agent")
        with patched_command_path(
            runner_module, fake_run_command_path("hooked")
        ):
            async for _ in runner.query_handler(
                user_command_msgs(), make_request()
            ):
                pass

    hook.on_completion.assert_called_once()
    assert hook.on_completion.call_args.kwargs["span"] is not None
