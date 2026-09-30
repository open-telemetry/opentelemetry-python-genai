# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CrewAI instrumentor lifecycle."""

import inspect
from unittest.mock import patch

import pytest

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


def _patched_methods():
    from crewai.agent.core import Agent
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    from crewai.experimental.agent_executor import AgentExecutor
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool

    return (
        (Agent, "execute_task"),
        (Agent, "aexecute_task"),
        (Agent, "kickoff"),
        (Agent, "kickoff_async"),
        (BaseTool, "run"),
        (CrewStructuredTool, "invoke"),
        (AgentExecutor, "_execute_single_native_tool_call"),
        (CrewAgentExecutor, "_execute_single_native_tool_call"),
    )


def test_instrumentation_dependencies() -> None:
    assert CrewAIInstrumentor().instrumentation_dependencies() == (
        "crewai >= 1.11.0, < 2",
    )


def test_instrument_uninstrument_cycle(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    instrumentor = CrewAIInstrumentor()

    originals = {
        (cls, method): inspect.getattr_static(cls, method)
        for cls, method in _patched_methods()
    }

    for _ in range(2):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        for cls, method in _patched_methods():
            assert (
                inspect.getattr_static(cls, method)
                is not originals[(cls, method)]
            )
        instrumentor.uninstrument()
        for cls, method in _patched_methods():
            assert (
                inspect.getattr_static(cls, method) is originals[(cls, method)]
            )


def test_instrument_with_global_providers() -> None:
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()


def test_instrumentation_leaves_crewai_telemetry_unchanged(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    from crewai.telemetry.telemetry import Telemetry

    original = inspect.getattr_static(Telemetry, "_safe_telemetry_operation")
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        assert (
            inspect.getattr_static(Telemetry, "_safe_telemetry_operation")
            is original
        )
    finally:
        instrumentor.uninstrument()


def test_partial_install_failure_restores_patched_method(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    from opentelemetry.instrumentation.genai import crewai as crewai_module

    originals = {
        (cls, method): inspect.getattr_static(cls, method)
        for cls, method in _patched_methods()
    }
    real_wrap = crewai_module.wrap_function_wrapper
    calls = 0

    def fail_on_second_patch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("patch failed")
        return real_wrap(*args, **kwargs)

    instrumentor = CrewAIInstrumentor()
    with (
        patch.object(
            crewai_module,
            "wrap_function_wrapper",
            side_effect=fail_on_second_patch,
        ),
        pytest.raises(RuntimeError, match="patch failed"),
    ):
        instrumentor._instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )

    for cls, method in _patched_methods():
        assert inspect.getattr_static(cls, method) is originals[(cls, method)]
