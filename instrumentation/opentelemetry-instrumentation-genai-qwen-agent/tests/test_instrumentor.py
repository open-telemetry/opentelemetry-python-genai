# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the QwenAgentInstrumentor lifecycle."""
# pylint: disable=redefined-outer-name

from unittest.mock import patch

from qwen_agent.agent import Agent
from qwen_agent.llm.schema import Message

from opentelemetry.instrumentation.genai.qwen_agent import (
    QwenAgentInstrumentor,
)


class _StubAgent(Agent):
    @classmethod
    def create(cls):
        obj = cls.__new__(cls)
        obj.name = "LifecycleBot"
        obj.description = ""
        obj.system_message = ""
        obj.llm = None
        obj.function_map = {}
        obj.extra_generate_cfg = {}
        return obj

    def _run(self, messages, **kwargs):
        raise NotImplementedError


def _run_agent_once():
    agent = _StubAgent.create()

    def fake_run(messages, **kwargs):
        yield [Message(role="assistant", content="ok")]

    with patch.object(_StubAgent, "_run", side_effect=fake_run):
        list(agent.run([Message(role="user", content="Hi")]))


def test_instrumentation_dependencies():
    instrumentor = QwenAgentInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()
    assert dependencies == ("qwen-agent >= 0.0.20",)


def test_uninstrument_stops_span_creation(
    span_exporter, tracer_provider, logger_provider, meter_provider
):
    instrumentor = QwenAgentInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        _run_agent_once()
        assert len(span_exporter.get_finished_spans()) == 1
    finally:
        instrumentor.uninstrument()

    span_exporter.clear()
    _run_agent_once()
    assert span_exporter.get_finished_spans() == ()
