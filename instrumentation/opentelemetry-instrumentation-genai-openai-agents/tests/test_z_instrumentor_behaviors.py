# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from agents.tracing import (
    get_trace_provider,
    set_trace_processors,
)

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.genai.openai_agents.package import (
    _instruments,
)
from opentelemetry.sdk.trace import TracerProvider


def test_double_instrument_is_noop():
    set_trace_processors([])
    provider = TracerProvider()
    instrumentor = OpenAIAgentsInstrumentor()

    instrumentor.instrument(tracer_provider=provider)
    trace_provider = get_trace_provider()
    assert len(trace_provider._multi_processor._processors) == 1

    instrumentor.instrument(tracer_provider=provider)
    assert len(trace_provider._multi_processor._processors) == 1

    instrumentor.uninstrument()
    instrumentor.uninstrument()
    set_trace_processors([])


def test_instrumentation_dependencies_exposed():
    instrumentor = OpenAIAgentsInstrumentor()
    assert instrumentor.instrumentation_dependencies() == _instruments


def test_default_agent_configuration():
    set_trace_processors([])
    provider = TracerProvider()
    instrumentor = OpenAIAgentsInstrumentor()

    try:
        instrumentor.instrument(tracer_provider=provider)
        processor = instrumentor._processor
        assert processor is not None
        assert getattr(processor, "_agent_name_default") == "OpenAI Agent"
        assert getattr(processor, "_agent_id_default") == "agent"
        assert (
            getattr(processor, "_agent_description_default")
            == "OpenAI Agents instrumentation"
        )
        assert processor.base_url == "https://api.openai.com"
        assert processor.server_address == "api.openai.com"
        assert processor.server_port == 443
    finally:
        instrumentor.uninstrument()
        set_trace_processors([])
