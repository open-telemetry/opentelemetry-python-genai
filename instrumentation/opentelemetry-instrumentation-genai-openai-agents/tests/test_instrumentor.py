# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import agents.tracing

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.genai.openai_agents.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.openai_agents.processor import (
    GenAITracingProcessor,
)


def _registered_processors() -> tuple:
    provider = agents.tracing.get_trace_provider()
    multi = getattr(provider, "_multi_processor", None)
    return tuple(getattr(multi, "_processors", ()))


def _our_processors():
    return [
        p
        for p in _registered_processors()
        if isinstance(p, GenAITracingProcessor)
    ]


def test_instrumentation_dependencies_exposed() -> None:
    instrumentor = OpenAIAgentsInstrumentor()
    assert instrumentor.instrumentation_dependencies() == _instruments


def test_instrument_adds_processor_alongside_default() -> None:
    instrumentor = OpenAIAgentsInstrumentor()
    pre_count = len(_registered_processors())
    try:
        instrumentor.instrument()
        post = _registered_processors()
        # Default processor stays in place, ours is appended.
        assert len(post) == pre_count + 1
        assert len(_our_processors()) == 1
    finally:
        instrumentor.uninstrument()
    assert len(_our_processors()) == 0


def test_instrument_with_disable_openai_trace_export_replaces_processors() -> (
    None
):
    # Make sure the default processor is registered before we start,
    # so the "replace" behavior is observable.
    agents.tracing.set_trace_processors(
        [agents.tracing.processors.default_processor()]
    )
    instrumentor = OpenAIAgentsInstrumentor()
    try:
        instrumentor.instrument(disable_openai_trace_export=True)
        post = _registered_processors()
        assert len(post) == 1
        assert isinstance(post[0], GenAITracingProcessor)
    finally:
        instrumentor.uninstrument()


def test_uninstrument_restores_processors_in_replace_mode() -> None:
    baseline = [
        agents.tracing.processors.default_processor(),
        agents.tracing.processors.default_processor(),
    ]
    agents.tracing.set_trace_processors(baseline)

    instrumentor = OpenAIAgentsInstrumentor()
    try:
        instrumentor.instrument(disable_openai_trace_export=True)
        replaced = _registered_processors()
        assert len(replaced) == 1
        assert isinstance(replaced[0], GenAITracingProcessor)
    finally:
        instrumentor.uninstrument()

    restored = _registered_processors()
    assert restored == tuple(baseline)


def test_double_instrument_is_noop() -> None:
    instrumentor = OpenAIAgentsInstrumentor()
    try:
        instrumentor.instrument()
        first = _our_processors()
        instrumentor.instrument()
        second = _our_processors()
        assert len(first) == 1 and len(second) == 1
        assert first[0] is second[0]
    finally:
        instrumentor.uninstrument()


def test_double_uninstrument_is_noop() -> None:
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument()
    instrumentor.uninstrument()
    instrumentor.uninstrument()  # must not raise
