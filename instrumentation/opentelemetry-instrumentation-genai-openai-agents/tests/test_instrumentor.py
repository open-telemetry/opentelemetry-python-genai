# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import agents
import agents.run
import agents.tracing
import pytest

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.genai.openai_agents.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.openai_agents.patch import (
    _current_conversation_id,
    _extract_conversation_id,
    _traced_agent_runner_run,
    _traced_agent_runner_run_streamed,
    _traced_agent_runner_run_sync,
    patch,
    unpatch,
)
from opentelemetry.instrumentation.genai.openai_agents.processor import (
    GenAITracingProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode


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


@pytest.mark.parametrize("disable_openai_trace_export", [False, True])
def test_uninstrument_after_reinstantiation(
    disable_openai_trace_export: bool,
) -> None:
    baseline = _registered_processors()
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument(
        disable_openai_trace_export=disable_openai_trace_export,
    )
    try:
        assert len(_our_processors()) == 1
    finally:
        OpenAIAgentsInstrumentor().uninstrument()
    assert len(_our_processors()) == 0
    assert _registered_processors() == baseline


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


def test_extract_conversation_id_from_kwargs() -> None:
    assert (
        _extract_conversation_id((), {"conversation_id": "conv-123"})
        == "conv-123"
    )


def test_extract_conversation_id_from_session() -> None:
    session = MagicMock(session_id="sess-456")
    assert _extract_conversation_id((), {"session": session}) == "sess-456"


def test_extract_conversation_id_prefers_conversation_id_over_session() -> (
    None
):
    session = MagicMock(session_id="sess-456")
    assert (
        _extract_conversation_id(
            (), {"conversation_id": "conv-123", "session": session}
        )
        == "conv-123"
    )


def test_extract_conversation_id_from_resumed_run_state() -> None:
    run_state = MagicMock(_conversation_id="state-789")
    # input as positional arg 1
    assert _extract_conversation_id(("agent", run_state), {}) == "state-789"
    # input as keyword arg
    assert _extract_conversation_id((), {"input": run_state}) == "state-789"


def test_extract_conversation_id_returns_none_when_empty() -> None:
    assert _extract_conversation_id((), {}) is None
    assert _extract_conversation_id((), {"conversation_id": None}) is None
    assert _extract_conversation_id((), {"session": None}) is None
    assert (
        _extract_conversation_id((), {"session": MagicMock(session_id=None)})
        is None
    )


def test_patch_and_unpatch_agent_runner() -> None:
    patch()
    try:
        assert hasattr(agents.run.AgentRunner.run, "__wrapped__")
        assert hasattr(agents.run.AgentRunner.run_sync, "__wrapped__")
        assert hasattr(agents.run.AgentRunner.run_streamed, "__wrapped__")
    finally:
        unpatch()
    assert not hasattr(agents.run.AgentRunner.run, "__wrapped__")
    assert not hasattr(agents.run.AgentRunner.run_sync, "__wrapped__")
    assert not hasattr(agents.run.AgentRunner.run_streamed, "__wrapped__")


def test_instrument_and_uninstrument_patches_agent_runner() -> None:
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument()
    try:
        assert hasattr(agents.run.AgentRunner.run, "__wrapped__")
        assert hasattr(agents.run.AgentRunner.run_sync, "__wrapped__")
        assert hasattr(agents.run.AgentRunner.run_streamed, "__wrapped__")
    finally:
        instrumentor.uninstrument()
    assert not hasattr(agents.run.AgentRunner.run, "__wrapped__")
    assert not hasattr(agents.run.AgentRunner.run_sync, "__wrapped__")
    assert not hasattr(agents.run.AgentRunner.run_streamed, "__wrapped__")


def test_agent_runner_run_sync_sets_and_resets_conversation_id() -> None:
    captured_conv_id = None

    def fake_run_sync(*args: Any, **kwargs: Any) -> str:
        nonlocal captured_conv_id
        captured_conv_id = _current_conversation_id.get()
        return "ok"

    assert _current_conversation_id.get() is None
    result = _traced_agent_runner_run_sync(
        fake_run_sync,
        None,
        (),
        {"conversation_id": "conv-sync-123"},
    )
    assert result == "ok"
    assert captured_conv_id == "conv-sync-123"
    assert _current_conversation_id.get() is None


def test_agent_runner_run_async_sets_and_resets_session_id() -> None:
    captured_conv_id = None

    async def fake_run(*args: Any, **kwargs: Any) -> str:
        nonlocal captured_conv_id
        captured_conv_id = _current_conversation_id.get()
        return "async_ok"

    assert _current_conversation_id.get() is None
    session = MagicMock(session_id="session-async-456")
    coro = _traced_agent_runner_run(
        fake_run,
        None,
        (),
        {"session": session},
    )
    result = None
    try:
        coro.send(None)
    except StopIteration as exc:
        result = exc.value

    assert result == "async_ok"
    assert captured_conv_id == "session-async-456"
    assert _current_conversation_id.get() is None


def test_agent_runner_run_streamed_sets_and_resets_conversation_id() -> None:
    captured_conv_id = None

    def fake_run_streamed(*args: Any, **kwargs: Any) -> str:
        nonlocal captured_conv_id
        captured_conv_id = _current_conversation_id.get()
        return "streamed_ok"

    assert _current_conversation_id.get() is None
    result = _traced_agent_runner_run_streamed(
        fake_run_streamed,
        None,
        (),
        {"conversation_id": "conv-stream-789"},
    )
    assert result == "streamed_ok"
    assert captured_conv_id == "conv-stream-789"
    assert _current_conversation_id.get() is None


def test_agent_runner_run_sync_resets_conversation_id_on_exception() -> None:
    def fake_run_sync_fail(*args: Any, **kwargs: Any) -> str:
        assert _current_conversation_id.get() == "conv-err"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _traced_agent_runner_run_sync(
            fake_run_sync_fail,
            None,
            (),
            {"conversation_id": "conv-err"},
        )
    assert _current_conversation_id.get() is None


@agents.function_tool
def _failing_tool() -> str:
    raise ValueError("database connection failed")


@agents.function_tool
def _get_weather(city: str) -> str:
    """Return a canned forecast for the requested city."""

    return f"The forecast for {city} is sunny."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_runner_records_tool_call_arguments(
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        agent = agents.Agent(
            name="test_agent",
            model="gpt-4o-mini",
            tools=[_get_weather],
        )
        await agents.Runner.run(agent, "what is the weather in Barcelona?")

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert "execute_tool _get_weather" in spans
    tool_span = spans["execute_tool _get_weather"]
    assert tool_span.attributes is not None
    assert (
        tool_span.attributes["gen_ai.tool.call.arguments"]
        == '{"city":"Barcelona"}'
    )
    assert (
        tool_span.attributes["gen_ai.tool.call.result"]
        == "The forecast for Barcelona is sunny."
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_runner_failing_tool_records_span_error(
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        agent = agents.Agent(
            name="test_agent",
            model="gpt-4o-mini",
            tools=[_failing_tool],
        )
        result = await agents.Runner.run(agent, "run tool")

    tool_outputs = [
        item.output
        for item in result.new_items
        if isinstance(item, agents.ToolCallOutputItem)
    ]
    assert len(tool_outputs) == 1

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    assert "execute_tool _failing_tool" in spans
    tool_span = spans["execute_tool _failing_tool"]
    assert tool_span.status.status_code is StatusCode.ERROR
    assert tool_span.status.description
    assert tool_span.attributes is not None
    assert tool_span.attributes["error.type"] == "_OTHER"
    assert tool_span.attributes["gen_ai.tool.call.result"] == tool_outputs[0]
