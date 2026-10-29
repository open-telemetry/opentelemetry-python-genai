# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from importlib import import_module
from typing import Any
from unittest.mock import AsyncMock, MagicMock, NonCallableMock

import agents
import agents.tracing
import pytest
from openai.types.responses import Response, ResponseCompletedEvent

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.instrumentation.genai.openai_agents.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.openai_agents.processor import (
    GenAITracingProcessor,
)
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
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


def _assert_agent_call_metrics(
    metric_reader: InMemoryMetricReader,
    span_exporter: InMemorySpanExporter,
    *,
    inference_calls: int | None = 2,
    tool_calls: int | None = 1,
    invocations: int = 1,
) -> None:
    data = metric_reader.get_metrics_data()
    assert data is not None
    metrics = {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    for name, calls in (
        ("gen_ai.invoke_agent.inference_calls", inference_calls),
        ("gen_ai.invoke_agent.tool_calls", tool_calls),
    ):
        if calls is None:
            assert name not in metrics
        else:
            (point,) = metrics[name].data.data_points
            assert point.attributes == {"gen_ai.agent.name": "test_agent"}
            assert point.count == invocations
            assert point.sum == calls
            assert isinstance(point.sum, int)
        assert all(
            name not in (span.attributes or {})
            for span in span_exporter.get_finished_spans()
        )


def _mock_model(*outputs: list[dict[str, Any]]) -> MagicMock:
    responses = [
        Response.model_validate(
            {
                "id": f"resp_{index}",
                "object": "response",
                "created_at": 0,
                "model": "test-model",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        for index, output in enumerate(outputs)
    ]
    model = MagicMock(spec=agents.Model)
    model.get_response.side_effect = [
        agents.ModelResponse(
            output=response.output,
            usage=agents.Usage(),
            response_id=response.id,
        )
        for response in responses
    ]

    async def stream_response(
        *args: Any, **kwargs: Any
    ) -> AsyncIterator[ResponseCompletedEvent]:
        yield ResponseCompletedEvent(
            type="response.completed",
            response=responses.pop(0),
            sequence_number=0,
        )

    model.stream_response.side_effect = stream_response
    return model


_FINAL_MESSAGE = {
    "type": "message",
    "id": "msg_final",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Done", "annotations": []}],
}


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
    runner_methods = {
        name: agents.Runner.__dict__[name]
        for name in ("run", "run_sync", "run_streamed")
    }
    retry_module = (
        import_module("agents.run_internal.run_loop")
        if hasattr(agents.ModelSettings, "retry")
        else None
    )
    retry_functions = (
        {
            name: getattr(retry_module, name)
            for name in (
                "get_response_with_retry",
                "stream_response_with_retry",
            )
        }
        if retry_module is not None
        else {}
    )
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument(
        disable_openai_trace_export=disable_openai_trace_export,
    )
    try:
        assert len(_our_processors()) == 1
        assert all(
            getattr(retry_module, name) is not function
            for name, function in retry_functions.items()
        )
    finally:
        OpenAIAgentsInstrumentor().uninstrument()
    assert len(_our_processors()) == 0
    assert _registered_processors() == baseline
    assert all(
        agents.Runner.__dict__[name] is method
        for name, method in runner_methods.items()
    )
    assert all(
        getattr(retry_module, name) is function
        for name, function in retry_functions.items()
    )


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
    metric_reader: InMemoryMetricReader,
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
    _assert_agent_call_metrics(metric_reader, span_exporter)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_runner_failing_tool_records_span_error(
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
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
    _assert_agent_call_metrics(metric_reader, span_exporter)


@pytest.mark.parametrize(
    "runner", ["run", "run_sync", "run_streamed", "run_streamed_positional"]
)
def test_runner_entry_points_preserve_hooks(
    runner: str,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    model = _mock_model([_FINAL_MESSAGE])
    application_hooks = MagicMock(spec=type(agents.RunHooks()))
    agent = agents.Agent(name="test_agent", model=model)
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        if runner == "run_sync":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                agents.Runner.run_sync(agent, "hello", hooks=application_hooks)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        else:

            async def run() -> None:
                if runner == "run":
                    await agents.Runner.run(
                        agent, "hello", hooks=application_hooks
                    )
                else:
                    if runner == "run_streamed_positional":
                        result = agents.Runner.run_streamed(
                            agent, "hello", None, 10, application_hooks
                        )
                    else:
                        result = agents.Runner.run_streamed(
                            agent, "hello", hooks=application_hooks
                        )
                    async for _ in result.stream_events():
                        pass

            asyncio.run(run())
    application_hooks.on_agent_start.assert_awaited_once()
    application_hooks.on_agent_end.assert_awaited_once()
    application_hooks.on_llm_start.assert_awaited_once()
    application_hooks.on_llm_end.assert_awaited_once()
    _assert_agent_call_metrics(
        metric_reader, span_exporter, inference_calls=1, tool_calls=0
    )


@pytest.mark.asyncio
async def test_manual_agent_spans_do_not_record_call_counts(
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    runner_finished = asyncio.Event()
    background_tasks: list[asyncio.Task[None]] = []

    async def emit_after_run() -> None:
        await runner_finished.wait()
        with agents.agent_span(name="test_agent"):
            pass

    async def on_agent_start(*args: Any, **kwargs: Any) -> None:
        with agents.agent_span(name="test_agent"):
            pass
        background_tasks.append(asyncio.create_task(emit_after_run()))

    application_hooks = MagicMock(spec=type(agents.RunHooks()))
    application_hooks.on_agent_start.side_effect = on_agent_start
    agent = agents.Agent(
        name="test_agent", model=_mock_model([_FINAL_MESSAGE])
    )
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        await agents.Runner.run(agent, "hello", hooks=application_hooks)
        runner_finished.set()
        await asyncio.gather(*background_tasks)
    _assert_agent_call_metrics(
        metric_reader, span_exporter, inference_calls=1, tool_calls=0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("runner", ["run", "run_streamed"])
async def test_pre_start_abort_omits_call_counts(
    runner: str,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    agent = agents.Agent(
        name="test_agent", model=_mock_model([_FINAL_MESSAGE])
    )
    application_hooks = MagicMock(spec=type(agents.RunHooks()))
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        with pytest.raises(agents.MaxTurnsExceeded):
            if runner == "run":
                await agents.Runner.run(
                    agent, "hello", max_turns=0, hooks=application_hooks
                )
            else:
                result = agents.Runner.run_streamed(
                    agent, "hello", max_turns=0, hooks=application_hooks
                )
                async for _ in result.stream_events():
                    pass
        with agents.trace("manual"), agents.agent_span(name="test_agent"):
            pass
    application_hooks.on_agent_start.assert_not_awaited()
    _assert_agent_call_metrics(
        metric_reader, span_exporter, inference_calls=None, tool_calls=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("used_tools", [0, 1])
async def test_unused_function_tools_preserve_call_counts(
    used_tools: int,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    tool_call = {
        "type": "function_call",
        "id": "fc_weather",
        "call_id": "call_weather",
        "name": _get_weather.name,
        "arguments": '{"city":"Barcelona"}',
    }
    outputs = (
        [[tool_call], [_FINAL_MESSAGE]] if used_tools else [[_FINAL_MESSAGE]]
    )
    agent = agents.Agent(
        name="test_agent",
        model=_mock_model(*outputs),
        tools=[_get_weather, _failing_tool],
    )
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        await agents.Runner.run(agent, "hello")
        agent.model = _mock_model([_FINAL_MESSAGE])
        await agents.Runner.run(agent, "hello again")
    _assert_agent_call_metrics(
        metric_reader,
        span_exporter,
        inference_calls=2 + used_tools,
        tool_calls=used_tools,
        invocations=2,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(agents.ModelSettings, "retry"),
    reason="This SDK does not support runner-managed model retries",
)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
async def test_model_retry_attempts_record_inference_calls(
    streaming: bool,
    exhausted: bool,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    model = _mock_model([_FINAL_MESSAGE])
    failures = 3 if exhausted else 2
    model.get_retry_advice = MagicMock(return_value=None)
    if streaming:
        original_stream = model.stream_response.side_effect

        async def stream_response(
            *args: Any, **kwargs: Any
        ) -> AsyncIterator[ResponseCompletedEvent]:
            if model.stream_response.call_count <= failures:
                raise ValueError("model failed")
            async for event in original_stream(*args, **kwargs):
                yield event

        model.stream_response.side_effect = stream_response
    else:
        model.get_response.side_effect = [
            ValueError("model failed") for _ in range(failures)
        ] + list(model.get_response.side_effect)
    retry_policy = MagicMock(return_value=True)
    agent = agents.Agent(
        name="test_agent",
        model=model,
        model_settings=agents.ModelSettings(
            retry=agents.ModelRetrySettings(
                max_retries=2,
                policy=retry_policy,
                backoff={"initial_delay": 0, "max_delay": 0, "jitter": False},
            )
        ),
    )
    application_hooks = MagicMock(spec=type(agents.RunHooks()))

    async def run() -> None:
        if streaming:
            result = agents.Runner.run_streamed(
                agent, "hello", hooks=application_hooks
            )
            async for _ in result.stream_events():
                pass
        else:
            await agents.Runner.run(agent, "hello", hooks=application_hooks)

    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        if exhausted:
            with pytest.raises(ValueError, match="model failed"):
                await run()
        else:
            await run()
    model_method = model.stream_response if streaming else model.get_response
    assert model_method.call_count == 3
    assert retry_policy.call_count == 2
    application_hooks.on_llm_start.assert_awaited_once()
    assert application_hooks.on_llm_end.await_count == (0 if exhausted else 1)
    _assert_agent_call_metrics(
        metric_reader, span_exporter, inference_calls=3, tool_calls=0
    )


@pytest.mark.asyncio
async def test_failed_inference_records_call_count(
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    model = _mock_model()
    model.get_response.side_effect = ValueError("model failed")
    model.get_retry_advice = MagicMock(return_value=None)
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        agent = agents.Agent(name="test_agent", model=model)
        with pytest.raises(ValueError, match="model failed"):
            await agents.Runner.run(agent, "hello")
    _assert_agent_call_metrics(
        metric_reader, span_exporter, inference_calls=1, tool_calls=0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_class", "tool_name"),
    [
        ("ComputerTool", "computer"),
        ("LocalShellTool", "local_shell"),
        ("ShellTool", "shell"),
        ("ShellTool", "run_commands"),
        ("ApplyPatchTool", "apply_patch"),
        ("ApplyPatchTool", "edit_files"),
        ("WebSearchTool", "web_search"),
    ],
)
async def test_runner_counts_client_side_tools(
    tool_class: str,
    tool_name: str,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    factory = getattr(agents, tool_class, None)
    if factory is None:
        pytest.skip(f"{tool_class} is unavailable in this SDK version")
    executor = AsyncMock(return_value="ok")
    if tool_class == "ComputerTool":
        computer = NonCallableMock(spec=agents.AsyncComputer)
        computer.environment = "browser"
        computer.dimensions = (100, 100)
        computer.screenshot.return_value = "c2NyZWVu"
        tool = factory(computer=computer)
        payload = {
            "type": "computer_call",
            "action": {"type": "click", "button": "left", "x": 1, "y": 1},
            "pending_safety_checks": [],
        }
    elif tool_class == "LocalShellTool":
        tool = factory(executor=executor)
        payload = {
            "type": "local_shell_call",
            "action": {"type": "exec", "command": ["echo", "test"], "env": {}},
        }
    elif tool_class == "ShellTool":
        tool = factory(executor=executor, name=tool_name)
        payload = {
            "type": "shell_call",
            "name": tool_name,
            "action": {"commands": ["echo test"]},
        }
    elif tool_class == "ApplyPatchTool":
        editor = MagicMock()
        editor.create_file = AsyncMock(return_value=None)
        tool = factory(editor=editor, name=tool_name)
        payload = {
            "type": "apply_patch_call",
            "name": tool_name,
            "operation": {
                "type": "create_file",
                "path": "test.txt",
                "diff": "+test",
            },
        }
    else:
        tool = factory()
        payload = {
            "type": "web_search_call",
            "action": {"type": "search", "query": "weather"},
        }
    payload.update(id="tool_1", call_id="call_1", status="completed")
    provider_executed = tool_class == "WebSearchTool"
    outputs = (
        [[payload, _FINAL_MESSAGE]]
        if provider_executed
        else [[payload], [_FINAL_MESSAGE]]
    )
    application_hooks = MagicMock(spec=type(agents.RunHooks()))
    with instrument(
        OpenAIAgentsInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        agent = agents.Agent(
            name="test_agent", model=_mock_model(*outputs), tools=[tool]
        )
        result = await agents.Runner.run(
            agent, "run tool", hooks=application_hooks
        )
    executed_calls = sum(
        isinstance(item, agents.ToolCallOutputItem)
        for item in result.new_items
    )
    assert application_hooks.on_tool_start.await_count == executed_calls
    assert application_hooks.on_tool_end.await_count == executed_calls
    if tool_class == "LocalShellTool":
        # SDK 0.3.3 accepts local-shell requests without dispatching them.
        assert executor.await_count == executed_calls
    elif tool_class == "ShellTool":
        executor.assert_awaited_once()
    elif tool_class == "ComputerTool":
        computer.click.assert_awaited_once_with(1, 1, "left")
    elif tool_class == "ApplyPatchTool":
        editor.create_file.assert_awaited_once()
    if tool_class != "LocalShellTool":
        assert executed_calls == (0 if provider_executed else 1)
    _assert_agent_call_metrics(
        metric_reader,
        span_exporter,
        inference_calls=1 if provider_executed else 2,
        tool_calls=executed_calls,
    )
