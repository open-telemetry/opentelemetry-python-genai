# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Workflow background execution instrumentation."""

from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.run.workflow import RunStatus

try:
    from agno.workflow.agent import WorkflowAgent
except ImportError:
    WorkflowAgent = None
from agno.workflow.workflow import Workflow
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_CONVERSATION_ID,
    GEN_AI_OPERATION_NAME,
)
from opentelemetry.trace.status import StatusCode


async def _wait_for_background_tasks(timeout: float = 5.0) -> None:
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current]
    if pending:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=timeout,
        )


def _setup_mock_db_if_needed(workflow: Workflow) -> None:
    """Mock DB on workflow for newer Agno versions where background streaming requires db."""
    if getattr(workflow, "db", None) is None:
        db_mock = MagicMock()
        workflow.db = db_mock
        if hasattr(workflow, "_has_async_db"):
            workflow._has_async_db = MagicMock(return_value=False)
        if hasattr(workflow, "read_or_create_session"):
            sess = MagicMock()
            sess.metadata = {}
            workflow.read_or_create_session = MagicMock(return_value=sess)


def test_workflow_background_execution_non_streaming(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Workflow.arun(background=True) traces the background task until completion."""

    async def async_step(step_input: str) -> str:
        await asyncio.sleep(0.05)
        return f"processed: {step_input}"

    workflow = Workflow(
        name="test-bg-workflow",
        steps=[async_step],
    )

    async def _test() -> None:
        placeholder = await workflow.arun("hello background", background=True)
        # Verify placeholder returned immediately is pending
        assert placeholder.status == RunStatus.pending

        # At this point, the span should NOT be finished yet
        assert len(span_exporter.get_finished_spans()) == 0

        # Wait for background task execution to complete
        await _wait_for_background_tasks()

        assert placeholder.status == RunStatus.completed

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-bg-workflow"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    assert span.status.status_code != StatusCode.ERROR


def test_workflow_background_execution_session_id(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that background workflow captures session_id attribute."""

    async def quick_step(step_input: str) -> str:
        return "quick done"

    workflow = Workflow(
        name="test-bg-id-workflow",
        steps=[quick_step],
    )

    async def _test() -> None:
        placeholder = await workflow.arun(
            "bg input",
            background=True,
            session_id="bg-sess-888",
        )
        assert placeholder.status == RunStatus.pending
        await _wait_for_background_tasks()
        assert placeholder.status == RunStatus.completed

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(GEN_AI_CONVERSATION_ID) == "bg-sess-888"


def test_workflow_background_execution_child_parentage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that steps/agents executed inside background workflow are parented under the workflow span."""
    agent = Agent(name="inner-bg-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Agent output inside background step")

    async def agent_step(step_input: str) -> str:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await agent.arun("step agent input")
            return str(getattr(res, "content", "ok"))

    workflow = Workflow(
        name="test-bg-parent-workflow",
        steps=[agent_step],
    )

    async def _test() -> None:
        placeholder = await workflow.arun("start parent test", background=True)
        assert placeholder.status == RunStatus.pending
        await _wait_for_background_tasks()
        assert placeholder.status == RunStatus.completed

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    # Find workflow span and agent span
    wf_spans = [
        s
        for s in spans
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    ]
    agent_spans = [
        s
        for s in spans
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_agent"
    ]
    assert len(wf_spans) == 1
    assert len(agent_spans) == 1

    wf_span = wf_spans[0]
    agent_span = agent_spans[0]

    assert agent_span.parent is not None
    assert agent_span.parent.span_id == wf_span.context.span_id


def test_workflow_background_execution_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error inside background workflow execution fails the span."""

    async def failing_custom_executor(execution_input: object) -> str:
        raise RuntimeError("boom in background")

    workflow = Workflow(
        name="test-bg-fail-workflow",
        steps=failing_custom_executor,
    )

    async def _test() -> None:
        placeholder = await workflow.arun("fail test", background=True)
        assert placeholder.status == RunStatus.pending
        await _wait_for_background_tasks()
        assert placeholder.status == RunStatus.error

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "RuntimeError"


def test_workflow_foreground_spawns_background_workflow(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that a foreground workflow spawning a background workflow does not suppress it."""
    child_wf = Workflow(
        name="child-bg-workflow", steps=[lambda x: "child done"]
    )

    async def parent_step(step_input: str) -> str:
        await child_wf.arun("child input", background=True)
        return "parent done"

    parent_wf = Workflow(name="parent-fg-workflow", steps=[parent_step])

    async def _test() -> None:
        await parent_wf.arun("start parent")
        await _wait_for_background_tasks()

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    wf_spans_by_name = {
        s.attributes.get("gen_ai.workflow.name"): s
        for s in spans
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    }
    assert "parent-fg-workflow" in wf_spans_by_name
    assert "child-bg-workflow" in wf_spans_by_name

    parent_span = wf_spans_by_name["parent-fg-workflow"]
    child_span = wf_spans_by_name["child-bg-workflow"]
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span.context.span_id


def test_workflow_background_streaming_happy_path(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that background streaming workflow emits an invoke_workflow span."""

    async def async_step(step_input: str):
        yield "chunk 1 "
        yield "chunk 2"

    workflow = Workflow(
        name="test-bg-stream-workflow",
        steps=[async_step],
    )
    _setup_mock_db_if_needed(workflow)

    async def _test() -> None:
        res = workflow.arun(
            "stream input",
            background=True,
            stream=True,
            websocket=AsyncMock(),
        )
        if inspect.isawaitable(res):
            res = await res
        if hasattr(res, "__aiter__"):
            async for _ in res:
                pass
        await _wait_for_background_tasks()

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-bg-stream-workflow"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    assert span.status.status_code != StatusCode.ERROR


def test_workflow_background_streaming_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that background streaming workflow execution error marks span as error."""

    async def failing_custom_executor(execution_input: object):
        yield "chunk 1"
        raise RuntimeError("boom in background stream")

    workflow = Workflow(
        name="test-bg-stream-fail-workflow",
        steps=failing_custom_executor,
    )
    _setup_mock_db_if_needed(workflow)

    async def _test() -> None:
        res = workflow.arun(
            "stream input",
            background=True,
            stream=True,
            websocket=AsyncMock(),
        )
        if inspect.isawaitable(res):
            res = await res
        if hasattr(res, "__aiter__"):
            try:
                async for _ in res:
                    pass
            except Exception:
                pass
        await _wait_for_background_tasks()

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "RuntimeError"


def test_workflow_background_streaming_caller_close(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that caller close / cancellation during background streaming finalizes the span."""

    async def slow_step(step_input: str):
        yield "chunk 1"
        await asyncio.sleep(1.0)
        yield "chunk 2"

    workflow = Workflow(
        name="test-bg-stream-close-workflow",
        steps=[slow_step],
    )
    _setup_mock_db_if_needed(workflow)

    async def _test() -> None:
        res = workflow.arun(
            "stream input",
            background=True,
            stream=True,
            websocket=AsyncMock(),
        )
        if inspect.isawaitable(res):
            res = await res
        if hasattr(res, "__aiter__"):
            async for _ in res:
                break
            if hasattr(res, "aclose"):
                await res.aclose()
        else:
            await asyncio.sleep(0.05)
            current = asyncio.current_task()
            for t in asyncio.all_tasks():
                if t is not current:
                    t.cancel()
        await _wait_for_background_tasks()

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-bg-stream-close-workflow"


@pytest.mark.skipif(
    not hasattr(Workflow, "_arun_background_stream_ws"),
    reason="Background streaming via SSE is not supported in this agno version",
)
def test_workflow_background_streaming_sse(
    instrument_agno,
    span_exporter,
) -> None:
    async def step(step_input: str):
        yield "chunk"

    workflow = Workflow(name="bg-sse", steps=[step])
    _setup_mock_db_if_needed(workflow)

    async def _test() -> None:
        stream = workflow.arun("input", background=True, stream=True)
        if inspect.isawaitable(stream):
            stream = await stream
        if hasattr(stream, "__aiter__"):
            async for _ in stream:
                pass
        await _wait_for_background_tasks()

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "invoke_workflow bg-sse"
    assert spans[0].status.status_code != StatusCode.ERROR


@pytest.mark.skipif(
    not hasattr(Workflow, "_arun_background_stream_ws"),
    reason="Background streaming via SSE is not supported in this agno version",
)
def test_workflow_background_streaming_sse_error(
    instrument_agno,
    span_exporter,
) -> None:
    async def failing(execution_input: object):
        yield "chunk"
        raise RuntimeError("boom")

    workflow = Workflow(name="bg-sse-fail", steps=failing)
    _setup_mock_db_if_needed(workflow)

    async def _test() -> None:
        stream = workflow.arun("input", background=True, stream=True)
        if inspect.isawaitable(stream):
            stream = await stream
        if hasattr(stream, "__aiter__"):
            try:
                async for _ in stream:
                    pass
            except Exception:
                pass
        await _wait_for_background_tasks()

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


@pytest.mark.skipif(
    WorkflowAgent is None,
    reason="WorkflowAgent not available in this agno version",
)
def test_workflow_background_workflow_agent(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that background workflow with WorkflowAgent emits workflow and agent spans."""
    import agno
    import packaging.version

    if packaging.version.parse(agno.__version__) < packaging.version.parse(
        "3.0.0"
    ):
        pytest.skip(
            "Agno < 3.0.0 execute_workflow_background does not await _aexecute_workflow_agent"
        )
    assert WorkflowAgent is not None
    wf_agent = WorkflowAgent(model=MockModel(id="mock-model"))
    workflow = Workflow(name="test-bg-wf-agent", agent=wf_agent)
    mock_output = ModelResponse(content="Agent output inside workflow agent")

    async def _test() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            placeholder = await workflow.arun("test input", background=True)
            assert placeholder.status == RunStatus.pending
            await _wait_for_background_tasks()

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    wf_spans = [
        s
        for s in spans
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    ]
    agent_spans = [
        s
        for s in spans
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_agent"
    ]
    assert len(wf_spans) == 1
    assert len(agent_spans) == 1

    wf_span = wf_spans[0]
    agent_span = agent_spans[0]

    assert agent_span.parent is not None
    assert agent_span.parent.span_id == wf_span.context.span_id


@pytest.mark.skipif(
    not hasattr(Workflow, "acontinue_run"),
    reason="Workflow.acontinue_run not available in this agno version",
)
def test_acontinue_run_background_without_websocket_has_span(
    instrument_agno, span_exporter
) -> None:
    from agno.run.workflow import WorkflowRunOutput

    workflow = Workflow(name="cont-bg", steps=[], session_id="s")
    paused = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="s",
        status=RunStatus.paused,
        paused_step_index=0,
        step_requirements=[],
    )

    async def _run() -> None:
        with (
            patch.object(
                Workflow,
                "aget_session",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                Workflow,
                "_acontinue_execute",
                new_callable=AsyncMock,
                return_value=paused,
            ) as execute,
        ):
            await workflow.acontinue_run(
                run_response=paused, input="go", background=True
            )
            execute.assert_awaited_once()  # ran inline

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["invoke_workflow cont-bg"]


class ToolCallingModel(Model):
    """Calls run_workflow once, then answers."""

    def _next(self, messages):
        if any(getattr(m, "role", None) == "tool" for m in messages or []):
            return ModelResponse(role="assistant", content="done")
        return ModelResponse(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_workflow",
                        "arguments": json.dumps({"query": "go"}),
                    },
                }
            ],
        )

    def _parse_provider_response(self, response, **kwargs):
        return response

    def _parse_provider_response_delta(self, response):
        return response

    def invoke(self, *args, **kwargs):
        return self._next(kwargs.get("messages"))

    async def ainvoke(self, *args, **kwargs):
        return self._next(kwargs.get("messages"))

    def invoke_stream(self, *args, **kwargs):
        yield self._next(kwargs.get("messages"))

    async def ainvoke_stream(self, *args, **kwargs):
        yield self._next(kwargs.get("messages"))


def _workflow_spans(span_exporter):
    return [
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(GEN_AI_OPERATION_NAME) == "invoke_workflow"
    ]


@pytest.mark.skipif(
    WorkflowAgent is None,
    reason="WorkflowAgent not available in this agno version",
)
@pytest.mark.parametrize("background", [False, True])
def test_workflow_agent_tool_call_single_workflow_span(
    instrument_agno, span_exporter, background
) -> None:
    import agno
    import packaging.version

    if background and packaging.version.parse(
        agno.__version__
    ) < packaging.version.parse("3.0.0"):
        pytest.skip(
            "Agno < 3.0.0 execute_workflow_background does not await _aexecute_workflow_agent"
        )

    async def step(step_input) -> str:
        return "step done"

    workflow = Workflow(
        name="wf-agent",
        steps=[step],
        agent=WorkflowAgent(model=ToolCallingModel(id="m")),
    )

    async def _run() -> None:
        await workflow.arun("hi", background=background)
        await _wait_for_background_tasks()

    asyncio.run(_run())

    names = sorted(s.name for s in span_exporter.get_finished_spans())
    assert "execute_tool run_workflow" in names
    assert len(_workflow_spans(span_exporter)) == 1, names


@pytest.mark.skipif(
    WorkflowAgent is None,
    reason="WorkflowAgent not available in this agno version",
)
def test_workflow_agent_tool_call_foreground_stream_single_workflow_span(
    instrument_agno, span_exporter
) -> None:
    async def step(step_input) -> str:
        return "step done"

    workflow = Workflow(
        name="wf-agent-stream",
        steps=[step],
        agent=WorkflowAgent(model=ToolCallingModel(id="m")),
    )

    async def _run() -> None:
        async for _ in workflow.arun("hi", stream=True):
            pass

    asyncio.run(_run())

    names = sorted(s.name for s in span_exporter.get_finished_spans())
    assert "execute_tool run_workflow" in names
    assert len(_workflow_spans(span_exporter)) == 1, names


def test_workflow_aexecute_stream_non_async_iterator_ends_invocation(
    span_exporter,
    tracer_provider,
) -> None:
    from opentelemetry.instrumentation.genai.agno.patch import (
        _workflow_aexecute_stream,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        instrumentation_scope_name="test-agno",
    )
    workflow = Workflow(name="wf-non-stream-result", steps=[])
    mock_fn = MagicMock(return_value="immediate result")
    traced_fn = _workflow_aexecute_stream(handler)

    res = traced_fn(
        mock_fn, workflow, (), {"user_input": "hi", "run_id": "r-123"}
    )
    assert res == "immediate result"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow wf-non-stream-result"
    assert span.status.status_code != StatusCode.ERROR
