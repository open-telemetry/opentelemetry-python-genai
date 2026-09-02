# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Agent instrumentation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.team import Team
from agno.tools.function import Function, FunctionCall
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace.status import StatusCode


def test_agent_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run emits an invoke_agent span."""
    agent = Agent(name="test-sync-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Hello back!")

    with (
        patch.object(Agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        res = agent.run("hello world")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-sync-agent"
    )


def test_agent_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun emits an invoke_agent span."""
    agent = Agent(name="test-async-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Async hello back!")

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await agent.arun("hello async world")
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-async-agent"
    )


def test_tool_call_execute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that FunctionCall.execute emits an execute_tool span."""

    def sample_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    func = Function.from_callable(sample_tool)
    func_call = FunctionCall(
        function=func,
        arguments={"x": 5},
        call_id="call-123",
    )
    res = func_call.execute()
    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool sample_tool"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME) == "sample_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID) == "call-123"
    )


def test_tool_call_aexecute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that FunctionCall.aexecute emits an execute_tool span."""

    def sample_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    func = Function.from_callable(sample_tool)
    func_call = FunctionCall(
        function=func,
        arguments={"x": 5},
        call_id="call-456",
    )

    async def _run_async() -> None:
        res = await func_call.aexecute()
        assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool sample_tool"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME) == "sample_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID) == "call-456"
    )


def test_team_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.run emits an invoke_agent span."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-sync-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )
    mock_output = ModelResponse(content="Hello back from team!")

    with patch("agno.models.base.Model.response", return_value=mock_output):
        res = team.run("hello team world")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-team"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-sync-team"
    )


def test_team_run_error_path(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.run records error.type and re-raises on failure."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-sync-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )
    with (
        patch.object(
            Team,
            "initialize_team",
            side_effect=RuntimeError("team failure"),
        ),
        pytest.raises(RuntimeError, match="team failure"),
    ):
        team.run("hello team world")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-team"
    assert span.attributes.get("error.type") == "RuntimeError"
    assert span.status.status_code == StatusCode.ERROR


def test_team_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.arun emits an invoke_agent span."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-async-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )
    mock_output = ModelResponse(content="Async hello back from team!")

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await team.arun("hello async team world")
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-team"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-async-team"
    )


def test_team_arun_error_path(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.arun records error.type and re-raises on failure."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-async-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )

    async def _run_async() -> None:
        with (
            patch.object(
                Team,
                "initialize_team",
                side_effect=RuntimeError("async team failure"),
            ),
            pytest.raises(RuntimeError, match="async team failure"),
        ):
            await team.arun("hello async team world")

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-team"
    assert span.attributes.get("error.type") == "RuntimeError"
    assert span.status.status_code == StatusCode.ERROR


def test_workflow_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.run emits an invoke_workflow span."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    workflow = Workflow(name="test-workflow", steps=[])
    workflow.run("test input")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-workflow"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_WORKFLOW_NAME)
        == "test-workflow"
    )


def test_workflow_run_error_path(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.run records error.type and re-raises on failure."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    workflow = Workflow(name="test-workflow", steps=[])
    with (
        patch.object(
            Workflow, "_execute", side_effect=RuntimeError("workflow failure")
        ),
        pytest.raises(RuntimeError, match="workflow failure"),
    ):
        workflow.run("test input")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-workflow"
    assert span.attributes.get("error.type") == "RuntimeError"
    assert span.status.status_code == StatusCode.ERROR


def test_workflow_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.arun emits an invoke_workflow span."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    workflow = Workflow(name="test-workflow-async", steps=[])

    async def _run_async() -> None:
        await workflow.arun("test input")

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-workflow-async"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_WORKFLOW_NAME)
        == "test-workflow-async"
    )


def test_workflow_arun_error_path(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.arun records error.type and re-raises on failure."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    workflow = Workflow(name="test-workflow-async", steps=[])

    async def _run_async() -> None:
        with (
            patch.object(
                Workflow,
                "_aexecute",
                side_effect=RuntimeError("async workflow failure"),
            ),
            pytest.raises(RuntimeError, match="async workflow failure"),
        ):
            await workflow.arun("test input")

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-workflow-async"
    assert span.attributes.get("error.type") == "RuntimeError"
    assert span.status.status_code == StatusCode.ERROR


def test_none_role_becomes_assistant_and_finish_reason_stop(
    tracer_provider,
) -> None:
    from dataclasses import dataclass

    from opentelemetry.instrumentation.genai.agno.patch import (
        _set_invocation_output,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    @dataclass
    class _Result:
        content: str = "hi"
        role: str | None = None
        finish_reason: str | None = None
        session_id: str | None = None

    invocation = TelemetryHandler(tracer_provider=tracer_provider).workflow(
        name="wf"
    )
    _set_invocation_output(invocation, _Result(), capture_content=True)
    invocation.stop()
    msg = invocation.output_messages[0]
    assert msg.role == "assistant"
    assert msg.finish_reason == "stop"


def test_status_error_becomes_finish_reason_error(
    tracer_provider,
) -> None:
    from dataclasses import dataclass

    from opentelemetry.instrumentation.genai.agno.patch import (
        _set_invocation_output,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    @dataclass
    class _StatusResult:
        content: str = "failed"
        role: str | None = None
        finish_reason: str | None = None
        status: str = "error"
        session_id: str | None = None

    invocation = TelemetryHandler(tracer_provider=tracer_provider).workflow(
        name="wf"
    )
    _set_invocation_output(invocation, _StatusResult(), capture_content=True)
    invocation.stop()
    msg = invocation.output_messages[0]
    assert msg.role == "assistant"
    assert msg.finish_reason == "error"


def test_agno_run_status_handling(
    tracer_provider,
) -> None:
    from dataclasses import dataclass

    from agno.run.base import RunStatus

    from opentelemetry.instrumentation.genai.agno.patch import (
        _set_invocation_output,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    @dataclass
    class _AgnoRunResult:
        content: str = "completed run"
        status: RunStatus = RunStatus.completed
        session_id: str = "session-abc"

    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.invoke_local_agent(agent_name="agent")
    _set_invocation_output(invocation, _AgnoRunResult(), capture_content=True)
    invocation.stop()
    msg = invocation.output_messages[0]
    assert msg.role == "assistant"
    assert msg.finish_reason == "stop"
    assert invocation.conversation_id == "session-abc"

    @dataclass
    class _AgnoErrorResult:
        content: str = "errored run"
        status: RunStatus = RunStatus.error
        session_id: str | None = None

    invocation_err = handler.invoke_local_agent(agent_name="agent")
    _set_invocation_output(
        invocation_err, _AgnoErrorResult(), capture_content=True
    )
    invocation_err.stop()
    msg_err = invocation_err.output_messages[0]
    assert msg_err.role == "assistant"
    assert msg_err.finish_reason == "error"
    assert invocation_err.conversation_id is None


def test_workflow_session_id_sets_conversation_id(
    tracer_provider,
) -> None:
    from dataclasses import dataclass

    from opentelemetry.instrumentation.genai.agno.patch import (
        _set_invocation_output,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    @dataclass
    class _WorkflowResult:
        content: str = "workflow finished"
        session_id: str = "wf-session-456"

    invocation = TelemetryHandler(tracer_provider=tracer_provider).workflow(
        name="wf"
    )
    _set_invocation_output(invocation, _WorkflowResult(), capture_content=True)
    invocation.stop()
    assert invocation.conversation_id == "wf-session-456"
