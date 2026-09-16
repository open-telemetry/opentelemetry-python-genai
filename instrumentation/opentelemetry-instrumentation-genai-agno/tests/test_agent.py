# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Agent instrumentation."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.team import Team
from agno.tools.function import Function, FunctionCall
from tests.mock_model import MockModel

from opentelemetry.instrumentation.genai.agno.patch import (
    _fail_tool_invocation,
    _set_tool_invocation_output,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ErrorTypeValues,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
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


def test_agent_arun_concurrent(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that concurrent Agent.arun calls emit unnested spans without context errors."""
    agent = Agent(name="test-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="output")

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            coro1 = agent.arun("input 1")
            coro2 = agent.arun("input 2")
            await asyncio.gather(coro1, coro2)

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].parent is None
    assert spans[1].parent is None


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


@pytest.mark.parametrize("async_execute", [False, True])
def test_tool_call_failure_spans(
    instrument_agno,
    span_exporter,
    async_execute: bool,
) -> None:
    def failing_tool() -> None:
        raise ValueError("tool failed")

    func_call = FunctionCall(
        function=Function.from_callable(failing_tool), arguments={}
    )
    result = (
        asyncio.run(func_call.aexecute())
        if async_execute
        else func_call.execute()
    )

    assert result.status == "failure"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "tool failed"
    assert (
        span.attributes.get(ErrorAttributes.ERROR_TYPE)
        == ErrorTypeValues.OTHER.value
    )


def test_failed_tool_result_is_not_captured() -> None:
    invocation = MagicMock()
    invocation.tool_result = None

    result = SimpleNamespace(status="failure", error="tool failed")
    _fail_tool_invocation(invocation, result)
    _set_tool_invocation_output(
        invocation,
        result,
        capture_content=True,
    )

    assert invocation.tool_result is None
    invocation.fail.assert_called_once()


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


def test_workflow_arun_concurrent(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that concurrent Workflow.arun calls emit unnested spans without context errors."""
    pytest.importorskip("fastapi")
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    workflow = Workflow(name="test-workflow-concurrent", steps=[])

    async def _run() -> None:
        coro1 = workflow.arun("input 1")
        coro2 = workflow.arun("input 2")
        await asyncio.gather(coro1, coro2)

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].parent is None
    assert spans[1].parent is None


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


def test_format_content_structured_types() -> None:
    import json
    from dataclasses import dataclass

    from pydantic import BaseModel

    from opentelemetry.instrumentation.genai.agno.utils import format_content

    class MyModel(BaseModel):
        name: str
        count: int

    @dataclass
    class MyDataClass:
        item: str

    assert format_content("plain text") == "plain text"
    assert format_content(None) == ""
    assert format_content(42) == "42"
    assert json.loads(format_content(MyModel(name="test", count=5))) == {
        "name": "test",
        "count": 5,
    }
    assert json.loads(format_content(MyDataClass(item="val"))) == {
        "item": "val"
    }
    assert json.loads(format_content({"a": 1, "b": "c"})) == {"a": 1, "b": "c"}
    assert json.loads(format_content([1, "x"])) == [1, "x"]
    assert json.loads(
        format_content({"nested": MyModel(name="sub", count=1)})
    ) == {"nested": {"name": "sub", "count": 1}}

    class MockV1Model:
        def __init__(self, key: str) -> None:
            self.key = key

        def dict(self) -> dict[str, Any]:
            return {"key": self.key}

    assert json.loads(format_content(MockV1Model("v1"))) == {"key": "v1"}
    assert json.loads(
        format_content({"nested_v1": MockV1Model("v1_nested")})
    ) == {"nested_v1": {"key": "v1_nested"}}
    assert json.loads(format_content([MockV1Model("v1_list")])) == [
        {"key": "v1_list"}
    ]
    assert format_content(MockV1Model) == str(MockV1Model)


def test_set_invocation_output_pydantic_structured_content(
    tracer_provider,
) -> None:
    import json

    from pydantic import BaseModel

    from opentelemetry.instrumentation.genai.agno.patch import (
        _set_invocation_output,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    class Movie(BaseModel):
        title: str
        year: int

    class _OutputResult:
        def __init__(self, content: object) -> None:
            self.content = content
            self.session_id = "sess-123"

    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.invoke_local_agent(agent_name="agent")
    _set_invocation_output(
        invocation,
        _OutputResult(content=Movie(title="Inception", year=2010)),
        capture_content=True,
    )
    invocation.stop()
    msg = invocation.output_messages[0]
    assert json.loads(msg.parts[0].content) == {
        "title": "Inception",
        "year": 2010,
    }


def test_set_tool_invocation_output_structured_result(
    tracer_provider,
) -> None:
    import json

    from pydantic import BaseModel

    from opentelemetry.instrumentation.genai.agno.patch import (
        _set_tool_invocation_output,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    class ToolOutput(BaseModel):
        status: str
        code: int

    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.tool(name="sample_tool")
    _set_tool_invocation_output(
        invocation,
        ToolOutput(status="ok", code=200),
        capture_content=True,
    )
    invocation.stop()
    assert json.loads(invocation.tool_result) == {
        "status": "ok",
        "code": 200,
    }


def test_agent_run_attributes(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run extracts description and model attributes."""
    agent = Agent(
        id="agent-custom-id",
        name="attribute-agent",
        description="Custom agent description",
        model=MockModel(id="custom-model", provider="custom_provider"),
    )

    def mock_response(*args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(content="Response")

    with (
        patch("agno.models.base.Model.response", side_effect=mock_response),
    ):
        res = agent.run("hello")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "attribute-agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_DESCRIPTION)
        == "Custom agent description"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "custom-model"
    )
    assert GenAIAttributes.GEN_AI_AGENT_ID not in span.attributes
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in span.attributes


def test_agent_arun_attributes(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun extracts description and model attributes."""
    agent = Agent(
        id="async-agent-id",
        name="async-attribute-agent",
        description="Async description",
        model=MockModel(id="custom-async-model", provider="custom_provider"),
    )

    async def mock_aresponse(*args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(content="Async response")

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", side_effect=mock_aresponse
        ):
            res = await agent.arun("async hello")
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "async-attribute-agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_DESCRIPTION)
        == "Async description"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "custom-async-model"
    )
    assert GenAIAttributes.GEN_AI_AGENT_ID not in span.attributes
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in span.attributes


def test_agent_run_attributes_model_from_kwargs(
    instrument_agno,
    span_exporter,
) -> None:
    """Test capturing model passed via kwargs at invocation start."""
    import agno.agent
    from agno.agent import RunOutput

    agent = Agent(name="kwargs-agent", model=MockModel(id=None))

    mock_run_output = RunOutput(
        agent_id="extracted-agent-id",
        content="Hello output",
    )

    dispatch_target = (
        "agno.agent._run.run_dispatch"
        if hasattr(agno.agent, "_run")
        else "agno.agent.agent.Agent._run"
    )

    with patch(dispatch_target, return_value=mock_run_output):
        res = agent.run("test", model=MockModel(id="kwargs-model"))
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "kwargs-model"
    )
    assert GenAIAttributes.GEN_AI_AGENT_ID not in span.attributes
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in span.attributes


def test_agent_run_does_not_extract_model_from_run_output(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that model from RunOutput is not used as a fallback."""
    import agno.agent
    from agno.agent import RunOutput

    agent = Agent(name="unconfigured-agent", model=MockModel(id=None))

    mock_run_output = RunOutput(
        agent_id="extracted-agent-id",
        model="extracted-model",
        content="Hello output",
    )

    dispatch_target = (
        "agno.agent._run.run_dispatch"
        if hasattr(agno.agent, "_run")
        else "agno.agent.agent.Agent._run"
    )

    with patch(dispatch_target, return_value=mock_run_output):
        res = agent.run("test")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert GenAIAttributes.GEN_AI_REQUEST_MODEL not in span.attributes
    assert GenAIAttributes.GEN_AI_AGENT_ID not in span.attributes
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in span.attributes


def test_agent_continue_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.continue_run emits an invoke_agent span correlated by conversation ID."""
    agent = Agent(
        name="test-continue-agent",
        model=MockModel(id="mock-model"),
        session_id="session-cont-123",
    )
    mock_run_output = ModelResponse(content="Initial output")
    mock_cont_output = ModelResponse(content="Continued output")

    with patch(
        "agno.models.base.Model.response", return_value=mock_run_output
    ):
        run_res = agent.run("hello")
        assert run_res is not None

    with patch(
        "agno.models.base.Model.response", return_value=mock_cont_output
    ):
        kwargs = (
            {"input": "continue instruction"}
            if "input" in inspect.signature(Agent.continue_run).parameters
            else {}
        )
        cont_res = agent.continue_run(run_res, **kwargs)
        assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    # First run span
    span1 = spans[0]
    assert span1.name == "invoke_agent test-continue-agent"
    assert (
        span1.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span1.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "session-cont-123"
    )

    # Continue run span
    span2 = spans[1]
    assert span2.name == "invoke_agent test-continue-agent"
    assert (
        span2.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span2.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "session-cont-123"
    )


def test_agent_acontinue_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.acontinue_run emits an invoke_agent span."""
    agent = Agent(
        name="test-async-continue-agent",
        model=MockModel(id="mock-model"),
        session_id="async-cont-session",
    )
    mock_run_output = ModelResponse(content="Initial async output")
    mock_cont_output = ModelResponse(content="Continued async output")

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_run_output
        ):
            run_res = await agent.arun("hello async")
            assert run_res is not None
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_cont_output
        ):
            kwargs = (
                {"input": "continue async"}
                if "input" in inspect.signature(Agent.acontinue_run).parameters
                else {}
            )
            cont_res = await agent.acontinue_run(run_res, **kwargs)
            assert cont_res is not None

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.name == "invoke_agent test-async-continue-agent"
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "invoke_agent"
        )
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
            == "async-cont-session"
        )


def test_team_continue_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.continue_run emits an invoke_agent span."""
    if not hasattr(Team, "continue_run"):
        pytest.skip(
            "Team.continue_run is not supported in this version of agno"
        )

    agent1 = Agent(name="m1", model=MockModel(id="mock-model"))
    agent2 = Agent(name="m2", model=MockModel(id="mock-model"))
    team = Team(
        name="test-continue-team",
        members=[agent1, agent2],
        model=MockModel(id="mock-model"),
        session_id="team-cont-session",
    )
    mock_run_output = ModelResponse(content="Team initial output")
    mock_cont_output = ModelResponse(content="Team continued output")

    with patch(
        "agno.models.base.Model.response", return_value=mock_run_output
    ):
        run_res = team.run("team run")
        assert run_res is not None

    with patch(
        "agno.models.base.Model.response", return_value=mock_cont_output
    ):
        cont_res = team.continue_run(run_res, input="team continue")
        assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    # At least team run and team continue_run spans
    team_spans = [
        s for s in spans if s.name == "invoke_agent test-continue-team"
    ]
    assert len(team_spans) == 2
    for span in team_spans:
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
            == "team-cont-session"
        )


def test_team_acontinue_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Team.acontinue_run emits an invoke_agent span."""
    if not hasattr(Team, "acontinue_run"):
        pytest.skip(
            "Team.acontinue_run is not supported in this version of agno"
        )

    agent1 = Agent(name="m1", model=MockModel(id="mock-model"))
    agent2 = Agent(name="m2", model=MockModel(id="mock-model"))
    team = Team(
        name="test-async-continue-team",
        members=[agent1, agent2],
        model=MockModel(id="mock-model"),
        session_id="async-team-cont-session",
    )
    mock_run_output = ModelResponse(content="Team initial output")
    mock_cont_output = ModelResponse(content="Team continued output")

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_run_output
        ):
            run_res = await team.arun("team arun")
            assert run_res is not None
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_cont_output
        ):
            cont_res = await team.acontinue_run(
                run_res, input="team acontinue"
            )
            assert cont_res is not None

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    team_spans = [
        s for s in spans if s.name == "invoke_agent test-async-continue-team"
    ]
    assert len(team_spans) == 2
    for span in team_spans:
        assert (
            span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
            == "async-team-cont-session"
        )


def test_workflow_continue_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.continue_run emits a workflow span."""
    from agno.workflow.workflow import Workflow

    if not hasattr(Workflow, "continue_run"):
        pytest.skip(
            "Workflow.continue_run is not supported in this version of agno"
        )

    from agno.run.workflow import RunStatus, WorkflowRunOutput

    workflow = Workflow(
        name="test-continue-workflow",
        steps=[],
        session_id="wf-session-cont",
    )
    mock_wf_output = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="wf-session-cont",
        status=RunStatus.paused,
        paused_step_index=0,
        content="Workflow continued",
        step_requirements=[],
    )

    with (
        patch.object(Workflow, "get_session", return_value=MagicMock()),
        patch.object(
            Workflow, "_continue_execute", return_value=mock_wf_output
        ),
    ):
        res = workflow.continue_run(
            run_response=mock_wf_output, input="continue wf"
        )
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-continue-workflow"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "wf-session-cont"
    )


def test_workflow_acontinue_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.acontinue_run emits a workflow span."""
    from agno.workflow.workflow import Workflow

    if not hasattr(Workflow, "acontinue_run"):
        pytest.skip(
            "Workflow.acontinue_run is not supported in this version of agno"
        )

    from unittest.mock import AsyncMock

    from agno.run.workflow import RunStatus, WorkflowRunOutput

    workflow = Workflow(
        name="test-async-continue-workflow",
        steps=[],
        session_id="wf-async-session-cont",
    )
    mock_wf_output = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="wf-async-session-cont",
        status=RunStatus.paused,
        paused_step_index=0,
        content="Workflow async continued",
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
                return_value=mock_wf_output,
            ),
        ):
            res = await workflow.acontinue_run(
                run_response=mock_wf_output, input="continue wf async"
            )
            assert res is not None

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-async-continue-workflow"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "wf-async-session-cont"
    )


def test_agent_continue_run_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error in Agent.continue_run records error telemetry."""
    import agno.agent

    agent = Agent(
        name="error-continue-agent", model=MockModel(id="mock-model")
    )

    if hasattr(agno.agent, "_run"):
        cm = patch(
            "agno.agent._run.continue_run_dispatch",
            side_effect=RuntimeError("continue boom"),
        )
    else:
        cm = patch.object(
            Agent,
            "_initialize_session",
            side_effect=RuntimeError("continue boom"),
        )

    with (
        cm,
        pytest.raises(RuntimeError, match="continue boom"),
    ):
        agent.continue_run(run_id="some-id", session_id="sess-err")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


def test_agent_acontinue_run_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error in Agent.acontinue_run records error telemetry."""
    import agno.agent

    agent = Agent(
        name="error-async-continue-agent", model=MockModel(id="mock-model")
    )

    if hasattr(agno.agent, "_run"):
        cm = patch(
            "agno.agent._run.acontinue_run_dispatch",
            side_effect=RuntimeError("async continue boom"),
        )
    else:
        cm = patch.object(
            Agent,
            "_initialize_session",
            side_effect=RuntimeError("async continue boom"),
        )

    async def _run() -> None:
        with (
            cm,
            pytest.raises(RuntimeError, match="async continue boom"),
        ):
            await agent.acontinue_run(run_id="some-id", session_id="sess-err")

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


def test_team_continue_run_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error in Team.continue_run records error telemetry."""
    if not hasattr(Team, "continue_run"):
        pytest.skip(
            "Team.continue_run is not supported in this version of agno"
        )

    agent1 = Agent(name="m1", model=MockModel(id="mock-model"))
    team = Team(
        name="test-err-team",
        members=[agent1],
        model=MockModel(id="mock-model"),
    )
    with (
        patch(
            "agno.team._run.continue_run_dispatch",
            side_effect=RuntimeError("team continue boom"),
        ),
        pytest.raises(RuntimeError, match="team continue boom"),
    ):
        team.continue_run(run_id="some-id")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


def test_team_acontinue_run_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error in Team.acontinue_run records error telemetry."""
    if not hasattr(Team, "acontinue_run"):
        pytest.skip(
            "Team.acontinue_run is not supported in this version of agno"
        )

    agent1 = Agent(name="m1", model=MockModel(id="mock-model"))
    team = Team(
        name="test-err-async-team",
        members=[agent1],
        model=MockModel(id="mock-model"),
    )

    async def _run() -> None:
        with (
            patch(
                "agno.team._run.acontinue_run_dispatch",
                side_effect=RuntimeError("async team continue boom"),
            ),
            pytest.raises(RuntimeError, match="async team continue boom"),
        ):
            await team.acontinue_run(run_id="some-id")

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


def test_workflow_continue_run_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error in Workflow.continue_run records error telemetry."""
    from agno.workflow.workflow import Workflow

    if not hasattr(Workflow, "continue_run"):
        pytest.skip(
            "Workflow.continue_run is not supported in this version of agno"
        )

    from agno.run.workflow import RunStatus, WorkflowRunOutput

    workflow = Workflow(
        name="test-err-workflow",
        steps=[],
        session_id="wf-err-sess",
    )
    mock_wf_output = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="wf-err-sess",
        status=RunStatus.paused,
        paused_step_index=0,
        content="Workflow continued",
        step_requirements=[],
    )
    with (
        patch.object(Workflow, "get_session", return_value=MagicMock()),
        patch.object(
            Workflow,
            "_continue_execute",
            side_effect=RuntimeError("workflow continue boom"),
        ),
        pytest.raises(RuntimeError, match="workflow continue boom"),
    ):
        workflow.continue_run(run_response=mock_wf_output)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


def test_workflow_acontinue_run_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error in Workflow.acontinue_run records error telemetry."""
    from agno.workflow.workflow import Workflow

    if not hasattr(Workflow, "acontinue_run"):
        pytest.skip(
            "Workflow.acontinue_run is not supported in this version of agno"
        )

    from unittest.mock import AsyncMock

    from agno.run.workflow import RunStatus, WorkflowRunOutput

    workflow = Workflow(
        name="test-err-async-workflow",
        steps=[],
        session_id="wf-async-err-sess",
    )
    mock_wf_output = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="wf-async-err-sess",
        status=RunStatus.paused,
        paused_step_index=0,
        content="Workflow async continued",
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
                side_effect=RuntimeError("async workflow continue boom"),
            ),
            pytest.raises(RuntimeError, match="async workflow continue boom"),
        ):
            await workflow.acontinue_run(run_response=mock_wf_output)

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


def test_agent_continue_run_with_additional_instructions(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Agent.continue_run with additional_instructions parameter."""
    if (
        "additional_instructions"
        not in inspect.signature(Agent.continue_run).parameters
    ):
        pytest.skip(
            "Agent.continue_run does not support additional_instructions in this version of agno"
        )

    agent = Agent(name="test-add-inst-agent", model=MockModel(id="mock-model"))
    mock_run_output = ModelResponse(content="Initial response")
    mock_cont_output = ModelResponse(content="Continued response")

    with patch.object(Agent, "run", wraps=agent.run):
        with patch(
            "agno.models.base.Model.response", return_value=mock_run_output
        ):
            run_res = agent.run("hello")
            assert run_res is not None

        with patch(
            "agno.models.base.Model.response", return_value=mock_cont_output
        ):
            cont_res = agent.continue_run(
                run_res, additional_instructions="be more concise"
            )
            assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in cont_span.attributes
    input_messages = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == "be more concise"


def test_agent_continue_run_with_additional_instructions_camel_case(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Agent.continue_run with camelCase additionalInstructions alias."""
    if (
        "additional_instructions"
        not in inspect.signature(Agent.continue_run).parameters
    ):
        pytest.skip(
            "Agent.continue_run does not support additional_instructions in this version of agno"
        )

    agent = Agent(
        name="test-camel-inst-agent", model=MockModel(id="mock-model")
    )
    mock_run_output = ModelResponse(content="Initial response")
    mock_cont_output = ModelResponse(content="Continued response")

    with patch.object(Agent, "run", wraps=agent.run):
        with patch(
            "agno.models.base.Model.response", return_value=mock_run_output
        ):
            run_res = agent.run("hello")
            assert run_res is not None

        with patch(
            "agno.models.base.Model.response", return_value=mock_cont_output
        ):
            cont_res = agent.continue_run(
                run_res, additionalInstructions="steer output"
            )
            assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in cont_span.attributes
    input_messages = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == "steer output"


def test_agent_continue_run_with_tools_json_string_results(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Agent.continue_run with tools passed as a JSON string of tool executions."""
    agent = Agent(
        name="test-tools-json-agent", model=MockModel(id="mock-model")
    )
    mock_run_output = ModelResponse(content="Initial response")
    mock_cont_output = ModelResponse(content="Continued response")
    tools_payload = json.dumps(
        [{"tool_call_id": "call_abc", "tool_name": "calc", "result": "42"}]
    )

    with patch.object(Agent, "run", wraps=agent.run):
        with patch(
            "agno.models.base.Model.response", return_value=mock_run_output
        ):
            run_res = agent.run("calculate something")
            assert run_res is not None

        with patch(
            "agno.models.base.Model.response", return_value=mock_cont_output
        ):
            cont_res = agent.continue_run(run_res, updated_tools=tools_payload)
            assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in cont_span.attributes
    input_messages = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "tool"
    assert input_messages[0]["parts"][0]["id"] == "call_abc"
    assert input_messages[0]["parts"][0]["response"] == "42"


def test_agent_continue_run_with_tools_kwarg_json_string(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Agent.continue_run with tools kwarg directly passed as JSON string."""
    import agno.agent

    if not hasattr(agno.agent, "_run"):
        pytest.skip(
            "Agent.continue_run kwargs not supported in this version of agno"
        )

    agent = Agent(
        name="test-tools-direct-agent", model=MockModel(id="mock-model")
    )
    mock_run_output = ModelResponse(content="Initial response")
    mock_cont_output = ModelResponse(content="Continued response")
    tools_payload = json.dumps(
        [{"tool_call_id": "call_direct", "tool_name": "calc", "result": "99"}]
    )

    with (
        patch.object(Agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_run_output),
    ):
        run_res = agent.run("calculate")

    with patch(
        "agno.agent._run.continue_run_dispatch", return_value=mock_cont_output
    ):
        cont_res = agent.continue_run(run_res, tools=tools_payload)
        assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in cont_span.attributes
    input_messages = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "tool"
    assert input_messages[0]["parts"][0]["id"] == "call_direct"
    assert input_messages[0]["parts"][0]["response"] == "99"


def test_agent_continue_run_with_tools_json_string_and_additional_instructions(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Agent.continue_run with both tools JSON string and additional_instructions."""
    if (
        "additional_instructions"
        not in inspect.signature(Agent.continue_run).parameters
    ):
        pytest.skip(
            "Agent.continue_run does not support additional_instructions in this version of agno"
        )

    agent = Agent(
        name="test-tools-hitl-agent", model=MockModel(id="mock-model")
    )
    mock_run_output = ModelResponse(content="Initial response")
    mock_cont_output = ModelResponse(content="Continued response")
    tools_payload = json.dumps(
        [{"tool_call_id": "call_hitl", "confirmed": True}]
    )

    with patch.object(Agent, "run", wraps=agent.run):
        with patch(
            "agno.models.base.Model.response", return_value=mock_run_output
        ):
            run_res = agent.run("start hitl")
            assert run_res is not None

        with patch(
            "agno.models.base.Model.response", return_value=mock_cont_output
        ):
            cont_res = agent.continue_run(
                run_res,
                updated_tools=tools_payload,
                additional_instructions="proceed with caution",
            )
            assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in cont_span.attributes
    input_messages = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_messages) == 2
    assert input_messages[0]["role"] == "tool"
    assert input_messages[0]["parts"][0]["id"] == "call_hitl"
    assert json.loads(input_messages[0]["parts"][0]["response"]) == {
        "confirmed": True
    }
    assert input_messages[1]["role"] == "user"
    assert input_messages[1]["parts"][0]["content"] == "proceed with caution"


def test_agent_continue_run_with_tools_json_string_tool_definitions(
    instrument_agno,
    span_exporter,
) -> None:
    """Test Agent with tools initialized as a JSON string of tool definitions."""
    tool_defs_json = json.dumps(
        [
            {
                "name": "weather_tool",
                "description": "Get weather info",
                "parameters": {"type": "object"},
            }
        ]
    )
    agent = Agent(
        name="test-tools-def-json-agent",
        model=MockModel(id="mock-model"),
        tools=tool_defs_json,
    )
    mock_run_output = ModelResponse(content="Initial response")
    mock_cont_output = ModelResponse(content="Continued response")

    with patch.object(Agent, "run", wraps=agent.run):
        with patch(
            "agno.models.base.Model.response", return_value=mock_run_output
        ):
            run_res = agent.run("initial run")
            assert run_res is not None

        with patch(
            "agno.models.base.Model.response", return_value=mock_cont_output
        ):
            kwargs = (
                {"input": "next"}
                if "input" in inspect.signature(Agent.continue_run).parameters
                else {}
            )
            cont_res = agent.continue_run(run_res, **kwargs)
            assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in cont_span.attributes
    tool_defs = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    )
    assert len(tool_defs) == 1
    assert tool_defs[0]["name"] == "weather_tool"
    assert tool_defs[0]["description"] == "Get weather info"


def test_agent_acontinue_run_with_tools_and_additional_instructions(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Agent.acontinue_run with tools JSON string and additional_instructions."""
    if (
        "additional_instructions"
        not in inspect.signature(Agent.acontinue_run).parameters
    ):
        pytest.skip(
            "Agent.acontinue_run does not support additional_instructions in this version of agno"
        )

    agent = Agent(
        name="test-async-tools-agent", model=MockModel(id="mock-model")
    )
    mock_run_output = ModelResponse(content="Async initial")
    mock_cont_output = ModelResponse(content="Async continued")
    tools_payload = json.dumps(
        [
            {
                "tool_call_id": "call_async_1",
                "tool_name": "calc",
                "result": "100",
            }
        ]
    )

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_run_output
        ):
            run_res = await agent.arun("async hello")
            assert run_res is not None

        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_cont_output
        ):
            cont_res = await agent.acontinue_run(
                run_res,
                updated_tools=tools_payload,
                additional_instructions="async extra guidance",
            )
            assert cont_res is not None

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in cont_span.attributes
    input_messages = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_messages) == 2
    assert input_messages[0]["role"] == "tool"
    assert input_messages[0]["parts"][0]["id"] == "call_async_1"
    assert input_messages[0]["parts"][0]["response"] == "100"
    assert input_messages[1]["role"] == "user"
    assert input_messages[1]["parts"][0]["content"] == "async extra guidance"
