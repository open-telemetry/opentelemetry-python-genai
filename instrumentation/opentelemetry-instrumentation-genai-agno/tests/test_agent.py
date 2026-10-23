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
from opentelemetry.util.genai._conversation_context import (
    get_ambient_conversation_id,
)


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


def test_agent_run_captures_keyword_input(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Agent.run captures input when passed as keyword argument."""
    agent = Agent(name="test-kw-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Hello back!")

    with (
        patch.object(Agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        res = agent.run(input="hello keyword")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    input_messages = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    assert input_messages is not None
    assert "hello keyword" in input_messages


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
    invocation.should_capture_content = True

    result = SimpleNamespace(status="failure", error="tool failed")
    _fail_tool_invocation(invocation, result)
    _set_tool_invocation_output(invocation, result)

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
        session_id="instance-session",
    )

    def fail_team_initialization(*args: Any, **kwargs: Any) -> None:
        assert get_ambient_conversation_id() == "call-session"
        raise RuntimeError("team failure")

    with (
        patch.object(
            Team,
            "initialize_team",
            side_effect=fail_team_initialization,
        ),
        pytest.raises(RuntimeError, match="team failure"),
    ):
        team.run("hello team world", session_id="call-session")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-team"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "call-session"
    )
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
        session_id="async-instance-session",
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
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "async-instance-session"
    )
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

    workflow = Workflow(
        name="test-workflow", steps=[], session_id="workflow-instance-session"
    )

    def fail_workflow_execution(*args: Any, **kwargs: Any) -> None:
        assert get_ambient_conversation_id() == "workflow-instance-session"
        raise RuntimeError("workflow failure")

    with (
        patch.object(
            Workflow, "_execute", side_effect=fail_workflow_execution
        ),
        pytest.raises(RuntimeError, match="workflow failure"),
    ):
        workflow.run("test input")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-workflow"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "workflow-instance-session"
    )
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

    workflow = Workflow(
        name="test-workflow-async",
        steps=[],
        session_id="workflow-instance-session",
    )

    async def _run_async() -> None:
        with (
            patch.object(
                Workflow,
                "_aexecute",
                side_effect=RuntimeError("async workflow failure"),
            ),
            pytest.raises(RuntimeError, match="async workflow failure"),
        ):
            await workflow.arun(
                "test input", session_id="workflow-call-session"
            )

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-workflow-async"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "workflow-call-session"
    )
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
    monkeypatch,
) -> None:
    from pydantic import BaseModel

    from opentelemetry.instrumentation.genai.agno.patch import (
        _set_tool_invocation_output,
    )
    from opentelemetry.util.genai.handler import TelemetryHandler

    class ToolOutput(BaseModel):
        status: str
        code: int

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.tool(name="sample_tool")
    _set_tool_invocation_output(
        invocation,
        ToolOutput(status="ok", code=200),
    )
    invocation.stop()
    assert invocation.tool_result == {
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


def test_team_continue_run_dict_requirement(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that Team.continue_run handles dict-form requirements."""
    if not hasattr(Team, "continue_run"):
        pytest.skip(
            "Team.continue_run is not supported in this version of agno"
        )
    try:
        from agno.run.requirement import RunRequirement
    except ImportError:
        pytest.skip("RunRequirement is not supported in this version of agno")

    req = {
        "id": "req-uuid",
        "tool_execution": {
            "tool_call_id": "call_1",
            "tool_name": "lookup",
            "tool_args": {},
        },
        "external_execution_result": "Delivered",
    }
    if (
        getattr(
            getattr(RunRequirement.from_dict(req), "tool_execution", None),
            "result",
            None,
        )
        != "Delivered"
    ):
        pytest.skip(
            "RunRequirement.from_dict does not support external_execution_result in this version"
        )

    team = Team(name="t", members=[], model=MockModel(id="mock-model"))
    with patch(
        "agno.team._run.continue_run_dispatch",
        return_value=ModelResponse(content="ok"),
    ):
        team.continue_run(run_id="r", session_id="s", requirements=[req])

    span = span_exporter.get_finished_spans()[-1]
    messages = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert messages[0]["parts"] == [
        {"type": "tool_call_response", "id": "call_1", "response": "Delivered"}
    ]


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
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in cont_span.attributes
    system_instructions = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    )
    assert len(system_instructions) == 1
    assert system_instructions[0]["type"] == "text"
    assert system_instructions[0]["content"] == "be more concise"
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in cont_span.attributes


def test_agent_continue_run_with_input_and_additional_instructions(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Agent.continue_run with both input and additional_instructions parameters."""
    if (
        "additional_instructions"
        not in inspect.signature(Agent.continue_run).parameters
        or "input" not in inspect.signature(Agent.continue_run).parameters
    ):
        pytest.skip(
            "Agent.continue_run does not support additional_instructions or input in this version of agno"
        )

    agent = Agent(
        name="test-both-input-inst-agent", model=MockModel(id="mock-model")
    )
    mock_run_output = ModelResponse(content="Initial response")
    mock_cont_output = ModelResponse(content="Continued response")

    import agno.agent

    if hasattr(agno.agent, "_run"):
        dispatch_patch = patch(
            "agno.agent._run.continue_run_dispatch",
            return_value=mock_cont_output,
        )
    else:
        dispatch_patch = patch(
            "agno.agent.agent.continue_run_dispatch",
            return_value=mock_cont_output,
        )

    with patch.object(Agent, "run", wraps=agent.run):
        with patch(
            "agno.models.base.Model.response", return_value=mock_run_output
        ):
            run_res = agent.run("hello")
            assert run_res is not None

        with dispatch_patch:
            cont_res = agent.continue_run(
                run_res,
                input="user follow-up",
                additional_instructions="be very concise",
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
    assert input_messages[0]["parts"][0]["content"] == "user follow-up"

    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in cont_span.attributes
    system_instructions = json.loads(
        cont_span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    )
    assert len(system_instructions) == 1
    assert system_instructions[0]["type"] == "text"
    assert system_instructions[0]["content"] == "be very concise"


def test_agent_continue_run_positional_run_id_not_recorded_as_input(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that a positional string run ID in continue_run is not captured as user input."""
    import agno.agent

    if not hasattr(agno.agent, "_run"):
        pytest.skip(
            "Agent.continue_run kwargs not supported in this version of agno"
        )

    agent = Agent(
        name="test-run-id-agent",
        model=MockModel(id="mock-model"),
    )
    mock_run_output = ModelResponse(content="Initial")
    mock_cont_output = ModelResponse(content="Continued")

    with patch(
        "agno.models.base.Model.response", return_value=mock_run_output
    ):
        run_res = agent.run("hello")
        assert run_res is not None

    with patch(
        "agno.agent._run.continue_run_dispatch", return_value=mock_cont_output
    ):
        # Call with positional string run ID without input argument
        cont_res = agent.continue_run("run-123")
        assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    cont_span = spans[-1]
    # "run-123" must not be captured as a user input message
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in cont_span.attributes


def test_continue_run_captures_external_tool_result(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test Agent.continue_run captures resumed tool results from run_response."""
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse, ToolExecution
    from agno.tools import tool

    if not hasattr(agno.agent, "_run") or not hasattr(Agent, "continue_run"):
        pytest.skip(
            "Agent.continue_run is not supported in this version of agno"
        )
    if not hasattr(ToolExecution, "external_execution_required"):
        pytest.skip(
            "ToolExecution.external_execution_required is not supported in this version of agno"
        )

    @tool(external_execution=True)
    def lookup_order(order_id: str) -> str:
        """Look up the delivery status of an order."""
        raise AssertionError("The caller executes this tool outside Agno")

    agent = Agent(
        name="support",
        model=MockModel(id="mock-model"),
        tools=[lookup_order],
        session_id="order-session",
    )
    paused_response = ModelResponse(
        tool_executions=[
            ToolExecution(
                tool_call_id="call_123",
                tool_name="lookup_order",
                tool_args={"order_id": "123"},
                external_execution_required=True,
            )
        ],
    )
    with patch(
        "agno.models.base.Model.response",
        side_effect=[
            paused_response,
            ModelResponse(content="Your order was delivered yesterday."),
        ],
    ):
        response = agent.run("Check order 123.")
        assert response.is_paused
        assert len(response.tools_awaiting_external_execution) == 1

        response.tools_awaiting_external_execution[
            0
        ].result = "Delivered yesterday"
        span_exporter.clear()
        result = agent.continue_run(run_response=response)

    assert result.content == "Your order was delivered yesterday."
    assert not result.is_paused
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent support"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID]
        == "order-session"
    )
    messages = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES, "[]")
    )
    assert messages == [
        {
            "role": "tool",
            "name": None,
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call_123",
                    "response": "Delivered yesterday",
                }
            ],
        }
    ]


def test_acontinue_run_captures_external_tool_result(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test Agent.acontinue_run captures resumed tool results from run_response."""
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse, ToolExecution
    from agno.tools import tool

    if not hasattr(agno.agent, "_run") or not hasattr(Agent, "acontinue_run"):
        pytest.skip(
            "Agent.acontinue_run is not supported in this version of agno"
        )
    if not hasattr(ToolExecution, "external_execution_required"):
        pytest.skip(
            "ToolExecution.external_execution_required is not supported in this version of agno"
        )

    @tool(external_execution=True)
    def lookup_order(order_id: str) -> str:
        """Look up the delivery status of an order."""
        raise AssertionError("The caller executes this tool outside Agno")

    agent = Agent(
        name="support",
        model=MockModel(id="mock-model"),
        tools=[lookup_order],
        session_id="order-session",
    )
    paused_response = ModelResponse(
        tool_executions=[
            ToolExecution(
                tool_call_id="call_123",
                tool_name="lookup_order",
                tool_args={"order_id": "123"},
                external_execution_required=True,
            )
        ],
    )

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse",
            side_effect=[
                paused_response,
                ModelResponse(content="Your order was delivered yesterday."),
            ],
        ):
            response = await agent.arun("Check order 123.")
            assert response.is_paused
            assert len(response.tools_awaiting_external_execution) == 1

            response.tools_awaiting_external_execution[
                0
            ].result = "Delivered yesterday"
            span_exporter.clear()
            result = await agent.acontinue_run(run_response=response)

        assert result.content == "Your order was delivered yesterday."
        assert not result.is_paused

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent support"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID]
        == "order-session"
    )
    messages = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES, "[]")
    )
    assert messages == [
        {
            "role": "tool",
            "name": None,
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call_123",
                    "response": "Delivered yesterday",
                }
            ],
        }
    ]


def test_second_continue_run_captures_only_new_tool_result(
    instrument_agno_content_capture, span_exporter
) -> None:
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse, ToolExecution
    from agno.tools import tool

    if not hasattr(agno.agent, "_run") or not hasattr(Agent, "continue_run"):
        pytest.skip(
            "Agent.continue_run is not supported in this version of agno"
        )
    if not hasattr(ToolExecution, "external_execution_required"):
        pytest.skip(
            "ToolExecution.external_execution_required is not supported in this version of agno"
        )

    @tool(external_execution=True)
    def lookup_order(order_id: str) -> str:
        """Look up the delivery status of an order."""
        raise AssertionError("external")

    @tool(external_execution=True)
    def refund_order(order_id: str) -> str:
        """Refund an order."""
        raise AssertionError("external")

    agent = Agent(
        name="support",
        model=MockModel(id="mock-model"),
        tools=[lookup_order, refund_order],
        session_id="order-session",
    )

    def paused(call_id: str, name: str) -> ModelResponse:
        return ModelResponse(
            tool_executions=[
                ToolExecution(
                    tool_call_id=call_id,
                    tool_name=name,
                    tool_args={"order_id": "123"},
                    external_execution_required=True,
                )
            ],
        )

    with patch(
        "agno.models.base.Model.response",
        side_effect=[
            paused("call_1", "lookup_order"),
            paused("call_2", "refund_order"),
            ModelResponse(content="Refunded."),
        ],
    ):
        response = agent.run("Refund order 123 if not delivered.")
        response.tools_awaiting_external_execution[0].result = "Not delivered"
        response = agent.continue_run(run_response=response)
        response.tools_awaiting_external_execution[0].result = "Refund issued"
        span_exporter.clear()
        agent.continue_run(run_response=response)

    span = span_exporter.get_finished_spans()[-1]
    messages = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert [m["parts"][0]["id"] for m in messages] == ["call_2"]


def test_continue_run_prefers_run_response_session_id(
    instrument_agno, span_exporter
) -> None:
    """Test that continue_run prefers run_response.session_id over the session_id argument."""
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse

    if not hasattr(agno.agent, "_run") or not hasattr(Agent, "continue_run"):
        pytest.skip(
            "Agent.continue_run is not supported in this version of agno"
        )

    agent = Agent(
        name="test-session-pref-agent",
        model=MockModel(id="mock-model"),
        session_id="agent-session",
    )
    mock_run_output = ModelResponse(content="Initial")
    mock_cont_output = ModelResponse(content="Continued")

    with patch(
        "agno.models.base.Model.response", return_value=mock_run_output
    ):
        run_res = agent.run("hello")
        assert run_res is not None

    run_res.session_id = "response-session"

    with patch(
        "agno.models.base.Model.response", return_value=mock_cont_output
    ):
        span_exporter.clear()
        cont_res = agent.continue_run(
            run_response=run_res, session_id="arg-session"
        )
        assert cont_res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "response-session"
    )


def test_extract_session_id_unit() -> None:
    """Unit tests for extract_session_id."""
    from opentelemetry.instrumentation.genai.agno.utils import (
        extract_session_id,
    )

    # Arguments priority
    assert (
        extract_session_id(kwargs={"session_id": "kw-session"}) == "kw-session"
    )
    assert extract_session_id(kwargs={"session_id": 0}) == "0"
    args = ("a", "b", "c", "d", "pos-session")
    assert extract_session_id(args=args) == "pos-session"
    assert extract_session_id(args=("a", "b", "c", "d", 0)) == "0"

    # Instance attribute
    class MockInstance:
        session_id = "inst-session"

    assert extract_session_id(instance=MockInstance()) == "inst-session"

    # Response attribute
    class MockResponse:
        session_id = "resp-session"

    assert extract_session_id(run_response=MockResponse()) == "resp-session"

    # Continue run prioritizes response over kwargs / instance
    assert (
        extract_session_id(
            instance=MockInstance(),
            kwargs={"session_id": "kw-session"},
            run_response=MockResponse(),
            is_continue=True,
        )
        == "resp-session"
    )

    # Continue run with run_response with None session_id falls back to instance, not kwargs
    class MockResponseNoneSession:
        session_id = None

    assert (
        extract_session_id(
            instance=MockInstance(),
            kwargs={"session_id": "kw-session"},
            run_response=MockResponseNoneSession(),
            is_continue=True,
        )
        == "inst-session"
    )

    # Continue run without run_response (e.g. continuing from run_id) uses kwargs
    assert (
        extract_session_id(
            instance=MockInstance(),
            kwargs={"session_id": "kw-session"},
            run_response=None,
            is_continue=True,
        )
        == "kw-session"
    )

    # When not continue, kwargs take precedence over response
    assert (
        extract_session_id(
            instance=MockInstance(),
            kwargs={"session_id": "kw-session"},
            run_response=MockResponse(),
            is_continue=False,
        )
        == "kw-session"
    )

    # Empty
    assert extract_session_id() is None


def test_workflow_nested_hitl_captures_only_active_step_requirement(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that Workflow.continue_run in nested HITL captures only the active (last) step requirement."""
    from agno.workflow.workflow import Workflow

    if not hasattr(Workflow, "continue_run"):
        pytest.skip(
            "Workflow.continue_run is not supported in this version of agno"
        )

    from agno.run.workflow import RunStatus, WorkflowRunOutput
    from agno.workflow.types import StepRequirement

    workflow = Workflow(
        name="test-nested-hitl-workflow",
        steps=[],
        session_id="wf-session-nested",
    )
    req1 = StepRequirement(
        step_id="step_1",
        step_name="first_step",
        requires_confirmation=True,
        confirmed=True,
    )
    req2 = StepRequirement(
        step_id="step_2",
        step_name="second_step",
        requires_confirmation=True,
        confirmed=True,
    )
    mock_wf_output = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="wf-session-nested",
        status=RunStatus.paused,
        paused_step_index=1,
        content="Workflow continued",
        step_requirements=[req1, req2],
    )

    with (
        patch.object(Workflow, "get_session", return_value=MagicMock()),
        patch.object(
            Workflow, "_continue_execute", return_value=mock_wf_output
        ),
    ):
        res = workflow.continue_run(run_response=mock_wf_output)
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Step requirements are not LLM tool calls and must not be captured as tool messages
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes


def test_workflow_nested_hitl_explicit_step_requirements(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that Workflow.continue_run does not capture explicit step requirements as tools."""
    from agno.workflow.workflow import Workflow

    if not hasattr(Workflow, "continue_run"):
        pytest.skip(
            "Workflow.continue_run is not supported in this version of agno"
        )

    from agno.run.workflow import RunStatus, WorkflowRunOutput
    from agno.workflow.types import StepRequirement

    workflow = Workflow(
        name="test-nested-hitl-workflow",
        steps=[],
        session_id="wf-session-nested",
    )
    req1 = StepRequirement(
        step_id="step_1",
        step_name="first_step",
        requires_confirmation=True,
        confirmed=True,
    )
    req2 = StepRequirement(
        step_id="step_2",
        step_name="second_step",
        requires_confirmation=True,
        confirmed=True,
    )
    mock_wf_output = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="wf-session-nested",
        status=RunStatus.paused,
        paused_step_index=1,
        content="Workflow continued",
    )

    with (
        patch.object(Workflow, "get_session", return_value=MagicMock()),
        patch.object(
            Workflow, "_continue_execute", return_value=mock_wf_output
        ),
    ):
        res = workflow.continue_run(
            run_response=mock_wf_output,
            step_requirements=[req1, req2],
        )
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Step requirements are not LLM tool calls and must not be captured as tool messages
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes


def test_workflow_nested_hitl_acontinue_run(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that Workflow.acontinue_run does not capture step requirements as tool calls."""
    from agno.workflow.workflow import Workflow

    if not hasattr(Workflow, "acontinue_run"):
        pytest.skip(
            "Workflow.acontinue_run is not supported in this version of agno"
        )

    from unittest.mock import AsyncMock

    from agno.run.workflow import RunStatus, WorkflowRunOutput
    from agno.workflow.types import StepRequirement

    workflow = Workflow(
        name="test-nested-hitl-workflow-async",
        steps=[],
        session_id="wf-session-nested-async",
    )
    req1 = StepRequirement(
        step_id="step_1",
        step_name="first_step",
        requires_confirmation=True,
        confirmed=True,
    )
    req2 = StepRequirement(
        step_id="step_2",
        step_name="second_step",
        requires_confirmation=True,
        confirmed=True,
    )
    mock_wf_output = WorkflowRunOutput(
        workflow_id="wf-1",
        session_id="wf-session-nested-async",
        status=RunStatus.paused,
        paused_step_index=1,
        content="Workflow async continued",
        step_requirements=[req1, req2],
    )

    mock_session = MagicMock()

    async def _run() -> None:
        with (
            patch.object(
                Workflow,
                "aget_session",
                new_callable=AsyncMock,
                return_value=mock_session,
            ),
            patch.object(
                Workflow, "_acontinue_execute", new_callable=AsyncMock
            ) as mock_exec,
        ):
            mock_exec.return_value = mock_wf_output
            res = await workflow.acontinue_run(run_response=mock_wf_output)
            assert res is not None

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Step requirements are not LLM tool calls and must not be captured as tool messages
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes


def test_agent_continue_run_with_user_input_schema(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that continue_run extracts answered fields from user_input_schema."""
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse, ToolExecution

    try:
        from agno.run.requirement import RunRequirement, UserInputField
    except ImportError:
        pytest.skip("UserInputField is not supported in this version of agno")

    agent = Agent(
        name="test-user-input-agent", model=MockModel(id="mock-model")
    )
    field = UserInputField(
        name="email", field_type="str", value="test@example.com"
    )
    tool_exec = ToolExecution(
        tool_call_id="call_form",
        tool_name="collect_info",
        user_input_schema=[field],
    )
    target_mod = getattr(
        agno.agent, "_run", getattr(agno.agent, "agent", None)
    )
    if not hasattr(target_mod, "continue_run_dispatch"):
        pytest.skip(
            "continue_run_dispatch is not supported in this version of agno"
        )
    target = (
        "agno.agent._run.continue_run_dispatch"
        if hasattr(agno.agent, "_run")
        else "agno.agent.agent.continue_run_dispatch"
    )
    with patch(target, return_value=ModelResponse(content="Got info")):
        agent.continue_run(
            run_id="mock-id",
            requirements=[RunRequirement(tool_execution=tool_exec)],
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    messages = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(messages) == 1
    assert messages[0]["parts"][0]["response"] == {"email": "test@example.com"}


def test_agent_continue_run_with_user_feedback_schema(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that continue_run extracts answered options from user_feedback_schema."""
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse, ToolExecution

    try:
        from agno.run.requirement import RunRequirement, UserFeedbackQuestion
    except ImportError:
        pytest.skip(
            "UserFeedbackQuestion is not supported in this version of agno"
        )

    agent = Agent(name="test-feedback-agent", model=MockModel(id="mock-model"))
    question = UserFeedbackQuestion(
        question="Satisfied?", selected_options=["Yes"]
    )
    tool_exec = ToolExecution(
        tool_call_id="call_feedback",
        tool_name="request_feedback",
        user_feedback_schema=[question],
    )
    target_mod = getattr(
        agno.agent, "_run", getattr(agno.agent, "agent", None)
    )
    if not hasattr(target_mod, "continue_run_dispatch"):
        pytest.skip(
            "continue_run_dispatch is not supported in this version of agno"
        )
    target = (
        "agno.agent._run.continue_run_dispatch"
        if hasattr(agno.agent, "_run")
        else "agno.agent.agent.continue_run_dispatch"
    )
    with patch(
        target,
        return_value=ModelResponse(content="Thanks for feedback"),
    ):
        agent.continue_run(
            run_id="mock-id",
            requirements=[RunRequirement(tool_execution=tool_exec)],
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    messages = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(messages) == 1
    assert messages[0]["parts"][0]["response"] == {"Satisfied?": ["Yes"]}


def test_agent_continue_run_with_confirmation_note(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that continue_run captures confirmation_note alongside confirmed flag."""
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse, ToolExecution

    try:
        from agno.run.requirement import RunRequirement
    except ImportError:
        pytest.skip("RunRequirement is not supported in this version of agno")

    agent = Agent(
        name="test-confirm-note-agent", model=MockModel(id="mock-model")
    )
    tool_exec = ToolExecution(
        tool_call_id="call_reject",
        tool_name="drop_db",
        confirmed=False,
    )
    tool_exec.confirmation_note = "Operation forbidden"
    if not hasattr(agno.agent, "_run"):
        pytest.skip(
            "continue_run dispatch is not supported in this version of agno"
        )
    with patch(
        "agno.agent._run.continue_run_dispatch",
        return_value=ModelResponse(content="Cancelled"),
    ):
        agent.continue_run(
            run_id="mock-id",
            requirements=[RunRequirement(tool_execution=tool_exec)],
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    messages = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(messages) == 1
    assert messages[0]["parts"][0]["response"] == {
        "confirmed": False,
        "confirmation_note": "Operation forbidden",
    }


def test_agent_continue_run_unresolved_tool_has_none_response(
    instrument_agno_content_capture, span_exporter
) -> None:
    """Test that continue_run with an unresolved tool leaves response as None rather than dumping to_dict."""
    import agno.agent
    from agno.agent import Agent
    from agno.models.response import ModelResponse, ToolExecution

    try:
        from agno.run.requirement import RunRequirement
    except ImportError:
        pytest.skip("RunRequirement is not supported in this version of agno")

    agent = Agent(name="test-none-agent", model=MockModel(id="mock-model"))
    tool_exec = ToolExecution(
        tool_call_id="call_empty",
        tool_name="empty_tool",
    )
    if not hasattr(agno.agent, "_run"):
        pytest.skip(
            "continue_run dispatch is not supported in this version of agno"
        )
    with patch(
        "agno.agent._run.continue_run_dispatch",
        return_value=ModelResponse(content="Done"),
    ):
        agent.continue_run(
            run_id="mock-id",
            requirements=[RunRequirement(tool_execution=tool_exec)],
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    messages = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(messages) == 1
    assert messages[0]["parts"][0]["response"] is None
