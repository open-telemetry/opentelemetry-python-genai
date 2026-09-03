# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno streaming instrumentation."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

import pytest
from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.team import Team
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace.status import StatusCode


def _patch_agent_stream(side_effect):
    try:
        import agno.agent._run  # noqa: F401

        return patch("agno.agent._run.run_dispatch", side_effect=side_effect)
    except (ImportError, AttributeError):
        return patch.object(Agent, "_run_stream", side_effect=side_effect)


def _patch_agent_astream(side_effect):
    try:
        import agno.agent._run  # noqa: F401

        return patch("agno.agent._run.arun_dispatch", side_effect=side_effect)
    except (ImportError, AttributeError):
        return patch.object(Agent, "_arun_stream", side_effect=side_effect)


def test_agent_run_stream_spans(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Agent.run with stream=True emits an invoke_agent span."""
    agent = Agent(name="test-stream-agent", model=MockModel(id="mock-model"))

    def fake_stream(*args, **kwargs):
        yield ModelResponse(content="chunk 1 ")
        yield ModelResponse(content="chunk 2")

    with patch(
        "agno.models.base.Model.response_stream", side_effect=fake_stream
    ):
        stream = agent.run("hello stream", stream=True)
        assert len(span_exporter.get_finished_spans()) == 0
        chunks = list(stream)
        assert len(chunks) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-stream-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-stream-agent"
    )
    assert span.status.status_code != StatusCode.ERROR

    output_messages = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert "chunk 1 chunk 2" in output_messages


def test_agent_run_stream_content_capture_disabled(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run with stream=True omits output content when content capture is off."""
    agent = Agent(
        name="test-stream-agent-no-content", model=MockModel(id="mock-model")
    )

    def fake_stream(*args, **kwargs):
        yield ModelResponse(content="chunk 1 ")
        yield ModelResponse(content="chunk 2")

    with patch(
        "agno.models.base.Model.response_stream", side_effect=fake_stream
    ):
        chunks = list(agent.run("hello stream", stream=True))
        assert len(chunks) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-stream-agent-no-content"
    assert span.status.status_code != StatusCode.ERROR
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_agent_run_stream_error_mid_iteration(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that a stream-side error raised mid-iteration re-raises and marks span error."""
    agent = Agent(
        name="test-stream-agent-error", model=MockModel(id="mock-model")
    )

    def failing_stream(*args, **kwargs):
        yield "chunk 1"
        raise ConnectionError("stream connection dropped")

    with (
        _patch_agent_stream(failing_stream),
        pytest.raises(ConnectionError, match="stream connection dropped"),
    ):
        list(agent.run("hello stream", stream=True))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-stream-agent-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ConnectionError"


def test_agent_run_stream_caller_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error raised by caller inside 'with stream:' re-raises and marks span error."""
    agent = Agent(
        name="test-stream-agent-caller-error", model=MockModel(id="mock-model")
    )

    def fake_stream(*args, **kwargs):
        yield ModelResponse(content="chunk 1")
        yield ModelResponse(content="chunk 2")

    with patch(
        "agno.models.base.Model.response_stream", side_effect=fake_stream
    ):
        stream = agent.run("hello stream", stream=True)
        with (
            pytest.raises(RuntimeError, match="caller aborted stream"),
            stream,
        ):
            for _ in stream:
                raise RuntimeError("caller aborted stream")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-stream-agent-caller-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "RuntimeError"


def test_agent_arun_stream_spans(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Agent.arun with stream=True returns an AsyncIterator and emits invoke_agent span."""
    agent = Agent(
        name="test-async-stream-agent", model=MockModel(id="mock-model")
    )

    async def fake_astream(*args, **kwargs):
        yield ModelResponse(content="async chunk 1 ")
        yield ModelResponse(content="async chunk 2")

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse_stream", side_effect=fake_astream
        ):
            stream = agent.arun("hello async stream", stream=True)
            assert len(span_exporter.get_finished_spans()) == 0
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert len(chunks) == 2

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-stream-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-async-stream-agent"
    )
    assert span.status.status_code != StatusCode.ERROR

    output_messages = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert "async chunk 1 async chunk 2" in output_messages


def test_agent_arun_stream_content_capture_disabled(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun with stream=True omits output content when content capture is off."""
    agent = Agent(
        name="test-async-stream-agent-no-content",
        model=MockModel(id="mock-model"),
    )

    async def fake_astream(*args, **kwargs):
        yield ModelResponse(content="async chunk 1 ")
        yield ModelResponse(content="async chunk 2")

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse_stream", side_effect=fake_astream
        ):
            async for _ in agent.arun("hello async stream", stream=True):
                pass

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-stream-agent-no-content"
    assert span.status.status_code != StatusCode.ERROR
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_agent_arun_stream_error_mid_iteration(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an async stream-side error mid-iteration re-raises and marks span error."""
    agent = Agent(
        name="test-async-stream-error", model=MockModel(id="mock-model")
    )

    async def failing_astream(*args, **kwargs):
        yield "chunk 1"
        raise ConnectionError("async stream dropped")

    async def _run() -> None:
        with (
            _patch_agent_astream(failing_astream),
            pytest.raises(ConnectionError, match="async stream dropped"),
        ):
            async for _ in agent.arun("hello async stream", stream=True):
                pass

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-stream-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ConnectionError"


def test_agent_arun_stream_caller_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error inside 'async with stream:' re-raises and marks span error."""
    agent = Agent(
        name="test-async-stream-caller-error",
        model=MockModel(id="mock-model"),
    )

    async def fake_astream(*args, **kwargs):
        yield ModelResponse(content="chunk 1")
        yield ModelResponse(content="chunk 2")

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse_stream", side_effect=fake_astream
        ):
            stream = agent.arun("hello async stream", stream=True)
            with (
                pytest.raises(RuntimeError, match="async caller abort"),
            ):
                async with stream:
                    async for _ in stream:
                        raise RuntimeError("async caller abort")

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-stream-caller-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "RuntimeError"


def test_team_run_stream_spans(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Team.run with stream=True emits an invoke_agent span."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-stream-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )

    def fake_stream(*args, **kwargs):
        yield ModelResponse(content="team chunk 1 ")
        yield ModelResponse(content="team chunk 2")

    with patch(
        "agno.models.base.Model.response_stream", side_effect=fake_stream
    ):
        stream = team.run("hello team stream", stream=True)
        assert len(span_exporter.get_finished_spans()) == 0
        chunks = list(stream)
        assert len(chunks) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-stream-team"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-stream-team"
    )
    assert span.status.status_code != StatusCode.ERROR

    output_messages = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert "team chunk 1 team chunk 2" in output_messages


def test_team_arun_stream_spans(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Team.arun with stream=True emits an invoke_agent span."""
    member = Agent(name="member-agent", model=MockModel(id="mock-model"))
    team = Team(
        name="test-async-stream-team",
        members=[member],
        model=MockModel(id="mock-model"),
    )

    async def fake_astream(*args, **kwargs):
        yield ModelResponse(content="async team chunk 1 ")
        yield ModelResponse(content="async team chunk 2")

    async def _run() -> None:
        with patch(
            "agno.models.base.Model.aresponse_stream", side_effect=fake_astream
        ):
            stream = team.arun("hello async team stream", stream=True)
            assert len(span_exporter.get_finished_spans()) == 0
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert len(chunks) == 2

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-stream-team"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-async-stream-team"
    )
    assert span.status.status_code != StatusCode.ERROR

    output_messages = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert "async team chunk 1 async team chunk 2" in output_messages


def test_workflow_run_stream_spans(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Workflow.run with stream=True emits an invoke_workflow span."""
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    def step1(step_input):
        return "step1 stream output"

    workflow = Workflow(name="test-stream-workflow", steps=[step1])

    stream = workflow.run("test input", stream=True)
    assert len(span_exporter.get_finished_spans()) == 0
    chunks = list(stream)
    assert len(chunks) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-stream-workflow"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_WORKFLOW_NAME)
        == "test-stream-workflow"
    )
    assert span.status.status_code != StatusCode.ERROR

    output_messages = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert "step1 stream output" in output_messages


def test_workflow_run_stream_caller_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that caller error inside 'with stream:' for Workflow re-raises and marks error."""
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    def step1(step_input):
        return "step1 output"

    workflow = Workflow(name="test-stream-wf-caller-error", steps=[step1])
    stream = workflow.run("test input", stream=True)

    with (
        pytest.raises(RuntimeError, match="wf caller abort"),
        stream,
    ):
        for _ in stream:
            raise RuntimeError("wf caller abort")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-stream-wf-caller-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "RuntimeError"


def test_workflow_run_stream_error_mid_iteration(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that stream-side error mid-iteration for Workflow re-raises and marks error."""
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    def failing_stream(*args, **kwargs):
        yield "wf chunk 1"
        raise ConnectionError("wf step connection dropped")

    workflow = Workflow(name="test-stream-wf-error", steps=[])

    with (
        patch.object(Workflow, "_execute_stream", side_effect=failing_stream),
        pytest.raises(ConnectionError, match="wf step connection dropped"),
    ):
        list(workflow.run("test input", stream=True))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-stream-wf-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ConnectionError"


def test_workflow_arun_stream_spans(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Workflow.arun with stream=True emits an invoke_workflow span."""
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    def step1(step_input):
        return "async step1 output"

    workflow = Workflow(name="test-async-stream-workflow", steps=[step1])

    async def _run() -> None:
        stream = workflow.arun("test input", stream=True)
        if inspect.isawaitable(stream):
            stream = await stream
        assert len(span_exporter.get_finished_spans()) == 0
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert len(chunks) > 0

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-async-stream-workflow"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_WORKFLOW_NAME)
        == "test-async-stream-workflow"
    )
    assert span.status.status_code != StatusCode.ERROR

    output_messages = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert "async step1 output" in output_messages


def test_workflow_arun_stream_caller_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that caller error inside 'async with stream:' for Workflow re-raises and marks error."""
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    def step1(step_input):
        return "async step1 output"

    workflow = Workflow(
        name="test-async-stream-wf-caller-error", steps=[step1]
    )

    async def _run() -> None:
        stream = workflow.arun("test input", stream=True)
        if inspect.isawaitable(stream):
            stream = await stream
        with pytest.raises(RuntimeError, match="async wf caller abort"):
            async with stream:
                async for _ in stream:
                    raise RuntimeError("async wf caller abort")

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-async-stream-wf-caller-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "RuntimeError"


def test_workflow_arun_stream_error_mid_iteration(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that async stream-side error for Workflow re-raises and marks error."""
    pytest.importorskip("agno.workflow.workflow")
    from agno.workflow.workflow import Workflow

    async def failing_astream(*args, **kwargs):
        yield "async wf chunk 1"
        raise ConnectionError("async wf connection dropped")

    workflow = Workflow(name="test-async-stream-wf-error", steps=[])

    async def _run() -> None:
        with (
            patch.object(
                Workflow, "_aexecute_stream", side_effect=failing_astream
            ),
            pytest.raises(
                ConnectionError, match="async wf connection dropped"
            ),
        ):
            stream = workflow.arun("test input", stream=True)
            if inspect.isawaitable(stream):
                stream = await stream
            async for _ in stream:
                pass

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow test-async-stream-wf-error"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ConnectionError"


def test_agent_run_stream_structured_pydantic_output(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Agent.run with stream=True serializes Pydantic model chunk content as JSON."""
    import json
    from dataclasses import dataclass

    from pydantic import BaseModel

    class MovieOutput(BaseModel):
        title: str
        year: int

    @dataclass
    class FakeCompletedChunk:
        content: MovieOutput
        event: str = "RunCompletedEvent"

    agent = Agent(
        name="test-stream-pydantic-agent", model=MockModel(id="mock-model")
    )

    def fake_stream(*args, **kwargs):
        yield "chunk delta"
        yield FakeCompletedChunk(
            content=MovieOutput(title="Inception", year=2010)
        )

    with _patch_agent_stream(fake_stream):
        chunks = list(agent.run("tell me about inception", stream=True))
        assert len(chunks) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    output_messages = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    parsed_messages = json.loads(output_messages)
    content_str = parsed_messages[0]["parts"][0]["content"]
    assert json.loads(content_str) == {
        "title": "Inception",
        "year": 2010,
    }
