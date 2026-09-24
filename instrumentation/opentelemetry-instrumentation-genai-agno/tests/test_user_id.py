# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for user ID capture in Agno."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.workflow.workflow import Workflow
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_CONVERSATION_ID,
)
from opentelemetry.semconv._incubating.attributes.user_attributes import (
    USER_ID,
)


def test_agent_run_user_id_kwargs(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run with user_id in kwargs records user.id attribute."""
    agent = Agent(name="test-user-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Response")

    with (
        patch.object(agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        agent.run("hello", user_id="user-kw-123")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(USER_ID) == "user-kw-123"


def test_agent_run_user_id_instance(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent initialized with user_id records user.id attribute."""
    agent = Agent(
        name="test-user-inst-agent",
        user_id="user-inst-456",
        model=MockModel(id="mock-model"),
    )
    mock_output = ModelResponse(content="Response")

    with (
        patch.object(agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        agent.run("hello")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(USER_ID) == "user-inst-456"


def test_agent_arun_user_id(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun records user.id attribute."""
    agent = Agent(
        name="test-user-async-agent", model=MockModel(id="mock-model")
    )
    mock_output = ModelResponse(content="Response")

    async def _test() -> None:
        with (
            patch.object(agent, "arun", wraps=agent.arun),
            patch(
                "agno.models.base.Model.aresponse", return_value=mock_output
            ),
        ):
            await agent.arun("hello", user_id="user-async-789")

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(USER_ID) == "user-async-789"


def test_workflow_run_user_id(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.run with user_id records user.id attribute."""

    def step1(step_input: str) -> str:
        return f"processed {step_input}"

    workflow = Workflow(
        name="test-user-workflow",
        user_id="wf-user-1",
        steps=[step1],
    )
    workflow.run("input data")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(USER_ID) == "wf-user-1"


def test_workflow_arun_user_id_kwargs(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Workflow.arun with user_id in kwargs records user.id attribute."""

    def step1(step_input: str) -> str:
        return f"processed {step_input}"

    workflow = Workflow(name="test-user-async-wf", steps=[step1])

    async def _test() -> None:
        coro = workflow.arun("input data", user_id="wf-async-user-99")
        if asyncio.iscoroutine(coro):
            await coro

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(USER_ID) == "wf-async-user-99"


def test_user_and_session_id_non_string_types(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that non-string user_id and session_id (e.g. ints) are converted to strings."""
    agent = Agent(name="test-user-agent-int", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Response")

    with patch("agno.models.base.Model.response", return_value=mock_output):
        agent.run("hello", user_id=12345, session_id=67890)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(USER_ID) == "12345"
    assert spans[0].attributes.get(GEN_AI_CONVERSATION_ID) == "67890"

    from opentelemetry.instrumentation.genai.agno.utils import (
        extract_session_id,
        extract_user_id,
    )

    # Test positional non-string user_id (arg index 2) and session_id (arg index 4)
    args = ("input", None, 9999, None, 8888)
    assert extract_user_id(args=args) == "9999"
    assert extract_session_id(args=args) == "8888"


def test_user_and_session_id_zero_value(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that numeric zero user_id and session_id are treated as present."""
    agent = Agent(
        name="test-user-agent-zero", model=MockModel(id="mock-model")
    )
    mock_output = ModelResponse(content="Response")

    with patch("agno.models.base.Model.response", return_value=mock_output):
        agent.run("hello", user_id=0, session_id=0)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(USER_ID) == "0"
    assert spans[0].attributes.get(GEN_AI_CONVERSATION_ID) == "0"

    from opentelemetry.instrumentation.genai.agno.utils import (
        extract_session_id,
        extract_user_id,
    )

    assert extract_user_id(kwargs={"user_id": 0}) == "0"
    assert extract_session_id(kwargs={"session_id": 0}) == "0"

    args = ("input", None, 0, None, 0)
    assert extract_user_id(args=args) == "0"
    assert extract_session_id(args=args) == "0"
