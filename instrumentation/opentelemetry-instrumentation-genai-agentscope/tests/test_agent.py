# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""AgentScope agent and tool instrumentation tests."""

from __future__ import annotations

import asyncio

import agentscope
import pytest
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, execute_shell_command

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ._test_helpers import (
    assert_no_removed_telemetry,
    find_spans_by_name_prefix,
)


async def _drain(response):
    try:
        is_stream = hasattr(response, "__aiter__")
    except (KeyError, AttributeError):
        is_stream = False
    if is_stream:
        result = []
        async for chunk in response:
            result.append(chunk)
        return result[-1] if result else response
    return response


async def _call_agent(agent, msg):
    return await _drain(await agent(msg))


class TestAgentBasic:
    @pytest.mark.vcr()
    def test_agent_simple_call(
        self, span_exporter, instrument_with_content, dashscope_model
    ):
        agentscope.init(project="test_agent_simple")
        agent = ReActAgent(
            name="TestAgent",
            sys_prompt="You are a helpful assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )
        msg = Msg("user", "Hello, how are you?", "user")

        response = asyncio.run(_call_agent(agent, msg))
        assert response is not None

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)
        assert len(spans) >= 1
        assert len(find_spans_by_name_prefix(spans, "chat ")) > 0

    @pytest.mark.vcr()
    def test_agent_with_tool(
        self, span_exporter, instrument_with_content, dashscope_model
    ):
        agentscope.init(project="test_agent_tool")
        toolkit = Toolkit()
        toolkit.register_tool_function(execute_shell_command)
        agent = ReActAgent(
            name="ToolAgent",
            sys_prompt="You are an assistant with tool access.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )
        msg = Msg("user", "compute 10+20 for me using shell", "user")

        async def run():
            try:
                return await _drain(await agent(msg))
            except Exception:
                return None

        asyncio.run(run())

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)
        assert len(spans) >= 1
        assert len(find_spans_by_name_prefix(spans, "chat ")) > 0

    @pytest.mark.vcr()
    def test_agent_multiple_turns(
        self, span_exporter, instrument_with_content, dashscope_model
    ):
        agentscope.init(project="test_agent_multi_turn")
        agent = ReActAgent(
            name="MultiTurnAgent",
            sys_prompt="You are a helpful assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )

        async def run(message: str):
            return await _drain(await agent(Msg("user", message, "user")))

        asyncio.run(run("Hello"))
        asyncio.run(run("What's 2+2?"))
        asyncio.run(run("Thank you"))

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)
        assert len(spans) >= 3


class TestAgentAttributes:
    @pytest.mark.vcr()
    def test_agent_span_attributes(
        self, span_exporter, instrument_with_content, dashscope_model
    ):
        agentscope.init(project="test_agent_attrs")
        agent = ReActAgent(
            name="AttributeAgent",
            sys_prompt="You are a test assistant.",
            model=dashscope_model,
            formatter=DashScopeChatFormatter(),
        )
        msg = Msg("user", "Simple test", "user")

        asyncio.run(_call_agent(agent, msg))

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)

        chat_spans = find_spans_by_name_prefix(spans, "chat ")
        assert len(chat_spans) > 0
        attrs = chat_spans[0].attributes
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs
        assert "gen_ai.provider.name" in attrs

        # invoke_agent spans carry only the standard operation-name attribute.
        agent_spans = find_spans_by_name_prefix(spans, "invoke_agent")
        for agent_span in agent_spans:
            assert (
                agent_span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
                == "invoke_agent"
            )
            assert "gen_ai.span.kind" not in agent_span.attributes


class TestAgentError:
    @pytest.mark.vcr()
    def test_agent_with_invalid_model(
        self, span_exporter, instrument_with_content
    ):
        agentscope.init(project="test_invalid_model")
        invalid_model = DashScopeChatModel(
            model_name="qwen-max",
            api_key="invalid_key_test",
        )
        agent = ReActAgent(
            name="InvalidAgent",
            sys_prompt="Test agent",
            model=invalid_model,
            formatter=DashScopeChatFormatter(),
        )
        msg = Msg("user", "Test", "user")

        async def run():
            try:
                return await _drain(await agent(msg))
            except Exception:
                return None

        asyncio.run(run())

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)
