# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Span content-capture tests for agent and LLM layers."""

from __future__ import annotations

import asyncio
import json

import pytest
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ._test_helpers import (
    assert_no_removed_telemetry,
    find_spans_by_operation,
)


class TestSpanContentCapture:
    @pytest.mark.vcr()
    def test_span_content_with_span_only(
        self, span_exporter, instrument_with_content, request
    ):
        """Content captured in SPAN_ONLY mode for both agent and LLM layers."""
        toolkit = Toolkit()
        agent = ReActAgent(
            name="ContentTest",
            sys_prompt="You are a test assistant.",
            model=DashScopeChatModel(
                api_key=request.config.option.api_key,
                model_name="qwen-max",
                stream=True,
            ),
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )
        msg = Msg("user", "Hello, please say 'Hi' to me", "user")

        response = asyncio.run(agent(msg))
        assert response is not None

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)

        agent_spans = find_spans_by_operation(spans, "invoke_agent")
        assert len(agent_spans) > 0
        agent_attrs = dict(agent_spans[0].attributes)
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in agent_attrs
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in agent_attrs

        chat_spans = find_spans_by_operation(spans, "chat")
        assert len(chat_spans) > 0
        chat_attrs = dict(chat_spans[0].attributes)
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_attrs
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_attrs

        llm_input = chat_attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
        if isinstance(llm_input, str):
            input_msgs = json.loads(llm_input)
            assert isinstance(input_msgs, list)
            assert len(input_msgs) > 0

        llm_output = chat_attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        if isinstance(llm_output, str):
            output_msgs = json.loads(llm_output)
            assert isinstance(output_msgs, list)
            assert len(output_msgs) > 0

        chat_span = chat_spans[0]
        assert chat_span.parent is not None
        assert chat_span.context.trace_id == agent_spans[0].context.trace_id

    @pytest.mark.vcr()
    def test_span_content_with_span_and_event(
        self,
        span_exporter,
        log_exporter,
        instrument_with_content_and_events,
        request,
    ):
        """Content captured on span in SPAN_AND_EVENT mode."""
        toolkit = Toolkit()
        agent = ReActAgent(
            name="EventTest",
            sys_prompt="You are a test assistant.",
            model=DashScopeChatModel(
                api_key=request.config.option.api_key,
                model_name="qwen-max",
                stream=True,
            ),
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )
        msg = Msg("user", "What is 2+2?", "user")

        response = asyncio.run(agent(msg))
        assert response is not None

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)

        chat_spans = find_spans_by_operation(spans, "chat")
        assert len(chat_spans) > 0
        chat_attrs = dict(chat_spans[0].attributes)
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in chat_attrs
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in chat_attrs

    @pytest.mark.vcr()
    def test_span_content_disabled(
        self, span_exporter, instrument_no_content, request
    ):
        """No content captured on either layer when capture is disabled."""
        toolkit = Toolkit()
        agent = ReActAgent(
            name="NoContentTest",
            sys_prompt="You are a test assistant.",
            model=DashScopeChatModel(
                api_key=request.config.option.api_key,
                model_name="qwen-max",
                stream=True,
            ),
            formatter=DashScopeChatFormatter(),
            toolkit=toolkit,
        )
        msg = Msg("user", "Say hello", "user")

        response = asyncio.run(agent(msg))
        assert response is not None

        spans = span_exporter.get_finished_spans()
        assert_no_removed_telemetry(spans)

        for agent_span in find_spans_by_operation(spans, "invoke_agent"):
            attrs = dict(agent_span.attributes)
            assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs
            assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs
            assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs

        for chat_span in find_spans_by_operation(spans, "chat"):
            attrs = dict(chat_span.attributes)
            assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs
            assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs
            assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs
            assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs
