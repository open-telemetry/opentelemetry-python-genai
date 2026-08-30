# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode


@pytest.mark.asyncio
async def test_agent_tool_and_inference_instrumentation_compose(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    openai_llm,
    vcr,
) -> None:
    # The cross-package composition is exercised in the latest matrix. The
    # oldest matrix intentionally retains the util-genai release lower bound.
    OpenAIInstrumentor = pytest.importorskip(
        "opentelemetry.instrumentation.genai.openai"
    ).OpenAIInstrumentor

    async def weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    def function_response(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="It is sunny.")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="weather-call",
                    tool_name="weather",
                    tool_kwargs={"city": "Paris"},
                )
            ],
        )

    function_agent = FunctionAgent(
        name="weather-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=function_response,
        ),
        tools=[FunctionTool.from_defaults(async_fn=weather)],
        streaming=False,
    )

    def react_response(messages, **kwargs):
        return ChatMessage(
            role="assistant",
            content="Thought: I know the answer.\nAnswer: Four.",
        )

    react_agent = ReActAgent(
        name="react-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=react_response,
        ),
        streaming=False,
    )
    provider_agent = FunctionAgent(
        name="provider-agent",
        llm=openai_llm,
        streaming=False,
    )
    provider_workflow = AgentWorkflow(agents=[provider_agent])

    providers = {
        "tracer_provider": tracer_provider,
        "logger_provider": logger_provider,
        "meter_provider": meter_provider,
    }
    with instrument(LlamaIndexInstrumentor(), **providers):
        with instrument(OpenAIInstrumentor(), **providers):
            await function_agent.run(user_msg="What is the weather in Paris?")
            await react_agent.run(user_msg="What is two plus two?")
            with vcr.use_cassette("inference.yaml"):
                await provider_workflow.run(user_msg="Hello!")

    spans = span_exporter.get_finished_spans()
    operations = [
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        for span in spans
    ]
    assert operations.count("invoke_agent") == 3
    assert operations.count("invoke_workflow") == 1
    assert operations.count("execute_tool") == 1
    assert operations.count("chat") == 1
    assert all(isinstance(operation, str) for operation in operations)

    spans_by_name = {span.name: span for span in spans}
    tool_span = spans_by_name["execute_tool weather"]
    function_span = spans_by_name["invoke_agent weather-agent"]
    inference_span = spans_by_name["chat gpt-4o-mini"]
    provider_span = spans_by_name["invoke_agent provider-agent"]
    workflow_span = spans_by_name["invoke_workflow AgentWorkflow"]

    assert tool_span.context.trace_id == function_span.context.trace_id
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == function_span.context.span_id
    assert inference_span.context.trace_id == provider_span.context.trace_id
    assert inference_span.parent is not None
    assert inference_span.parent.span_id == provider_span.context.span_id
    assert provider_span.context.trace_id == workflow_span.context.trace_id
    assert provider_span.parent is not None
    assert provider_span.parent.span_id == workflow_span.context.span_id


@pytest.mark.asyncio
async def test_agent_and_inference_provider_errors_compose(
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
    openai_llm_without_retries,
    vcr,
) -> None:
    OpenAIInstrumentor = pytest.importorskip(
        "opentelemetry.instrumentation.genai.openai"
    ).OpenAIInstrumentor
    RateLimitError = pytest.importorskip("openai").RateLimitError

    agent = FunctionAgent(
        name="rate-limited-agent",
        llm=openai_llm_without_retries,
        streaming=False,
    )
    providers = {
        "tracer_provider": tracer_provider,
        "logger_provider": logger_provider,
        "meter_provider": meter_provider,
    }
    with instrument(LlamaIndexInstrumentor(), **providers):
        with instrument(OpenAIInstrumentor(), **providers):
            with vcr.use_cassette("rate_limit.yaml"):
                with pytest.raises(RateLimitError):
                    await agent.run(user_msg="Hello!")

    spans_by_name = {
        span.name: span for span in span_exporter.get_finished_spans()
    }
    agent_span = spans_by_name["invoke_agent rate-limited-agent"]
    inference_span = spans_by_name["chat gpt-4o-mini"]
    assert agent_span.status.status_code == StatusCode.ERROR
    assert inference_span.status.status_code == StatusCode.ERROR
    assert (
        agent_span.attributes[ErrorAttributes.ERROR_TYPE]
        == "openai.RateLimitError"
    )
    assert (
        inference_span.attributes[ErrorAttributes.ERROR_TYPE]
        == "openai.RateLimitError"
    )
    assert inference_span.context.trace_id == agent_span.context.trace_id
    assert inference_span.parent is not None
    assert inference_span.parent.span_id == agent_span.context.span_id
