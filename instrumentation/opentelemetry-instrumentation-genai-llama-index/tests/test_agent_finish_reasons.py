# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Sequence
from typing import Any

import llama_index.core.agent.workflow.multi_agent_workflow as multi_agent_workflow
import pytest
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow.errors import WorkflowRuntimeError

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode


@pytest.fixture(params=["NO_CONTENT", "SPAN_ONLY"])
def capture_content(
    request: pytest.FixtureRequest,
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> Iterator[bool]:
    with instrument(
        LlamaIndexInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture=request.param,
    ):
        yield request.param == "SPAN_ONLY"


def _llm(
    responses: Sequence[ChatMessage | Exception],
) -> MockFunctionCallingLLM:
    remaining = iter(responses)

    def respond(messages: list[ChatMessage], **kwargs: Any) -> ChatMessage:
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        return response

    return MockFunctionCallingLLM(
        is_chat_model=True, response_generator=respond
    )


def _answer(agent_type: type[FunctionAgent | ReActAgent]) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=(
            "Thought: I can answer.\nAnswer: complete"
            if agent_type is ReActAgent
            else "complete"
        ),
    )


def _tool_call(
    agent_type: type[FunctionAgent | ReActAgent],
    name: str,
    arguments: dict[str, str],
) -> ChatMessage:
    if agent_type is ReActAgent:
        return ChatMessage(
            role="assistant",
            content=(
                f"Thought: Use the tool.\nAction: {name}\n"
                f"Action Input: {json.dumps(arguments)}"
            ),
        )
    return ChatMessage(
        role="assistant",
        blocks=[
            ToolCallBlock(
                tool_call_id="tool-call",
                tool_name=name,
                tool_kwargs=arguments,
            )
        ],
    )


def _agent_span(
    exporter: InMemorySpanExporter, name: str = "member"
) -> ReadableSpan:
    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == f"invoke_agent {name}"
    ]
    assert len(spans) == 1
    return spans[0]


def _assert_finish_reason(
    span: ReadableSpan, expected: str, capture_content: bool
) -> None:
    attributes = dict(span.attributes or {})
    reasons = attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
    assert isinstance(reasons, tuple)
    assert reasons == (expected,)
    assert all(isinstance(reason, str) for reason in reasons)
    assert span.status.status_code == StatusCode.UNSET
    if capture_content:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attributes
        messages = json.loads(
            attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        )
        assert len(messages) == 1
    else:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [FunctionAgent, ReActAgent])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("use_tool", [False, True])
async def test_agent_finish_reasons_completion(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
    agent_type: type[FunctionAgent | ReActAgent],
    streaming: bool,
    use_tool: bool,
) -> None:
    responses = [_answer(agent_type)]
    if use_tool:
        responses.insert(0, _tool_call(agent_type, "echo", {"value": "hi"}))
    agent = agent_type(
        name="member",
        llm=_llm(responses),
        tools=[FunctionTool.from_defaults(lambda value: value, name="echo")],
        streaming=streaming,
    )

    result = await AgentWorkflow(agents=[agent]).run(user_msg="Answer")

    assert result.response.content == "complete"
    if use_tool:
        assert result.tool_calls
    _assert_finish_reason(_agent_span(span_exporter), "stop", capture_content)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [FunctionAgent, ReActAgent])
@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_finish_reasons_return_direct(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
    agent_type: type[FunctionAgent | ReActAgent],
    streaming: bool,
) -> None:
    agent = agent_type(
        name="member",
        llm=_llm([_tool_call(agent_type, "echo", {"value": "direct answer"})]),
        tools=[
            FunctionTool.from_defaults(
                lambda value: value, name="echo", return_direct=True
            )
        ],
        streaming=streaming,
    )

    result = await AgentWorkflow(agents=[agent]).run(user_msg="Use echo")

    assert result.response.content == "direct answer"
    _assert_finish_reason(_agent_span(span_exporter), "stop", capture_content)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [FunctionAgent, ReActAgent])
@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_finish_reasons_handoff(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
    agent_type: type[FunctionAgent | ReActAgent],
    streaming: bool,
) -> None:
    router = agent_type(
        name="router",
        description="Routes the request.",
        llm=_llm(
            [
                _tool_call(
                    agent_type,
                    "handoff",
                    {"to_agent": "member", "reason": "Let the member answer."},
                )
            ]
        ),
        streaming=streaming,
    )
    member = ReActAgent(
        name="member",
        description="Answers the request.",
        llm=_llm([_answer(ReActAgent)]),
        streaming=streaming,
    )

    result = await AgentWorkflow(
        agents=[router, member], root_agent="router"
    ).run(user_msg="Answer")

    assert result.response.content == "complete"
    _assert_finish_reason(
        _agent_span(span_exporter, "router"), "tool_calls", capture_content
    )
    _assert_finish_reason(_agent_span(span_exporter), "stop", capture_content)


@pytest.mark.asyncio
async def test_agent_finish_reasons_rejected_handoff(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
) -> None:
    agent = FunctionAgent(
        name="member",
        description="Handles the request and attempts a handoff.",
        llm=_llm(
            [
                _tool_call(
                    FunctionAgent,
                    "handoff",
                    {"to_agent": "missing", "reason": "Try another agent."},
                ),
                _answer(FunctionAgent),
            ]
        ),
    )
    other = FunctionAgent(
        name="other",
        description="Provides an alternative answer.",
        llm=_llm([_answer(FunctionAgent)]),
    )

    result = await AgentWorkflow(
        agents=[agent, other], root_agent="member"
    ).run(user_msg="Answer")

    assert result.response.content == "complete"
    assert not any(
        span.name == "invoke_agent other"
        for span in span_exporter.get_finished_spans()
    )
    _assert_finish_reason(
        _agent_span(span_exporter), "stop", capture_content
    )


@pytest.mark.asyncio
async def test_agent_finish_reasons_handoff_with_return_direct_sibling(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def direct_answer() -> str:
        return "direct answer"

    original_handoff = multi_agent_workflow.handoff

    async def delayed_handoff(
        ctx: Any, to_agent: str, reason: str
    ) -> str:
        await asyncio.sleep(0.02)
        return await original_handoff(ctx, to_agent, reason)

    monkeypatch.setattr(multi_agent_workflow, "handoff", delayed_handoff)

    router = FunctionAgent(
        name="router",
        description="Routes the request to another agent.",
        can_handoff_to=["member"],
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=lambda messages, **kwargs: ChatMessage(
                role="assistant",
                blocks=[
                    ToolCallBlock(
                        tool_call_id="direct-answer",
                        tool_name="direct_answer",
                        tool_kwargs={},
                    ),
                    ToolCallBlock(
                        tool_call_id="handoff-member",
                        tool_name="handoff",
                        tool_kwargs={
                            "to_agent": "member",
                            "reason": "Route the request.",
                        },
                    ),
                ],
            ),
        ),
        tools=[
            FunctionTool.from_defaults(
                async_fn=direct_answer,
                name="direct_answer",
                return_direct=True,
            )
        ],
    )
    member = FunctionAgent(
        name="member",
        description="Provides the requested answer.",
        llm=_llm([_answer(FunctionAgent)]),
    )

    result = await AgentWorkflow(
        agents=[router, member], root_agent="router"
    ).run(user_msg="Answer")

    assert result.response.content == "direct answer"
    _assert_finish_reason(
        _agent_span(span_exporter, "router"), "stop", capture_content
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_finish_reasons_retry(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
    streaming: bool,
) -> None:
    agent = ReActAgent(
        name="member",
        llm=_llm(
            [ChatMessage(role="assistant", content=""), _answer(ReActAgent)]
        ),
        streaming=streaming,
    )

    result = await AgentWorkflow(agents=[agent]).run(user_msg="Answer")

    assert result.response.content == "complete"
    _assert_finish_reason(_agent_span(span_exporter), "stop", capture_content)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [FunctionAgent, ReActAgent])
@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_finish_reasons_error_after_tool(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
    agent_type: type[FunctionAgent | ReActAgent],
    streaming: bool,
) -> None:
    error = ConnectionError("agent failed after the tool returned")
    agent = agent_type(
        name="member",
        llm=_llm([_tool_call(agent_type, "echo", {"value": "hi"}), error]),
        tools=[FunctionTool.from_defaults(lambda value: value, name="echo")],
        streaming=streaming,
    )

    with pytest.raises(ConnectionError) as caught:
        await AgentWorkflow(agents=[agent]).run(user_msg="Use echo")

    assert caught.value is error
    span = _agent_span(span_exporter)
    attributes = dict(span.attributes or {})
    assert span.status.status_code == StatusCode.ERROR
    assert attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"
    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("early_stopping_method", ["force", "generate"])
async def test_agent_finish_reasons_iteration_limit(
    span_exporter: InMemorySpanExporter,
    capture_content: bool,
    early_stopping_method: str,
) -> None:
    agent = FunctionAgent(
        name="member",
        llm=_llm([_answer(FunctionAgent), _answer(FunctionAgent)]),
        streaming=False,
    )
    workflow = AgentWorkflow(agents=[agent])

    if early_stopping_method == "force":
        with pytest.raises(WorkflowRuntimeError, match="Max iterations"):
            await workflow.run(user_msg="Answer", max_iterations=1)
        span = _agent_span(span_exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in (
            span.attributes or {}
        )
    else:
        result = await workflow.run(
            user_msg="Answer",
            max_iterations=1,
            early_stopping_method=early_stopping_method,
        )
        assert result.response.content == "complete"
        _assert_finish_reason(
            _agent_span(span_exporter), "stop", capture_content
        )