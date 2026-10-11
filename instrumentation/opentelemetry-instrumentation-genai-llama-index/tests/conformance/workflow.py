# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, MockFunctionCallingLLM


def function_response(
    messages: list[ChatMessage], **kwargs: Any
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        blocks=[
            ToolCallBlock(
                tool_call_id="handoff-call",
                tool_name="handoff",
                tool_kwargs={
                    "to_agent": "react-member",
                    "reason": "The ReAct agent should answer.",
                },
            )
        ],
    )


def react_response(messages: list[ChatMessage], **kwargs: Any) -> ChatMessage:
    return ChatMessage(
        role="assistant", content="Thought: I can answer.\nAnswer: complete"
    )


function_agent = FunctionAgent(
    name="function-member",
    description="Routes the request.",
    llm=MockFunctionCallingLLM(
        is_chat_model=True, response_generator=function_response
    ),
    streaming=False,
)
react_agent = ReActAgent(
    name="react-member",
    description="Answers the request.",
    llm=MockFunctionCallingLLM(
        is_chat_model=True, response_generator=react_response
    ),
    streaming=False,
)
workflow = AgentWorkflow(
    agents=[function_agent, react_agent], root_agent="function-member"
)


async def run_workflow() -> None:
    await workflow.run(user_msg="Complete the request")


asyncio.run(run_workflow())
