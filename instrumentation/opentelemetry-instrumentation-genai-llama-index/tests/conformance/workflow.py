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

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class WorkflowScenario(Scenario):
    expected_spans = {
        "invoke_workflow": 1,
        "invoke_agent": 2,
        "execute_tool": 1,
    }
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
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

        def react_response(
            messages: list[ChatMessage], **kwargs: Any
        ) -> ChatMessage:
            return ChatMessage(
                role="assistant",
                content="Thought: I can answer.\nAnswer: complete",
            )

        function_agent = FunctionAgent(
            name="function-member",
            description="Routes the request.",
            llm=MockFunctionCallingLLM(
                is_chat_model=True,
                response_generator=function_response,
            ),
            streaming=False,
        )
        react_agent = ReActAgent(
            name="react-member",
            description="Answers the request.",
            llm=MockFunctionCallingLLM(
                is_chat_model=True,
                response_generator=react_response,
            ),
            streaming=False,
        )
        workflow = AgentWorkflow(
            agents=[function_agent, react_agent],
            root_agent="function-member",
        )

        with instrument(
            LlamaIndexInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            async def run_workflow() -> None:
                await workflow.run(user_msg="Complete the request")

            asyncio.run(run_workflow())
