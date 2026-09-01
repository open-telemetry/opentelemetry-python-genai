# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import patch

from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import (
    ChatMessage,
    MockFunctionCallingLLM,
)
from llama_index.llms.openai import OpenAI

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class AgentScenario(Scenario):
    expected_spans = {"invoke_agent": 3, "execute_tool": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        async def weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Sunny in {city}"

        async def run_agent(agent: FunctionAgent, prompt: str) -> None:
            await agent.run(user_msg=prompt)

        def response_generator(
            messages: list[ChatMessage], **kwargs: Any
        ) -> ChatMessage:
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

        def react_response_generator(
            messages: list[ChatMessage], **kwargs: Any
        ) -> ChatMessage:
            return ChatMessage(
                role="assistant",
                content="Thought: I know the answer.\nAnswer: Four.",
            )

        with instrument(
            LlamaIndexInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            agent = FunctionAgent(
                name="weather-agent",
                description="Answers weather questions.",
                llm=MockFunctionCallingLLM(
                    is_chat_model=True,
                    response_generator=response_generator,
                ),
                tools=[weather],
                streaming=False,
            )
            asyncio.run(run_agent(agent, "What is the weather?"))

            react_agent = ReActAgent(
                name="react-agent",
                llm=MockFunctionCallingLLM(
                    is_chat_model=True,
                    response_generator=react_response_generator,
                ),
                streaming=False,
            )
            asyncio.run(run_agent(react_agent, "What is two plus two?"))

            with patch.dict(
                os.environ, {"OPENAI_API_KEY": "test_openai_api_key"}
            ):
                provider_agent = FunctionAgent(
                    name="provider-agent",
                    llm=OpenAI(model="gpt-4o-mini", temperature=0.1),
                    streaming=False,
                )
                with vcr.use_cassette("inference.yaml"):
                    asyncio.run(run_agent(provider_agent, "Hello!"))
