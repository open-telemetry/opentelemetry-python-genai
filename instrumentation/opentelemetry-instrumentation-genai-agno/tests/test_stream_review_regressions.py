# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest
from agno.agent import Agent
from agno.models.response import ModelResponse
from pydantic import BaseModel
from tests.mock_model import MockModel

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util.genai.types import TextPart


class UncapturableAnswer(BaseModel):
    answer: str = "final"

    def model_dump(self, *args, **kwargs):
        raise ValueError("cannot capture")

    def __str__(self) -> str:
        raise ValueError("cannot capture")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_public_agent_stream_preserves_partial_output_on_conversion_failure(
    tracer_provider, logger_provider, meter_provider, asynchronous
) -> None:
    answer = UncapturableAnswer()
    hook = MagicMock()
    agent = Agent(
        name="partial-answer",
        model=MockModel(id="mock-model"),
        output_schema=UncapturableAnswer,
    )

    def response_stream(*args, **kwargs):
        yield ModelResponse(content="partial")
        yield ModelResponse(content=answer)

    async def aresponse_stream(*args, **kwargs):
        yield ModelResponse(content="partial")
        yield ModelResponse(content=answer)

    async def consume():
        stream = agent.arun("Give an answer", stream=True, stream_events=True)
        if inspect.isawaitable(stream):
            stream = await stream
        return [chunk async for chunk in stream]

    with (
        instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        ),
        patch(
            "agno.models.base.Model.response_stream",
            side_effect=response_stream,
        ),
        patch(
            "agno.models.base.Model.aresponse_stream",
            side_effect=aresponse_stream,
        ),
    ):
        chunks = (
            asyncio.run(consume())
            if asynchronous
            else list(
                agent.run("Give an answer", stream=True, stream_events=True)
            )
        )

    assert any(getattr(chunk, "content", None) is answer for chunk in chunks)
    hook.on_completion.assert_called_once()
    outputs = hook.on_completion.call_args.kwargs["outputs"]
    captured_text = "".join(
        part.content
        for message in outputs
        for part in message.parts
        if isinstance(part, TextPart)
    )
    assert captured_text == "partial"
