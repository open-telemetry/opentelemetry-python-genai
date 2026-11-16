# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline LangChain components for public-API instrumentation tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TypeVar

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import Field

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import StatusCode

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


class FakeChatModel(FakeMessagesListChatModel):
    model_name: str = "test-model"
    generation_info: dict[str, Any] | None = Field(
        default_factory=lambda: {"finish_reason": "stop"}
    )
    generations: list[ChatGeneration] | None = None
    llm_output: dict[str, Any] | None = None
    chunks: list[ChatGenerationChunk] | None = None
    error: Exception | None = None
    stream_error: Exception | None = None

    @property
    def _identifying_params(self) -> dict[str, object]:
        return {"model_name": self.model_name}

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self.bind(tools=tools, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.error is not None:
            raise self.error
        result = super()._generate(messages, stop, run_manager, **kwargs)
        for generation in result.generations:
            generation.generation_info = self.generation_info
        if self.generations is not None:
            result.generations = self.generations
        result.llm_output = self.llm_output
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.error is not None:
            raise self.error
        chunks = self.chunks
        if chunks is None:
            response = self.responses[0]
            chunks = [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=response.content,
                        name=response.name,
                        response_metadata=response.response_metadata,
                        usage_metadata=getattr(
                            response, "usage_metadata", None
                        ),
                    ),
                    generation_info=self.generation_info,
                )
            ]
        yield from chunks
        if self.stream_error is not None:
            raise self.stream_error


class FakeRetriever(BaseRetriever):
    documents: list[Document] = Field(default_factory=list)
    error: Exception | None = None
    model: FakeChatModel | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.error is not None:
            raise self.error
        if self.model is not None:
            self.model.invoke(
                query, config={"callbacks": run_manager.get_child()}
            )
        return self.documents


async def invoke(
    runnable: Runnable[_Input, _Output],
    value: _Input,
    asynchronous: bool,
    config: RunnableConfig | None = None,
) -> _Output:
    if asynchronous:
        return await runnable.ainvoke(value, config=config)
    return runnable.invoke(value, config=config)


def finished_span(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    assert span.start_time is not None
    assert span.end_time is not None
    assert span.end_time >= span.start_time
    assert span.attributes is not None
    return span


def finished_chat(exporter: InMemorySpanExporter) -> ReadableSpan:
    span = finished_span(exporter, "chat test-model")
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes is not None
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS]
    return span
