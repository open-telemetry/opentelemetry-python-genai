# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from types import AsyncGeneratorType, GeneratorType
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.runnables import (
    RunnableGenerator,
    RunnableLambda,
    RunnableParallel,
)
from langchain_openai import ChatOpenAI

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.langchain import _run_context
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode

from .test_execution_context import (
    _HTTP_SCOPE,
    _LC_SCOPE,
    _OPENAI_SCOPE,
    _child,
    _spans,
    clients,
    no_detach_errors,
)

__all__ = ["clients", "no_detach_errors"]


_RESPONSES_API = "use_responses_api" in ChatOpenAI.model_fields


def _model(clients: Any, **kwargs: Any) -> ChatOpenAI:
    return ChatOpenAI(
        model="test-model",
        api_key="test",
        http_client=clients.http,
        http_async_client=clients.ahttp,
        **kwargs,
    )


def _assert_no_runs() -> None:
    handler = next(
        h
        for h in CallbackManager.configure().handlers
        if isinstance(h, OpenTelemetryLangChainCallbackHandler)
    )
    assert handler._invocation_manager._invocations == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_chat_stream_is_lazy_and_restores_consumer_context(
    clients, span_exporter, tracer_provider, asynchronous: bool
) -> None:
    model = _model(clients)
    config = {"run_id": uuid4(), "tags": ["test"]}
    original_config = dict(config)
    with tracer_provider.get_tracer("test").start_as_current_span(
        "request"
    ) as root:
        before = context.get_current()
        stream = (
            model.astream("hello", config=config)
            if asynchronous
            else model.stream("hello", config=config)
        )
        assert isinstance(
            stream, AsyncGeneratorType if asynchronous else GeneratorType
        )
        assert clients.requests == []
        assert span_exporter.get_finished_spans() == ()
        if asynchronous:
            first = await asyncio.create_task(anext(stream))
        else:
            first = next(stream)
        assert first.content == "answer"
        assert context.get_current() is before
        assert _spans(span_exporter, _LC_SCOPE) == []
        clients.http.get("https://example.test/consumer")
        if asynchronous:
            rest = [chunk async for chunk in stream]
        else:
            rest = list(stream)
        assert "".join(chunk.content for chunk in rest) == ""
        assert context.get_current() is before
    assert config == original_config
    (model_span,) = _spans(span_exporter, _LC_SCOPE)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    http, consumer_http = _spans(span_exporter, _HTTP_SCOPE)
    assert model_span.parent == root.get_span_context()
    _child(inference, model_span)
    _child(http, inference)
    assert consumer_http.parent == root.get_span_context()
    _assert_no_runs()


class _RunIdRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.run_ids: list[UUID] = []

    def on_chat_model_start(
        self, *args: Any, run_id: UUID, **kwargs: Any
    ) -> None:
        self.run_ids.append(run_id)

    def on_chain_start(self, *args: Any, run_id: UUID, **kwargs: Any) -> None:
        self.run_ids.append(run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["model", "lambda"])
async def test_stream_reads_config_on_first_advancement(
    clients, span_exporter, asynchronous: bool, kind: str
) -> None:
    runnable = (
        _model(clients)
        if kind == "model"
        else RunnableLambda(lambda text: [text], afunc=None)
    )
    recorder = _RunIdRecorder()
    config: dict[str, Any] = {"run_id": uuid4(), "callbacks": [recorder]}
    stream = (
        runnable.astream("hello", config=config)
        if asynchronous
        else runnable.stream("hello", config=config)
    )
    # Nothing ran yet, so a caller may still edit the config it passed.
    assert clients.requests == []
    late_run_id = uuid4()
    config["run_id"] = late_run_id
    if asynchronous:
        assert [chunk async for chunk in stream]
    else:
        assert list(stream)
    assert recorder.run_ids == [late_run_id]
    (span,) = _spans(span_exporter, _LC_SCOPE)
    assert span.name.startswith(
        "chat " if kind == "model" else "invoke_workflow "
    )
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _RESPONSES_API, reason="langchain-openai predates the Responses API"
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_stream_restores_consumer_context(
    clients, span_exporter, tracer_provider, asynchronous: bool
) -> None:
    model = _model(clients, use_responses_api=True)
    with tracer_provider.get_tracer("test").start_as_current_span(
        "request"
    ) as root:
        before = context.get_current()
        stream = (
            model.astream("hello") if asynchronous else model.stream("hello")
        )
        if asynchronous:
            first = await asyncio.create_task(anext(stream))
        else:
            first = next(stream)
        assert context.get_current() is before
        clients.http.get("https://example.test/consumer")

        async def drain() -> list[Any]:
            return [chunk async for chunk in stream]

        # Later reads move to another task, so each read must attach and
        # detach within its own frame.
        rest = (
            await asyncio.create_task(drain())
            if asynchronous
            else list(stream)
        )
        assert "answer" in "".join(
            str(chunk.content) for chunk in (first, *rest)
        )
        assert context.get_current() is before
    (model_span,) = _spans(span_exporter, _LC_SCOPE)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    http, consumer_http = _spans(span_exporter, _HTTP_SCOPE)
    assert model_span.parent == root.get_span_context()
    _child(inference, model_span)
    _child(http, inference)
    assert consumer_http.parent == root.get_span_context()
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ending", ["provider_error", "caller_error", "close"])
async def test_chat_stream_failure_and_close(
    clients, span_exporter, asynchronous: bool, ending: str
) -> None:
    error = ConnectionError("stream failed")
    if ending == "provider_error":
        clients.stream_error = error
    model = _model(clients)
    stream = model.astream("hello") if asynchronous else model.stream("hello")
    before = context.get_current()
    if ending == "caller_error":
        with pytest.raises(ConnectionError) as raised:
            if asynchronous:
                async with stream:
                    await anext(stream)
                    raise error
            else:
                with stream:
                    next(stream)
                    raise error
        assert raised.value is error
    else:
        if asynchronous:
            await anext(stream)
        else:
            next(stream)
        assert context.get_current() is before
        if ending == "close":
            if asynchronous:
                await stream.aclose()
            else:
                stream.close()
        else:
            with pytest.raises(ConnectionError) as raised:
                if asynchronous:
                    await anext(stream)
                else:
                    next(stream)
            assert raised.value is error
    assert context.get_current() is before
    (model_span,) = _spans(span_exporter, _LC_SCOPE)
    if ending != "close":
        assert model_span.status.status_code == StatusCode.ERROR
        assert (
            model_span.attributes[error_attributes.ERROR_TYPE]
            == "ConnectionError"
        )
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_interleaved_chat_streams_do_not_share_parents(
    clients, span_exporter, tracer_provider, asynchronous: bool
) -> None:
    model = _model(clients)
    one = model.astream("one") if asynchronous else model.stream("one")
    two = model.astream("two") if asynchronous else model.stream("two")
    with tracer_provider.get_tracer("test").start_as_current_span(
        "one"
    ) as first_parent:
        if asynchronous:
            await anext(one)
        else:
            next(one)
        with tracer_provider.get_tracer("test").start_as_current_span(
            "two"
        ) as second_parent:
            if asynchronous:
                await anext(two)
                assert [chunk async for chunk in one]
                assert [chunk async for chunk in two]
            else:
                next(two)
                assert list(one)
                assert list(two)
            assert trace.get_current_span() is second_parent
        assert trace.get_current_span() is first_parent
    model_spans = _spans(span_exporter, _LC_SCOPE)
    assert {span.parent.span_id for span in model_spans} == {
        first_parent.get_span_context().span_id,
        second_parent.get_span_context().span_id,
    }
    inferences = _spans(span_exporter, _OPENAI_SCOPE)
    assert len(inferences) == 2
    for inference in inferences:
        _child(
            inference,
            next(
                span
                for span in model_spans
                if span.context == inference.parent
            ),
        )
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["lambda", "generator"])
async def test_runnable_stream_correlates_each_step(
    clients, span_exporter, tracer_provider, asynchronous: bool, kind: str
) -> None:
    def produce(text: str) -> Iterator[str]:
        clients.http.get("https://example.test/produce/one")
        yield text
        clients.http.get("https://example.test/produce/two")
        yield "done"

    async def aproduce(text: str) -> AsyncIterator[str]:
        await clients.ahttp.get("https://example.test/produce/one")
        yield text
        await clients.ahttp.get("https://example.test/produce/two")
        yield "done"

    def transform(inputs: Iterator[str]) -> Iterator[str]:
        for text in inputs:
            yield from produce(text)

    async def atransform(inputs: AsyncIterator[str]) -> AsyncIterator[str]:
        async for text in inputs:
            async for chunk in aproduce(text):
                yield chunk

    if kind == "lambda":
        runnable = RunnableLambda(produce, afunc=aproduce)
    else:
        runnable = RunnableGenerator(transform, atransform=atransform)
    with tracer_provider.get_tracer("test").start_as_current_span(
        "request"
    ) as root:
        before = context.get_current()
        stream = (
            runnable.astream("hello")
            if asynchronous
            else runnable.stream("hello")
        )
        assert clients.requests == []
        if asynchronous:
            assert await anext(stream) == "hello"
        else:
            assert next(stream) == "hello"
        assert context.get_current() is before
        clients.http.get("https://example.test/consumer")
        if asynchronous:
            assert [chunk async for chunk in stream] == ["done"]
        else:
            assert list(stream) == ["done"]
        assert context.get_current() is before
    (workflow,) = _spans(span_exporter, _LC_SCOPE)
    one, consumer, two = _spans(span_exporter, _HTTP_SCOPE)
    _child(one, workflow)
    _child(two, workflow)
    assert consumer.parent == root.get_span_context()
    _assert_no_runs()


@pytest.mark.asyncio
async def test_runnable_stream_task_cancellation(
    clients, span_exporter
) -> None:
    waiting = asyncio.Event()
    errors: list[BaseException] = []

    async def produce(text: str) -> AsyncIterator[str]:
        yield text
        await clients.ahttp.get("https://example.test/waiting")
        waiting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            errors.append(error)
            raise

    stream = RunnableLambda(produce).astream("hello")
    before = context.get_current()
    assert await anext(stream) == "hello"

    async def consume() -> None:
        try:
            await anext(stream)
        except asyncio.CancelledError as error:
            assert error is errors[0]
            raise
        finally:
            assert context.get_current() is before

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(waiting.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert context.get_current() is before
    (workflow,) = _spans(span_exporter, _LC_SCOPE)
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    _child(http, workflow)
    assert (
        workflow.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )
    _assert_no_runs()


@pytest.mark.asyncio
async def test_chat_stream_task_cancellation(clients, span_exporter) -> None:
    waiting = asyncio.Event()
    clients.stream_waiting = waiting
    stream = _model(clients).astream("hello")
    before = context.get_current()
    assert (await anext(stream)).content == "answer"

    async def consume() -> None:
        try:
            await anext(stream)
        finally:
            assert context.get_current() is before

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(waiting.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert context.get_current() is before
    (model_span,) = _spans(span_exporter, _LC_SCOPE)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    _child(inference, model_span)
    assert (
        model_span.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["lambda", "generator"])
async def test_runnable_stream_close(
    clients, span_exporter, asynchronous: bool, kind: str
) -> None:
    def produce(text: str) -> Iterator[str]:
        clients.http.get("https://example.test/produce")
        yield text
        yield "done"

    async def aproduce(text: str) -> AsyncIterator[str]:
        await clients.ahttp.get("https://example.test/produce")
        yield text
        yield "done"

    def transform(inputs: Iterator[str]) -> Iterator[str]:
        for text in inputs:
            yield from produce(text)

    async def atransform(inputs: AsyncIterator[str]) -> AsyncIterator[str]:
        async for text in inputs:
            async for chunk in aproduce(text):
                yield chunk

    runnable = (
        RunnableLambda(produce, afunc=aproduce)
        if kind == "lambda"
        else RunnableGenerator(transform, atransform=atransform)
    )
    before = context.get_current()
    if asynchronous:
        stream = runnable.astream("hello")
        assert await anext(stream) == "hello"
        await stream.aclose()
    else:
        stream = runnable.stream("hello")
        assert next(stream) == "hello"
        stream.close()
    assert context.get_current() is before
    (workflow,) = _spans(span_exporter, _LC_SCOPE)
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    _child(http, workflow)
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("composite", ["sequence", "parallel"])
async def test_composite_stream_attaches_each_context_once(
    clients, span_exporter, monkeypatch, asynchronous: bool, composite: str
) -> None:
    attached: list[bool] = []
    original = _run_context.attach

    def attach(ctx: context.Context) -> object:
        # Attaching the current context again is a redundant token.
        attached.append(context.get_current() is ctx)
        return original(ctx)

    monkeypatch.setattr(_run_context, "attach", attach)
    steps = [RunnableLambda(lambda text: text) for _ in range(2)]
    chain = (
        steps[0] | steps[1]
        if composite == "sequence"
        else RunnableParallel(one=steps[0], two=steps[1])
    )
    before = context.get_current()
    if asynchronous:
        chunks = [chunk async for chunk in chain.astream("hello")]
    else:
        chunks = list(chain.stream("hello"))
    assert chunks
    assert context.get_current() is before
    assert attached
    assert not any(attached)
    _assert_no_runs()
