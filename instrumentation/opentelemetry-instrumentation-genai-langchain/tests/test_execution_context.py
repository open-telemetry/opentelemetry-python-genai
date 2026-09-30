# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack
from contextvars import copy_context
from typing import Any, TypedDict

import httpx
import pytest
import pytest_asyncio
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables import base as runnables_base
from langchain_core.tools import BaseTool, StructuredTool, Tool
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI

from opentelemetry import baggage, context, trace
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode

_LC_SCOPE = "opentelemetry.instrumentation.genai.langchain"
_OPENAI_SCOPE = "opentelemetry.instrumentation.genai.openai"
_HTTP_SCOPE = "opentelemetry.instrumentation.httpx"


@pytest.fixture(autouse=True)
def no_detach_errors(caplog) -> Iterator[None]:
    yield
    assert not [
        record
        for phase in ("setup", "call", "teardown")
        for record in caplog.get_records(phase)
        if record.name == "opentelemetry.context" and record.levelno >= 40
    ]


class _Clients:
    def __init__(self, http: httpx.Client, ahttp: httpx.AsyncClient) -> None:
        self.http = http
        self.ahttp = ahttp
        self.requests: list[httpx.Request] = []
        self.stream_error: BaseException | None = None
        self.stream_waiting: asyncio.Event | None = None
        self.openai = OpenAI(api_key="test", http_client=http, max_retries=0)
        self.aopenai = AsyncOpenAI(
            api_key="test", http_client=ahttp, max_retries=0
        )

    def infer(self, text: str) -> str:
        result = self.openai.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": text}]
        )
        return result.choices[0].message.content

    async def ainfer(self, text: str) -> str:
        result = await self.aopenai.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": text}]
        )
        return result.choices[0].message.content


@pytest_asyncio.fixture
async def clients(
    tracer_provider, meter_provider, logger_provider
) -> AsyncIterator[_Clients]:
    class ResponseStream(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield sse_chunk("answer", None)
            if client_set.stream_error is not None:
                raise client_set.stream_error
            yield sse_chunk("", "stop")
            yield b"data: [DONE]\n\n"

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for chunk in self:
                await asyncio.sleep(0)
                yield chunk
                if client_set.stream_waiting is not None:
                    client_set.stream_waiting.set()
                    await asyncio.Event().wait()

    class ResponsesApiStream(ResponseStream):
        def __iter__(self) -> Iterator[bytes]:
            response = {
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "test-model",
                "status": "in_progress",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
            message = {
                "type": "message",
                "id": "msg-test",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [],
                    }
                ],
            }
            yield sse_event("response.created", response=response)
            yield sse_event(
                "response.output_text.delta",
                item_id="msg-test",
                output_index=0,
                content_index=0,
                delta="answer",
                logprobs=[],
            )
            if client_set.stream_error is not None:
                raise client_set.stream_error
            yield sse_event(
                "response.completed",
                response={
                    **response,
                    "status": "completed",
                    "output": [message],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

    def sse_event(event_type: str, **payload: Any) -> bytes:
        data = json.dumps(
            {"type": event_type, "sequence_number": 0, **payload}
        )
        return f"event: {event_type}\ndata: {data}\n\n".encode()

    def sse_chunk(content: str, finish_reason: str | None) -> bytes:
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    def respond(request: httpx.Request) -> httpx.Response:
        client_set.requests.append(request)
        current = trace.get_current_span().get_span_context()
        assert request.headers["traceparent"] == (
            f"00-{current.trace_id:032x}-{current.span_id:016x}-{current.trace_flags:02x}"
        )
        if request.method == "POST" and json.loads(request.content).get(
            "stream"
        ):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=(
                    ResponsesApiStream()
                    if request.url.path.endswith("/responses")
                    else ResponseStream()
                ),
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    with ExitStack() as stack:
        for instrumentor in (LangChainInstrumentor(), OpenAIInstrumentor()):
            stack.enter_context(
                instrument(
                    instrumentor,
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                    logger_provider=logger_provider,
                )
            )
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            ) as ahttp:
                for client in (http, ahttp):
                    HTTPXClientInstrumentor.instrument_client(
                        client,
                        tracer_provider=tracer_provider,
                        meter_provider=meter_provider,
                    )
                    stack.callback(
                        HTTPXClientInstrumentor.uninstrument_client, client
                    )
                client_set = _Clients(http, ahttp)
                yield client_set


def _spans(exporter: Any, scope: str) -> list[ReadableSpan]:
    return [
        s
        for s in exporter.get_finished_spans()
        if s.instrumentation_scope.name == scope
    ]


def _child(child: ReadableSpan, parent: ReadableSpan) -> None:
    assert child.parent == parent.context
    assert child.context.trace_id == parent.context.trace_id
    assert (
        parent.start_time
        <= child.start_time
        <= child.end_time
        <= parent.end_time
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["structured", "simple"])
@pytest.mark.parametrize("mode", ["sync", "async", "executor"])
async def test_tool_correlates_inference_and_http(
    clients,
    span_exporter,
    tracer_provider,
    kind: str,
    mode: str,
) -> None:
    def lookup(text: str) -> str:
        assert baggage.get_baggage("request") == "test-request"
        clients.http.get("https://example.test/lookup")
        return clients.infer(text)

    async def alookup(text: str) -> str:
        assert baggage.get_baggage("request") == "test-request"
        await clients.ahttp.get("https://example.test/lookup")
        await asyncio.sleep(0)
        return await clients.ainfer(text)

    cls = StructuredTool if kind == "structured" else Tool
    tool = cls.from_function(
        func=lookup,
        coroutine=alookup if mode == "async" else None,
        name="lookup",
        description="Look up a value.",
    )
    token = context.attach(baggage.set_baggage("request", "test-request"))
    try:
        with tracer_provider.get_tracer("test").start_as_current_span(
            "request"
        ) as root:
            before = context.get_current()
            output = (
                tool.invoke("hello")
                if mode == "sync"
                else await tool.ainvoke("hello")
            )
            assert output == "answer"
            assert context.get_current() is before
            assert trace.get_current_span() is root
    finally:
        context.detach(token)

    (tool_span,) = _spans(span_exporter, _LC_SCOPE)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    http, model_http = _spans(span_exporter, _HTTP_SCOPE)
    assert tool_span.name == "execute_tool lookup"
    assert tool_span.parent == root.get_span_context()
    _child(inference, tool_span)
    _child(http, tool_span)
    _child(model_http, inference)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_chat_model_correlates_sdk_and_http(
    clients,
    span_exporter,
    asynchronous: bool,
) -> None:
    model = ChatOpenAI(
        model="test-model",
        api_key="test",
        http_client=clients.http,
        http_async_client=clients.ahttp,
    )
    before = context.get_current()
    result = (
        await model.ainvoke("hello") if asynchronous else model.invoke("hello")
    )
    assert result.content == "answer"
    assert context.get_current() is before
    (model_span,) = _spans(span_exporter, _LC_SCOPE)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    _child(inference, model_span)
    _child(http, inference)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "executor"])
async def test_runnable_correlates_inference(
    clients,
    span_exporter,
    mode: str,
) -> None:
    runnable = RunnableLambda(
        clients.infer, afunc=clients.ainfer if mode == "async" else None
    )
    before = context.get_current()
    result = (
        runnable.invoke("hello")
        if mode == "sync"
        else await runnable.ainvoke("hello")
    )
    assert result == "answer"
    assert context.get_current() is before
    (workflow,) = _spans(span_exporter, _LC_SCOPE)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    _child(inference, workflow)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_vector_retriever_correlates_http(
    clients,
    span_exporter,
    asynchronous: bool,
) -> None:
    class Store(VectorStore):
        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
            raise NotImplementedError

        def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
        ) -> list[Document]:
            clients.http.get("https://example.test/search")
            return [Document(page_content="found")]

        async def asimilarity_search(
            self, query: str, k: int = 4, **kwargs: Any
        ) -> list[Document]:
            await clients.ahttp.get("https://example.test/search")
            return [Document(page_content="found")]

    retriever = Store().as_retriever()
    before = context.get_current()
    result = (
        await retriever.ainvoke("query")
        if asynchronous
        else retriever.invoke("query")
    )
    assert result[0].page_content == "found"
    assert context.get_current() is before
    (retrieval,) = _spans(span_exporter, _LC_SCOPE)
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    _child(http, retrieval)


class _State(TypedDict):
    text: str


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_graph_node_and_nested_tool_correlate_http(
    clients,
    span_exporter,
    asynchronous: bool,
) -> None:
    graph_module = pytest.importorskip("langgraph.graph")
    helper = pytest.importorskip("langgraph._internal._runnable")
    if not hasattr(helper, "set_config_context"):
        pytest.skip("LangGraph version has no scoped node execution helper")

    tool = StructuredTool.from_function(
        func=clients.infer,
        coroutine=clients.ainfer,
        name="lookup",
        description="Look up a value.",
    )

    def node(state: _State) -> _State:
        clients.http.get("https://example.test/node")
        return {"text": tool.invoke(state["text"])}

    async def anode(state: _State) -> _State:
        await clients.ahttp.get("https://example.test/node")
        return {"text": await tool.ainvoke(state["text"])}

    builder = graph_module.StateGraph(_State)
    builder.add_node("lookup", anode if asynchronous else node)
    builder.add_edge(graph_module.START, "lookup")
    builder.add_edge("lookup", graph_module.END)
    graph = builder.compile()
    before = context.get_current()
    result = (
        await graph.ainvoke({"text": "hello"})
        if asynchronous
        else graph.invoke({"text": "hello"})
    )
    assert result == {"text": "answer"}
    assert context.get_current() is before
    tool_span, workflow = _spans(span_exporter, _LC_SCOPE)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    node_http, inference_http = _spans(span_exporter, _HTTP_SCOPE)
    _child(tool_span, workflow)
    _child(node_http, workflow)
    _child(inference, tool_span)
    _child(inference_http, inference)


@pytest.mark.asyncio
async def test_concurrent_tools_do_not_share_context(
    clients, span_exporter, tracer_provider
) -> None:
    ready = asyncio.Event()
    entered = 0

    async def lookup(text: str) -> str:
        nonlocal entered
        entered += 1
        if entered == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 5)
        return await clients.ainfer(text)

    tool = StructuredTool.from_function(
        coroutine=lookup, name="lookup", description="Look up a value."
    )

    async def request(name: str) -> None:
        with tracer_provider.get_tracer("test").start_as_current_span(
            name
        ) as root:
            assert await tool.ainvoke(name) == "answer"
            assert trace.get_current_span() is root

    await asyncio.gather(request("one"), request("two"))
    tools = _spans(span_exporter, _LC_SCOPE)
    inferences = _spans(span_exporter, _OPENAI_SCOPE)
    assert len(tools) == len(inferences) == 2
    assert tools[0].context.trace_id != tools[1].context.trace_id
    for inference in inferences:
        _child(
            inference,
            next(tool for tool in tools if tool.context == inference.parent),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "cancel", "sync_cancel"])
@pytest.mark.parametrize("kind", ["structured", "simple"])
async def test_execution_failure_restores_context(
    clients, span_exporter, mode: str, kind: str
) -> None:
    error = (
        asyncio.CancelledError()
        if "cancel" in mode
        else RuntimeError("tool failed")
    )

    def fail(text: str) -> str:
        clients.http.get("https://example.test/fail")
        raise error

    async def afail(text: str) -> str:
        await clients.ahttp.get("https://example.test/fail")
        await asyncio.sleep(0)
        raise error

    cls = StructuredTool if kind == "structured" else Tool
    tool = cls.from_function(
        func=fail, coroutine=afail, name="fail", description="Fail after HTTP."
    )
    before = context.get_current()
    with pytest.raises(type(error)) as raised:
        if mode in ("sync", "sync_cancel"):
            tool.invoke("hello")
        else:
            await tool.ainvoke("hello")
    assert raised.value is error
    assert context.get_current() is before
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    (tool_span,) = _spans(span_exporter, _LC_SCOPE)
    _child(http, tool_span)
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.attributes[error_attributes.ERROR_TYPE] == (
        "asyncio.exceptions.CancelledError"
        if "cancel" in mode
        else "RuntimeError"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["structured", "simple"])
async def test_task_cancellation_finalizes_tool_once(
    clients,
    span_exporter,
    tracer_provider,
    kind: str,
) -> None:
    started = asyncio.Event()
    errors: list[BaseException] = []
    other_errors: list[BaseException] = []

    class OtherHandler(BaseCallbackHandler):
        def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
            other_errors.append(error)

    async def lookup(text: str) -> str:
        await clients.ahttp.get("https://example.test/lookup")
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            errors.append(error)
            raise
        return text

    cls = StructuredTool if kind == "structured" else Tool
    tool = cls.from_function(
        func=None,
        coroutine=lookup,
        name="lookup",
        description="Wait for cancellation.",
    )
    handler = next(
        h
        for h in CallbackManager.configure().handlers
        if isinstance(h, OpenTelemetryLangChainCallbackHandler)
    )

    async def request() -> None:
        with tracer_provider.get_tracer("test").start_as_current_span(
            "request"
        ) as root:
            before = context.get_current()
            try:
                await tool.ainvoke(
                    "hello", config={"callbacks": [OtherHandler()]}
                )
            except asyncio.CancelledError as error:
                assert error is errors[0]
                assert error.args == ("cancelled by test",)
                raise
            finally:
                assert context.get_current() is before
                assert trace.get_current_span() is root

    task = asyncio.create_task(request())
    try:
        await asyncio.wait_for(started.wait(), 5)
        (run_id,) = handler._invocation_manager._invocations
        task.cancel("cancelled by test")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert handler._invocation_manager._invocations == {}
    assert other_errors == []
    handler.on_tool_error(errors[0], run_id=run_id)
    (tool_span,) = _spans(span_exporter, _LC_SCOPE)
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    _child(http, tool_span)
    assert tool_span.status.status_code == StatusCode.ERROR
    assert (
        tool_span.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


@pytest.mark.asyncio
async def test_handled_child_cancellation_does_not_fail_parent_tool(
    clients,
    span_exporter,
) -> None:
    async def child(text: str) -> str:
        raise asyncio.CancelledError()

    async def lookup(text: str) -> str:
        before = context.get_current()
        with pytest.raises(asyncio.CancelledError):
            await RunnableLambda(child).ainvoke(text)
        assert context.get_current() is before
        return await clients.ainfer(text)

    tool = StructuredTool.from_function(
        coroutine=lookup,
        name="lookup",
        description="Handle a cancelled child.",
    )
    assert await tool.ainvoke("hello") == "answer"
    (tool_span,) = [
        span
        for span in _spans(span_exporter, _LC_SCOPE)
        if span.name == "execute_tool lookup"
    ]
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    _child(inference, tool_span)
    assert tool_span.status.status_code != StatusCode.ERROR
    assert error_attributes.ERROR_TYPE not in tool_span.attributes


def test_callbacks_in_different_contexts_never_attach(
    clients,
    tracer_provider,
    span_exporter,
) -> None:
    with tracer_provider.get_tracer("test").start_as_current_span(
        "request"
    ) as root:
        start_context = copy_context()
        finish_context = copy_context()
        callbacks = CallbackManager.configure()
        run = start_context.run(
            callbacks.on_chain_start,
            {"name": "LangGraph", "id": ["langgraph"]},
            {},
        )
        assert start_context.run(trace.get_current_span) is root
        finish_context.run(run.on_chain_end, {})
        assert start_context.run(trace.get_current_span) is root
        assert finish_context.run(trace.get_current_span) is root
        assert trace.get_current_span() is root

    (workflow,) = _spans(span_exporter, _LC_SCOPE)
    assert workflow.parent == root.get_span_context()


def test_uninstrument_restores_execution_methods(
    tracer_provider,
    meter_provider,
    logger_provider,
) -> None:
    targets = [
        (BaseTool, "run"),
        (BaseTool, "arun"),
        (BaseRetriever, "invoke"),
        (BaseRetriever, "ainvoke"),
        (BaseChatModel, "stream"),
        (BaseChatModel, "astream"),
        (Runnable, "_call_with_config"),
        (Runnable, "_acall_with_config"),
        (Runnable, "_transform_stream_with_config"),
        (Runnable, "_atransform_stream_with_config"),
        (runnables_base, "set_config_context"),
        (CallbackManager, "on_chain_start"),
    ]
    try:
        from langgraph.pregel import Pregel

        targets.extend([(Pregel, "stream"), (Pregel, "astream")])
    except ImportError:
        pass
    originals = [getattr(owner, name) for owner, name in targets]
    for _ in range(2):
        with instrument(
            LangChainInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        ):
            for (owner, name), original in zip(targets, originals):
                assert getattr(owner, name) is not original
        for (owner, name), original in zip(targets, originals):
            assert getattr(owner, name) is original


@pytest.mark.parametrize(
    ("module_name", "class_name", "method"),
    [
        (
            "langchain_core.language_models.chat_models",
            "BaseChatModel",
            "_agenerate_with_cache",
        ),
        (
            "langchain_core.runnables.base",
            "Runnable",
            "_atransform_stream_with_config",
        ),
        ("langchain_core.runnables.base", None, "set_config_context"),
        ("langgraph._internal._runnable", None, "set_config_context"),
        ("langgraph._internal._runnable", None, None),
    ],
)
def test_instrument_skips_missing_execution_boundary(
    monkeypatch,
    caplog,
    tracer_provider,
    meter_provider,
    logger_provider,
    module_name: str,
    class_name: str | None,
    method: str | None,
) -> None:
    module = pytest.importorskip(module_name)
    if method is None:
        monkeypatch.setitem(sys.modules, module_name, None)
    else:
        owner = getattr(module, class_name) if class_name else module
        monkeypatch.delattr(owner, method)
    run = BaseTool.run
    generate = BaseChatModel._generate_with_cache
    with caplog.at_level(logging.DEBUG, logger=_LC_SCOPE):
        with instrument(
            LangChainInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        ):
            assert BaseTool.run is not run
            assert BaseChatModel._generate_with_cache is not generate
        assert BaseTool.run is run
        assert BaseChatModel._generate_with_cache is generate
    skipped = [
        record.getMessage()
        for record in caplog.records
        if record.name == f"{_LC_SCOPE}._execution_context"
        and record.levelno == logging.DEBUG
    ]
    target = ".".join(
        part
        for part in (module_name, class_name, method or "set_config_context")
        if part
    )
    assert f"Skipping execution boundary {target}: not found" in skipped
