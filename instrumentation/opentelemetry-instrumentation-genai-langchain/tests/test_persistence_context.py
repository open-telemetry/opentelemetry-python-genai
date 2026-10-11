# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from inspect import signature
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.store.memory import InMemoryStore

from opentelemetry import context
from opentelemetry.semconv.attributes import error_attributes

from .test_execution_context import (
    _HTTP_SCOPE,
    _LC_SCOPE,
    _child,
    _spans,
    clients,
    no_detach_errors,
)
from .test_stream_context import _assert_no_runs

__all__ = ["clients", "no_detach_errors"]

# LangGraph 0.6 replaced checkpoint_during with durability modes.
_durability = pytest.mark.skipif(
    "durability" not in signature(Pregel.stream).parameters,
    reason="durability modes need LangGraph >= 0.6",
)


class _State(TypedDict):
    text: str


def _saver(
    clients: Any,
    failure: str | None = None,
    error: BaseException | None = None,
) -> InMemorySaver:
    class Saver(InMemorySaver):
        def observe(self, op: str, config: dict) -> None:
            clients.http.get(
                f"https://example.test/checkpoint/{op}",
                params={"thread": config["configurable"]["thread_id"]},
            )
            if op == failure:
                raise error

        async def aobserve(self, op: str, config: dict) -> None:
            await clients.ahttp.get(
                f"https://example.test/checkpoint/{op}",
                params={"thread": config["configurable"]["thread_id"]},
            )
            await asyncio.sleep(0)
            if op == failure:
                raise error

        def get_tuple(self, config: dict) -> Any:
            self.observe("get", config)
            return super().get_tuple(config)

        def put(
            self,
            config: dict,
            checkpoint: Any,
            metadata: Any,
            new_versions: Any,
        ) -> Any:
            self.observe("put", config)
            return super().put(config, checkpoint, metadata, new_versions)

        def put_writes(
            self, config: dict, writes: Any, task_id: str, task_path: str = ""
        ) -> None:
            self.observe("writes", config)
            super().put_writes(config, writes, task_id, task_path)

        async def aget_tuple(self, config: dict) -> Any:
            await self.aobserve("get", config)
            return InMemorySaver.get_tuple(self, config)

        async def aput(
            self,
            config: dict,
            checkpoint: Any,
            metadata: Any,
            new_versions: Any,
        ) -> Any:
            await self.aobserve("put", config)
            return InMemorySaver.put(
                self, config, checkpoint, metadata, new_versions
            )

        async def aput_writes(
            self, config: dict, writes: Any, task_id: str, task_path: str = ""
        ) -> None:
            await self.aobserve("writes", config)
            InMemorySaver.put_writes(self, config, writes, task_id, task_path)

    return Saver()


def _store(clients: Any, error: BaseException | None = None) -> InMemoryStore:
    class Store(InMemoryStore):
        def batch(self, ops: Any) -> Any:
            ops = list(ops)
            for op in ops:
                clients.http.get(
                    f"https://example.test/store/{type(op).__name__}"
                )
                if error is not None:
                    raise error
            return super().batch(ops)

        async def abatch(self, ops: Any) -> Any:
            ops = list(ops)
            for op in ops:
                await clients.ahttp.get(
                    f"https://example.test/store/{type(op).__name__}"
                )
                if error is not None:
                    raise error
            return await super().abatch(ops)

    return Store()


def _graph(
    saver: InMemorySaver, store: InMemoryStore, asynchronous: bool
) -> Any:
    def lookup(text: str) -> str:
        result = store.get(("memory",), text)
        assert result is not None
        return result.value["text"]

    async def alookup(text: str) -> str:
        result = await store.aget(("memory",), text)
        assert result is not None
        return result.value["text"]

    tool = StructuredTool.from_function(
        func=lookup,
        coroutine=alookup,
        name="lookup",
        description="Read stored state.",
    )

    def node(state: _State) -> _State:
        store.put(("memory",), state["text"], {"text": state["text"]})
        assert store.search(("memory",))
        return {"text": tool.invoke(state["text"])}

    async def anode(state: _State) -> _State:
        await store.aput(("memory",), state["text"], {"text": state["text"]})
        assert await store.asearch(("memory",))
        return {"text": await tool.ainvoke(state["text"])}

    graph = StateGraph(_State)
    graph.add_node("lookup", anode if asynchronous else node)
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", END)
    return graph.compile(checkpointer=saver, store=store)


def _assert_persistence_parents(
    clients: Any, span_exporter: Any, writes_expected: bool = True
) -> None:
    lc_spans = _spans(span_exporter, _LC_SCOPE)
    by_id = {span.context.span_id: span for span in lc_spans}
    http_spans = {
        span.context.span_id: span
        for span in _spans(span_exporter, _HTTP_SCOPE)
    }
    operations: set[str] = set()
    for request in clients.requests:
        http = http_spans[
            int(request.headers["traceparent"].split("-")[2], 16)
        ]
        parent = by_id[http.parent.span_id]
        if request.url.path.startswith("/checkpoint/"):
            assert parent.name.startswith("invoke_workflow")
            assert (
                parent.attributes["gen_ai.conversation.id"]
                == request.url.params["thread"]
            )
        elif request.url.path == "/store/GetOp":
            assert parent.name == "execute_tool lookup"
        else:
            assert parent.name.startswith("invoke_workflow")
        _child(http, parent)
        operations.add(request.url.path)
    assert {
        "/checkpoint/get",
        "/checkpoint/put",
        "/store/GetOp",
        "/store/PutOp",
        "/store/SearchOp",
    } <= operations
    assert ("/checkpoint/writes" in operations) is writes_expected
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "durability",
    [
        # The default mode; older LangGraph checkpoints the same way.
        None,
        pytest.param("async", marks=_durability),
        pytest.param("exit", marks=_durability),
    ],
)
async def test_checkpoint_and_store_context(
    clients,
    span_exporter,
    asynchronous: bool,
    streaming: bool,
    durability: str | None,
) -> None:
    saver = _saver(clients)
    graph = _graph(saver, _store(clients), asynchronous)
    before = context.get_current()
    config = {"configurable": {"thread_id": "one"}}
    kwargs = {} if durability is None else {"durability": durability}
    for text in ("first", "second"):
        if streaming:
            if asynchronous:
                async for _ in graph.astream({"text": text}, config, **kwargs):
                    assert context.get_current() is before
            else:
                for _ in graph.stream({"text": text}, config, **kwargs):
                    assert context.get_current() is before
        elif asynchronous:
            assert await graph.ainvoke({"text": text}, config, **kwargs) == {
                "text": text
            }
        else:
            assert graph.invoke({"text": text}, config, **kwargs) == {
                "text": text
            }
        assert context.get_current() is before
        assert (
            InMemorySaver.get_tuple(saver, config).checkpoint[
                "channel_values"
            ]["text"]
            == text
        )
    _assert_persistence_parents(clients, span_exporter, durability != "exit")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_shared_saver_keeps_concurrent_workflows_separate(
    clients, span_exporter, asynchronous: bool
) -> None:
    graph = _graph(_saver(clients), _store(clients), asynchronous)

    async def run(name: str) -> None:
        config = {"configurable": {"thread_id": name}}
        before = context.get_current()
        if asynchronous:
            assert await graph.ainvoke({"text": name}, config) == {
                "text": name
            }
        else:
            assert await asyncio.to_thread(
                graph.invoke, {"text": name}, config
            ) == {"text": name}
        assert context.get_current() is before

    await asyncio.gather(run("one"), run("two"))
    roots = [
        span
        for span in _spans(span_exporter, _LC_SCOPE)
        if span.parent is None
    ]
    assert len(roots) == 2
    assert roots[0].context.trace_id != roots[1].context.trace_id
    _assert_persistence_parents(clients, span_exporter)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["get", "put", "writes"])
async def test_checkpoint_error_restores_context(
    clients, span_exporter, asynchronous: bool, operation: str
) -> None:
    error = ConnectionError("checkpoint failed")
    graph = _graph(
        _saver(clients, operation, error), _store(clients), asynchronous
    )
    before = context.get_current()
    with pytest.raises(ConnectionError) as raised:
        if asynchronous:
            await graph.ainvoke(
                {"text": "one"}, {"configurable": {"thread_id": "one"}}
            )
        else:
            graph.invoke(
                {"text": "one"}, {"configurable": {"thread_id": "one"}}
            )
    assert raised.value is error
    assert context.get_current() is before
    (workflow,) = [
        span
        for span in _spans(span_exporter, _LC_SCOPE)
        if span.parent is None
    ]
    assert (
        workflow.attributes[error_attributes.ERROR_TYPE] == "ConnectionError"
    )
    by_id = {
        span.context.span_id: span for span in _spans(span_exporter, _LC_SCOPE)
    }
    for span in _spans(span_exporter, _HTTP_SCOPE):
        _child(span, by_id[span.parent.span_id])
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_store_error_restores_context(
    clients, span_exporter, asynchronous: bool
) -> None:
    error = ConnectionError("store failed")
    graph = _graph(_saver(clients), _store(clients, error), asynchronous)
    before = context.get_current()
    config = {"configurable": {"thread_id": "one"}}
    with pytest.raises(ConnectionError) as raised:
        if asynchronous:
            await graph.ainvoke({"text": "one"}, config)
        else:
            graph.invoke({"text": "one"}, config)
    assert raised.value is error
    assert context.get_current() is before
    by_id = {
        span.context.span_id: span for span in _spans(span_exporter, _LC_SCOPE)
    }
    for span in _spans(span_exporter, _HTTP_SCOPE):
        _child(span, by_id[span.parent.span_id])
    (workflow,) = [span for span in by_id.values() if span.parent is None]
    assert (
        workflow.attributes[error_attributes.ERROR_TYPE] == "ConnectionError"
    )
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_graph_stream_close_restores_context(
    clients, span_exporter, asynchronous: bool
) -> None:
    graph = _graph(_saver(clients), _store(clients), asynchronous)
    config = {"configurable": {"thread_id": "one"}}
    before = context.get_current()
    if asynchronous:
        stream = graph.astream({"text": "one"}, config)
        assert await anext(stream)
        assert context.get_current() is before
        await stream.aclose()
    else:
        stream = graph.stream({"text": "one"}, config)
        assert next(stream)
        assert context.get_current() is before
        stream.close()
    assert context.get_current() is before
    by_id = {
        span.context.span_id: span for span in _spans(span_exporter, _LC_SCOPE)
    }
    for span in _spans(span_exporter, _HTTP_SCOPE):
        _child(span, by_id[span.parent.span_id])
    _assert_no_runs()
