# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from opentelemetry import context
from opentelemetry.semconv.attributes import error_attributes

from .test_execution_context import (
    _HTTP_SCOPE,
    _LC_SCOPE,
    _OPENAI_SCOPE,
    _child,
    _spans,
    clients,
    no_detach_errors,
)
from .test_stream_context import _assert_no_runs

__all__ = ["clients", "no_detach_errors"]


def _custom_operation(
    kind: str,
    call: Callable[[str], str],
    acall: Callable[[str], Awaitable[str]] | None = None,
) -> BaseTool | BaseRetriever:
    class CustomTool(BaseTool):
        name: str = "custom"
        description: str = "Custom tool execution."

        def _run(self, text: str) -> str:
            return call(text)

    class AsyncCustomTool(CustomTool):
        async def _arun(self, text: str) -> str:
            assert acall is not None
            return await acall(text)

    class CustomRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> list[Document]:
            return [Document(page_content=call(query))]

    class AsyncCustomRetriever(CustomRetriever):
        async def _aget_relevant_documents(self, query: str) -> list[Document]:
            assert acall is not None
            return [Document(page_content=await acall(query))]

    if kind == "tool":
        return CustomTool() if acall is None else AsyncCustomTool()
    return CustomRetriever() if acall is None else AsyncCustomRetriever()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "retriever"])
@pytest.mark.parametrize("mode", ["sync", "async", "executor"])
@pytest.mark.parametrize("parented", [False, True])
async def test_custom_execution_correlates_inference_and_http(
    clients: Any,
    span_exporter: Any,
    tracer_provider: Any,
    kind: str,
    mode: str,
    parented: bool,
) -> None:
    def call(text: str) -> str:
        clients.http.get("https://example.test/custom")
        return clients.infer(text)

    async def acall(text: str) -> str:
        await clients.ahttp.get("https://example.test/custom")
        return await clients.ainfer(text)

    operation = _custom_operation(
        kind, call, acall if mode == "async" else None
    )
    parent = (
        tracer_provider.get_tracer("test").start_as_current_span("request")
        if parented
        else nullcontext()
    )
    with parent as root:
        before = context.get_current()
        result = (
            operation.invoke("hello")
            if mode == "sync"
            else await operation.ainvoke("hello")
        )
        assert (
            result if kind == "tool" else result[0].page_content
        ) == "answer"
        assert context.get_current() is before
    (execution,) = _spans(span_exporter, _LC_SCOPE)
    assert execution.parent == (root.get_span_context() if root else None)
    (inference,) = _spans(span_exporter, _OPENAI_SCOPE)
    direct_http, provider_http = _spans(span_exporter, _HTTP_SCOPE)
    _child(direct_http, execution)
    _child(inference, execution)
    _child(provider_http, inference)
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "retriever"])
@pytest.mark.parametrize("mode", ["sync", "async", "executor", "sync_cancel"])
async def test_custom_execution_error_restores_context(
    clients: Any,
    span_exporter: Any,
    kind: str,
    mode: str,
) -> None:
    error = (
        asyncio.CancelledError("cancelled")
        if mode == "sync_cancel"
        else ConnectionError("failed")
    )

    def call(text: str) -> str:
        clients.http.get("https://example.test/custom")
        raise error

    async def acall(text: str) -> str:
        await clients.ahttp.get("https://example.test/custom")
        raise error

    operation = _custom_operation(
        kind, call, acall if mode == "async" else None
    )
    before = context.get_current()
    with pytest.raises(type(error)) as raised:
        if mode in ("sync", "sync_cancel"):
            operation.invoke("hello")
        else:
            await operation.ainvoke("hello")
    assert raised.value is error
    assert context.get_current() is before
    (execution,) = _spans(span_exporter, _LC_SCOPE)
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    _child(http, execution)
    assert execution.attributes[error_attributes.ERROR_TYPE] == (
        "asyncio.exceptions.CancelledError"
        if mode == "sync_cancel"
        else "ConnectionError"
    )
    _assert_no_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "retriever"])
async def test_custom_execution_task_cancellation(
    clients: Any,
    span_exporter: Any,
    tracer_provider: Any,
    kind: str,
) -> None:
    started = asyncio.Event()
    errors: list[BaseException] = []

    async def acall(text: str) -> str:
        await clients.ahttp.get("https://example.test/custom")
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            errors.append(error)
            raise
        return text

    operation = _custom_operation(kind, lambda text: text, acall)
    with tracer_provider.get_tracer("test").start_as_current_span(
        "request"
    ) as root:
        before = context.get_current()

        async def request() -> None:
            try:
                await operation.ainvoke("hello")
            except asyncio.CancelledError as error:
                assert error is errors[0]
                assert error.args == ("cancelled by test",)
                raise
            finally:
                assert context.get_current() is before

        task = asyncio.create_task(request())
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel("cancelled by test")
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert context.get_current() is before
    (execution,) = _spans(span_exporter, _LC_SCOPE)
    (http,) = _spans(span_exporter, _HTTP_SCOPE)
    assert execution.parent == root.get_span_context()
    _child(http, execution)
    assert (
        execution.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )
    _assert_no_runs()
