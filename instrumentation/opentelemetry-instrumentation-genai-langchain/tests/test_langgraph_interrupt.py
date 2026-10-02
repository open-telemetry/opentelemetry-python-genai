# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""LangGraph interrupt and resume must not log callback AttributeErrors."""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class _State(TypedDict):
    answer: str


def _build_graph() -> Any:
    def ask(_: _State) -> _State:
        return {"answer": interrupt("approve?")}

    builder = StateGraph(_State)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


def _assert_no_callback_warnings(caplog: pytest.LogCaptureFixture) -> None:
    assert (
        "OpenTelemetryLangChainCallbackHandler.on_interrupt" not in caplog.text
    )
    assert "OpenTelemetryLangChainCallbackHandler.on_resume" not in caplog.text


def test_invoke_interrupt_and_resume_do_not_log_callback_warnings(
    caplog: pytest.LogCaptureFixture, start_instrumentation
) -> None:
    graph = _build_graph()
    config: Any = {"configurable": {"thread_id": "sync"}}

    with caplog.at_level("WARNING"):
        interrupted = graph.invoke({"answer": ""}, config)
        resumed = graph.invoke(Command(resume="yes"), config)

    assert "__interrupt__" in interrupted
    assert resumed == {"answer": "yes"}
    _assert_no_callback_warnings(caplog)


def test_ainvoke_interrupt_and_resume_do_not_log_callback_warnings(
    caplog: pytest.LogCaptureFixture, start_instrumentation
) -> None:
    graph = _build_graph()
    config: Any = {"configurable": {"thread_id": "async"}}

    async def scenario() -> tuple[Any, Any]:
        interrupted = await graph.ainvoke({"answer": ""}, config)
        resumed = await graph.ainvoke(Command(resume="yes"), config)
        return interrupted, resumed

    with caplog.at_level("WARNING"):
        interrupted, resumed = asyncio.run(scenario())

    assert "__interrupt__" in interrupted
    assert resumed == {"answer": "yes"}
    _assert_no_callback_warnings(caplog)
