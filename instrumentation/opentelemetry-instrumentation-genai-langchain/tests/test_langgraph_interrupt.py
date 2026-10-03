# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""LangGraph interrupt and resume must not log callback warnings."""

from __future__ import annotations

import sys
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from opentelemetry.test_util_genai.logs import assert_no_warnings

_CALLBACK_LOGGER = "langchain_core.callbacks.manager"


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


def test_invoke_interrupt_and_resume_do_not_log_callback_warnings(
    caplog: pytest.LogCaptureFixture, start_instrumentation
) -> None:
    graph = _build_graph()
    config: Any = {"configurable": {"thread_id": "sync"}}

    with assert_no_warnings(caplog, _CALLBACK_LOGGER):
        interrupted = graph.invoke({"answer": ""}, config)
        resumed = graph.invoke(Command(resume="yes"), config)

    assert "__interrupt__" in interrupted
    assert resumed == {"answer": "yes"}


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="LangGraph async context propagation requires Python 3.11+",
)
@pytest.mark.asyncio
async def test_ainvoke_interrupt_and_resume_do_not_log_callback_warnings(
    caplog: pytest.LogCaptureFixture, start_instrumentation
) -> None:
    graph = _build_graph()
    config: Any = {"configurable": {"thread_id": "async"}}

    with assert_no_warnings(caplog, _CALLBACK_LOGGER):
        interrupted = await graph.ainvoke({"answer": ""}, config)
        resumed = await graph.ainvoke(Command(resume="yes"), config)

    assert "__interrupt__" in interrupted
    assert resumed == {"answer": "yes"}
