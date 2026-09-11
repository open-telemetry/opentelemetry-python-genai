# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Identify graph roots outside the LangChain callback API.

Callback metadata cannot distinguish a nested graph root from its internal
runnables because parent config is inherited. Graph entry points announce the
root on a context stack so only the matching callback can claim it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

from opentelemetry.instrumentation.genai.langchain.operation_mapping import (
    create_agent_graph_name,
)

_REACT_AGENT_MODULE = "langgraph.prebuilt.chat_agent_executor"


@dataclass
class _PendingGraph:
    name: str | None
    is_agent: bool
    metadata: dict[str, Any]
    claimed: bool = False


_pending: ContextVar[tuple[_PendingGraph, ...]] = ContextVar(
    "otel_genai_pending_graphs", default=()
)


def claim_graph() -> _PendingGraph | None:
    """Claim the innermost graph announcement for its root callback.

    Only the newest announcement is claimable, and only once, so nested graphs
    do not leak their classification to internal callbacks.
    """
    pending = _pending.get()
    if not pending:
        return None
    entry = pending[-1]
    if entry.claimed:
        return None
    entry.claimed = True
    return entry


def _react_agent_name(graph: Any) -> str | None:
    """Return the name of a deprecated ``create_react_agent`` graph.

    Checking the builder function's runtime callable avoids import-order
    dependence without treating every user graph with a node named ``agent`` as
    an agent.
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, Mapping):
        return None
    agent_node = cast("Mapping[str, Any]", nodes).get("agent")
    bound = getattr(agent_node, "bound", None)
    function = getattr(bound, "func", None)
    if getattr(function, "__module__", None) != _REACT_AGENT_MODULE:
        return None
    name = getattr(graph, "name", None)
    return str(name) if name and name != "LangGraph" else ""


def _bound_metadata(graph: Any) -> dict[str, Any]:
    config = getattr(graph, "config", None)
    if not isinstance(config, Mapping):
        return {}
    metadata = cast("Mapping[str, Any]", config).get("metadata")
    if not isinstance(metadata, Mapping):
        return {}
    return dict(cast("Mapping[str, Any]", metadata))


def _agent_name(
    graph: Any,
    metadata: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Return whether ``graph`` is an agent and its application-provided name."""
    config = getattr(graph, "config", None)
    create_agent_name = create_agent_graph_name(config)
    if create_agent_name:
        return True, create_agent_name
    if metadata.get("ls_integration") == "langchain_create_agent":
        return True, None
    react_name = _react_agent_name(graph)
    if react_name is not None:
        return True, react_name or None
    if (
        metadata.get("otel_agent_span")
        or metadata.get("agent_type")
        or metadata.get("agent_name")
    ):
        name = metadata.get("agent_name")
        return True, str(name) if name else None
    return False, None


def _push(
    name: str | None,
    is_agent: bool,
    metadata: dict[str, Any],
) -> _PendingGraph:
    entry = _PendingGraph(
        name=name,
        is_agent=is_agent,
        metadata=metadata,
    )
    _pending.set(_pending.get() + (entry,))
    return entry


def _pop(entry: _PendingGraph) -> None:
    """Withdraw ``entry``, tolerating a stack the caller's context no longer owns."""
    pending = _pending.get()
    if pending and pending[-1] is entry:
        _pending.set(pending[:-1])


def wrap_stream(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Announce a graph for the duration of ``Pregel.stream``.

    ``Pregel.invoke`` runs through ``stream``, so this covers both entry points.
    """
    metadata = _bound_metadata(instance)
    is_agent, name = _agent_name(instance, metadata)
    return _announce_at_stream_start(
        wrapped(*args, **kwargs),
        name,
        is_agent,
        metadata,
    )


def wrap_astream(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Announce a graph for the duration of ``Pregel.astream``.

    ``Pregel.ainvoke`` runs through ``astream``, so this covers both entry points.
    """
    metadata = _bound_metadata(instance)
    is_agent, name = _agent_name(instance, metadata)
    return _announce_at_astream_start(
        wrapped(*args, **kwargs),
        name,
        is_agent,
        metadata,
    )


def _announce_at_stream_start(
    stream: Iterator[Any],
    name: str | None,
    is_agent: bool,
    metadata: dict[str, Any],
) -> Iterator[Any]:
    """Announce ``name`` for the first step of ``stream`` only.

    The graph opens its root run before it can produce anything, so the
    announcement has served its purpose once the first item is in hand.
    Withdrawing it there keeps an abandoned stream from leaving the
    announcement standing until the generator is garbage collected - by which
    point ``_pop`` would run in an unrelated context and quietly do nothing.
    """
    # A generator body runs in the consumer's context, so the announcement lands
    # where the callbacks fire - on the first ``next()``, not here.
    iterator = iter(stream)
    entry = _push(name, is_agent, metadata)
    try:
        first = next(iterator)
    except StopIteration:
        return
    finally:
        _pop(entry)
    yield first
    yield from iterator


async def _announce_at_astream_start(
    stream: AsyncIterator[Any],
    name: str | None,
    is_agent: bool,
    metadata: dict[str, Any],
) -> AsyncIterator[Any]:
    """Announce ``name`` for the first step of ``stream`` only."""
    iterator = stream.__aiter__()
    entry = _push(name, is_agent, metadata)
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        return
    finally:
        _pop(entry)
    yield first
    async for chunk in iterator:
        yield chunk
