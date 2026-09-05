# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Callback-to-semconv operation mapping for LangChain callbacks.

Maps each LangChain callback to the correct GenAI semantic convention
operation name.  Direct callbacks (``on_chat_model_start``,
``on_llm_start``, ``on_tool_start``, ``on_retriever_start``) have a
fixed 1-to-1 mapping.  ``on_chain_start`` requires heuristic
classification because LangChain emits this callback for agents,
workflows, and internal plumbing alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

__all__ = [
    "OperationName",
    "classify_chain_run",
    "resolve_agent_name",
]

# ---------------------------------------------------------------------------
# Operation name constants (sourced from the GenAI semconv enum where
# available, with string fallbacks for values not yet in the enum).
# ---------------------------------------------------------------------------


class OperationName:
    """Canonical GenAI semantic convention operation names."""

    INVOKE_AGENT: str = GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
    INVOKE_WORKFLOW: str = GenAI.GenAiOperationNameValues.INVOKE_WORKFLOW.value


# ---------------------------------------------------------------------------
# LangGraph markers – names and prefixes produced by LangGraph that must
# be recognized when classifying ``on_chain_start`` callbacks.
# ---------------------------------------------------------------------------

LANGGRAPH_NODE_KEY = "langgraph_node"
LANGGRAPH_START_NODE = "__start__"
MIDDLEWARE_PREFIX = "Middleware."
LANGGRAPH_IDENTIFIER = "LangGraph"

# Metadata keys used by callers to override classification.
_META_AGENT_SPAN = "otel_agent_span"
_META_WORKFLOW_SPAN = "otel_workflow_span"
_META_AGENT_NAME = "agent_name"
_META_AGENT_TYPE = "agent_type"
_META_AGENT_ID = "agent_id"
_META_AGENT_DESCRIPTION = "agent_description"
_META_WORKFLOW_NAME = "workflow_name"
_META_LANGCHAIN_AGENT_NAME = "lc_agent_name"
_META_LANGCHAIN_INTEGRATION = "ls_integration"
_META_OTEL_TRACE = "otel_trace"
_LANGCHAIN_CREATE_AGENT = "langchain_create_agent"
_OPERATION_METADATA_KEYS = (
    _META_AGENT_SPAN,
    _META_WORKFLOW_SPAN,
    _META_AGENT_NAME,
    _META_AGENT_TYPE,
    _META_AGENT_ID,
    _META_AGENT_DESCRIPTION,
    _META_WORKFLOW_NAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_agent_graph_name(config: Any) -> str | None:
    """Return the agent name if ``config`` is a compiled ``create_agent`` graph's.

    Unlike the metadata reaching the callbacks, a graph's own bound config is
    never merged with an enclosing agent's, so this identifies nested agents.
    """
    if not isinstance(config, Mapping):
        return None
    metadata = cast("Mapping[str, Any]", config).get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    typed_metadata = cast("Mapping[str, Any]", metadata)
    if (
        typed_metadata.get(_META_LANGCHAIN_INTEGRATION)
        != _LANGCHAIN_CREATE_AGENT
    ):
        return None
    name = typed_metadata.get(_META_LANGCHAIN_AGENT_NAME)
    return str(name) if name else None


def operation_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return operation markers inherited by graph child callbacks."""
    if not metadata:
        return {}
    return {
        key: metadata[key]
        for key in _OPERATION_METADATA_KEYS
        if key in metadata
    }


def without_inherited_operation_metadata(
    metadata: dict[str, Any] | None,
    inherited_operation_metadata: tuple[Mapping[str, Any], ...] | None,
) -> dict[str, Any] | None:
    if not metadata or not inherited_operation_metadata:
        return metadata
    filtered_metadata = dict(metadata)
    for inherited_metadata in inherited_operation_metadata:
        for key, value in inherited_metadata.items():
            if key in filtered_metadata and filtered_metadata[key] == value:
                filtered_metadata.pop(key)
    return filtered_metadata


def resolve_agent_name(
    serialized: dict[str, Any],
    metadata: dict[str, Any] | None,
    kwargs: dict[str, Any],
    declared_agent_name: str | None = None,
    ancestor_agent_names: set[str] | None = None,
    announced_agent: bool = False,
) -> str | None:
    """Derive the best-effort agent name from callback arguments.

    Checks (in priority order):
    1. ``metadata["agent_name"]`` unless inherited from an enclosing agent
    2. ``declared_agent_name`` (the name a ``create_agent`` graph announced)
    3. ``kwargs["name"]``
    4. ``serialized["name"]``
    5. ``metadata["langgraph_node"]`` (if present and not a start node)
    """
    if metadata:
        name = metadata.get(_META_AGENT_NAME)
        if name:
            metadata_name = str(name)
            if not (
                ancestor_agent_names
                and metadata_name.lower() in ancestor_agent_names
            ):
                return metadata_name

    if declared_agent_name:
        return declared_agent_name
    if announced_agent:
        return None

    name = kwargs.get("name")
    if name:
        return str(name)

    name = serialized.get("name") if serialized else None
    if name:
        return str(name)

    if metadata:
        node = metadata.get(LANGGRAPH_NODE_KEY)
        if node and node != LANGGRAPH_START_NODE:
            return str(node)

    return None


def _has_agent_signals(
    metadata: dict[str, Any] | None,
    ancestor_agent_names: set[str] | None = None,
) -> bool:
    """Return True when metadata contains any signal that the chain is an agent.

    ``create_agent`` graphs are recognized by their announcement instead - the
    metadata reaching a nested agent's callbacks describes its enclosing agent.
    """
    if not metadata:
        return False
    metadata_name = metadata.get(_META_AGENT_NAME)
    inherited_name = bool(
        metadata_name
        and ancestor_agent_names
        and str(metadata_name).lower() in ancestor_agent_names
    )
    return bool(
        (metadata_name and not inherited_name)
        or metadata.get(_META_AGENT_TYPE)
    )


def _looks_like_workflow(
    serialized: dict[str, Any],
    parent_run_id: UUID | None,
) -> bool:
    """Return True if the chain looks like a top-level workflow/graph."""
    if parent_run_id is not None:
        return False

    # Heuristic: check for LangGraph identifier in the serialized repr.
    if serialized:
        name = serialized.get("name", "")
        graph_id = (
            serialized.get("graph", {}).get("id", "")
            if isinstance(serialized.get("graph"), dict)
            else ""
        )
        return LANGGRAPH_IDENTIFIER in name or LANGGRAPH_IDENTIFIER in graph_id

    # No serialized data to inspect, but this is a top-level chain
    # (parent_run_id is None). When we have zero information about a root-level
    # chain we prefer to emit a span rather than silently drop it — more data
    # is better than missing the outermost invocation entirely. Treat it as a
    # workflow so the outermost operation always gets a span even when the
    # chain didn't populate its serialized representation.
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _should_ignore_chain(
    metadata: dict[str, Any] | None,
    agent_name: str | None,
    kwargs: dict[str, Any],
    declared_agent_name: str | None = None,
) -> bool:
    """Return True if the chain callback should be silently suppressed.

    Suppression happens when:
    * The node is the LangGraph ``__start__`` node.
    * The name carries the ``Middleware.`` prefix.
    * ``metadata["otel_trace"]`` is explicitly ``False``.
    * ``metadata["otel_agent_span"]`` is explicitly ``False`` and no other
      agent signals are present.
    """
    if metadata:
        node = metadata.get(LANGGRAPH_NODE_KEY)
        if node == LANGGRAPH_START_NODE:
            return True

        if metadata.get(_META_OTEL_TRACE) is False:
            return True

        if (
            metadata.get(_META_AGENT_SPAN) is False
            and not metadata.get(_META_AGENT_NAME)
            and not metadata.get(_META_AGENT_TYPE)
            and not metadata.get(_META_WORKFLOW_SPAN)
        ):
            return True

    if not declared_agent_name:
        if agent_name and agent_name.startswith(MIDDLEWARE_PREFIX):
            return True

        name_from_kwargs = kwargs.get("name", "")
        if isinstance(name_from_kwargs, str) and name_from_kwargs.startswith(
            MIDDLEWARE_PREFIX
        ):
            return True

    return False


def classify_chain_run(
    serialized: dict[str, Any],
    metadata: dict[str, Any] | None,
    kwargs: dict[str, Any],
    parent_run_id: UUID | None = None,
    declared_agent_name: str | None = None,
    announced_agent: bool = False,
    ancestor_agent_names: set[str] | None = None,
    announced_workflow: bool = False,
    inherited_operation_metadata: tuple[Mapping[str, Any], ...] | None = None,
) -> str | None:
    """Classify a ``on_chain_start`` callback into a semconv operation.

    Returns one of the :class:`OperationName` constants, or ``None`` when
    the chain should be suppressed (no span emitted).

    Classification order:
    1. Check for explicit suppression signals.
    2. Honor graph announcements.
    3. Honor explicit agent and workflow overrides.
    4. Check remaining agent and workflow signals.
    5. Suppress unclassified chains.
    """
    effective_metadata = without_inherited_operation_metadata(
        metadata,
        inherited_operation_metadata,
    )
    agent_name = resolve_agent_name(
        serialized,
        effective_metadata,
        kwargs,
        declared_agent_name,
        ancestor_agent_names,
        announced_agent,
    )

    # 1. Suppress known noise.
    if _should_ignore_chain(
        effective_metadata,
        agent_name,
        kwargs,
        declared_agent_name,
    ):
        return None

    # 2. Graph announcements come from the graph's own bound config.
    if announced_agent or declared_agent_name:
        return OperationName.INVOKE_AGENT

    if announced_workflow:
        return OperationName.INVOKE_WORKFLOW

    # 3. Explicit callback metadata.
    if effective_metadata and effective_metadata.get(_META_AGENT_SPAN):
        return OperationName.INVOKE_AGENT

    if effective_metadata and effective_metadata.get(_META_WORKFLOW_SPAN):
        return OperationName.INVOKE_WORKFLOW

    # 4. Remaining callback signals.
    if _has_agent_signals(effective_metadata, ancestor_agent_names):
        return OperationName.INVOKE_AGENT

    if _looks_like_workflow(serialized, parent_run_id):
        return OperationName.INVOKE_WORKFLOW

    # 5. Default: suppress unclassified chains.
    return None
