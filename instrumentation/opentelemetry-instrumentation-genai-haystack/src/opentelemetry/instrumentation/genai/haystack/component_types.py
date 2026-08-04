# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-effort classification of Haystack components into GenAI operations.

Haystack exposes a single duck-typed ``Component`` protocol — there is no
built-in way to ask "is this an LLM?" Classification is inferred from the
component class name and the type hints on its ``run`` method (Haystack's
``@component`` decorator populates an ``_output_types_cache`` used here).

Only classifications with a direct ``opentelemetry-util-genai`` invocation
type are recognized (``GENERATOR`` -> inference, ``EMBEDDER`` -> embedding,
``RANKER`` / ``RETRIEVER`` -> retrieval, ``AGENT`` -> agent invocation).
Components that don't match any of these (prompt builders, routers,
converters, joiners, ...) classify as ``UNKNOWN`` and are not wrapped:
there is no util-genai invocation type for a generic component step, and
inventing one here would violate the "telemetry only through
opentelemetry-util-genai public types" rule.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)


class ComponentType(Enum):
    GENERATOR = auto()
    EMBEDDER = auto()
    RANKER = auto()
    RETRIEVER = auto()
    AGENT = auto()
    UNKNOWN = auto()


def get_component_run_method(component: Any) -> Optional[Callable[..., Any]]:
    """Return the component's ``run`` method, if it has one."""
    if callable(run_method := getattr(component, "run", None)):
        return run_method
    return None


def _get_run_method_output_types(
    run_method: Callable[..., Any],
) -> Optional[Dict[str, type]]:
    """Read the ``@component(output_types=...)``-populated cache off ``run``.

    See https://github.com/deepset-ai/haystack/blob/main/haystack/core/component/component.py
    """
    output_types_cache = getattr(run_method, "_output_types_cache", None)
    if isinstance(output_types_cache, dict):
        return {key: value.type for key, value in output_types_cache.items()}
    return None


def _get_run_method_input_types(
    run_method: Callable[..., Any],
) -> Optional[Dict[str, type]]:
    try:
        return get_type_hints(run_method)
    except Exception:  # pylint: disable=broad-except
        # get_type_hints can raise on forward refs it can't resolve; treat as unknown.
        return None


def _is_list_of_documents_type(type_hint: Any) -> bool:
    from haystack import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
        Document,
    )

    origin = get_origin(type_hint)
    if origin is not list:
        return False
    args = get_args(type_hint)
    return len(args) == 1 and args[0] is Document


def _has_generator_output_type(run_method: Callable[..., Any]) -> bool:
    from haystack.dataclasses.chat_message import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
        ChatMessage,
    )

    output_types = _get_run_method_output_types(run_method)
    if (
        output_types is None
        or (replies := output_types.get("replies")) is None
    ):
        return False
    return replies == list[ChatMessage] or replies == list[str]


def _has_ranker_io_types(run_method: Callable[..., Any]) -> bool:
    input_types = _get_run_method_input_types(run_method)
    output_types = _get_run_method_output_types(run_method)
    if input_types is None or output_types is None:
        return False
    has_documents_param = _is_list_of_documents_type(
        input_types.get("documents")
    )
    outputs_documents = _is_list_of_documents_type(
        output_types.get("documents")
    )
    return has_documents_param and outputs_documents


def _has_retriever_io_types(run_method: Callable[..., Any]) -> bool:
    """A retriever outputs ``List[Document]`` without taking documents as input.

    This also catches retrievers with no document input at all, e.g.
    ``SerperDevWebSearch``, which produces documents from a search query.
    """
    input_types = _get_run_method_input_types(run_method)
    output_types = _get_run_method_output_types(run_method)
    if input_types is None or output_types is None:
        return False
    has_documents_param = "documents" in input_types
    outputs_documents = _is_list_of_documents_type(
        output_types.get("documents")
    )
    return not has_documents_param and outputs_documents


def get_component_type(component: Any) -> ComponentType:
    """Classify a component, given either its class or an instance.

    Classification is wired up while walking the Haystack component
    *class* registry (before any instance exists), so this must work off a
    bare class: the type-hint/``_output_types_cache`` heuristics below are
    class-level artifacts set by the ``@component`` decorator, so they work
    identically either way -- only the class-name lookup needs to branch.
    """
    is_class = isinstance(component, type)
    component_name = (
        component.__name__ if is_class else component.__class__.__name__
    )
    run_method = (
        getattr(component, "run", None)
        if is_class
        else get_component_run_method(component)
    )
    if not callable(run_method):
        return ComponentType.UNKNOWN
    if "Agent" in component_name:
        return ComponentType.AGENT
    if "Generator" in component_name or _has_generator_output_type(run_method):
        return ComponentType.GENERATOR
    if "Embedder" in component_name:
        return ComponentType.EMBEDDER
    if "Ranker" in component_name and _has_ranker_io_types(run_method):
        return ComponentType.RANKER
    if (
        "Retriever" in component_name or "WebSearch" in component_name
    ) and _has_retriever_io_types(run_method):
        return ComponentType.RETRIEVER
    return ComponentType.UNKNOWN
