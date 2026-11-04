# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Agno instrumentation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agno.knowledge.document.base import Document

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    RetrievalDocument,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    get_argument,
    object_to_any_value,
)


def format_retrieval_document(doc: Document) -> RetrievalDocument:
    """Format an Agno Document into a RetrievalDocument."""
    score: float | None = None
    if doc.reranking_score is not None:
        try:
            score = float(doc.reranking_score)
        except (ValueError, TypeError):
            pass
    return RetrievalDocument(
        id=str(doc.id) if doc.id is not None else None,
        score=score,
    )


def format_content(val: object) -> str:
    """Format content into a string, converting structured objects to JSON."""
    if isinstance(val, str):
        return val
    any_value = object_to_any_value(val)
    if any_value is None:
        return str(val)
    return gen_ai_json_dumps(any_value, ensure_ascii=False)


def _get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)

    return getattr(obj, property_name, None)


def _extract_desc(tool: Any) -> str | None:
    desc = _get_property_value(tool, "description")
    if not desc:
        entrypoint = _get_property_value(tool, "entrypoint")
        if entrypoint:
            desc = _get_property_value(entrypoint, "__doc__")
    return str(desc).strip() if desc else None


def prepare_tool_definitions(
    tools: Iterable[Any] | None,
) -> list[ToolDefinition] | None:
    """Extract tool definitions from Agno Agent tools."""
    if not tools:
        return None

    seen_names: set[str] = set()
    definitions: list[ToolDefinition] = []

    def _add_def(name: str, desc: str | None, params: Any) -> None:
        if not name or name in seen_names:
            return
        seen_names.add(name)
        definitions.append(
            FunctionToolDefinition(
                name=name,
                description=desc,
                parameters=params,
            )
        )

    for tool in tools:
        if isinstance(tool, dict):
            if (
                "type" in tool
                and _get_property_value(tool, "type") == "function"
                and isinstance(_get_property_value(tool, "function"), dict)
            ):
                func_dict = _get_property_value(tool, "function")
                _add_def(
                    str(_get_property_value(func_dict, "name") or ""),
                    str(_get_property_value(func_dict, "description"))
                    if _get_property_value(func_dict, "description")
                    is not None
                    else None,
                    _get_property_value(func_dict, "parameters"),
                )
            elif "name" in tool:
                _add_def(
                    str(_get_property_value(tool, "name") or ""),
                    str(_get_property_value(tool, "description"))
                    if _get_property_value(tool, "description") is not None
                    else None,
                    _get_property_value(tool, "parameters"),
                )
        elif hasattr(tool, "functions") or hasattr(tool, "get_functions"):
            try:
                funcs = None
                if hasattr(tool, "get_functions") and callable(
                    _get_property_value(tool, "get_functions")
                ):
                    funcs_fn = _get_property_value(tool, "get_functions")
                    if callable(funcs_fn):
                        funcs = funcs_fn()
                else:
                    funcs = _get_property_value(tool, "functions")
                if isinstance(funcs, dict):
                    sub_defs = prepare_tool_definitions(
                        list(cast(dict[str, Any], funcs).values())
                    )
                    if sub_defs:
                        for defn in sub_defs:
                            _add_def(
                                _get_property_value(defn, "name") or "",
                                _get_property_value(defn, "description"),
                                _get_property_value(defn, "parameters"),
                            )
            except Exception:
                pass
        elif hasattr(tool, "name") and hasattr(tool, "parameters"):
            name = _get_property_value(tool, "name") or ""
            desc = _extract_desc(tool)
            params = _get_property_value(tool, "parameters")
            _add_def(
                str(name),
                desc,
                params,
            )
        elif callable(tool):
            try:
                import agno.tools.function  # pylint: disable=import-outside-toplevel

                fn_cls = _get_property_value(agno.tools.function, "Function")
                func = fn_cls.from_callable(tool)
                name = _get_property_value(func, "name") or ""
                desc = _extract_desc(func)
                params = _get_property_value(func, "parameters")
                _add_def(
                    str(name),
                    desc,
                    params,
                )
            except Exception:
                name = _get_property_value(tool, "__name__") or str(tool)
                desc = _get_property_value(tool, "__doc__")
                _add_def(
                    str(name),
                    str(desc).strip() if desc is not None else None,
                    None,
                )

    return definitions or None


def extract_session_id(
    instance: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_response: Any = None,
    wrapped: Callable[..., Any] | None = None,
    *,
    is_continue: bool = False,
) -> str | None:
    """Extract session_id from call arguments, instance, or response."""
    if is_continue:
        if run_response is None:
            if wrapped is not None and (args or kwargs):
                run_response = get_argument(
                    "run_response", wrapped, args or (), kwargs or {}
                )
            elif kwargs:
                run_response = kwargs.get("run_response")

        if run_response is not None:
            if (
                session_id := getattr(run_response, "session_id", None)
            ) is not None:
                return str(session_id)
            if instance is not None and (
                (session_id := getattr(instance, "session_id", None))
                is not None
            ):
                return str(session_id)
            return None

    if wrapped is not None and (args or kwargs):
        session_id = get_argument(
            "session_id", wrapped, args or (), kwargs or {}
        )
        if session_id is not None:
            return str(session_id)
    else:
        if kwargs and (session_id := kwargs.get("session_id")) is not None:
            return str(session_id)
        if args and len(args) > 4 and args[4] is not None:
            return str(args[4])
    if instance is not None and (
        (session_id := getattr(instance, "session_id", None)) is not None
    ):
        return str(session_id)
    if run_response is not None and (
        (session_id := getattr(run_response, "session_id", None)) is not None
    ):
        return str(session_id)
    return None
