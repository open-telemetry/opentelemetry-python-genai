# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Agno instrumentation."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from agno.knowledge.document.base import Document

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    RetrievalDocument,
    ToolDefinition,
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


@runtime_checkable
class _ModelDumpJson(Protocol):
    def model_dump_json(self) -> str: ...


@runtime_checkable
class _JsonDump(Protocol):
    def json(self) -> str: ...


@runtime_checkable
class _ModelDump(Protocol):
    def model_dump(self) -> Any: ...


@runtime_checkable
class _DictDump(Protocol):
    def dict(self) -> Any: ...


def _json_default(obj: object) -> object:
    if isinstance(obj, type):
        return str(obj)
    if isinstance(obj, _ModelDump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, _DictDump):
        try:
            return obj.dict()
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            pass
    return str(obj)


def format_content(val: object) -> str:
    """Format content into a string, converting structured objects to JSON."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, type):
        return str(val)
    if isinstance(val, _ModelDumpJson):
        try:
            return str(val.model_dump_json())
        except Exception:
            pass
    if isinstance(val, _JsonDump):
        try:
            return str(val.json())
        except Exception:
            pass
    if isinstance(val, _ModelDump):
        try:
            return json.dumps(
                val.model_dump(), ensure_ascii=False, default=_json_default
            )
        except Exception:
            pass
    if isinstance(val, _DictDump):
        try:
            return json.dumps(
                val.dict(), ensure_ascii=False, default=_json_default
            )
        except Exception:
            pass
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        try:
            return json.dumps(
                dataclasses.asdict(val),
                ensure_ascii=False,
                default=_json_default,
            )
        except Exception:
            pass
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False, default=_json_default)
        except Exception:
            pass
    return str(cast(object, val))


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


def _normalize_tools(
    tools: Iterable[Any] | str | None,
) -> list[Any] | None:
    """Normalize tool collection input into a list of tool objects."""
    if not tools:
        return None
    if isinstance(tools, str):
        try:
            parsed: Any = json.loads(tools)
            if isinstance(parsed, list):
                return cast(list[Any], parsed)
            if isinstance(parsed, dict):
                return [cast(dict[str, Any], parsed)]
        except Exception:
            pass
        return None
    if (
        isinstance(tools, (list, tuple))
        and tools
        and all(isinstance(c, str) and len(c) == 1 for c in tools)
    ):
        return _normalize_tools("".join(tools))
    if isinstance(tools, dict):
        return [cast(dict[str, Any], tools)]
    try:
        return list(cast(Iterable[Any], tools))
    except TypeError:
        return None


def _extract_tool_definitions(tool: Any) -> list[FunctionToolDefinition]:
    """Extract tool definitions from a single tool item."""
    if isinstance(tool, str):
        try:
            parsed: Any = json.loads(tool)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [
                d
                for d in (
                    prepare_tool_definitions(cast(list[Any], parsed)) or []
                )
                if isinstance(d, FunctionToolDefinition)
            ]
        if isinstance(parsed, dict):
            tool = cast(dict[str, Any], parsed)
        else:
            return []

    # Skip tool execution records (which have tool_call_id)
    if isinstance(tool, dict):
        if "tool_call_id" in tool:
            return []
    elif getattr(tool, "tool_call_id", None) is not None:
        return []

    # 1. Dict format
    if isinstance(tool, dict):
        tool_dict = cast(dict[str, Any], tool)
        func_dict: dict[str, Any]
        if tool_dict.get("type") == "function" and isinstance(
            tool_dict.get("function"), dict
        ):
            func_dict = cast(dict[str, Any], tool_dict["function"])
        elif "name" in tool_dict:
            func_dict = tool_dict
        else:
            return []
        name = func_dict.get("name")
        if not name:
            return []
        desc = func_dict.get("description")
        return [
            FunctionToolDefinition(
                name=str(name),
                description=str(desc) if desc is not None else None,
                parameters=func_dict.get("parameters"),
            )
        ]

    # 2. Toolkit with functions / get_functions
    if hasattr(tool, "functions") or hasattr(tool, "get_functions"):
        try:
            funcs_fn = _get_property_value(tool, "get_functions")
            funcs = (
                funcs_fn()
                if callable(funcs_fn)
                else _get_property_value(tool, "functions")
            )
            if isinstance(funcs, dict):
                return [
                    d
                    for d in (
                        prepare_tool_definitions(
                            list(cast(dict[str, Any], funcs).values())
                        )
                        or []
                    )
                    if isinstance(d, FunctionToolDefinition)
                ]
        except Exception:
            pass
        return []

    # 3. Agno Function object (has name and parameters attributes)
    if hasattr(tool, "name") and hasattr(tool, "parameters"):
        name = _get_property_value(tool, "name")
        if name:
            return [
                FunctionToolDefinition(
                    name=str(name),
                    description=_extract_desc(tool),
                    parameters=_get_property_value(tool, "parameters"),
                )
            ]
        return []

    # 4. Callable
    if callable(tool):
        try:
            import agno.tools.function  # pylint: disable=import-outside-toplevel

            fn_cls = _get_property_value(agno.tools.function, "Function")
            func = fn_cls.from_callable(tool)
            name = _get_property_value(func, "name")
            if name:
                return [
                    FunctionToolDefinition(
                        name=str(name),
                        description=_extract_desc(func),
                        parameters=_get_property_value(func, "parameters"),
                    )
                ]
        except Exception:
            pass
        name = _get_property_value(tool, "__name__") or str(tool)
        desc = _get_property_value(tool, "__doc__")
        return [
            FunctionToolDefinition(
                name=str(name),
                description=str(desc).strip() if desc is not None else None,
                parameters=None,
            )
        ]

    return []


def prepare_tool_definitions(
    tools: Iterable[Any] | str | None,
) -> list[ToolDefinition] | None:
    """Extract tool definitions from Agno Agent tools."""
    raw_tools = _normalize_tools(tools)
    if not raw_tools:
        return None

    seen_names: set[str] = set()
    definitions: list[ToolDefinition] = []

    for item in raw_tools:
        for defn in _extract_tool_definitions(item):
            name = defn.name
            if name and name not in seen_names:
                seen_names.add(name)
                definitions.append(defn)

    return definitions or None
