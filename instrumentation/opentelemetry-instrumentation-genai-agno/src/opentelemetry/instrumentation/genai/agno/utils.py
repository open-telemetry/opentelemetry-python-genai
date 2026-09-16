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
    """Format an Agno Document into a RetrievalDocument model."""
    score: float | None = None
    if doc.reranking_score is not None:
        try:
            score = float(doc.reranking_score)
        except (ValueError, TypeError):
            pass

    metadata: dict[str, Any] | None = None
    if doc.meta_data:
        metadata = dict(doc.meta_data)

    return RetrievalDocument(
        content=doc.content,
        id=str(doc.id) if doc.id is not None else None,
        score=score,
        metadata=metadata,
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


def prepare_tool_definitions(
    tools: Iterable[Any] | str | None,
) -> list[ToolDefinition] | None:
    """Extract tool definitions from Agno Agent tools."""
    if not tools:
        return None

    raw_tools: list[Any]
    if isinstance(tools, str):
        try:
            parsed_str_val: Any = json.loads(tools)
            if isinstance(parsed_str_val, list):
                raw_tools = cast(list[Any], parsed_str_val)
            elif isinstance(parsed_str_val, dict):
                raw_tools = [cast(dict[str, Any], parsed_str_val)]
            else:
                return None
        except Exception:
            return None
    elif (
        isinstance(tools, (list, tuple))
        and tools
        and all(isinstance(c, str) and len(c) == 1 for c in tools)
    ):
        try:
            parsed_coerced: Any = json.loads("".join(tools))
            if isinstance(parsed_coerced, list):
                raw_tools = cast(list[Any], parsed_coerced)
            elif isinstance(parsed_coerced, dict):
                raw_tools = [cast(dict[str, Any], parsed_coerced)]
            else:
                return None
        except Exception:
            return None
    elif isinstance(tools, dict):
        raw_tools = [cast(dict[str, Any], tools)]
    else:
        try:
            raw_tools = list(tools)
        except TypeError:
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

    for tool_item in raw_tools:
        tool: Any = tool_item
        if isinstance(tool, str):
            try:
                parsed_tool: Any = json.loads(tool)
                if isinstance(parsed_tool, dict):
                    tool = cast(dict[str, Any], parsed_tool)
                elif isinstance(parsed_tool, list):
                    sub_defs = prepare_tool_definitions(
                        cast(list[Any], parsed_tool)
                    )
                    if sub_defs:
                        for defn in sub_defs:
                            _add_def(
                                str(_get_property_value(defn, "name") or ""),
                                _get_property_value(defn, "description"),
                                _get_property_value(defn, "parameters"),
                            )
                    continue
                else:
                    continue
            except Exception:
                continue

        # Skip tool execution records (which have tool_call_id)
        if isinstance(tool, dict) and "tool_call_id" in cast(
            dict[str, Any], tool
        ):
            continue
        if (
            not isinstance(tool, dict)
            and getattr(cast(object, tool), "tool_call_id", None) is not None
        ):
            continue

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
