# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for DSPy instrumentation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    ToolDefinition,
)

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.primitives.prediction import Prediction


def extract_input_content(input_args: dict[str, Any]) -> str:
    """Extract input content string from agent invocation arguments."""
    if not input_args:
        return ""
    if len(input_args) == 1:
        val: Any = next(iter(input_args.values()))
        if isinstance(val, str):
            return val
        if isinstance(val, (int, float, bool)):
            return str(val)
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    try:
        return json.dumps(input_args, ensure_ascii=False)
    except Exception:
        return str(input_args)


def extract_output_content(
    result: Prediction | None,
    signature: Any = None,
) -> str:
    """Extract output content string from a DSPy prediction result."""
    if result is None:
        return ""

    output_dict: dict[str, Any] = {}
    if hasattr(result, "items") and callable(getattr(result, "items")):
        try:
            items_func: Callable[[], Any] = getattr(result, "items")
            for k, v in cast(Iterable[tuple[Any, Any]], items_func()):
                output_dict[str(k)] = v
        except Exception:
            pass

    if output_dict:
        filtered_dict: dict[str, Any] = {}
        output_fields: Any = (
            getattr(signature, "output_fields", None) if signature else None
        )
        if isinstance(output_fields, (dict, list, set, tuple)):
            for key in cast(Iterable[Any], output_fields):
                key_str = str(key)
                if key_str in output_dict:
                    filtered_dict[key_str] = output_dict[key_str]
        if not filtered_dict:
            for k_str, v_val in output_dict.items():
                if k_str not in (
                    "trajectory",
                    "history",
                    "termination_reason",
                ):
                    filtered_dict[k_str] = v_val

        if filtered_dict:
            if len(filtered_dict) == 1:
                val: Any = next(iter(filtered_dict.values()))
                if isinstance(val, str):
                    return val
                if isinstance(val, (int, float, bool)):
                    return str(val)
                try:
                    return json.dumps(val, ensure_ascii=False)
                except Exception:
                    return str(val)
            try:
                return json.dumps(filtered_dict, ensure_ascii=False)
            except Exception:
                return str(filtered_dict)

    return str(result)


def prepare_tool_definitions(
    tools: Sequence[Tool | Callable[..., Any]]
    | Mapping[str, Tool | Callable[..., Any]]
    | None,
) -> list[ToolDefinition] | None:
    """Prepare FunctionToolDefinition instances from a tools collection."""
    if not tools:
        return None

    if isinstance(tools, Mapping):
        tool_items = list(tools.values())
    elif isinstance(tools, (list, tuple, set)):
        tool_items = list(tools)
    else:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tool_items:
        tool_name = getattr(tool, "name", None)
        if not tool_name:
            func = getattr(tool, "func", None)
            tool_name = (
                getattr(func, "__name__", None)
                if func
                else getattr(tool, "__name__", None)
            )
        if not tool_name:
            tool_name = "tool"

        name_str = str(tool_name)
        if name_str in ("finish", "submit"):
            continue

        desc = (
            getattr(tool, "desc", None)
            or getattr(tool, "description", None)
            or getattr(tool, "__doc__", None)
        )
        args_schema = getattr(tool, "args", None)

        definitions.append(
            FunctionToolDefinition(
                name=name_str,
                description=str(desc).strip() if desc is not None else None,
                parameters=args_schema,
            )
        )

    return definitions or None
