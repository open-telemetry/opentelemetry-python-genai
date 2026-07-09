# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from enum import Enum
from inspect import signature
from typing import Any, Callable, Mapping, cast

from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SafeJSONEncoder(json.JSONEncoder):
    """Safely encode non-JSON-serializable CrewAI objects."""

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            if hasattr(o, "dict") and callable(o.dict):
                return o.dict()
            if hasattr(o, "model_dump") and callable(o.model_dump):
                return o.model_dump(mode="json")
            return repr(o)


_AGENT_MODEL_DUMP_EXCLUDE: dict[str, Any] = {
    "crew": True,
    "llm": True,
    "function_calling_llm": True,
    "agent_executor": True,
    "executor_class": True,
    "tools_handler": True,
    "callbacks": True,
    "step_callback": True,
    "guardrail": True,
    "tools": {"__all__": {"args_schema": True, "cache_function": True}},
}


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, cls=SafeJSONEncoder, ensure_ascii=False)


def _coerce_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _coerce_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_json_value(v) for v in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump(mode="json")
        except Exception:
            logger.debug("Failed to serialize model via model_dump", exc_info=True)
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return value.dict()
        except Exception:
            logger.debug("Failed to serialize model via dict", exc_info=True)
    return repr(value)


def serialize_agent_input(agent: Any) -> dict[str, Any]:
    model_dump = getattr(agent, "model_dump", None)
    if callable(model_dump):
        try:
            serialized_agent = cast(
                dict[str, Any],
                model_dump(mode="json", exclude=_AGENT_MODEL_DUMP_EXCLUDE),
            )
            tools = serialized_agent.get("tools")
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict):
                        tool.pop("args_schema", None)
                        tool.pop("cache_function", None)
            agent_key = getattr(agent, "key", None)
            if agent_key is not None:
                serialized_agent["key"] = str(agent_key)
            return serialized_agent
        except Exception:
            logger.debug("Failed to serialize CrewAI agent", exc_info=True)

    serialized_agent: dict[str, Any] = {
        "role": str(getattr(agent, "role", "") or ""),
        "goal": str(getattr(agent, "goal", "") or ""),
        "backstory": str(getattr(agent, "backstory", "") or ""),
        "verbose": bool(getattr(agent, "verbose", False)),
        "allow_delegation": bool(getattr(agent, "allow_delegation", False)),
        "max_iter": getattr(agent, "max_iter", None),
        "max_rpm": getattr(agent, "max_rpm", None),
    }
    if (agent_id := getattr(agent, "id", None)) is not None:
        serialized_agent["id"] = str(agent_id)
    if (agent_key := getattr(agent, "key", None)) is not None:
        serialized_agent["key"] = str(agent_key)
    return serialized_agent


def serialize_bound_arguments(
    method: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    method_signature = signature(method)
    first_parameter_name = next(iter(method_signature.parameters), None)
    bound_arguments = method_signature.bind(
        *([None] if first_parameter_name == "self" else []),
        *args,
        **kwargs,
    )
    bound_arguments.apply_defaults()

    serialized: dict[str, Any] = {}
    for argument_name, argument_value in bound_arguments.arguments.items():
        if argument_name in {"self", "kwargs"}:
            continue
        if argument_name == "agent" and all(
            hasattr(argument_value, attr)
            for attr in ("role", "goal", "backstory")
        ):
            serialized[argument_name] = serialize_agent_input(argument_value)
        else:
            serialized[argument_name] = _coerce_json_value(argument_value)

    extra_kwargs = bound_arguments.arguments.get("kwargs", {})
    if isinstance(extra_kwargs, Mapping):
        for argument_name, argument_value in extra_kwargs.items():
            serialized[str(argument_name)] = _coerce_json_value(argument_value)
    return serialized


def text_message(role: str, value: Any) -> InputMessage:
    content = value if isinstance(value, str) else safe_json_dumps(value)
    return InputMessage(
        role=role,
        parts=cast(list[MessagePart], [Text(content=content)]),
    )


def output_text_message(role: str, value: Any) -> OutputMessage:
    content = value if isinstance(value, str) else safe_json_dumps(value)
    return OutputMessage(
        role=role,
        parts=cast(list[MessagePart], [Text(content=content)]),
        finish_reason="stop",
    )


def get_crew_name(crew: Any) -> str:
    name = str(getattr(crew, "name", "") or "").strip()
    if name and name.lower() != "crew":
        return name
    if (crew_id := getattr(crew, "id", None)) is not None:
        return f"Crew_{crew_id}"
    return "Crew"


def get_flow_name(flow: Any) -> str:
    name = str(getattr(flow, "name", "") or "").strip()
    if name and name.lower() != "flow":
        return name
    if (flow_id := getattr(flow, "flow_id", None)) is not None:
        return f"Flow_{flow_id}"
    return "Flow"


def get_tool_name(tool: Any) -> str:
    if hasattr(tool, "name") and getattr(tool, "name"):
        return str(getattr(tool, "name"))
    if isinstance(tool, Mapping) and tool.get("name"):
        return str(tool["name"])
    return str(tool)
