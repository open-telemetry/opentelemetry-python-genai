# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    ToolDefinition,
)

if TYPE_CHECKING:
    from openai.types.responses import Tool


def get_tool_definitions(tools: Sequence[Tool]) -> list[ToolDefinition]:
    """Map Responses API ``Response.tools`` entries onto tool definitions."""
    definitions: list[ToolDefinition] = []
    for tool in tools:
        if tool.type == "function":
            definitions.append(
                FunctionToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
            continue
        if tool.type == "mcp":
            name = tool.server_label
        elif tool.type == "custom" or tool.type == "namespace":
            name = tool.name
        else:
            name = tool.type
        definitions.append(GenericToolDefinition(name=name, type=tool.type))
    return definitions
