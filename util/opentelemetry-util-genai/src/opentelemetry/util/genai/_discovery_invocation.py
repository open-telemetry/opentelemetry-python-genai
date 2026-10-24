# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from opentelemetry._logs import Logger
from opentelemetry.context import Context
from opentelemetry.semconv._incubating.attributes.mcp_attributes import (
    MCP_METHOD_NAME,
    McpMethodNameValues,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
    get_content_attributes,
)
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import ContentCapturingMode
from opentelemetry.util.types import AttributeValue


def _to_tool_definitions(
    items: Sequence[ToolDefinition | str | Mapping[str, object] | object],
) -> list[ToolDefinition]:
    defs: list[ToolDefinition] = []
    for item in items:
        if isinstance(item, (FunctionToolDefinition, GenericToolDefinition)):
            defs.append(item)
        elif isinstance(item, str):
            defs.append(GenericToolDefinition(name=item, type="function"))
        elif isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            raw_name = mapping.get("name")
            if not raw_name:
                continue
            name_str = str(raw_name)
            raw_desc = mapping.get("description")
            desc = str(raw_desc) if raw_desc is not None else None
            params = mapping.get("parameters") or mapping.get("input_schema")
            if desc is not None or params is not None:
                defs.append(
                    FunctionToolDefinition(
                        name=name_str,
                        description=desc,
                        parameters=params,
                    )
                )
            else:
                raw_type = mapping.get("type", "function")
                defs.append(
                    GenericToolDefinition(
                        name=name_str,
                        type=(
                            str(raw_type)
                            if raw_type is not None
                            else "function"
                        ),
                    )
                )
        elif hasattr(item, "name"):
            raw_name = getattr(item, "name", None)
            if not raw_name:
                continue
            name_str = str(raw_name)
            raw_desc = getattr(item, "description", None)
            desc = str(raw_desc) if raw_desc is not None else None
            params = getattr(
                item, "input_schema", getattr(item, "parameters", None)
            )
            if desc is not None or params is not None:
                defs.append(
                    FunctionToolDefinition(
                        name=name_str,
                        description=desc,
                        parameters=params,
                    )
                )
            else:
                raw_type = getattr(item, "type", "function")
                defs.append(
                    GenericToolDefinition(
                        name=name_str,
                        type=(
                            str(raw_type)
                            if raw_type is not None
                            else "function"
                        ),
                    )
                )
    return defs


class DiscoveryInvocation(GenAIInvocation):
    """Invocation for discovery calls (e.g. MCP tools/list).

    Use handler.discovery() or handler.mcp_list_tools() rather than constructing
    this directly.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        mcp_method: str = McpMethodNameValues.TOOLS_LIST.value,
        server_address: str | None = None,
        server_port: int | None = None,
        tool_definitions: (
            Sequence[ToolDefinition | str | Mapping[str, object] | object]
            | None
        ) = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        start_attributes: dict[str, AttributeValue] = {
            MCP_METHOD_NAME: mcp_method,
        }
        if server_address is not None:
            start_attributes[server_attributes.SERVER_ADDRESS] = server_address
        if server_port is not None:
            start_attributes[server_attributes.SERVER_PORT] = server_port

        super().__init__(
            tracer=tracer,
            instruments=instruments,
            logger=logger,
            completion_hook=completion_hook,
            operation_name=None,
            span_name=mcp_method,
            span_kind=SpanKind.CLIENT,
            start_attributes=start_attributes,
            context=context,
            _attach_to_context=_attach_to_context,
            content_capturing_mode=content_capturing_mode,
        )
        self._tool_definitions: list[ToolDefinition] | None = (
            _to_tool_definitions(tool_definitions)
            if tool_definitions is not None
            else None
        )
        self.tool_result: Any = None

    @property
    def tool_definitions(self) -> list[ToolDefinition] | None:
        return self._tool_definitions

    @tool_definitions.setter
    def tool_definitions(
        self,
        value: (
            Sequence[ToolDefinition | str | Mapping[str, object] | object]
            | None
        ),
    ) -> None:
        self._tool_definitions = (
            _to_tool_definitions(value) if value is not None else None
        )

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        attributes: dict[str, AttributeValue] = {}
        if self._tool_definitions:
            content_attrs = get_content_attributes(
                input_messages=[],
                output_messages=[],
                system_instruction=[],
                tool_definitions=self._tool_definitions,
                for_span=True,
                content_capturing_mode=self._content_capturing_mode,
            )
            attributes.update(content_attrs)
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._call_completion_hook(
            tool_definitions=self._tool_definitions,
        )


MCPListToolsInvocation = DiscoveryInvocation

__all__ = [
    "DiscoveryInvocation",
    "MCPListToolsInvocation",
]
