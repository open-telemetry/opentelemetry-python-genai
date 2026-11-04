# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool telemetry exercised through LangChain's public APIs, without network calls."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import Tool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from tests._api import FakeChatModel as _ToolChatModel
from tests._api import invoke as _invoke

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.genai.langchain.utils import (
    _get_property_value,
    prepare_tool_definitions,
)
from opentelemetry.instrumentation.genai.langchain.version import __version__
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    FunctionToolDefinition,
)

# ---------------------------------------------------------------------------
# Unit tests for _get_property_value
# ---------------------------------------------------------------------------


def test_get_property_value_from_dict():
    assert _get_property_value({"name": "my_tool"}, "name") == "my_tool"


def test_get_property_value_from_dict_missing_key():
    assert _get_property_value({}, "name") is None


def test_get_property_value_from_object():
    obj = MagicMock()
    obj.name = "obj_tool"
    assert _get_property_value(obj, "name") == "obj_tool"


def test_get_property_value_from_object_missing_attr():
    class Plain:
        pass

    assert _get_property_value(Plain(), "missing") is None


# ---------------------------------------------------------------------------
# Unit tests for prepare_tool_definitions
# ---------------------------------------------------------------------------


def test_prepare_tool_definitions_returns_none_for_empty():
    assert prepare_tool_definitions([]) is None


def test_prepare_tool_definitions_dict_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                },
            },
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    defn = result[0]
    assert isinstance(defn, FunctionToolDefinition)
    assert defn.name == "multiply"
    assert defn.description == "Multiply two numbers"
    assert defn.parameters is not None


def test_prepare_tool_definitions_skips_non_function_type():
    tools = [{"type": "retrieval", "retrieval": {}}]
    result = prepare_tool_definitions(tools)
    assert result is None


def test_prepare_tool_definitions_multiple_tools():
    tools = [
        {
            "type": "function",
            "function": {"name": "add", "description": "Add numbers"},
        },
        {
            "type": "function",
            "function": {
                "name": "subtract",
                "description": "Subtract numbers",
            },
        },
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 2
    assert result[0].name == "add"
    assert result[1].name == "subtract"


def test_prepare_tool_definitions_missing_name_defaults_to_empty_string():
    tools = [
        {
            "type": "function",
            "function": {"description": "No name tool"},
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == ""


def test_prepare_tool_definitions_none_description_stays_none():
    tools = [
        {
            "type": "function",
            "function": {"name": "no_desc"},
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert result[0].description is None


def test_prepare_tool_definitions_object_tools():
    """Tools may be objects (e.g. pydantic models) rather than dicts."""

    class FuncDef:
        name = "get_weather"
        description = "Get current weather"
        parameters = {"type": "object"}

    class ToolDef:
        type = "function"
        function = FuncDef()

    result = prepare_tool_definitions([ToolDef()])
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "get_weather"
    assert result[0].description == "Get current weather"


# ---------------------------------------------------------------------------
# Unit tests for prepare_tool_definitions - legacy OpenAI ``functions`` shape
# ---------------------------------------------------------------------------


def test_prepare_tool_definitions_legacy_flat_function():
    """Pre-tools OpenAI ``functions`` entries are flat mappings with no
    ``type``/nested ``function`` wrapper. They must still be surfaced as
    function tool definitions."""
    tools = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    defn = result[0]
    assert isinstance(defn, FunctionToolDefinition)
    assert defn.type == "function"
    assert defn.name == "get_current_weather"
    assert defn.description == "Get the current weather in a given location."
    assert defn.parameters is not None
    assert defn.parameters["properties"]["location"]["type"] == "string"


def test_prepare_tool_definitions_legacy_none_description_stays_none():
    tools = [{"name": "no_desc"}]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "no_desc"
    assert result[0].description is None
    assert result[0].parameters is None


def test_prepare_tool_definitions_legacy_object_function():
    """Legacy flat definitions may also arrive as objects rather than dicts."""

    class LegacyFuncDef:
        name = "get_weather"
        description = "Get current weather"
        parameters = {"type": "object"}

    result = prepare_tool_definitions([LegacyFuncDef()])
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "get_weather"
    assert result[0].description == "Get current weather"


def test_prepare_tool_definitions_mixed_modern_and_legacy():
    """A mix of modern wrapped ``tools`` and legacy flat ``functions`` entries
    should both be surfaced."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "modern_tool",
                "description": "Modern wrapped tool",
            },
        },
        {
            "name": "legacy_tool",
            "description": "Legacy flat function",
        },
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 2
    assert result[0].name == "modern_tool"
    assert result[1].name == "legacy_tool"


# ---------------------------------------------------------------------------
# Unit tests for prepare_tool_definitions - malformed / edge-case inputs
# ---------------------------------------------------------------------------


def test_prepare_tool_definitions_wrapped_function_takes_precedence():
    """An entry with ``type == 'function'`` is parsed via the wrapped path even
    if it also carries a top-level ``name`` (the legacy branch must not fire)."""
    tools = [
        {
            "type": "function",
            "name": "top_level_ignored",
            "function": {"name": "wrapped_name"},
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "wrapped_name"


def test_prepare_tool_definitions_function_type_missing_function_body():
    """``type == 'function'`` with no ``function`` body and no top-level
    ``name`` yields nothing (neither branch produces a definition)."""
    tools = [{"type": "function"}]
    result = prepare_tool_definitions(tools)
    assert result is None


def test_prepare_tool_definitions_non_function_type_with_name_skipped():
    tools = [
        {
            "type": "web_search",
            "name": "search_the_web",
            "description": "Vendor-specific tool",
        }
    ]
    result = prepare_tool_definitions(tools)
    assert result is None


def test_prepare_tool_definitions_non_function_type_with_name_skipped_in_mix():
    tools = [
        {
            "type": "function",
            "function": {"name": "modern_tool"},
        },
        {
            "type": "web_search",
            "name": "search_the_web",
        },
        {
            "name": "legacy_tool",
        },
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert [defn.name for defn in result] == ["modern_tool", "legacy_tool"]


def test_prepare_tool_definitions_entry_without_type_or_name_skipped():
    tools = [{"description": "no type, no name"}]
    result = prepare_tool_definitions(tools)
    assert result is None


def test_prepare_tool_definitions_skips_malformed_keeps_valid():
    """Malformed entries are skipped without discarding valid ones."""
    tools = [
        {"description": "malformed - no name/type"},
        {"type": "retrieval", "retrieval": {}},
        {"name": "legacy_valid", "description": "ok"},
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "legacy_valid"


def _tool_span(span_exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == f"execute_tool {name}"
    ]
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes is not None
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == "execute_tool"
    )
    assert span.attributes[gen_ai_attributes.GEN_AI_TOOL_NAME] == name
    assert span.end_time is not None
    assert span.start_time is not None
    assert span.end_time >= span.start_time
    return span


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "with_call_id", [False, True], ids=["raw", "tool-call"]
)
async def test_tool_invocation(
    asynchronous: bool,
    with_call_id: bool,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    tool_input = (
        {
            "type": "tool_call",
            "name": "multiply",
            "id": "call_multiply",
            "args": {"a": 3, "b": 4},
        }
        if with_call_id
        else {"a": 3, "b": 4}
    )
    result = await _invoke(multiply, tool_input, asynchronous)
    if with_call_id:
        assert isinstance(result, ToolMessage)
        assert result.content == "12"
        assert result.tool_call_id == "call_multiply"
    else:
        assert result == 12
        assert isinstance(result, int)

    span = _tool_span(span_exporter, "multiply")
    assert len(span_exporter.get_finished_spans()) == 1
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs is not None
    assert (
        attrs[gen_ai_attributes.GEN_AI_TOOL_DESCRIPTION]
        == "Multiply two integers."
    )
    assert gen_ai_attributes.GEN_AI_AGENT_NAME not in attrs
    assert gen_ai_attributes.GEN_AI_CONVERSATION_ID not in attrs
    if with_call_id:
        assert attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] == "call_multiply"
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_ID not in attrs
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        arguments = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        assert isinstance(arguments, str)
        assert json.loads(arguments) == {"a": 3, "b": 4}
        captured_result = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT]
        assert captured_result == "12"
        assert type(captured_result) is str
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("value", ["Paris weather", '{"city": "Berlin"}'])
async def test_tool_string_input(
    asynchronous: bool,
    value: str,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    def lookup(query: str) -> ToolMessage:
        assert query == value
        return ToolMessage(content="Sunny", tool_call_id="call_lookup")

    lookup_tool = Tool(
        name="lookup", description="Look up weather.", func=lookup
    )
    result = await _invoke(lookup_tool, value, asynchronous)
    assert isinstance(result, ToolMessage)
    assert result.content == "Sunny"
    span = _tool_span(span_exporter, "lookup")
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs is not None
    assert attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] == "call_lookup"
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        arguments = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        assert isinstance(arguments, str)
        assert arguments == (
            '{"city":"Berlin"}' if value.startswith("{") else value
        )
        assert attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT] == "Sunny"
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "with_call_id", [False, True], ids=["raw", "tool-call"]
)
@pytest.mark.parametrize(
    "output",
    [
        {"status": "ok", "items": [1, 2, 3], "meta": {"count": 3}},
        ["apple", "banana", "cherry"],
    ],
    ids=["dict", "list"],
)
async def test_tool_structured_arguments_and_result(
    asynchronous: bool,
    with_call_id: bool,
    output: dict[str, object] | list[str],
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    @tool
    def search(
        query: str, filters: dict[str, object]
    ) -> dict[str, object] | list[str]:
        """Search with filters."""
        assert query == "Paris weather"
        assert filters == {"lang": "en", "limit": 5}
        return output

    arguments = {
        "query": "Paris weather",
        "filters": {"lang": "en", "limit": 5},
    }
    tool_input = (
        {
            "type": "tool_call",
            "name": "search",
            "id": "call_search",
            "args": arguments,
        }
        if with_call_id
        else arguments
    )
    result = await _invoke(search, tool_input, asynchronous)
    if with_call_id:
        assert isinstance(result, ToolMessage)
        if isinstance(output, dict):
            assert isinstance(result.content, str)
            assert json.loads(result.content) == output
        else:
            assert result.content == output
    else:
        assert result is output
    span = _tool_span(span_exporter, "search")
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs is not None
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        captured_arguments = attrs[
            gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS
        ]
        captured_result = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT]
        assert isinstance(captured_arguments, str)
        assert isinstance(captured_result, str)
        assert json.loads(captured_arguments) == arguments
        assert json.loads(captured_result) == output
        if with_call_id and isinstance(result.content, str):
            assert captured_result == result.content
        else:
            assert captured_result == json.dumps(output, separators=(",", ":"))
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("output", "expected_result"),
    [
        ("answer", "answer"),
        ("", ""),
        (0, "0"),
        (1.5, "1.5"),
        (False, "false"),
        (None, None),
    ],
)
async def test_tool_raw_scalar_result(
    asynchronous: bool,
    output: str | float | bool | None,
    expected_result: str | None,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    @tool
    def raw_result() -> str | int | float | bool | None:
        """Return a raw scalar."""
        return output

    result = await _invoke(raw_result, {}, asynchronous)
    assert result == output
    assert type(result) is type(output)
    span = _tool_span(span_exporter, "raw_result")
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs is not None
    if (
        content_capture == ContentCapturingMode.SPAN_ONLY
        and output is not None
    ):
        captured_result = attrs[gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT]
        assert captured_result == expected_result
        assert type(captured_result) is str
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
async def test_tool_error(
    asynchronous: bool,
    error_type: type[Exception],
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    error = error_type("something went wrong")

    @tool
    def failing_tool(value: str) -> str:
        """Fail with the original exception."""
        raise error

    with pytest.raises(error_type) as raised:
        await _invoke(failing_tool, {"value": "bad input"}, asynchronous)
    assert raised.value is error

    span = _tool_span(span_exporter, "failing_tool")
    assert len(span_exporter.get_finished_spans()) == 1
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes[error_attributes.ERROR_TYPE] == error_type.__name__


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_tool_parents_to_workflow_instead_of_ambient_span(
    asynchronous: bool,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    def calculate(
        inputs: dict[str, int], config: RunnableConfig
    ) -> dict[str, int]:
        with tracer.start_as_current_span("ambient"):
            return {"result": multiply.invoke(inputs, config=config)}

    async def acalculate(
        inputs: dict[str, int], config: RunnableConfig
    ) -> dict[str, int]:
        with tracer.start_as_current_span("ambient"):
            return {"result": await multiply.ainvoke(inputs, config=config)}

    workflow = RunnableLambda(calculate, afunc=acalculate).with_config(
        run_name="arithmetic_workflow",
        metadata={"otel_workflow_span": True, "thread_id": "t1"},
    )
    assert await _invoke(workflow, {"a": 3, "b": 4}, asynchronous) == {
        "result": 12
    }

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    workflow_span = next(
        s for s in spans if s.name == "invoke_workflow arithmetic_workflow"
    )
    ambient_span = next(s for s in spans if s.name == "ambient")
    tool_span = _tool_span(span_exporter, "multiply")
    assert workflow_span.status.status_code == StatusCode.UNSET
    assert tool_span.status.status_code == StatusCode.UNSET
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == workflow_span.context.span_id
    assert tool_span.parent.span_id != ambient_span.context.span_id
    assert tool_span.context.trace_id == workflow_span.context.trace_id
    assert tool_span.attributes is not None
    assert gen_ai_attributes.GEN_AI_CONVERSATION_ID not in tool_span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "nested", [False, True], ids=["agent", "nested-agents"]
)
async def test_tool_carries_nearest_agent_name(
    asynchronous: bool,
    nested: bool,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    def calculate(
        inputs: dict[str, int], config: RunnableConfig
    ) -> dict[str, int]:
        return {"result": multiply.invoke(inputs, config=config)}

    async def acalculate(
        inputs: dict[str, int], config: RunnableConfig
    ) -> dict[str, int]:
        return {"result": await multiply.ainvoke(inputs, config=config)}

    specialist = RunnableLambda(calculate, afunc=acalculate).with_config(
        run_name="arithmetic_specialist",
        metadata={
            "agent_name": "arithmetic_specialist",
            "otel_agent_span": True,
        },
    )
    if nested:

        def coordinate(
            inputs: dict[str, int], config: RunnableConfig
        ) -> dict[str, int]:
            return specialist.invoke(inputs, config=config)

        async def acoordinate(
            inputs: dict[str, int], config: RunnableConfig
        ) -> dict[str, int]:
            return await specialist.ainvoke(inputs, config=config)

        agent = RunnableLambda(coordinate, afunc=acoordinate).with_config(
            run_name="coordinator",
            metadata={"agent_name": "coordinator", "otel_agent_span": True},
        )
    else:
        agent = specialist
    assert await _invoke(agent, {"a": 3, "b": 4}, asynchronous) == {
        "result": 12
    }

    spans = span_exporter.get_finished_spans()
    assert len(spans) == (3 if nested else 2)
    tool_span = _tool_span(span_exporter, "multiply")
    assert tool_span.attributes is not None
    assert (
        tool_span.attributes[gen_ai_attributes.GEN_AI_AGENT_NAME]
        == "arithmetic_specialist"
    )
    specialist_span = next(
        s for s in spans if s.name == "invoke_agent arithmetic_specialist"
    )
    assert all(s.status.status_code == StatusCode.UNSET for s in spans)
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == specialist_span.context.span_id
    if nested:
        coordinator_span = next(
            s for s in spans if s.name == "invoke_agent coordinator"
        )
        assert specialist_span.parent is not None
        assert (
            specialist_span.parent.span_id == coordinator_span.context.span_id
        )


def _model_response(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> AIMessage:
    return AIMessage(
        content="" if tool_calls else "12",
        id="response_123",
        tool_calls=tool_calls or [],
        response_metadata={
            "stopReason"
            if finish_reason == "tool_use"
            else "finish_reason": finish_reason,
            "model_name": "served-model",
            "id": "response_123",
        },
        usage_metadata={
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
        },
    )


def _chat_span(
    span_exporter: InMemorySpanExporter, finish_reason: str
) -> ReadableSpan:
    (span,) = span_exporter.get_finished_spans()
    assert span.name == "chat test-model"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs is not None
    assert attrs[gen_ai_attributes.GEN_AI_RESPONSE_MODEL] == "served-model"
    assert attrs[gen_ai_attributes.GEN_AI_RESPONSE_ID] == "response_123"
    assert attrs[gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        finish_reason,
    )
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert attrs[gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    return span


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("definitions_key", ["tools", "functions", None])
async def test_chat_tool_definitions(
    asynchronous: bool,
    definitions_key: str | None,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    model = _ToolChatModel(responses=[_model_response()])
    function = convert_to_openai_function(multiply)
    if definitions_key == "tools":
        runnable = model.bind(
            tools=[{"type": "function", "function": function}]
        )
    elif definitions_key == "functions":
        runnable = model.bind(functions=[function])
    else:
        runnable = model
    result = await _invoke(runnable, "What is 3 * 4?", asynchronous)
    assert isinstance(result, AIMessage)
    assert result.content == "12"

    span = _chat_span(span_exporter, "stop")
    attrs = span.attributes
    assert attrs is not None
    if definitions_key and content_capture == ContentCapturingMode.SPAN_ONLY:
        definitions = attrs[gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS]
        assert isinstance(definitions, str)
        assert json.loads(definitions) == [{"type": "function", **function}]
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("finish_reason", ["tool_calls", "tool_use"])
@pytest.mark.parametrize(
    "multiple", [False, True], ids=["one-tool", "two-tools"]
)
async def test_chat_tool_call_requests(
    asynchronous: bool,
    finish_reason: str,
    multiple: bool,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    calls = [{"name": "multiply", "id": "call_001", "args": {"a": 3, "b": 4}}]
    if multiple:
        calls.append(
            {"name": "add", "id": "call_002", "args": {"a": 5, "b": 6}}
        )
    model = _ToolChatModel(
        responses=[
            _model_response(tool_calls=calls, finish_reason=finish_reason)
        ],
        generation_info=(
            None
            if finish_reason == "tool_use"
            else {"finish_reason": finish_reason}
        ),
    )
    result = await _invoke(model, "Calculate the answer.", asynchronous)
    assert isinstance(result, AIMessage)
    assert [call["id"] for call in result.tool_calls] == [
        call["id"] for call in calls
    ]

    span = _chat_span(span_exporter, finish_reason)
    attrs = span.attributes
    assert attrs is not None
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        messages = attrs[gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES]
        assert isinstance(messages, str)
        (message,) = json.loads(messages)
        assert message["role"] == "assistant"
        assert message["finish_reason"] == finish_reason
        assert message["parts"] == [
            {
                "type": "tool_call",
                "id": call["id"],
                "name": call["name"],
                "arguments": call["args"],
            }
            for call in calls
        ]
    else:
        assert gen_ai_attributes.GEN_AI_OUTPUT_MESSAGES not in attrs


class _ToolStartRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.call_ids: list[str | None] = []

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self.call_ids.append(kwargs.get("tool_call_id"))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "fail", [False, True], ids=["conflicting-id", "error"]
)
async def test_tool_call_id_from_real_callbacks(
    asynchronous: bool,
    fail: bool,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    error = RuntimeError("boom")
    recorder = _ToolStartRecorder()

    @tool
    def mytool() -> ToolMessage:
        """Return a message with its own call ID."""
        if fail:
            raise error
        return ToolMessage(content="ok", tool_call_id="from_end")

    tool_input = {
        "type": "tool_call",
        "name": "mytool",
        "id": "from_start",
        "args": {},
    }
    if fail:
        with pytest.raises(RuntimeError) as raised:
            await _invoke(
                mytool,
                tool_input,
                asynchronous,
                config={"callbacks": [recorder]},
            )
        assert raised.value is error
    else:
        result = await _invoke(
            mytool, tool_input, asynchronous, config={"callbacks": [recorder]}
        )
        assert isinstance(result, ToolMessage)
        assert result.content == "ok"
        assert result.tool_call_id == "from_end"

    # Older LangChain does not forward tool_call_id to the start callback.
    (start_call_id,) = recorder.call_ids
    assert start_call_id in (None, "from_start")
    expected_id = start_call_id if fail else start_call_id or "from_end"
    span = _tool_span(span_exporter, "mytool")
    assert span.attributes is not None
    assert (
        span.attributes.get(gen_ai_attributes.GEN_AI_TOOL_CALL_ID)
        == expected_id
    )
    if expected_id is not None:
        assert (
            type(span.attributes[gen_ai_attributes.GEN_AI_TOOL_CALL_ID]) is str
        )
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_CALL_ID not in span.attributes
    assert span.status.status_code == (
        StatusCode.ERROR if fail else StatusCode.UNSET
    )
    if fail:
        assert span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"


@pytest.fixture
def callback_handler(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    content_capture: ContentCapturingMode,
) -> OpenTelemetryLangChainCallbackHandler:
    return OpenTelemetryLangChainCallbackHandler(
        TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            instrumentation_scope_name="opentelemetry.instrumentation.genai.langchain",
            instrumentation_scope_version=__version__,
        )
    )


# BaseTool always supplies serialized metadata and paired start/end callbacks.
# Keep direct coverage only for defensive inputs its public API cannot emit.
def test_tool_start_without_serialized_metadata(
    callback_handler: OpenTelemetryLangChainCallbackHandler,
    span_exporter: InMemorySpanExporter,
) -> None:
    run_id = uuid4()
    callback_handler.on_tool_start(None, "some input", run_id=run_id)
    callback_handler.on_tool_end(
        ToolMessage(content="result", tool_call_id="call_unknown"),
        run_id=run_id,
    )
    span = _tool_span(span_exporter, "unknown")
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes is not None
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_TOOL_CALL_ID]
        == "call_unknown"
    )


@pytest.mark.parametrize("error", [False, True], ids=["end", "error"])
def test_tool_completion_without_start(
    error: bool,
    callback_handler: OpenTelemetryLangChainCallbackHandler,
    span_exporter: InMemorySpanExporter,
) -> None:
    if error:
        callback_handler.on_tool_error(RuntimeError("boom"), run_id=uuid4())
    else:
        callback_handler.on_tool_end(
            ToolMessage(content="result", tool_call_id="call_unknown"),
            run_id=uuid4(),
        )
    assert not span_exporter.get_finished_spans()
