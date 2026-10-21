# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline public-API callback integration tests and message-helper unit tests."""

from __future__ import annotations

import base64
import json
import math
import uuid
from unittest import mock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    LLMResult,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
)
from tests._api import (
    FakeChatModel,
    FakeRetriever,
    finished_chat,
    finished_span,
    invoke,
)

from opentelemetry import trace
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
    _document_to_retrieval_document,
    _extract_document_score,
    _extract_prompt_context,
)
from opentelemetry.instrumentation.genai.langchain.utils import (
    _ai_message_parts,
    _legacy_function_call_request,
    _media_part,
    _normalize_role,
    extract_token_details,
    make_input_message,
    make_last_output_message,
    serialize,
    to_input_messages,
    to_output_messages,
)
from opentelemetry.instrumentation.genai.langchain.version import __version__
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
    LocalAgentInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    ContentCapturingMode,
    FilePart,
    InputMessage,
    RetrievalDocument,
    TextPart,
    ToolCallRequestPart,
    UriPart,
)


def _make_agent_inv_mock() -> mock.MagicMock:
    agent_inv = mock.MagicMock(spec=LocalAgentInvocation)
    agent_inv.agent_id = None
    return agent_inv


def _make_handler():
    telemetry = mock.MagicMock()
    workflow_inv = mock.MagicMock(spec=WorkflowInvocation)
    telemetry.workflow.return_value = workflow_inv
    agent_inv = _make_agent_inv_mock()
    telemetry.invoke_local_agent.return_value = agent_inv
    handler = OpenTelemetryLangChainCallbackHandler(telemetry)
    return handler, telemetry, workflow_inv, agent_inv


def _run_id():
    return uuid.uuid4()


class TestExtractPromptContext:
    @pytest.mark.parametrize(
        "template_type", ["PromptTemplate", "ChatPromptTemplate"]
    )
    def test_extracts_name_and_declared_variables(
        self, template_type: str
    ) -> None:
        history = [HumanMessage(content="Earlier question")]
        context = _extract_prompt_context(
            {
                "id": ["langchain", "prompts", template_type],
                "name": "weather",
                "kwargs": {
                    "input_variables": ["question"],
                    "optional_variables": ["history"],
                    "partial_variables": {"language": "French"},
                },
            },
            {
                "question": "Will it rain?",
                "history": history,
                "unused": "ignored",
            },
            None,
        )

        assert context is not None
        assert context.name == "weather"
        assert context.variables == {
            "question": "Will it rain?",
            "history": history,
            "language": "French",
        }

    def test_name_precedence_and_generic_name_exclusion(self):
        serialized = {
            "id": ["langchain", "prompts", "PromptTemplate"],
            "name": "PromptTemplate",
            "kwargs": {"name": "serialized-name"},
        }

        context = _extract_prompt_context(
            serialized, {}, {"prompt_name": "metadata-name"}
        )
        assert context is not None
        assert context.name == "metadata-name"

        unnamed = _extract_prompt_context(
            {**serialized, "kwargs": {}}, {}, None
        )
        assert unnamed is not None
        assert unnamed.name is None

    @pytest.mark.parametrize("source", ["metadata", "kwargs"])
    def test_explicit_template_class_name_is_preserved(self, source: str):
        serialized = {
            "id": ["langchain", "prompts", "PromptTemplate"],
            "name": "PromptTemplate",
            "kwargs": {},
        }
        metadata = None
        if source == "metadata":
            metadata = {"prompt_name": "PromptTemplate"}
        else:
            serialized["kwargs"] = {"name": "PromptTemplate"}

        context = _extract_prompt_context(serialized, {}, metadata)

        assert context is not None
        assert context.name == "PromptTemplate"

    def test_runtime_values_override_partials_and_scalar_is_mapped(self):
        serialized = {
            "id": ["langchain", "prompts", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["question"],
                "optional_variables": ["history"],
                "partial_variables": {"language": "French"},
            },
        }

        context = _extract_prompt_context(serialized, "runtime", None)

        assert context is not None
        assert context.variables == {
            "question": "runtime",
            "language": "French",
        }

    def test_preserves_none_as_json_null(self):
        context = _extract_prompt_context(
            {
                "id": ["langchain", "prompts", "PromptTemplate"],
                "kwargs": {"input_variables": ["value"]},
            },
            {"value": None},
            None,
        )

        assert context is not None
        assert context.variables == {"value": None}

    def test_passes_declared_values_without_serializing(self):
        callable_value = lambda: "value"
        unsupported_value = object()
        serialized = {
            "id": ["langchain", "prompts", "PromptTemplate"],
            "kwargs": {
                "partial_variables": {
                    "callable": callable_value,
                    "unresolved": {
                        "lc": 1,
                        "type": "not_implemented",
                    },
                },
                "input_variables": ["unsupported"],
            },
        }

        context = _extract_prompt_context(
            serialized, {"unsupported": unsupported_value}, None
        )

        assert context is not None
        assert context.variables == {
            "callable": callable_value,
            "unsupported": unsupported_value,
        }

    def test_ignores_non_prompt_runs(self):
        assert (
            _extract_prompt_context(
                {
                    "id": ["langchain", "schema", "RunnableSequence"],
                    "name": "PromptTemplate",
                },
                {"question": "ignored"},
                {"prompt_name": "ignored"},
            )
            is None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("run_name", "metadata", "expected_name"),
    [
        ("MyLangGraph", {}, "invoke_workflow MyLangGraph"),
        (
            "MyLangGraph",
            {"workflow_name": "custom_workflow"},
            "invoke_workflow custom_workflow",
        ),
        ("math_agent", {}, "invoke_agent math_agent"),
        (
            "calculate",
            {
                "agent_name": "math_agent",
                "agent_description": "does math",
                "agent_id": "ignored",
            },
            "invoke_agent math_agent",
        ),
        ("AgentExecutor", {}, "invoke_agent AgentExecutor"),
    ],
)
async def test_chain_classification_and_completion(
    asynchronous: bool,
    run_name: str,
    metadata: dict[str, object],
    expected_name: str,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    chain = RunnableLambda(
        lambda state: {"messages": [AIMessage(content="Solved")]}
    ).with_config(run_name=run_name, metadata=metadata)
    result = await invoke(
        chain, {"messages": [HumanMessage(content="Solve this")]}, asynchronous
    )
    assert result["messages"][0].content == "Solved"
    span = finished_span(span_exporter, expected_name)
    assert len(span_exporter.get_finished_spans()) == 1
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs is not None
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == expected_name.split()[0]
    if "agent_description" in metadata:
        assert attrs[GenAI.GEN_AI_AGENT_DESCRIPTION] == "does math"
        assert GenAI.GEN_AI_AGENT_ID not in attrs
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        inputs = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
        outputs = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
        assert inputs[0]["parts"] == [
            {"type": "text", "content": "Solve this"}
        ]
        assert outputs[0]["parts"] == [{"type": "text", "content": "Solved"}]
    else:
        assert GenAI.GEN_AI_INPUT_MESSAGES not in attrs
        assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("agent", [False, True], ids=["workflow", "agent"])
@pytest.mark.parametrize("inputs", [{}, {"user_query": "plan a trip"}])
@pytest.mark.parametrize(
    "outputs",
    [
        {},
        {"messages": [HumanMessage(content="hi")]},
        {
            "messages": [
                AIMessage(content="intermediate"),
                AIMessage(content="final answer"),
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "final answer"},
            ]
        },
    ],
    ids=["empty", "no-assistant", "last-assistant", "dict-assistant"],
)
async def test_chain_input_and_output_capture(
    asynchronous: bool,
    agent: bool,
    inputs: dict[str, object],
    outputs: dict[str, object],
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    chain = RunnableLambda(lambda state: outputs).with_config(
        run_name="math_agent" if agent else "MyLangGraph"
    )
    assert await invoke(chain, inputs, asynchronous) == outputs
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs is not None
    if content_capture == ContentCapturingMode.SPAN_ONLY and inputs:
        captured = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
        assert json.loads(captured[0]["parts"][0]["content"]) == inputs
    else:
        assert GenAI.GEN_AI_INPUT_MESSAGES not in attrs
    last = (outputs.get("messages") or [None])[-1]
    if content_capture == ContentCapturingMode.SPAN_ONLY and (
        isinstance(last, AIMessage)
        or (isinstance(last, dict) and last.get("role") == "assistant")
    ):
        captured = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
        assert len(captured) == 1
        assert captured[0]["parts"] == [
            {"type": "text", "content": "final answer"}
        ]
    else:
        assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("parent_name", ["MyLangGraph", "math_agent"])
@pytest.mark.parametrize(
    "child_name", ["weather_agent", "math_agent", "Math_Agent"]
)
async def test_nested_agent_parenting_and_deduplication(
    asynchronous: bool,
    parent_name: str,
    child_name: str,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    child = RunnableLambda(
        lambda state: {"messages": [AIMessage(content="done")]}
    ).with_config(run_name=child_name)

    def run(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return child.invoke(state, config)

    async def arun(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return await child.ainvoke(state, config)

    # The intermediate runnable has no agent/workflow classification of its own.
    intermediate = RunnableLambda(run, afunc=arun).with_config(
        run_name="internal"
    )

    def root(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return intermediate.invoke(state, config)

    async def aroot(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return await intermediate.ainvoke(state, config)

    parent = RunnableLambda(root, afunc=aroot).with_config(
        run_name=parent_name
    )
    result = await invoke(parent, {}, asynchronous)
    assert result["messages"][0].content == "done"
    spans = span_exporter.get_finished_spans()
    deduplicated = parent_name.lower() == child_name.lower()
    assert len(spans) == (1 if deduplicated else 2)
    assert all(s.status.status_code == StatusCode.UNSET for s in spans)
    if not deduplicated:
        child_span = finished_span(span_exporter, f"invoke_agent {child_name}")
        parent_span = next(s for s in spans if s is not child_span)
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span.context.span_id
        assert child_span.context.trace_id == parent_span.context.trace_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("run_name", ["MyLangGraph", "math_agent"])
async def test_chain_error_and_subsequent_run(
    asynchronous: bool,
    run_name: str,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    error = ValueError("chain failed")

    def fail(state: dict[str, object]) -> dict[str, object]:
        raise error

    failing = RunnableLambda(fail).with_config(run_name=run_name)
    with pytest.raises(ValueError) as raised:
        await invoke(failing, {}, asynchronous)
    assert raised.value is error
    successful = RunnableLambda(lambda state: {}).with_config(
        run_name=run_name
    )
    assert await invoke(successful, {}, asynchronous) == {}
    failed, succeeded = span_exporter.get_finished_spans()
    assert failed.status.status_code == StatusCode.ERROR
    assert failed.attributes[error_attributes.ERROR_TYPE] == "ValueError"
    assert succeeded.status.status_code == StatusCode.UNSET
    assert succeeded.parent is None
    assert failed.context.trace_id != succeeded.context.trace_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_prompt_sequence_calls_real_callback_managers(
    asynchronous: bool,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Calculate accurately."), ("human", "{question}")]
    )
    model = FakeChatModel(
        responses=[
            AIMessage(
                content="12",
                usage_metadata={
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7,
                },
            )
        ]
    )
    chain = (prompt | model).with_config(
        run_name="arithmetic", metadata={"otel_workflow_span": True}
    )
    assert (
        await invoke(chain, {"question": "What is 3 * 4?"}, asynchronous)
    ).content == "12"
    assert len(span_exporter.get_finished_spans()) == 2
    chat = finished_chat(span_exporter)
    workflow = finished_span(span_exporter, "invoke_workflow arithmetic")
    assert workflow.status.status_code == StatusCode.UNSET
    assert chat.parent is not None
    assert chat.parent.span_id == workflow.context.span_id
    assert chat.attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert chat.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        inputs = json.loads(chat.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
        assert [m["role"] for m in inputs] == ["system", "user"]
        assert inputs[1]["parts"] == [
            {"type": "text", "content": "What is 3 * 4?"}
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("reason_key", "reason", "legacy", "with_text"),
    [
        ("finish_reason", "tool_calls", False, False),
        ("finish_reason", "tool_calls", False, True),
        ("stopReason", "tool_use", False, False),
        ("stop_reason", "tool_use", False, False),
        ("done_reason", "stop", False, True),
        ("finish_reason", "function_call", True, False),
    ],
)
async def test_chat_response_parts_and_finish_reason(
    asynchronous: bool,
    reason_key: str,
    reason: str,
    legacy: bool,
    with_text: bool,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    call = {"name": "get_weather", "id": "call_123", "args": {"city": "Paris"}}
    response = AIMessage(
        content="Checking the weather..." if with_text else "",
        name="weather_bot",
        tool_calls=[] if legacy else [call],
        additional_kwargs=(
            {
                "function_call": {
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                }
            }
            if legacy
            else {}
        ),
        response_metadata={reason_key: reason}
        if reason_key.startswith("stop")
        else {},
    )
    model = FakeChatModel(
        responses=[response],
        generation_info=None
        if reason_key.startswith("stop")
        else {reason_key: reason},
    )
    result = await invoke(
        model, [HumanMessage(content="Weather?", name="Alice")], asynchronous
    )
    assert result.content == response.content
    span = finished_chat(span_exporter)
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (reason,)
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        inputs = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
        assert inputs[0]["name"] == "Alice"
        (output,) = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
        assert output["name"] == "weather_bot"
        assert output["role"] == "assistant"
        assert output["finish_reason"] == reason
        parts = output["parts"]
        assert parts[-1]["type"] == "tool_call"
        assert parts[-1]["name"] == "get_weather"
        assert parts[-1]["arguments"] == {"city": "Paris"}
        if not legacy:
            assert parts[-1]["id"] == "call_123"
        assert len(parts) == (2 if with_text else 1)
        if with_text:
            assert parts[0] == {
                "type": "text",
                "content": "Checking the weather...",
            }
    else:
        assert GenAI.GEN_AI_INPUT_MESSAGES not in attrs
        assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_chat_multiple_generations_keep_finish_reason_alignment(
    asynchronous: bool,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    generations = [
        ChatGeneration(
            message=AIMessage(content="First"),
            generation_info={"finish_reason": "stop"},
        ),
        ChatGeneration(
            message=AIMessage(content="Second"), generation_info=None
        ),
        ChatGeneration(message=AIMessage(content="Third"), generation_info={}),
        ChatGeneration(
            message=AIMessage(content="Fourth"),
            generation_info={"finish_reason": "length"},
        ),
    ]
    model = FakeChatModel(
        responses=[AIMessage(content="unused")], generations=generations
    )
    messages = [[HumanMessage(content="Give alternatives.")]]
    result = (
        await model.agenerate(messages)
        if asynchronous
        else model.generate(messages)
    )
    assert len(result.generations[0]) == 4
    attrs = finished_chat(span_exporter).attributes
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
        "error",
        "error",
        "length",
    )
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        outputs = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
        assert [m["finish_reason"] for m in outputs] == [
            "stop",
            "error",
            "error",
            "length",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (
            {
                "input_token_details": {"cache_creation": 3, "cache_read": 2},
                "output_token_details": {"reasoning": 5},
            },
            {
                "gen_ai.usage.cache_write.input_tokens": 3,
                GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 2,
                GenAI.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: 5,
            },
        ),
        (
            {"input_token_details": {"cache_write": 7, "cache_read": 3}},
            {
                "gen_ai.usage.cache_write.input_tokens": 7,
                GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 3,
            },
        ),
        (
            {
                "input_token_details": {"text": 7, "image": 2, "audio": 1},
                "output_token_details": {"text": 14, "image": 4, "audio": 2},
            },
            {
                "gen_ai.usage.text.input_tokens": 7,
                "gen_ai.usage.image.input_tokens": 2,
                "gen_ai.usage.audio.input_tokens": 1,
                "gen_ai.usage.text.output_tokens": 14,
                "gen_ai.usage.image.output_tokens": 4,
                "gen_ai.usage.audio.output_tokens": 2,
            },
        ),
        ({}, {}),
    ],
    ids=["cache-creation", "cache-write", "modalities", "no-details"],
)
async def test_chat_token_details(
    asynchronous: bool,
    details: dict[str, object],
    expected: dict[str, int],
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    model = FakeChatModel(
        responses=[
            AIMessage(
                content="hi",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    **details,
                },
            )
        ]
    )
    assert (await invoke(model, "Hello", asynchronous)).content == "hi"
    attrs = finished_chat(span_exporter).attributes
    assert attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
    for name, value in expected.items():
        assert attrs[name] == value
        assert type(attrs[name]) is int
    detail_names = [
        "gen_ai.usage.cache_write.input_tokens",
        GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        GenAI.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        *[
            f"gen_ai.usage.{modality}.{direction}_tokens"
            for modality in ("text", "image", "audio")
            for direction in ("input", "output")
        ],
    ]
    assert all(
        name not in attrs for name in detail_names if name not in expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    (
        "metadata",
        "generation_info",
        "llm_output",
        "expected_model",
        "expected_id",
    ),
    [
        (
            {"headers": {"x-ms-served-model": "served"}},
            {},
            None,
            "served",
            None,
        ),
        (
            {"headers": {"X-MS-Served-Model": "served"}},
            {},
            {"model_name": "body", "id": "resp_123"},
            "served",
            "resp_123",
        ),
        ({}, {}, {"model": "body"}, "body", None),
        (
            {"model_name": "message-model", "id": "msg_123"},
            {"model_name": "generation-model"},
            None,
            "message-model",
            "msg_123",
        ),
        (
            {},
            {"model_name": "generation-model", "id": "gen_123"},
            None,
            "generation-model",
            "gen_123",
        ),
        (
            {},
            {"model_name": "generation-model"},
            {"model_name": "body"},
            "body",
            None,
        ),
        ({"headers": {"x-request-id": "ignored"}}, {}, None, None, None),
        ({"headers": {"x-ms-served-model": " "}}, {}, None, None, None),
    ],
)
async def test_chat_response_model_and_id_resolution(
    asynchronous: bool,
    metadata: dict[str, object],
    generation_info: dict[str, object],
    llm_output: dict[str, object] | None,
    expected_model: str | None,
    expected_id: str | None,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    model = FakeChatModel(
        responses=[AIMessage(content="hello", response_metadata=metadata)],
        generation_info={"finish_reason": "stop", **generation_info},
        llm_output=llm_output,
    )
    assert (await invoke(model, "Hello", asynchronous)).content == "hello"
    attrs = finished_chat(span_exporter).attributes
    assert attrs.get(GenAI.GEN_AI_RESPONSE_MODEL) == expected_model
    assert attrs.get(GenAI.GEN_AI_RESPONSE_ID) == expected_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("metadata", "generation_info", "expected_model", "expected_id"),
    [
        (
            {"model_name": "message-model", "id": "msg_123"},
            {},
            "message-model",
            "msg_123",
        ),
        (
            {},
            {"model_name": "generation-model", "id": "gen_123"},
            "generation-model",
            "gen_123",
        ),
        (
            {"model_name": "message-model"},
            {"model_name": "generation-model"},
            "message-model",
            None,
        ),
        (
            {"headers": {"x-ms-served-model": "served"}},
            {"model_name": "generation-model"},
            "served",
            None,
        ),
        ({}, {}, None, None),
    ],
)
async def test_streamed_chat_response(
    asynchronous: bool,
    metadata: dict[str, object],
    generation_info: dict[str, object],
    expected_model: str | None,
    expected_id: str | None,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    model = FakeChatModel(
        responses=[AIMessage(content="unused")],
        chunks=[
            ChatGenerationChunk(
                message=AIMessageChunk(
                    content="hello", response_metadata=metadata
                ),
                generation_info={"finish_reason": "stop", **generation_info},
            )
        ],
    )
    chunks = (
        [chunk async for chunk in model.astream("Hello")]
        if asynchronous
        else list(model.stream("Hello"))
    )
    assert "".join(str(chunk.content) for chunk in chunks) == "hello"
    attrs = finished_chat(span_exporter).attributes
    assert attrs.get(GenAI.GEN_AI_RESPONSE_MODEL) == expected_model
    assert attrs.get(GenAI.GEN_AI_RESPONSE_ID) == expected_id
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        (output,) = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
        assert output["role"] == "assistant"
        assert output["parts"] == [{"type": "text", "content": "hello"}]
    else:
        assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "stream"])
async def test_chat_error(
    asynchronous: bool,
    streaming: bool,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    error = ConnectionError("provider unavailable")
    model = FakeChatModel(
        responses=[AIMessage(content="first chunk")],
        error=None if streaming else error,
        stream_error=error if streaming else None,
    )
    with pytest.raises(ConnectionError) as raised:
        if streaming:
            if asynchronous:
                async for _ in model.astream("Hello"):
                    pass
            else:
                list(model.stream("Hello"))
        else:
            await invoke(model, "Hello", asynchronous)
    assert raised.value is error
    span = finished_span(span_exporter, "chat test-model")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[error_attributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"ls_vector_store_provider": "Chroma"},
        {
            "ls_vector_store_provider": "Chroma",
            "ls_embedding_model": "embedding-model",
        },
    ],
)
@pytest.mark.parametrize("score", [None, 0.85, 0, True, "high", math.inf])
async def test_retriever_invocation(
    asynchronous: bool,
    metadata: dict[str, str],
    score: object,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    instrument_langchain: None,
) -> None:
    documents = [
        Document(
            page_content="doc one", id="doc-1", metadata={"score": score}
        ),
        Document(page_content="doc two"),
    ]
    retriever = FakeRetriever(documents=documents).with_config(
        metadata=metadata
    )
    assert (
        await invoke(retriever, "semantic search query", asynchronous)
        == documents
    )
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "retrieval"
    assert attrs.get(GenAI.GEN_AI_PROVIDER_NAME) == metadata.get(
        "ls_vector_store_provider"
    )
    assert attrs.get(GenAI.GEN_AI_REQUEST_MODEL) == metadata.get(
        "ls_embedding_model"
    )
    assert GenAI.GEN_AI_CONVERSATION_ID not in attrs
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        assert (
            attrs[GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT] == "semantic search query"
        )
        captured = json.loads(attrs[GenAI.GEN_AI_RETRIEVAL_DOCUMENTS])
        assert len(captured) == 2
        assert captured[0]["id"] == "doc-1"
        expected_score = (
            score
            if type(score) in (int, float) and math.isfinite(score)
            else None
        )
        assert captured[0].get("score") == expected_score
        assert captured[1].get("id") is None
        assert all(d.get("content") is None for d in captured)
    else:
        assert GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT not in attrs
        assert GenAI.GEN_AI_RETRIEVAL_DOCUMENTS not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "error"])
async def test_retriever_parenting_and_error(
    asynchronous: bool,
    fail: bool,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    error = RuntimeError("retrieval failed")
    retriever = FakeRetriever(error=error if fail else None)

    def retrieve(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {"documents": retriever.invoke("query", config=config)}

    async def aretrieve(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {"documents": await retriever.ainvoke("query", config=config)}

    workflow = RunnableLambda(retrieve, afunc=aretrieve).with_config(
        run_name="MyLangGraph"
    )
    if fail:
        with pytest.raises(RuntimeError) as raised:
            await invoke(workflow, {}, asynchronous)
        assert raised.value is error
    else:
        assert await invoke(workflow, {}, asynchronous) == {"documents": []}
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    parent = finished_span(span_exporter, "invoke_workflow MyLangGraph")
    child = next(
        s
        for s in spans
        if s.attributes[GenAI.GEN_AI_OPERATION_NAME] == "retrieval"
    )
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id
    assert all(
        s.status.status_code
        == (StatusCode.ERROR if fail else StatusCode.UNSET)
        for s in spans
    )
    if fail:
        assert all(
            s.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"
            for s in spans
        )


class TestPromptContextLifecycle:
    @pytest.mark.parametrize("with_transform", [False, True])
    def test_real_lcel_prompt_context_reaches_model(
        self, with_transform: bool
    ) -> None:
        handler, telemetry, _, _ = _make_handler()
        telemetry.should_capture_content.return_value = True
        prompt = ChatPromptTemplate.from_messages(
            [("human", "Hello {name}")]
        ).model_copy(update={"name": "greeting"})
        model = FakeListChatModel(responses=["Hello Ada"]).with_config(
            metadata={"ls_model_name": "fake-model"}
        )
        if with_transform:
            (prompt | RunnableLambda(lambda value: value) | model).invoke(
                {"name": "Ada"}, config={"callbacks": [handler]}
            )
        else:
            (prompt | model).invoke(
                {"name": "Ada"}, config={"callbacks": [handler]}
            )

        invocation = telemetry.inference.return_value
        assert invocation.prompt_name == "greeting"
        assert invocation.prompt_variables == {"name": "Ada"}

    def test_real_parallel_lcel_keeps_prompt_contexts_isolated(self) -> None:
        handler, telemetry, _, _ = _make_handler()
        telemetry.should_capture_content.return_value = True
        invocations = [
            mock.MagicMock(spec=InferenceInvocation),
            mock.MagicMock(spec=InferenceInvocation),
        ]
        telemetry.inference.side_effect = invocations

        def branch(prompt_name: str, response: str):
            prompt = ChatPromptTemplate.from_messages(
                [("human", "{value}")]
            ).model_copy(update={"name": prompt_name})
            model = FakeListChatModel(responses=[response]).with_config(
                metadata={"ls_model_name": "fake-model"}
            )
            return prompt | model

        chain = RunnableParallel(
            first=branch("first-prompt", "first-response"),
            second=branch("second-prompt", "second-response"),
        )

        chain.invoke({"value": "shared"}, config={"callbacks": [handler]})

        contexts = {
            (invocation.prompt_name, invocation.prompt_variables["value"])
            for invocation in invocations
        }
        assert contexts == {
            ("first-prompt", "shared"),
            ("second-prompt", "shared"),
        }

    def test_successful_prompt_context_is_applied_to_next_model(self):
        handler, telemetry, _, _ = _make_handler()
        telemetry.should_capture_content.return_value = True
        sequence_id = _run_id()
        prompt_id = _run_id()
        handler._invocation_manager.add_invocation_state(
            sequence_id, None, None
        )
        handler.on_chain_start(
            serialized={
                "id": ["langchain", "prompts", "PromptTemplate"],
                "name": "greeting",
                "kwargs": {"input_variables": ["name"]},
            },
            inputs={"name": "Ada"},
            run_id=prompt_id,
            parent_run_id=sequence_id,
        )
        handler.on_chain_end(outputs={"text": "Hello Ada"}, run_id=prompt_id)

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="Hello Ada")]],
            run_id=_run_id(),
            parent_run_id=sequence_id,
            metadata={"ls_provider": "openai"},
            invocation_params={"model_name": "gpt-4"},
        )

        invocation = telemetry.inference.return_value
        assert invocation.prompt_name == "greeting"
        assert invocation.prompt_variables == {"name": "Ada"}
        assert (
            handler._invocation_manager.consume_prompt_context(sequence_id)
            is None
        )

    def test_unsupported_model_does_not_consume_prompt_context(self):
        handler, telemetry, _, _ = _make_handler()
        telemetry.should_capture_content.return_value = True
        sequence_id = _run_id()
        prompt_id = _run_id()
        handler._invocation_manager.add_invocation_state(
            sequence_id, None, None
        )
        handler.on_chain_start(
            serialized={
                "id": ["langchain", "prompts", "PromptTemplate"],
                "name": "greeting",
                "kwargs": {"input_variables": ["name"]},
            },
            inputs={"name": "Ada"},
            run_id=prompt_id,
            parent_run_id=sequence_id,
        )
        handler.on_chain_end(outputs={"text": "Hello Ada"}, run_id=prompt_id)

        handler.on_chat_model_start(
            serialized={"name": "UnsupportedModel"},
            messages=[[HumanMessage(content="Hello Ada")]],
            run_id=_run_id(),
            parent_run_id=sequence_id,
        )

        telemetry.inference.assert_not_called()

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="Hello Ada")]],
            run_id=_run_id(),
            parent_run_id=sequence_id,
            metadata={"ls_provider": "openai"},
            invocation_params={"model_name": "gpt-4"},
        )

        invocation = telemetry.inference.return_value
        assert invocation.prompt_name == "greeting"
        assert invocation.prompt_variables == {"name": "Ada"}

    def test_prompt_variables_are_not_processed_when_content_disabled(self):
        handler, telemetry, _, _ = _make_handler()
        telemetry.should_capture_content.return_value = False
        sequence_id = _run_id()
        prompt_id = _run_id()
        handler._invocation_manager.add_invocation_state(
            sequence_id, None, None
        )
        handler.on_chain_start(
            serialized={
                "id": ["langchain", "prompts", "PromptTemplate"],
                "name": "greeting",
                "kwargs": {"input_variables": ["name"]},
            },
            inputs={"name": object()},
            run_id=prompt_id,
            parent_run_id=sequence_id,
        )
        handler.on_chain_end(outputs={}, run_id=prompt_id)

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="Hello")]],
            run_id=_run_id(),
            parent_run_id=sequence_id,
            metadata={"ls_provider": "openai"},
            invocation_params={"model_name": "gpt-4"},
        )

        invocation = telemetry.inference.return_value
        assert invocation.prompt_name == "greeting"
        assert invocation.prompt_variables == {}

    def test_failed_prompt_context_is_not_applied(self):
        handler, telemetry, _, _ = _make_handler()
        telemetry.should_capture_content.return_value = True
        telemetry.inference.return_value.prompt_name = None
        telemetry.inference.return_value.prompt_variables = None
        sequence_id = _run_id()
        prompt_id = _run_id()
        handler._invocation_manager.add_invocation_state(
            sequence_id, None, None
        )
        handler.on_chain_start(
            serialized={
                "id": ["langchain", "prompts", "PromptTemplate"],
                "name": "failed",
                "kwargs": {"input_variables": ["value"]},
            },
            inputs={"value": "secret"},
            run_id=prompt_id,
            parent_run_id=sequence_id,
        )
        handler.on_chain_error(RuntimeError("format failed"), run_id=prompt_id)

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="unrelated")]],
            run_id=_run_id(),
            parent_run_id=sequence_id,
            metadata={"ls_provider": "openai"},
            invocation_params={"model_name": "gpt-4"},
        )

        invocation = telemetry.inference.return_value
        assert invocation.prompt_name is None
        assert invocation.prompt_variables is None


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_multimodal_input_content_capture(
    asynchronous: bool,
    span_exporter: InMemorySpanExporter,
    content_capture: ContentCapturingMode,
    monkeypatch: pytest.MonkeyPatch,
    instrument_langchain: None,
) -> None:
    if content_capture == ContentCapturingMode.NO_CONTENT:

        def unexpected_decode(block: object) -> None:
            pytest.fail("Content-disabled calls must not decode media")

        monkeypatch.setattr(
            "opentelemetry.instrumentation.genai.langchain.utils._media_part",
            unexpected_decode,
        )
    model = FakeChatModel(responses=[AIMessage(content="a red pixel")])
    message = HumanMessage(
        name="Alice",
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_REAL_PNG_B64}"},
            },
        ],
    )
    assert (
        await invoke(model, [message], asynchronous)
    ).content == "a red pixel"
    attrs = finished_chat(span_exporter).attributes
    if content_capture == ContentCapturingMode.SPAN_ONLY:
        (captured,) = json.loads(attrs[GenAI.GEN_AI_INPUT_MESSAGES])
        assert captured["name"] == "Alice"
        assert captured["parts"][0] == {
            "type": "text",
            "content": "What's in this image?",
        }
        blob = captured["parts"][1]
        assert blob["type"] == "blob"
        assert base64.b64decode(blob["content"]) == _REAL_PNG_BYTES
    else:
        assert GenAI.GEN_AI_INPUT_MESSAGES not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_sync_and_async_execution_context(
    asynchronous: bool,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    before = trace.get_current_span().get_span_context()
    observed = []

    def run(state: dict[str, object]) -> dict[str, object]:
        observed.append(trace.get_current_span().get_span_context())
        return {}

    async def arun(state: dict[str, object]) -> dict[str, object]:
        observed.append(trace.get_current_span().get_span_context())
        return {}

    chain = RunnableLambda(run, afunc=arun).with_config(run_name="MyLangGraph")
    assert await invoke(chain, {}, asynchronous) == {}
    span = finished_span(span_exporter, "invoke_workflow MyLangGraph")
    assert span.status.status_code == StatusCode.UNSET
    assert observed == [span.context]
    assert trace.get_current_span().get_span_context() == before


# A completed callback without a preceding start is not emitted by LangChain.
@pytest.mark.parametrize("operation", ["chain", "retriever", "llm"])
@pytest.mark.parametrize("error", [False, True], ids=["end", "error"])
def test_unknown_completion_is_ignored(
    operation: str,
    error: bool,
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    handler = OpenTelemetryLangChainCallbackHandler(
        TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            instrumentation_scope_name="opentelemetry.instrumentation.genai.langchain",
            instrumentation_scope_version=__version__,
        )
    )
    run_id = uuid.uuid4()
    if error:
        getattr(handler, f"on_{operation}_error")(
            RuntimeError("boom"), run_id=run_id
        )
    elif operation == "chain":
        handler.on_chain_end({}, run_id=run_id)
    elif operation == "retriever":
        handler.on_retriever_end([], run_id=run_id)
    else:
        handler.on_llm_end(LLMResult(generations=[]), run_id=run_id)
    assert not span_exporter.get_finished_spans()


# Runnable APIs always supply a name. A nameless, unannounced agent callback
# can only be exercised directly; announced agents are covered via create_agent.
@pytest.mark.parametrize("error", [False, True], ids=["end", "error"])
def test_nameless_unannounced_agent_completion(
    error: bool,
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    handler = OpenTelemetryLangChainCallbackHandler(
        TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            instrumentation_scope_name="opentelemetry.instrumentation.genai.langchain",
            instrumentation_scope_version=__version__,
        )
    )
    run_id = uuid.uuid4()
    handler.on_chain_start(
        {}, {}, run_id=run_id, metadata={"otel_agent_span": True}
    )
    assert run_id in handler._invocation_manager._invocations
    assert handler._invocation_manager.get_invocation(run_id) is None
    if error:
        handler.on_chain_error(RuntimeError("boom"), run_id=run_id)
    else:
        handler.on_chain_end({}, run_id=run_id)
    assert run_id not in handler._invocation_manager._invocations
    assert not span_exporter.get_finished_spans()


# ---------------------------------------------------------------------------
# utils.make_input_message
# ---------------------------------------------------------------------------


class TestMakeInputMessage:
    def test_returns_empty_list_for_non_dict(self):
        assert make_input_message("not a dict") == []
        assert make_input_message(None) == []
        assert make_input_message(42) == []

    def test_empty_dict_returns_empty_list(self):
        assert make_input_message({}) == []

    def test_messages_key_with_human_message(self):
        msg = HumanMessage(content="Hello")
        result = make_input_message({"messages": [msg]})

        assert len(result) == 1
        assert isinstance(result[0], InputMessage)
        assert result[0].role == "user"
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].content == "Hello"

    def test_messages_key_skips_empty_content(self):
        msg_empty = HumanMessage(content="")
        msg_valid = HumanMessage(content="Hi")
        result = make_input_message({"messages": [msg_empty, msg_valid]})

        assert len(result) == 1
        assert result[0].parts[0].content == "Hi"

    def test_messages_key_multiple_messages(self):
        msgs = [HumanMessage(content="First"), HumanMessage(content="Second")]
        result = make_input_message({"messages": msgs})

        assert len(result) == 2
        assert result[0].parts[0].content == "First"
        assert result[1].parts[0].content == "Second"

    def test_messages_key_takes_priority_over_other_fields(self):
        msg = HumanMessage(content="hello")
        result = make_input_message(
            {"messages": [msg], "user_query": "should be ignored"}
        )

        assert len(result) == 1
        assert result[0].parts[0].content == "hello"

    def test_messages_key_converts_raw_role_content_tuples(self):
        result = make_input_message(
            {"messages": [("human", "Hello"), HumanMessage(content="Hi")]}
        )

        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].parts[0].content == "Hello"
        assert result[1].parts[0].content == "Hi"

    def test_messages_key_converts_role_content_dicts(self):
        result = make_input_message(
            {"messages": [{"role": "user", "content": "Hello"}]}
        )

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].parts[0].content == "Hello"

    @pytest.mark.parametrize(
        "bad_entry",
        [
            ("human",),  # too few items for (role, content)
            ("human", "hi", "extra"),  # too many items
            ("not_a_real_role", "hi"),  # unknown role string
            {"content": "no role key"},  # dict missing role
            {"role": "user"},  # dict missing content
            None,  # not a message at all
            42,  # not a message at all
            object(),  # arbitrary object
        ],
    )
    def test_messages_key_malformed_entry_does_not_raise(self, bad_entry):
        result = make_input_message(
            {"messages": [bad_entry, HumanMessage(content="Hi")]}
        )

        assert len(result) == 1
        assert result[0].parts[0].content == "Hi"

    def test_messages_key_malformed_entry_keeps_convertible_entries(self):
        result = make_input_message(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    ("human",),
                    HumanMessage(content="Hi"),
                ]
            }
        )

        assert [m.parts[0].content for m in result] == ["Hello", "Hi"]

    def test_messages_key_all_malformed_returns_empty_without_raising(self):
        result = make_input_message(
            {"messages": [("human",), None, {"content": "x"}]}
        )

        assert result == []

    def test_to_input_messages_never_raises_on_mixed_bad_input(self):
        result = to_input_messages(
            [object(), ("bad", "role", "shape"), HumanMessage(content="ok")]
        )

        assert len(result) == 1
        assert result[0].parts[0].content == "ok"

    def test_to_input_messages_empty_iterable_returns_empty(self):
        assert to_input_messages([]) == []

    def test_fallback_serializes_non_message_state_fields(self):
        result = make_input_message({"user_query": "what is 2+2?"})

        assert len(result) == 1
        assert result[0].role == "user"
        # The content should be a JSON serialization of the dict
        assert "user_query" in result[0].parts[0].content
        assert "what is 2+2?" in result[0].parts[0].content

    def test_fallback_excludes_intermediate_steps_key(self):
        # messages key absent → fallback path runs; intermediate_steps excluded
        result = make_input_message(
            {
                "user_query": "hi",
                "intermediate_steps": [("tool", "result")],
            }
        )

        assert len(result) == 1
        content = result[0].parts[0].content
        assert "intermediate_steps" not in content
        assert "user_query" in content

    def test_messages_key_present_skips_fallback_even_with_other_fields(self):
        # messages key present (even empty list) → early return, fallback not reached
        result = make_input_message(
            {
                "user_query": "ignored",
                "messages": [],
                "intermediate_steps": [("tool", "result")],
            }
        )

        assert result == []

    def test_fallback_returns_empty_when_all_values_are_none(self):
        result = make_input_message({"user_query": None, "context": None})
        assert result == []

    def test_fallback_returns_empty_when_only_excluded_keys(self):
        result = make_input_message(
            {"messages": None, "intermediate_steps": None}
        )
        assert result == []

    def test_messages_key_with_empty_list(self):
        # messages key present but empty → return empty list (no fallback)
        result = make_input_message({"messages": [], "user_query": "ignored"})
        assert result == []


# ---------------------------------------------------------------------------
# utils.to_output_messages / make_last_output_message
# ---------------------------------------------------------------------------


class TestToOutputMessages:
    def test_to_output_messages_propagates_explicit_finish_reason(self):
        # The inference path passes the provider's finish_reason through
        # ``to_output_messages``; the converter must forward it onto every
        # ``OutputMessage`` so util-genai can aggregate it into
        # ``gen_ai.response.finish_reasons``.
        msgs = [
            AIMessage(content="first"),
            AIMessage(content="second"),
        ]
        result = to_output_messages(msgs, finish_reason="tool_calls")
        assert [m.finish_reason for m in result] == [
            "tool_calls",
            "tool_calls",
        ]

    def test_to_output_messages_skips_non_ai_when_finish_reason_set(self):
        # finish_reason should never bleed onto non-AI messages: those are
        # filtered out entirely on the output side.
        result = to_output_messages(
            [HumanMessage(content="q"), AIMessage(content="a")],
            finish_reason="length",
        )
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].finish_reason == "length"


class TestMakeLastOutputMessage:
    def test_returns_only_last_ai_message(self):
        msgs = [
            AIMessage(content="intermediate"),
            AIMessage(content="final answer"),
        ]
        result = make_last_output_message({"messages": msgs})

        assert len(result) == 1
        assert result[0].parts[0].content == "final answer"

    def test_returns_last_assistant_message_in_dict_form(self):
        msgs = [
            AIMessage(content="intermediate"),
            {"role": "assistant", "content": "final answer"},
        ]
        result = make_last_output_message({"messages": msgs})

        assert len(result) == 1
        assert result[0].parts[0].content == "final answer"

    def test_returns_empty_when_no_ai_messages(self):
        result = make_last_output_message(
            {"messages": [HumanMessage(content="hi")]}
        )
        assert result == []

    def test_returns_empty_for_empty_outputs(self):
        assert make_last_output_message({}) == []

    @pytest.mark.parametrize(
        "data",
        ["not a dict", None, {"output": "hi"}, {"messages": None}],
    )
    def test_returns_empty_without_messages_list(self, data):
        assert make_last_output_message(data) == []

    def test_tuple_assistant_entry_produces_assistant_output(self):
        result = make_last_output_message({"messages": [("ai", "Hello")]})

        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].parts[0].content == "Hello"

    def test_finish_reason_default_is_empty_string(self):
        # Workflow/agent output spans must not fabricate a finish reason:
        # util-genai filters empty strings out of
        # ``gen_ai.response.finish_reasons`` so the rollup span stays silent
        # while the per-call inference spans report the real values.
        result = make_last_output_message(
            {"messages": [AIMessage(content="hi")]}
        )
        assert result[0].finish_reason == ""

    def test_single_ai_message_returned(self):
        ai_msg = AIMessage(content="only response")
        result = make_last_output_message({"messages": [ai_msg]})

        assert len(result) == 1
        assert result[0].parts[0].content == "only response"

    def test_skips_trailing_empty_ai_message(self):
        msgs = [
            AIMessage(content="final answer"),
            AIMessage(content=""),
        ]
        result = make_last_output_message({"messages": msgs})

        assert len(result) == 1
        assert result[0].parts[0].content == "final answer"

    @pytest.mark.parametrize(
        "trailing_entry",
        [
            HumanMessage(content="next question"),
            ToolMessage(content="tool result", tool_call_id="call_1"),
            {"role": "not_a_real_role", "content": "dropped"},
            {"content": "no role key"},
            ("human",),
            None,
        ],
    )
    def test_skips_trailing_non_output_entry(self, trailing_entry):
        msgs = [AIMessage(content="final answer"), trailing_entry]
        result = make_last_output_message({"messages": msgs})

        assert len(result) == 1
        assert result[0].parts[0].content == "final answer"

    def test_accepts_generator_messages(self):
        msgs = (AIMessage(content=text) for text in ("intermediate", "final"))
        result = make_last_output_message({"messages": msgs})

        assert len(result) == 1
        assert result[0].parts[0].content == "final"

    def test_does_not_convert_earlier_ai_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        converted: list[str] = []

        def recording_parts(message: AIMessage):
            converted.append(str(message.content))
            return _ai_message_parts(message)

        monkeypatch.setattr(
            "opentelemetry.instrumentation.genai.langchain.utils._ai_message_parts",
            recording_parts,
        )
        msgs = [
            AIMessage(content="first"),
            HumanMessage(content="next question"),
            AIMessage(content="final answer"),
        ]
        result = make_last_output_message({"messages": msgs})

        assert result[0].parts[0].content == "final answer"
        assert converted == ["final answer"]


# ---------------------------------------------------------------------------
# utils.serialize
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_none_returns_none(self):
        assert serialize(None) is None

    def test_dict_serialized_to_json(self):
        result = serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_list_serialized_to_json(self):
        result = serialize([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_non_serializable_falls_back_to_str(self):
        class Custom:
            def __str__(self):
                return "custom_repr"

        result = serialize({"obj": Custom()})
        assert result is not None
        assert "custom_repr" in result

    def test_string_value(self):
        result = serialize("hello")
        assert result == '"hello"'


class TestExtractDocumentScore:
    def test_attribute_score(self):
        class Obj:
            score = 0.88

        assert _extract_document_score(Obj()) == 0.88

    def test_metadata_score(self):
        class Obj:
            metadata = {"score": 0.75}

        assert _extract_document_score(Obj()) == 0.75

    def test_metadata_relevance_score_fallback(self):
        class Obj:
            metadata = {"relevance_score": 0.82}

        assert _extract_document_score(Obj()) == 0.82

    def test_attr_relevance_score_fallback(self):
        class Obj:
            relevance_score = 0.91

        assert _extract_document_score(Obj()) == 0.91

    def test_precedence_score_over_relevance_score(self):
        class Obj:
            metadata = {"score": 0.85, "relevance_score": 0.42}

        assert _extract_document_score(Obj()) == 0.85

    def test_precedence_attr_over_metadata(self):
        class Obj:
            score = 0.9
            metadata = {"score": 0.4}

        assert _extract_document_score(Obj()) == 0.9

    def test_attr_none_falls_back_to_metadata(self):
        class Obj:
            score = None
            metadata = {"score": 0.65}

        assert _extract_document_score(Obj()) == 0.65

    def test_zero_scores(self):
        class ObjFloat:
            score = 0.0

        class ObjInt:
            score = 0

        assert _extract_document_score(ObjFloat()) == 0.0
        assert _extract_document_score(ObjInt()) == 0

    def test_negative_score(self):
        class Obj:
            score = -1.25

        assert _extract_document_score(Obj()) == -1.25

    def test_non_numeric_and_bool(self):
        class ObjBool:
            score = True

        class ObjStr:
            score = "0.9"

        assert _extract_document_score(ObjBool()) is None
        assert _extract_document_score(ObjStr()) is None

    def test_non_finite_float_score(self):
        class ObjNaN:
            score = float("nan")

        class ObjInf:
            score = float("inf")

        class ObjNegInf:
            score = float("-inf")

        class ObjMathNaN:
            score = math.nan

        class ObjMathInf:
            score = math.inf

        class ObjMathNegInf:
            score = -math.inf

        assert _extract_document_score(ObjNaN()) is None
        assert _extract_document_score(ObjInf()) is None
        assert _extract_document_score(ObjNegInf()) is None
        assert _extract_document_score(ObjMathNaN()) is None
        assert _extract_document_score(ObjMathInf()) is None
        assert _extract_document_score(ObjMathNegInf()) is None
        assert _extract_document_score({"score": float("nan")}) is None
        assert _extract_document_score({"score": math.nan}) is None
        assert (
            _extract_document_score({"metadata": {"score": float("inf")}})
            is None
        )
        assert (
            _extract_document_score({"metadata": {"score": math.inf}}) is None
        )
        assert (
            _extract_document_score({"metadata": {"score": float("-inf")}})
            is None
        )
        assert (
            _extract_document_score({"metadata": {"score": -math.inf}}) is None
        )

    def test_dict_score(self):
        assert _extract_document_score({"score": 0.82}) == 0.82
        assert _extract_document_score({"metadata": {"score": 0.91}}) == 0.91
        assert (
            _extract_document_score(
                {"score": 0.99, "metadata": {"score": 0.1}}
            )
            == 0.99
        )

    def test_no_score(self):
        assert _extract_document_score(object()) is None
        assert _extract_document_score({}) is None
        assert _extract_document_score({"metadata": None}) is None
        assert _extract_document_score({"metadata": "str"}) is None


class TestDocumentToRetrievalDocument:
    def test_document_with_id_and_score(self):
        doc = Document(
            page_content="doc content", id="d1", metadata={"score": 0.85}
        )
        assert _document_to_retrieval_document(doc) == RetrievalDocument(
            id="d1", score=0.85
        )

    def test_duck_typed_content_is_ignored(self):
        class DuckNoPageContent:
            content = "fallback content"
            id = "d2"
            score = 0.9

        assert _document_to_retrieval_document(
            DuckNoPageContent()
        ) == RetrievalDocument(id="d2", score=0.9)

    def test_duck_typed_none_page_content_is_ignored(self):
        class DuckNonePageContent:
            page_content = None
            content = "fallback content when page_content is None"
            id = "d3"
            score = 0.75

        assert _document_to_retrieval_document(
            DuckNonePageContent()
        ) == RetrievalDocument(id="d3", score=0.75)

    def test_mapping_content_is_ignored(self):
        doc_map = {"content": "mapping fallback", "id": "m1", "score": 0.8}
        assert _document_to_retrieval_document(doc_map) == RetrievalDocument(
            id="m1", score=0.8
        )

    def test_mapping_none_page_content_is_ignored(self):
        doc_map = {
            "page_content": None,
            "content": "mapping fallback when page_content is None",
            "id": "m2",
            "score": 0.7,
        }
        assert _document_to_retrieval_document(doc_map) == RetrievalDocument(
            id="m2", score=0.7
        )

    def test_does_not_access_document_content(self) -> None:
        class LazyDocument:
            id = "lazy"
            score = 0.0

            @property
            def page_content(self) -> str:
                raise AssertionError("document content must not be read")

            @property
            def content(self) -> str:
                raise AssertionError("document content must not be read")

        assert _document_to_retrieval_document(
            LazyDocument()
        ) == RetrievalDocument(id="lazy", score=0.0)

    @pytest.mark.parametrize(
        "doc_id", [None, 42, True, ["doc"], {"id": "doc"}]
    )
    def test_non_string_ids_are_not_recorded(self, doc_id: object) -> None:
        assert (
            _document_to_retrieval_document({"id": doc_id})
            == RetrievalDocument()
        )

    def test_non_finite_scores_omitted(self):
        doc_nan = Document(page_content="c", metadata={"score": math.nan})
        doc_inf = Document(page_content="c", metadata={"score": float("inf")})
        doc_neginf = Document(
            page_content="c", metadata={"score": float("-inf")}
        )

        assert _document_to_retrieval_document(doc_nan) == RetrievalDocument()
        assert _document_to_retrieval_document(doc_inf) == RetrievalDocument()
        assert (
            _document_to_retrieval_document(doc_neginf) == RetrievalDocument()
        )


# ---------------------------------------------------------------------------
# utils.extract_token_details
# ---------------------------------------------------------------------------


def test_extract_token_details_cache_and_reasoning():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "input_token_details": {"cache_creation": 3, "cache_read": 2},
        "output_token_details": {"reasoning": 5},
    }
    details = extract_token_details(usage)
    assert details == {
        "cache_write_input_tokens": 3,
        "cache_read_input_tokens": 2,
        "reasoning_tokens": 5,
    }


def test_extract_token_details_cache_write_key():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "input_token_details": {"cache_write": 4, "cache_read": 2},
        "output_token_details": {"reasoning": 5},
    }
    details = extract_token_details(usage)
    assert details == {
        "cache_write_input_tokens": 4,
        "cache_read_input_tokens": 2,
        "reasoning_tokens": 5,
    }


def test_extract_token_details_modalities():
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "input_token_details": {
            "text": 70,
            "image": 20,
            "audio": 10,
            "cache_write": 15,
            "cache_read": 25,
        },
        "output_token_details": {
            "text": 30,
            "image": 15,
            "audio": 5,
            "reasoning": 10,
        },
    }
    # Modality break-downs are forwarded to InferenceInvocation rather than
    # flattened here, so only cache and reasoning remain.
    details = extract_token_details(usage)
    assert details == {
        "cache_write_input_tokens": 15,
        "cache_read_input_tokens": 25,
        "reasoning_tokens": 10,
    }


def test_extract_token_details_zero_values_omitted():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "input_token_details": {
            "cache_creation": 0,
            "cache_write": 0,
            "cache_read": 0,
            "text": 0,
            "image": 0,
            "audio": 0,
        },
        "output_token_details": {
            "reasoning": 0,
            "text": 0,
            "image": 0,
            "audio": 0,
        },
    }
    assert extract_token_details(usage) == {}


def test_extract_token_details_no_details_key():
    assert extract_token_details({"input_tokens": 1, "output_tokens": 2}) == {}


def test_extract_token_details_boolean_and_non_int_omitted():
    usage = {
        "input_token_details": {
            "cache_write": True,
            "cache_read": False,
            "text": "10",
            "image": None,
            "audio": -5,
        },
        "output_token_details": {
            "reasoning": True,
            "text": [10],
            "image": 0,
        },
    }
    assert extract_token_details(usage) == {}


# ---------------------------------------------------------------------------
# utils - legacy OpenAI function_call (_legacy_function_call_request)
# ---------------------------------------------------------------------------


def test_legacy_function_call_dict_arguments():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "get_weather",
                "arguments": {"city": "New York"},
            }
        },
    )
    call = _legacy_function_call_request(message)
    assert isinstance(call, ToolCallRequestPart)
    assert call.name == "get_weather"
    assert call.arguments == {"city": "New York"}


def test_legacy_function_call_string_arguments_parsed():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "get_weather",
                "arguments": '{"city": "New York"}',
            }
        },
    )
    call = _legacy_function_call_request(message)
    assert isinstance(call, ToolCallRequestPart)
    assert call.arguments == {"city": "New York"}


def test_legacy_function_call_absent_returns_none():
    assert _legacy_function_call_request(AIMessage(content="hi")) is None


def test_to_input_messages_includes_legacy_function_call():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {"name": "f", "arguments": {"x": 1}},
        },
    )
    messages = to_input_messages([message])
    assert len(messages) == 1
    assert any(
        isinstance(p, ToolCallRequestPart) and p.name == "f"
        for p in messages[0].parts
    )


# ---------------------------------------------------------------------------
# utils._normalize_role
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_role",
    [
        (AIMessage(content="hi"), "assistant"),
        (AIMessageChunk(content="hi"), "assistant"),
        (HumanMessage(content="hi"), "user"),
        (HumanMessageChunk(content="hi"), "user"),
        (SystemMessage(content="hi"), "system"),
        (SystemMessageChunk(content="hi"), "system"),
        (ToolMessage(content="hi", tool_call_id="call_1"), "tool"),
        (ToolMessageChunk(content="hi", tool_call_id="call_1"), "tool"),
        (FunctionMessage(content="hi", name="f"), "tool"),
        (FunctionMessageChunk(content="hi", name="f"), "tool"),
    ],
)
def test_normalize_role_resolves_chunk_variants(message, expected_role):
    """Chunk classes report their class name as ``.type``
    (``AIMessageChunk.type == "AIMessageChunk"``), so they only resolve to a
    spec role when matched by class."""
    assert _normalize_role(message) == expected_role


@pytest.mark.parametrize(
    "message,expected_role",
    [
        (ChatMessage(content="hi", role="assistant"), "assistant"),
        (ChatMessage(content="hi", role="custom"), "custom"),
        (ChatMessageChunk(content="hi", role="custom"), "custom"),
    ],
)
def test_normalize_role_reads_the_chat_message_role(message, expected_role):
    """``ChatMessage`` keeps its speaker in ``role`` rather than in the class."""
    assert _normalize_role(message) == expected_role


def test_normalize_role_returns_none_for_unmapped_class():
    assert _normalize_role(RemoveMessage(id="abc")) is None


def test_chat_message_role_reaches_the_input_side():
    (message,) = to_input_messages([ChatMessage(content="hi", role="custom")])
    assert message.role == "custom"


# ---------------------------------------------------------------------------
# utils._media_part - LangChain multimodal image block parsing
# ---------------------------------------------------------------------------

_REAL_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
    b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc"
    b"\xf8\xcf\xc0\xf0\x1f\x00\x05\x05\x02\x00\xa1\r\xf7\xdf\x00\x00\x00"
    b"\x00IEND\xaeB`\x82"
)
_REAL_PNG_B64 = base64.b64encode(_REAL_PNG_BYTES).decode("ascii")


def test_media_part_openai_image_url_dict():
    item = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/a.png"


def test_media_part_openai_image_url_string():
    item = {"type": "image_url", "image_url": "https://example.com/b.png"}
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/b.png"


def test_media_part_anthropic_base64_source_returns_blob():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "R0lGODlh",
        },
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == b"GIF89a"


def test_media_part_anthropic_base64_source_without_media_type():
    item = {
        "type": "image",
        "source": {"type": "base64", "data": "QUJD"},
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type is None
    assert part.content == b"ABC"


def test_media_part_base64_source_decodes_real_png():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _REAL_PNG_B64,
        },
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_url_source_returns_uri():
    item = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/c.png"},
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/c.png"


def test_media_part_unrecognized_returns_none():
    assert _media_part({"type": "text", "text": "hi"}) is None
    assert _media_part({"type": "image_url", "image_url": {}}) is None
    assert (
        _media_part({"type": "image_url", "image_url": {"url": 123}}) is None
    )
    assert _media_part({"type": "image", "source": "nope"}) is None
    assert (
        _media_part({"type": "image", "source": {"type": "base64", "data": 5}})
        is None
    )
    assert (
        _media_part({"type": "image", "source": {"type": "url", "url": ""}})
        is None
    )


def test_media_part_malformed_base64_returns_none():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "not!valid!base64!",
        },
    }
    assert _media_part(item) is None


def test_media_part_openai_real_png_data_uri_returns_blob():
    item = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_REAL_PNG_B64}"},
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_real_png_source_returns_blob():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _REAL_PNG_B64,
        },
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_real_png_corrupted_base64_returns_none():
    corrupted = _REAL_PNG_B64[:10] + "@@@@" + _REAL_PNG_B64[14:]
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": corrupted,
        },
    }
    assert _media_part(item) is None


def test_media_part_real_png_url_source_returns_uri():
    item = {
        "type": "image",
        "source": {
            "type": "url",
            "url": "https://example.com/real-image.png",
        },
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/real-image.png"


def test_media_part_standard_v1_base64_block_returns_blob():
    item = {
        "type": "image",
        "base64": _REAL_PNG_B64,
        "mime_type": "image/png",
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_standard_v1_url_block_returns_uri():
    item = {"type": "image", "url": "https://example.com/a.png"}
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/a.png"


def test_media_part_standard_block_without_mime_type_returns_blob():
    part = _media_part({"type": "image", "base64": _REAL_PNG_B64})
    assert isinstance(part, BlobPart)
    assert part.mime_type is None
    assert part.content == _REAL_PNG_BYTES


def test_media_part_standard_v03_block_without_mime_type_returns_blob():
    part = _media_part(
        {"type": "image", "source_type": "base64", "data": _REAL_PNG_B64}
    )
    assert isinstance(part, BlobPart)
    assert part.mime_type is None


def test_media_part_non_dict_source_falls_through_to_standard_keys():
    """A non-dict ``source`` must not shadow a standard payload."""
    part = _media_part(
        {
            "type": "image",
            "source": "not-a-dict",
            "base64": _REAL_PNG_B64,
            "mime_type": "image/png",
        }
    )
    assert isinstance(part, BlobPart)
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_source_takes_precedence_over_standard_keys():
    """Anthropic's ``source`` is checked before the standard top-level keys."""
    part = _media_part(
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/from.png"},
            "url": "https://example.com/ignored.png",
        }
    )
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/from.png"


def test_media_part_v03_url_source_type_does_not_fall_through_to_base64():
    """An explicit ``source_type`` pins the variant, even when it is broken."""
    part = _media_part(
        {
            "type": "image",
            "source_type": "url",
            "url": None,
            "base64": _REAL_PNG_B64,
        }
    )
    assert part is None


def test_media_part_standard_v03_base64_block_returns_blob():
    item = {
        "type": "image",
        "source_type": "base64",
        "data": _REAL_PNG_B64,
        "mime_type": "image/png",
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_standard_v03_url_block_returns_uri():
    item = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/b.png",
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/b.png"


def test_media_part_standard_block_corrupted_base64_returns_none():
    corrupted = _REAL_PNG_B64[:10] + "@@@@" + _REAL_PNG_B64[14:]
    assert (
        _media_part(
            {"type": "image", "base64": corrupted, "mime_type": "image/png"}
        )
        is None
    )


def test_media_part_standard_block_without_payload_returns_none():
    assert _media_part({"type": "image", "mime_type": "image/png"}) is None


@pytest.mark.parametrize(
    "block,expected,expected_value",
    [
        (
            {
                "type": "image",
                "base64": _REAL_PNG_B64,
                "mime_type": "image/png",
            },
            BlobPart,
            _REAL_PNG_BYTES,
        ),
        (
            {"type": "image", "url": "https://example.com/a.png"},
            UriPart,
            "https://example.com/a.png",
        ),
        (
            {
                "type": "image",
                "source_type": "base64",
                "data": _REAL_PNG_B64,
                "mime_type": "image/png",
            },
            BlobPart,
            _REAL_PNG_BYTES,
        ),
        (
            {
                "type": "image",
                "source_type": "url",
                "url": "https://e/b.png",
            },
            UriPart,
            "https://e/b.png",
        ),
        (
            {"type": "image", "source_type": "id", "id": "file-123"},
            FilePart,
            "file-123",
        ),
        (
            {"type": "image", "id": "file-456"},
            FilePart,
            "file-456",
        ),
        (
            {
                "type": "image",
                "source": {"type": "file", "file_id": "file-789"},
            },
            FilePart,
            "file-789",
        ),
    ],
)
def test_langchain_standard_image_blocks_are_captured(
    block, expected, expected_value
):
    messages = to_input_messages([HumanMessage(content=[block])])
    assert messages, "message dropped entirely - no parts extracted"
    part = next(p for p in messages[0].parts if isinstance(p, expected))
    assert part.modality == "image"
    assert _part_payload(part) == expected_value


@pytest.mark.parametrize(
    "block,expected,expected_value",
    [
        (
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{_REAL_PNG_B64}",
            },
            BlobPart,
            _REAL_PNG_BYTES,
        ),
        (
            {"type": "input_image", "image_url": "https://example.com/a.png"},
            UriPart,
            "https://example.com/a.png",
        ),
        (
            {
                "type": "input_image",
                "image_url": {"url": "https://example.com/a.png"},
            },
            UriPart,
            "https://example.com/a.png",
        ),
        (
            {"type": "input_image", "file_id": "file-123"},
            FilePart,
            "file-123",
        ),
    ],
)
def test_responses_api_input_image_blocks_are_captured(
    block, expected, expected_value
):
    messages = to_input_messages([HumanMessage(content=[block])])
    assert messages, "message dropped entirely - no parts extracted"
    part = next(p for p in messages[0].parts if isinstance(p, expected))
    assert part.modality == "image"
    assert _part_payload(part) == expected_value


def _part_payload(part):
    if isinstance(part, BlobPart):
        return part.content
    if isinstance(part, UriPart):
        return part.uri
    return part.file_id


def test_media_part_responses_api_input_image_base64_url():
    part = _media_part(
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{_REAL_PNG_B64}",
        }
    )
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.modality == "image"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_responses_api_input_image_file_id_returns_file_part():
    # Provider-hosted image: no bytes and no URL, but the reference is still
    # worth recording.
    part = _media_part({"type": "input_image", "file_id": "file-123"})
    assert isinstance(part, FilePart)
    assert part.file_id == "file-123"
    assert part.modality == "image"
    assert part.mime_type is None


def test_media_part_anthropic_file_source_returns_file_part():
    part = _media_part(
        {
            "type": "image",
            "source": {
                "type": "file",
                "file_id": "file-789",
                "media_type": "image/png",
            },
        }
    )
    assert isinstance(part, FilePart)
    assert part.file_id == "file-789"
    assert part.mime_type == "image/png"


def test_media_part_anthropic_file_source_without_file_id_returns_none():
    assert _media_part({"type": "image", "source": {"type": "file"}}) is None


def test_responses_api_input_text_block_is_captured():
    messages = to_input_messages(
        [
            HumanMessage(
                content=[{"type": "input_text", "text": "what is this?"}]
            )
        ]
    )
    assert len(messages) == 1
    parts = messages[0].parts
    assert len(parts) == 1
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "what is this?"


def test_responses_api_output_text_block_is_captured():
    # langchain-openai accepts "output_text" on an assistant turn fed back in
    # as request input.
    messages = to_input_messages(
        [AIMessage(content=[{"type": "output_text", "text": "the answer"}])]
    )
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    parts = messages[0].parts
    assert len(parts) == 1
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "the answer"


def test_message_survives_unconvertible_image_block():
    # An image block with no bytes, url, or file id has nothing to record.
    messages = to_input_messages(
        [HumanMessage(content=[{"type": "image", "detail": "high"}])]
    )
    assert messages, "message dropped entirely"
    assert messages[0].role == "user"
    assert messages[0].parts == []


def test_text_kept_when_image_block_is_unconvertible():
    messages = to_input_messages(
        [
            HumanMessage(
                content=[
                    {"type": "input_text", "text": "what is in this image?"},
                    {"type": "input_image", "detail": "high"},
                ]
            )
        ]
    )
    assert messages, "message dropped entirely"
    assert any(isinstance(p, TextPart) for p in messages[0].parts)


def test_message_with_only_unknown_blocks_is_kept():
    messages = to_input_messages(
        [HumanMessage(content=[{"type": "input_file", "file_id": "file-1"}])]
    )
    assert messages
    assert messages[0].parts == []


def test_empty_message_is_still_dropped():
    assert to_input_messages([HumanMessage(content="")]) == []
    assert to_input_messages([HumanMessage(content=[])]) == []


@pytest.mark.parametrize("content", [[""], ["", ""], [{}], ["", {}]])
def test_message_of_blank_blocks_is_dropped(content):
    # Blank blocks are empty content, not content we failed to convert.
    assert to_input_messages([HumanMessage(content=content)]) == []


def test_output_message_survives_unconvertible_blocks():
    # A dropped output message would also lose its finish_reason.
    messages = to_output_messages(
        [AIMessage(content=[{"type": "input_file", "file_id": "file-1"}])],
        finish_reason="stop",
    )
    assert messages
    assert messages[0].parts == []
    assert messages[0].finish_reason == "stop"


def test_empty_output_message_is_still_dropped():
    assert to_output_messages([AIMessage(content="")]) == []


def test_to_input_messages_extracts_image_part():
    image_url = "data:image/jpeg;base64,QUJD"
    content = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    messages = to_input_messages([HumanMessage(content=content)])
    assert len(messages) == 1
    parts = messages[0].parts

    assert any(isinstance(p, BlobPart) for p in parts)
    blob = next(p for p in parts if isinstance(p, BlobPart))
    assert blob.mime_type == "image/jpeg"
    assert blob.content == b"ABC"


def test_instrumentor_shares_handler_across_callback_managers(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManager,
        CallbackManager,
    )

    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        cm = CallbackManager([])
        sync_handlers = [
            h
            for h in cm.handlers
            if isinstance(h, OpenTelemetryLangChainCallbackHandler)
        ]
        assert len(sync_handlers) == 1
        assert sync_handlers[0].run_inline is False

        acm = AsyncCallbackManager([])
        async_handlers = [
            h
            for h in acm.handlers
            if isinstance(h, OpenTelemetryLangChainCallbackHandler)
        ]
        assert len(async_handlers) == 1
        assert sync_handlers[0] is async_handlers[0]
        assert async_handlers[0].run_inline is False
