# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch
from uuid import UUID

import pytest
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatResponse,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
from llama_index.core.llms import ChatMessage, MockFunctionCallingLLM
from llama_index.core.tools import (
    AsyncBaseTool,
    BaseTool,
    FunctionTool,
    ToolMetadata,
    ToolOutput,
)
from llama_index.core.workflow.errors import WorkflowRuntimeError
from openai import RateLimitError

from opentelemetry.instrumentation.genai.llama_index._handler import (
    _agent_input,
    _input_message,
    _method_name,
    _output_message,
    _set_agent_output,
    _tool_definition,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.types import (
    BlobPart,
    GenericToolDefinition,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    UriPart,
)


def _spans_named(span_exporter, name: str):
    return [s for s in span_exporter.get_finished_spans() if s.name == name]


def _assert_error_type(span: ReadableSpan, expected: str) -> None:
    error_type = span.attributes[ErrorAttributes.ERROR_TYPE]
    assert isinstance(error_type, str)
    assert error_type.rsplit(".", 1)[-1] == expected


@pytest.mark.parametrize(
    ("span_id", "expected"),
    [
        (
            "FunctionAgent.run-550e8400-e29b-41d4-a716-446655440000",
            "run",
        ),
        (
            "BaseWorkflowAgent.call_tool-550e8400-e29b-41d4-a716-446655440000",
            "call_tool",
        ),
        (
            "FunctionTool.acall-550e8400-e29b-41d4-a716-446655440000",
            "acall",
        ),
    ],
)
def test_method_name_extracts_dispatcher_method(
    span_id: str, expected: str
) -> None:
    assert _method_name(span_id) == expected


def test_input_message_preserves_tool_call_blocks() -> None:
    message = ChatMessage(
        role="assistant",
        blocks=[
            TextBlock(text="I will check the weather."),
            ToolCallBlock(
                tool_call_id="weather-call",
                tool_name="weather",
                tool_kwargs={"city": "Paris"},
            ),
        ],
    )

    converted = _input_message(message)

    assert converted.parts == [
        TextPart(content="I will check the weather."),
        ToolCallRequestPart(
            id="weather-call",
            name="weather",
            arguments={"city": "Paris"},
        ),
    ]


def test_output_message_preserves_tool_call_blocks() -> None:
    message = ChatMessage(
        role="assistant",
        blocks=[
            ToolCallBlock(
                tool_call_id="weather-call",
                tool_name="weather",
                tool_kwargs={"city": "Paris"},
            )
        ],
    )

    converted = _output_message(message)

    assert converted.parts == [
        ToolCallRequestPart(
            id="weather-call",
            name="weather",
            arguments={"city": "Paris"},
        )
    ]
    assert converted.finish_reason == "tool_calls"


def test_input_message_preserves_multimodal_blocks() -> None:
    image = b"\x89PNG\r\n\x1a\n"
    message = ChatMessage(
        role="user",
        blocks=[
            ImageBlock(image=image, image_mimetype="image/png"),
            AudioBlock(url="https://example.com/audio.mp3", format="mp3"),
            DocumentBlock(
                url="https://example.com/document.pdf",
                document_mimetype="application/pdf",
            ),
            ThinkingBlock(content="Consider the available evidence."),
        ],
    )

    converted = _input_message(message)

    assert converted.parts == [
        BlobPart(content=image, mime_type="image/png", modality="image"),
        UriPart(
            uri="https://example.com/audio.mp3",
            mime_type="audio/mpeg",
            modality="audio",
        ),
        UriPart(
            uri="https://example.com/document.pdf",
            mime_type="application/pdf",
            modality="document",
        ),
        ReasoningPart(content="Consider the available evidence."),
    ]


def test_tool_definition_skips_invalid_parameter_schema() -> None:
    tool = FunctionTool.from_defaults(lambda value: value, name="echo")

    with patch.object(
        ToolMetadata,
        "get_parameters_dict",
        side_effect=ValueError("invalid schema"),
    ):
        assert _tool_definition(tool) is None


def test_generic_tool_definition_is_preserved() -> None:
    class SearchTool(BaseTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(name="search", description="Search documents.")

        def __call__(self, input: Any) -> ToolOutput:
            return ToolOutput(
                tool_name="search",
                content=str(input),
                raw_input={"input": input},
                raw_output=input,
            )

    assert _tool_definition(SearchTool()) == GenericToolDefinition(
        name="search", type="SearchTool"
    )


@pytest.mark.asyncio
async def test_agent_tool_span_uses_generic_tool_metadata(
    span_exporter, instrument_llama_index
) -> None:
    class StoreTool(AsyncBaseTool):
        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(name="store", description="Store a value.")

        def call(self, value: str) -> ToolOutput:
            return ToolOutput(
                tool_name="store",
                content=value,
                raw_input={"value": value},
                raw_output=value,
            )

        async def acall(self, value: str) -> ToolOutput:
            return self.call(value)

    def response_generator(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="stored")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="store-call",
                    tool_name="store",
                    tool_kwargs={"value": "hello"},
                )
            ],
        )

    agent = FunctionAgent(
        name="storage-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        tools=[StoreTool()],
        streaming=False,
    )

    result = await agent.run(user_msg="Store hello")
    assert result.response.content == "stored"

    tool_span = _spans_named(span_exporter, "execute_tool store")[0]
    assert (
        tool_span.attributes[GenAIAttributes.GEN_AI_TOOL_TYPE] == "StoreTool"
    )
    assert (
        tool_span.attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION]
        == "Store a value."
    )


@pytest.mark.asyncio
async def test_agent_run_span(
    span_exporter, instrument_llama_index_with_content
) -> None:
    agent = FunctionAgent(
        name="math-agent",
        description="Answers math questions.",
        system_prompt="Be concise.",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    result = await agent.run(user_msg="What is two plus two?")
    assert result.response.content

    spans = _spans_named(span_exporter, "invoke_agent math-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == SpanKind.INTERNAL
    assert span.status.status_code != StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_NAME] == "math-agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_DESCRIPTION] == (
        "Answers math questions."
    )
    assert isinstance(attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], str)
    input_messages = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages == [
        {
            "parts": [{"content": "What is two plus two?", "type": "text"}],
            "role": "user",
        }
    ]
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert json.loads(attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]) == [
        {"content": "Be concise.", "type": "text"}
    ]
    assert GenAIAttributes.GEN_AI_PROVIDER_NAME not in attrs


@pytest.mark.asyncio
async def test_agent_content_extraction_is_opt_in(
    span_exporter, instrument_llama_index
) -> None:
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    agent = FunctionAgent(
        name="no-content-agent",
        system_prompt="Do not capture this.",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        tools=[FunctionTool.from_defaults(echo)],
        streaming=False,
    )

    with (
        patch(
            "opentelemetry.instrumentation.genai.llama_index._handler._agent_input",
            wraps=_agent_input,
        ) as input_spy,
        patch(
            "opentelemetry.instrumentation.genai.llama_index._handler._set_agent_output",
            wraps=_set_agent_output,
        ) as output_spy,
    ):
        await agent.run(user_msg="Do not capture this either.")

    input_spy.assert_not_called()
    output_spy.assert_not_called()
    span = _spans_named(span_exporter, "invoke_agent no-content-agent")[0]
    attrs = dict(span.attributes or {})
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in attrs
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in attrs


@pytest.mark.asyncio
async def test_agent_run_captures_chat_message_input(
    span_exporter, instrument_llama_index_with_content
) -> None:
    agent = FunctionAgent(
        name="chat-message-agent",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    await agent.run(
        user_msg=ChatMessage(
            role="user", blocks=[TextBlock(text="Hello from a ChatMessage")]
        )
    )

    span = _spans_named(span_exporter, "invoke_agent chat-message-agent")[0]
    assert json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    ) == [
        {
            "parts": [{"content": "Hello from a ChatMessage", "type": "text"}],
            "role": "user",
        }
    ]


@pytest.mark.asyncio
async def test_agent_span_skips_unnamed_tool_definition(
    span_exporter, instrument_llama_index_with_content
) -> None:
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    unnamed_tool = FunctionTool(
        fn=lambda: None,
        metadata=ToolMetadata(
            description="A tool without a name.",
            name=None,
            fn_schema=None,
        ),
    )
    agent = FunctionAgent(
        name="agent-with-unnamed-tool",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        tools=[FunctionTool.from_defaults(echo), unnamed_tool],
        streaming=False,
    )

    result = await agent.run(user_msg="Hello!")
    assert result.response.content

    span = _spans_named(span_exporter, "invoke_agent agent-with-unnamed-tool")[
        0
    ]
    definitions = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    )
    assert len(definitions) == 1
    assert definitions[0]["name"] == "echo"


@pytest.mark.asyncio
async def test_agent_metadata_error_does_not_leak_span_context(
    span_exporter, instrument_llama_index
) -> None:
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    def response_generator(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="done")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="echo-call",
                    tool_name="echo",
                    tool_kwargs={"value": "hello"},
                )
            ],
        )

    agent = FunctionAgent(
        name="metadata-error-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        tools=[FunctionTool.from_defaults(echo)],
        streaming=False,
    )

    with patch(
        "opentelemetry.instrumentation.genai.llama_index._handler._tool_definitions",
        side_effect=ValueError("metadata extraction failed"),
    ):
        result = await agent.run(user_msg="Call echo")
    assert result.response.content == "done"

    assert not _spans_named(span_exporter, "invoke_agent metadata-error-agent")
    tool_span = _spans_named(span_exporter, "execute_tool echo")[0]
    assert tool_span.parent is None


@pytest.mark.asyncio
async def test_delegated_inference_is_not_reemitted(
    span_exporter, instrument_llama_index, openai_llm, vcr
) -> None:
    agent = FunctionAgent(
        name="provider-agent",
        llm=openai_llm,
        streaming=False,
    )

    with vcr.use_cassette("inference.yaml"):
        result = await agent.run(user_msg="Hello!")
    assert result.response.content == "Hello! How can I assist you today?"

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent provider-agent"]
    assert all(
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] != "chat"
        for span in spans
    )


@pytest.mark.asyncio
async def test_agent_error_re_raises(
    span_exporter, instrument_llama_index
) -> None:
    agent = FunctionAgent(
        name="broken-agent",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    async def fail(*args, **kwargs):
        raise ValueError("agent failed")

    with patch.object(FunctionAgent, "take_step", side_effect=fail):
        with pytest.raises(ValueError, match="agent failed"):
            await agent.run(user_msg="fail")

    spans = _spans_named(span_exporter, "invoke_agent broken-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_react_agent_run_span(
    span_exporter, instrument_llama_index_with_content
) -> None:
    def response_generator(messages, **kwargs):
        return ChatMessage(
            role="assistant",
            content="Thought: I know the answer.\nAnswer: Four.",
        )

    agent = ReActAgent(
        name="react-math-agent",
        description="Answers math questions with ReAct reasoning.",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        streaming=False,
    )

    result = await agent.run(user_msg="What is two plus two?")
    assert result.response.content == "Four."

    spans = _spans_named(span_exporter, "invoke_agent react-math-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == SpanKind.INTERNAL
    assert span.status.status_code != StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_NAME] == "react-math-agent"
    assert attrs[GenAIAttributes.GEN_AI_AGENT_DESCRIPTION] == (
        "Answers math questions with ReAct reasoning."
    )
    assert isinstance(attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], str)
    assert json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]) == [
        {
            "parts": [{"content": "What is two plus two?", "type": "text"}],
            "role": "user",
        }
    ]
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"] == [
        {"content": "Four.", "type": "text"}
    ]


@pytest.mark.asyncio
async def test_react_agent_error_re_raises(
    span_exporter, instrument_llama_index
) -> None:
    agent = ReActAgent(
        name="broken-react-agent",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    with patch.object(
        ReActAgent,
        "take_step",
        side_effect=ValueError("react agent failed"),
    ):
        with pytest.raises(ValueError, match="react agent failed"):
            await agent.run(user_msg="fail")

    spans = _spans_named(span_exporter, "invoke_agent broken-react-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_agent_missing_input_re_raises(
    span_exporter, instrument_llama_index
) -> None:
    agent = FunctionAgent(
        name="no-input-agent",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=False,
    )

    with pytest.raises(
        ValueError, match="Must provide either user_msg or chat_history"
    ):
        await agent.run()

    spans = _spans_named(span_exporter, "invoke_agent no-input-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_agent_max_iterations_re_raises(
    span_exporter, instrument_llama_index
) -> None:
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    def response_generator(messages, **kwargs):
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="echo-call",
                    tool_name="echo",
                    tool_kwargs={"value": "again"},
                )
            ],
        )

    agent = FunctionAgent(
        name="looping-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        tools=[FunctionTool.from_defaults(echo)],
        streaming=False,
    )

    with pytest.raises(WorkflowRuntimeError, match="Max iterations of 2"):
        await agent.run(user_msg="loop forever", max_iterations=2)

    spans = _spans_named(span_exporter, "invoke_agent looping-agent")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    _assert_error_type(span, "WorkflowRuntimeError")

    tool_span = _spans_named(span_exporter, "execute_tool echo")[0]
    assert tool_span.status.status_code != StatusCode.ERROR
    assert tool_span.context.trace_id == span.context.trace_id
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == span.context.span_id


@pytest.mark.asyncio
async def test_react_agent_malformed_response_reaches_max_iterations(
    span_exporter, instrument_llama_index, openai_llm_without_retries, vcr
) -> None:
    agent = ReActAgent(
        name="malformed-react-agent",
        llm=openai_llm_without_retries,
        streaming=False,
    )

    with vcr.use_cassette("react_malformed.yaml"):
        with pytest.raises(WorkflowRuntimeError, match="Max iterations of 1"):
            await agent.run(user_msg="What is two plus two?", max_iterations=1)

    span = _spans_named(span_exporter, "invoke_agent malformed-react-agent")[0]
    assert span.status.status_code == StatusCode.ERROR
    _assert_error_type(span, "WorkflowRuntimeError")


@pytest.mark.asyncio
async def test_provider_error_re_raises(
    span_exporter, instrument_llama_index, openai_llm_without_retries, vcr
) -> None:
    agent = FunctionAgent(
        name="rate-limited-agent",
        llm=openai_llm_without_retries,
        streaming=False,
    )

    with vcr.use_cassette("rate_limit.yaml"):
        with pytest.raises(RateLimitError):
            await agent.run(user_msg="Hello!")

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent rate-limited-agent"]
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    _assert_error_type(span, "RateLimitError")


@pytest.mark.asyncio
async def test_streaming_provider_error_re_raises(
    span_exporter, instrument_llama_index, openai_llm_without_retries, vcr
) -> None:
    agent = FunctionAgent(
        name="streaming-rate-limited-agent",
        llm=openai_llm_without_retries,
        streaming=True,
    )

    with vcr.use_cassette("rate_limit_streaming.yaml"):
        with pytest.raises(RateLimitError):
            await agent.run(user_msg="Hello!")

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "invoke_agent streaming-rate-limited-agent"
    ]
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    _assert_error_type(span, "RateLimitError")


@pytest.mark.asyncio
async def test_stream_iteration_error_re_raises(
    span_exporter, instrument_llama_index
) -> None:
    error = ConnectionError("stream disconnected")

    async def response_stream():
        yield ChatResponse(
            message=ChatMessage(role="assistant", content="partial")
        )
        raise error

    async def failing_stream(*args, **kwargs):
        return response_stream()

    agent = FunctionAgent(
        name="stream-error-agent",
        llm=MockFunctionCallingLLM(is_chat_model=True),
        streaming=True,
    )

    with patch.object(
        MockFunctionCallingLLM,
        "astream_chat_with_tools",
        side_effect=failing_stream,
    ):
        with pytest.raises(ConnectionError) as exc_info:
            await agent.run(user_msg="Hello!")

    assert exc_info.value is error
    span = _spans_named(span_exporter, "invoke_agent stream-error-agent")[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio
async def test_agent_recovers_from_tool_error(
    span_exporter,
    instrument_llama_index,
    openai_llm_without_retries,
    framework_span_handler: SimpleSpanHandler,
    vcr,
) -> None:
    def broken_tool() -> None:
        """Raise a tool failure."""
        raise RuntimeError("tool failed")

    agent = FunctionAgent(
        name="recovering-provider-agent",
        llm=openai_llm_without_retries,
        tools=[FunctionTool.from_defaults(broken_tool)],
        streaming=False,
    )

    with vcr.use_cassette("tool_error_recovery.yaml"):
        result = await agent.run(user_msg="Run the tool")
    assert result.response.content

    agent_span = _spans_named(
        span_exporter, "invoke_agent recovering-provider-agent"
    )[0]
    assert agent_span.status.status_code != StatusCode.ERROR
    tool_span = _spans_named(span_exporter, "execute_tool broken_tool")[0]
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
    assert tool_span.context.trace_id == agent_span.context.trace_id
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id

    expected_framework_spans = {
        "FunctionAgent.run": "run",
        "BaseWorkflowAgent.call_tool": "call_tool",
        "FunctionTool.acall": "acall",
    }
    completed_span_ids = [
        span.id_
        for span in (
            framework_span_handler.completed_spans
            + framework_span_handler.dropped_spans
        )
    ]
    qualified_names = {
        span_id.partition("-")[0] for span_id in completed_span_ids
    }
    assert expected_framework_spans.keys() <= qualified_names, qualified_names
    for qualified_name, method_name in expected_framework_spans.items():
        span_id = next(
            span_id
            for span_id in completed_span_ids
            if span_id.startswith(f"{qualified_name}-")
        )
        generated_name, separator, generated_id = span_id.partition("-")
        assert generated_name == qualified_name
        assert separator == "-"
        assert UUID(generated_id).version == 4
        assert _method_name(span_id) == method_name


@pytest.mark.asyncio
async def test_unknown_tool_output_marks_tool_span_as_error(
    span_exporter, instrument_llama_index
) -> None:
    def response_generator(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="done")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="unknown-tool-call",
                    tool_name="nonexistent",
                    tool_kwargs={},
                )
            ],
        )

    agent = FunctionAgent(
        name="unknown-tool-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        streaming=False,
    )

    result = await agent.run(user_msg="Call a tool")
    assert result.response.content == "done"

    tool_span = _spans_named(span_exporter, "execute_tool nonexistent")[0]
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio
async def test_agent_workflow_emits_tool_span(
    span_exporter, instrument_llama_index
) -> None:
    def echo(value: str) -> str:
        """Echo a value."""
        return value

    def response_generator(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="done")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="workflow-tool-call",
                    tool_name="echo",
                    tool_kwargs={"value": "hello"},
                )
            ],
        )

    agent = FunctionAgent(
        name="workflow-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        tools=[FunctionTool.from_defaults(echo)],
        streaming=False,
    )
    workflow = AgentWorkflow(agents=[agent])

    result = await workflow.run(user_msg="Call echo")
    assert result.response.content == "done"
    tool_spans = _spans_named(span_exporter, "execute_tool echo")
    assert len(tool_spans) == 1
    assert tool_spans[0].parent is None

    FunctionTool.from_defaults(echo)(value="after workflow")
    assert len(_spans_named(span_exporter, "execute_tool echo")) == 2


def test_sync_tool_span(
    span_exporter, instrument_llama_index_with_content
) -> None:
    def multiply(value: int) -> int:
        """Multiply a number by two."""
        return value * 2

    tool = FunctionTool.from_defaults(multiply)
    result = tool(4)
    assert result.raw_output == 8

    spans = _spans_named(span_exporter, "execute_tool multiply")
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "multiply"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
    assert isinstance(attrs[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION], str)
    assert json.loads(attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]) == {
        "value": 4
    }
    assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT] == 8


@pytest.mark.asyncio
async def test_async_tool_span_nested_under_agent(
    span_exporter, instrument_llama_index_with_content
) -> None:
    async def weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    tool = FunctionTool.from_defaults(async_fn=weather)

    def response_generator(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="It is sunny.")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="weather-call",
                    tool_name="weather",
                    tool_kwargs={"city": "Paris"},
                )
            ],
        )

    agent = FunctionAgent(
        name="weather-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        tools=[tool],
        streaming=False,
    )
    await agent.run(user_msg="What is the weather?")

    agent_span = _spans_named(span_exporter, "invoke_agent weather-agent")[0]
    tool_spans = _spans_named(span_exporter, "execute_tool weather")
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]
    assert tool_span.context.trace_id == agent_span.context.trace_id
    assert tool_span.parent is not None
    tool_attrs = dict(tool_span.attributes or {})
    assert tool_attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID] == "weather-call"
    assert isinstance(tool_attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID], str)
    assert tool_attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"


def test_tool_content_is_opt_in(span_exporter, instrument_llama_index) -> None:
    tool = FunctionTool.from_defaults(lambda value: value * 2, name="double")
    tool(value=3)

    span = _spans_named(span_exporter, "execute_tool double")[0]
    attrs = dict(span.attributes or {})
    assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
    assert GenAIAttributes.GEN_AI_TOOL_CALL_RESULT not in attrs


def test_tool_error_re_raises(span_exporter, instrument_llama_index) -> None:
    def broken() -> None:
        raise RuntimeError("tool failed")

    tool = FunctionTool.from_defaults(broken)
    with pytest.raises(RuntimeError, match="tool failed"):
        tool()

    spans = _spans_named(span_exporter, "execute_tool broken")
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
