# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy LM (inference) instrumentation."""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest import mock

import dspy
import pytest
from dspy.utils import DummyLM

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.instrumentation.genai.dspy.utils import (
    parse_provider_and_model,
    resolve_provider,
    resolve_request_model,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode


class FakeLM(dspy.LM):
    """Test helper inheriting directly from dspy.LM."""

    def __init__(
        self,
        responses: list[str] | None = None,
        model: str = "openai/gpt-4o",
        model_type: str = "chat",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key="fake-api-key",
            model_type=model_type,
            **kwargs,
        )
        self._responses = list(responses or ["Paris"])
        self._idx = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        resp_text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        mock_resp = mock.MagicMock()
        mock_choice = mock.MagicMock()
        mock_choice.message.content = str(resp_text)
        mock_choice.finish_reason = "stop"
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o-2024-05-13"
        mock_resp.id = "chatcmpl-123"
        mock_resp.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        return mock_resp

    async def aforward(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


def test_lm_call_sync_prompt(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        lm = FakeLM(responses=["Paris"])
        res = lm("What is the capital of France?")

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat gpt-4o"
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "openai"
    assert span.attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert span.attributes[GenAI.GEN_AI_RESPONSE_MODEL] == "gpt-4o-2024-05-13"
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert span.attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 5

    input_messages = json.loads(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert (
        input_messages[0]["parts"][0]["content"]
        == "What is the capital of France?"
    )

    output_messages = json.loads(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == "Paris"
    assert output_messages[0]["finish_reason"] == "stop"


def test_lm_call_sync_messages(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        lm = FakeLM(responses=["Berlin"])
        res = lm(messages=[{"role": "user", "content": "Capital of Germany?"}])

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_messages = json.loads(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == "Capital of Germany?"


def test_lm_call_request_parameters(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM(responses=["42"])
    res = lm(
        "Compute meaning",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.2,
        stop=["STOP"],
        seed=42,
        n=2,
    )

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes[GenAI.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert span.attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert span.attributes[GenAI.GEN_AI_REQUEST_TOP_P] == 0.9
    assert span.attributes[GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY] == 0.5
    assert span.attributes[GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY] == 0.2
    assert span.attributes[GenAI.GEN_AI_REQUEST_STOP_SEQUENCES] == ("STOP",)
    assert span.attributes[GenAI.GEN_AI_REQUEST_SEED] == 42
    assert span.attributes[GenAI.GEN_AI_REQUEST_CHOICE_COUNT] == 2


def test_lm_call_typed_response(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        lm = FakeLM(responses=["Rome"])
        with dspy.context(experimental=True):
            res = lm("Capital of Italy?")

    assert hasattr(res, "outputs")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes[GenAI.GEN_AI_RESPONSE_MODEL] == "gpt-4o-2024-05-13"
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert span.attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 5

    output_messages = json.loads(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"


def test_lm_call_text_model_type(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM(responses=["Madrid"], model_type="text")
    res = lm("Capital of Spain?")

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat gpt-4o"
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"


def test_lm_call_gemini_provider(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM(responses=["Madrid"], model="gemini/gemini-1.5-pro")
    res = lm("Capital of Spain?")

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat gemini-1.5-pro"
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"


def test_lm_call_vertex_provider(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM(responses=["Madrid"], model="vertex_ai/gemini-1.5-pro")
    res = lm("Capital of Spain?")

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat gemini-1.5-pro"
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.vertex_ai"


def test_lm_call_model_name_only_provider(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM(responses=["Madrid"], model="gemini-1.5-flash")
    res = lm("Capital of Spain?")

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat gemini-1.5-flash"
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"


def test_lm_call_error(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM()

    with mock.patch.object(
        lm, "forward", side_effect=RuntimeError("Model unreachable")
    ):
        with pytest.raises(RuntimeError, match="Model unreachable"):
            lm("Hello")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.anyio
async def test_lm_acall_async(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        lm = FakeLM(responses=["Tokyo"])
        res = await lm.acall("Capital of Japan?")

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat gpt-4o"
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    input_messages = json.loads(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["parts"][0]["content"] == "Capital of Japan?"


@pytest.mark.anyio
async def test_lm_acall_async_messages(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        lm = FakeLM(responses=["London"])
        res = await lm.acall(
            messages=[{"role": "user", "content": "Capital of UK?"}]
        )

    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_messages = json.loads(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["parts"][0]["content"] == "Capital of UK?"


@pytest.mark.anyio
async def test_lm_acall_async_typed_response(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM(responses=["Ottawa"])
    with dspy.context(experimental=True):
        res = await lm.acall("Capital of Canada?")

    assert hasattr(res, "outputs")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)


@pytest.mark.anyio
async def test_lm_acall_async_error(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM()

    with mock.patch.object(
        lm, "aforward", side_effect=ConnectionError("Network timeout")
    ):
        with pytest.raises(ConnectionError, match="Network timeout"):
            await lm.acall("Hello")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[error_attributes.ERROR_TYPE] == "ConnectionError"


def test_lm_content_capture_disabled(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="false",
    ):
        lm = FakeLM(responses=["Canberra"])
        lm("Capital of Australia?")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GenAI.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_dummy_lm_not_instrumented(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    dummy = DummyLM([{"answer": "Paris"}])
    res = dummy("What is the capital of France?")
    assert res == ["[[ ## answer ## ]]\nParis"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0


def test_provider_and_model_resolution() -> None:
    assert parse_provider_and_model("openai/gpt-4o") == ("openai", "gpt-4o")
    assert parse_provider_and_model("anthropic/claude-3-5-sonnet") == (
        "anthropic",
        "claude-3-5-sonnet",
    )
    assert parse_provider_and_model("bedrock/anthropic.claude-3-sonnet") == (
        "bedrock",
        "anthropic.claude-3-sonnet",
    )
    assert parse_provider_and_model("text-completion-openai/davinci") == (
        "openai",
        "davinci",
    )
    assert parse_provider_and_model(None) == (None, None)
    assert parse_provider_and_model("gpt-4o") == (None, "gpt-4o")

    class MockLM:
        def __init__(
            self,
            model: str | None = None,
            model_name: str | None = None,
            provider: object | None = None,
        ):
            self.model = model
            self.model_name = model_name
            self.provider = provider

    class OpenAIProvider:
        pass

    assert resolve_provider(MockLM("openai/gpt-4o")) == "openai"
    assert resolve_request_model(MockLM("openai/gpt-4o")) == "gpt-4o"

    assert resolve_provider(MockLM("anthropic/claude-3")) == "anthropic"
    assert resolve_request_model(MockLM("anthropic/claude-3")) == "claude-3"

    assert (
        resolve_provider(MockLM("bedrock/anthropic.claude")) == "aws.bedrock"
    )
    assert (
        resolve_request_model(MockLM("bedrock/anthropic.claude"))
        == "anthropic.claude"
    )

    assert resolve_provider(MockLM("vertex_ai/gemini-pro")) == "gcp.vertex_ai"
    assert resolve_provider(MockLM("gemini/gemini-pro")) == "gcp.gemini"

    assert (
        resolve_provider(MockLM("custom-model", provider=OpenAIProvider()))
        == "openai"
    )
    assert (
        resolve_request_model(MockLM("custom-model", model_name="my-custom"))
        == "my-custom"
    )

    assert resolve_provider(DummyLM([])) == "dummy"
    assert resolve_request_model(DummyLM([])) == "dummy"


def test_copy_and_deepcopy_lm(
    instrument_dspy: DSPyInstrumentor,
    span_exporter,
) -> None:
    lm = FakeLM(responses=["Copy test"])
    lm_copy = copy.copy(lm)
    lm_deepcopy = copy.deepcopy(lm)

    assert lm_copy("test 1") is not None
    assert lm_deepcopy("test 2") is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.name == "chat gpt-4o"


def test_extract_message_rich_parts() -> None:
    from dspy.core.types import (
        LMMessage,
        LMResponse,
        LMTextPart,
        LMThinkingPart,
        LMToolCallPart,
    )

    from opentelemetry.instrumentation.genai.dspy.utils import (
        _extract_single_message,
        extract_lm_output_messages,
    )
    from opentelemetry.util.genai.types import (
        ReasoningPart,
        TextPart,
        ToolCallRequestPart,
        ToolCallResponsePart,
    )

    # 1. Tool result message dict
    tool_msg = _extract_single_message(
        {"role": "tool", "tool_call_id": "call_123", "content": "42"}
    )
    assert tool_msg is not None
    assert tool_msg.role == "tool"
    assert len(tool_msg.parts) == 1
    assert isinstance(tool_msg.parts[0], ToolCallResponsePart)
    assert tool_msg.parts[0].id == "call_123"
    assert tool_msg.parts[0].response == "42"

    # 2. Assistant message dict with tool calls and reasoning
    asst_msg = _extract_single_message(
        {
            "role": "assistant",
            "content": "Let me check.",
            "reasoning_content": "Thinking about the question...",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query": "weather"}',
                    },
                }
            ],
        }
    )
    assert asst_msg is not None
    assert asst_msg.role == "assistant"
    assert len(asst_msg.parts) == 3
    assert isinstance(asst_msg.parts[0], ToolCallRequestPart)
    assert asst_msg.parts[0].name == "lookup"
    assert asst_msg.parts[0].arguments == {"query": "weather"}
    assert isinstance(asst_msg.parts[1], ReasoningPart)
    assert asst_msg.parts[1].content == "Thinking about the question..."
    assert isinstance(asst_msg.parts[2], TextPart)
    assert asst_msg.parts[2].content == "Let me check."

    # 3. LMMessage object with thinking and tool call
    lm_msg = LMMessage(
        role="assistant",
        parts=[
            LMThinkingPart(text="Analyzing request"),
            LMToolCallPart(id="call_99", name="fetch", args={"id": 1}),
            LMTextPart(text="Done"),
        ],
    )
    extracted_lm_msg = _extract_single_message(lm_msg)
    assert extracted_lm_msg is not None
    assert len(extracted_lm_msg.parts) == 3
    assert isinstance(extracted_lm_msg.parts[0], ReasoningPart)
    assert extracted_lm_msg.parts[0].content == "Analyzing request"
    assert isinstance(extracted_lm_msg.parts[1], ToolCallRequestPart)
    assert extracted_lm_msg.parts[1].id == "call_99"
    assert extracted_lm_msg.parts[1].name == "fetch"
    assert isinstance(extracted_lm_msg.parts[2], TextPart)
    assert extracted_lm_msg.parts[2].content == "Done"

    # 4. extract_lm_output_messages with LMResponse
    lm_resp = LMResponse.from_text("Result text")
    lm_resp.outputs[0].parts.insert(0, LMThinkingPart(text="Output thinking"))
    lm_resp.outputs[0].parts.append(
        LMToolCallPart(id="call_out", name="calc", args={"a": 2})
    )
    output_msgs = extract_lm_output_messages(
        lm_resp, finish_reason="tool_calls"
    )
    assert len(output_msgs) == 1
    assert len(output_msgs[0].parts) == 3
    assert isinstance(output_msgs[0].parts[0], ReasoningPart)
    assert output_msgs[0].parts[0].content == "Output thinking"
    assert isinstance(output_msgs[0].parts[1], TextPart)
    assert output_msgs[0].parts[1].content == "Result text"
    assert isinstance(output_msgs[0].parts[2], ToolCallRequestPart)
    assert output_msgs[0].parts[2].name == "calc"
    assert output_msgs[0].finish_reason == "tool_calls"

    # 5. extract_lm_output_messages with legacy dict item
    legacy_msgs = extract_lm_output_messages(
        [
            {
                "text": "Answer",
                "reasoning_content": "Deep thought",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
                "finish_reason": "stop",
            }
        ]
    )
    assert len(legacy_msgs) == 1
    assert len(legacy_msgs[0].parts) == 3
    assert isinstance(legacy_msgs[0].parts[0], TextPart)
    assert isinstance(legacy_msgs[0].parts[1], ReasoningPart)
    assert isinstance(legacy_msgs[0].parts[2], ToolCallRequestPart)

    # 6. capture_content=False omits tool call arguments
    no_content_msg = _extract_single_message(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query": "weather"}',
                    },
                }
            ],
        },
        capture_content=False,
    )
    assert no_content_msg is not None
    assert isinstance(no_content_msg.parts[0], ToolCallRequestPart)
    assert no_content_msg.parts[0].arguments is None

    no_content_out = extract_lm_output_messages(
        lm_resp, finish_reason="tool_calls", capture_content=False
    )
    assert len(no_content_out) == 1
    assert isinstance(no_content_out[0].parts[2], ToolCallRequestPart)
    assert no_content_out[0].parts[2].arguments is None


def test_extract_multimodal_and_generic_parts() -> None:
    from dspy.core.types import (
        LMAudioPart,
        LMBinaryPart,
        LMCitationPart,
        LMDocumentPart,
        LMImagePart,
        LMMessage,
        LMRefusalPart,
        LMResponse,
        LMSourcePart,
        LMVideoPart,
    )

    from opentelemetry.instrumentation.genai.dspy.utils import (
        _extract_single_message,
        extract_lm_output_messages,
    )
    from opentelemetry.util.genai.types import (
        BlobPart,
        GenericPart,
        UriPart,
    )

    # 1. UriPart for URL specified (image, audio, video, document, dict)
    lm_msg = LMMessage(
        role="user",
        parts=[
            LMImagePart(
                url="https://example.com/img.png", media_type="image/png"
            ),
            LMAudioPart(
                url="https://example.com/audio.mp3", media_type="audio/mp3"
            ),
            LMVideoPart(
                url="https://example.com/video.mp4", media_type="video/mp4"
            ),
            LMDocumentPart(
                url="https://example.com/doc.pdf", media_type="application/pdf"
            ),
        ],
    )
    msg = _extract_single_message(lm_msg)
    assert msg is not None
    assert len(msg.parts) == 4
    assert msg.parts[0] == UriPart(
        mime_type="image/png",
        modality="image",
        uri="https://example.com/img.png",
    )
    assert msg.parts[1] == UriPart(
        mime_type="audio/mp3",
        modality="audio",
        uri="https://example.com/audio.mp3",
    )
    assert msg.parts[2] == UriPart(
        mime_type="video/mp4",
        modality="video",
        uri="https://example.com/video.mp4",
    )
    assert msg.parts[3] == UriPart(
        mime_type="application/pdf",
        modality="document",
        uri="https://example.com/doc.pdf",
    )

    # Dict with image_url
    dict_msg = _extract_single_message(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/pic.jpg"},
                }
            ],
        }
    )
    assert dict_msg is not None
    assert dict_msg.parts[0] == UriPart(
        mime_type=None, modality="image", uri="https://example.com/pic.jpg"
    )

    # Data URL (data:<mime>;base64,...) decoded into BlobPart
    data_url_dict_msg = _extract_single_message(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                }
            ],
        }
    )
    assert data_url_dict_msg is not None
    assert data_url_dict_msg.parts[0] == BlobPart(
        mime_type="image/png", modality="image", content=b"hello"
    )

    data_url_part_msg = _extract_single_message(
        LMMessage(
            role="user",
            parts=[LMImagePart(url="data:image/png;base64,aGVsbG8=")],
        )
    )
    assert data_url_part_msg is not None
    assert data_url_part_msg.parts[0] == BlobPart(
        mime_type="image/png", modality="image", content=b"hello"
    )

    # 2. BlobPart for inline data (images, documents, binary, source)
    blob_msg = LMMessage(
        role="user",
        parts=[
            LMImagePart(data="aGVsbG8=", media_type="image/png"),
            LMDocumentPart(data="aGVsbG8=", media_type="application/pdf"),
            LMBinaryPart(
                data="aGVsbG8=", media_type="application/octet-stream"
            ),
        ],
    )
    extracted_blob = _extract_single_message(blob_msg)
    assert extracted_blob is not None
    assert len(extracted_blob.parts) == 3
    assert extracted_blob.parts[0] == BlobPart(
        mime_type="image/png", modality="image", content=b"hello"
    )
    assert extracted_blob.parts[1] == BlobPart(
        mime_type="application/pdf", modality="document", content=b"hello"
    )
    assert extracted_blob.parts[2] == BlobPart(
        mime_type="application/octet-stream",
        modality="document",
        content=b"hello",
    )

    # LMSourcePart extracted directly and from dict
    from opentelemetry.instrumentation.genai.dspy.utils import _extract_part

    assert _extract_part(
        LMSourcePart(type="source", data="aGVsbG8=", media_type="text/html")
    ) == BlobPart(mime_type="text/html", modality="document", content=b"hello")
    source_dict_msg = _extract_single_message(
        {
            "role": "user",
            "parts": [
                {
                    "type": "source",
                    "data": "aGVsbG8=",
                    "media_type": "text/html",
                }
            ],
        }
    )
    assert source_dict_msg is not None
    assert source_dict_msg.parts[0] == BlobPart(
        mime_type="text/html", modality="document", content=b"hello"
    )

    # 3. Inline audio / video omitted as GenericPart
    av_msg = LMMessage(
        role="user",
        parts=[
            LMAudioPart(data="aGVsbG8=", media_type="audio/wav"),
            LMVideoPart(data="aGVsbG8=", media_type="video/mp4"),
        ],
    )
    extracted_av = _extract_single_message(av_msg)
    assert extracted_av is not None
    assert len(extracted_av.parts) == 2
    assert extracted_av.parts[0] == GenericPart(type="audio")
    assert extracted_av.parts[1] == GenericPart(type="video")

    # 4. GenericPart fallback for other parts (refusal, citation, custom dict, invalid b64)
    refusal_msg = LMMessage(
        role="assistant",
        parts=[
            LMRefusalPart(text="I cannot fulfill this request."),
        ],
    )
    extracted_refusal = _extract_single_message(refusal_msg)
    assert extracted_refusal is not None
    assert extracted_refusal.parts[0] == GenericPart(type="refusal")

    refusal_dict_msg = _extract_single_message(
        {
            "role": "assistant",
            "parts": [{"type": "refusal", "text": "I refuse."}],
        }
    )
    assert refusal_dict_msg is not None
    assert refusal_dict_msg.parts[0] == GenericPart(type="refusal")

    invalid_b64_msg = _extract_single_message(
        {
            "role": "user",
            "parts": [{"type": "image", "data": "invalid-base64-!@#$"}],
        }
    )
    assert invalid_b64_msg is not None
    assert invalid_b64_msg.parts[0] == GenericPart(type="image")

    other_msg = LMMessage(
        role="user",
        parts=[
            LMCitationPart(text="cite", title="title"),
        ],
    )
    extracted_other = _extract_single_message(other_msg)
    assert extracted_other is not None
    assert len(extracted_other.parts) == 1
    assert extracted_other.parts[0] == GenericPart(type="citation")

    custom_dict_msg = _extract_single_message(
        {
            "role": "user",
            "content": [
                {"type": "custom_extension", "val": 123},
            ],
        }
    )
    assert custom_dict_msg is not None
    assert len(custom_dict_msg.parts) == 1
    assert custom_dict_msg.parts[0] == GenericPart(type="custom_extension")

    # 5. Output messages with multimodal and generic parts in LMResponse
    lm_resp = LMResponse.from_text("Text output")
    lm_resp.outputs[0].parts.append(
        LMImagePart(url="https://example.com/out.png", media_type="image/png")
    )
    lm_resp.outputs[0].parts.append(
        LMCitationPart(text="source", title="title")
    )
    out_msgs = extract_lm_output_messages(lm_resp)
    assert len(out_msgs) == 1
    assert len(out_msgs[0].parts) == 3
    assert out_msgs[0].parts[1] == UriPart(
        mime_type="image/png",
        modality="image",
        uri="https://example.com/out.png",
    )
    assert out_msgs[0].parts[2] == GenericPart(type="citation")
