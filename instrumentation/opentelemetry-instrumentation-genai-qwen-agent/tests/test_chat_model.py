# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseChatModel.chat() instrumentation (chat spans)."""
# pylint: disable=redefined-outer-name

import json
from unittest.mock import patch

import pytest
from qwen_agent.llm import get_chat_model
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import FunctionCall, Message

from opentelemetry.sdk.metrics.export import Histogram as HistogramData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode


class _StubChatModel(BaseChatModel):
    """A minimal BaseChatModel subclass for tests without network access."""

    def __init__(self, model="qwen-max", model_type="qwen_dashscope"):
        super().__init__({"model": model, "model_type": model_type})
        # Disable raw_api mode which requires stream-only and an API key.
        self.use_raw_api = False

    def _chat_no_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_with_functions(self, messages, functions, **kwargs):
        raise NotImplementedError


def _get_chat_spans(span_exporter):
    return [
        s
        for s in span_exporter.get_finished_spans()
        if s.name.startswith("chat")
    ]


def _get_metric_data_points(metric_reader, metric_name):
    metrics_data = metric_reader.get_metrics_data()
    points = []
    for resource_metrics in metrics_data.resource_metrics or []:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == metric_name and isinstance(
                    metric.data, HistogramData
                ):
                    points.extend(metric.data.data_points)
    return points


@pytest.mark.vcr()
def test_stream_chat(span_exporter, metric_reader, instrument_with_content):
    """A streaming chat() records one chat span once the stream is drained."""
    llm = get_chat_model({"model": "qwen-max", "model_type": "qwen_dashscope"})
    responses = list(
        llm.chat(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            stream=True,
        )
    )
    assert responses

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    span = chat_spans[0]
    assert span.name == "chat qwen-max"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code != StatusCode.ERROR

    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "qwen-max"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "qwen-max"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert isinstance(attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS], int)
    assert isinstance(attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS], int)

    input_messages = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[-1]["role"] == "user"
    assert input_messages[-1]["parts"][0]["type"] == "text"
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"

    duration_points = _get_metric_data_points(
        metric_reader, "gen_ai.client.operation.duration"
    )
    assert len(duration_points) == 1
    token_points = _get_metric_data_points(
        metric_reader, "gen_ai.client.token.usage"
    )
    token_types = {
        point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]
        for point in token_points
    }
    assert token_types == {"input", "output"}


@pytest.mark.vcr()
def test_non_stream_chat(span_exporter, instrument_with_content):
    """A non-streaming chat() records one chat span."""
    llm = get_chat_model(
        {
            "model": "qwen-max",
            "model_type": "qwen_dashscope",
            # use_raw_api only supports stream=True on newer qwen-agent.
            "generate_cfg": {"use_raw_api": False},
        }
    )
    responses = list(
        llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2? Answer with just the number.",
                }
            ],
            stream=False,
        )
    )
    assert responses

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    span = chat_spans[0]
    assert span.name == "chat qwen-max"
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "dashscope"
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "qwen-max"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "qwen-max"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    # gen_ai.response.id is the DashScope request id.
    response_id = attrs[GenAIAttributes.GEN_AI_RESPONSE_ID]
    assert isinstance(response_id, str)
    assert response_id


def test_non_stream_chat_no_content(span_exporter, instrument_no_content):
    """Without content capture, no message content lands on the span."""
    model = _StubChatModel()
    fake_response = [Message(role="assistant", content="Hello there!")]

    with patch.object(
        _StubChatModel, "_chat_no_stream", return_value=fake_response
    ):
        model.chat(messages=[Message(role="user", content="Hi")], stream=False)

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    attrs = dict(chat_spans[0].attributes or {})
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attrs
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attrs


def test_non_stream_chat_records_token_usage(
    span_exporter, instrument_no_content
):
    """Token usage is extracted from Message.extra.model_service_info."""
    model = _StubChatModel(model="qwen-plus")
    fake_response = [
        Message(
            role="assistant",
            content="4",
            extra={
                "model_service_info": {
                    "usage": {
                        "input_tokens": 21,
                        "output_tokens": 1,
                        "total_tokens": 22,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    }
                }
            },
        )
    ]

    with patch.object(
        _StubChatModel, "_chat_no_stream", return_value=fake_response
    ):
        model.chat(
            messages=[Message(role="user", content="What is 2+2?")],
            stream=False,
        )

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    attrs = dict(chat_spans[0].attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 21
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 1
    assert attrs[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 4


def test_stream_chat_keeps_most_complete_usage(
    span_exporter, instrument_no_content
):
    """Streaming chunks carry cumulative usage; the largest one wins."""
    model = _StubChatModel(model="qwen-plus")

    def fake_stream(messages, **kwargs):
        yield [
            Message(
                role="assistant",
                content="The",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 1}
                    }
                },
            )
        ]
        yield [
            Message(
                role="assistant",
                content="The answer",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 5}
                    }
                },
            )
        ]
        yield [Message(role="assistant", content="The answer is 4.")]

    with patch.object(_StubChatModel, "_chat_stream", side_effect=fake_stream):
        list(
            model.chat(
                messages=[Message(role="user", content="What is 2+2?")],
                stream=True,
            )
        )

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    attrs = dict(chat_spans[0].attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 18
    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5


def test_chat_with_function_call_response(
    span_exporter, instrument_with_content
):
    """A function_call response yields finish_reason tool_calls and a
    tool_call output message part."""
    model = _StubChatModel()
    fake_response = [
        Message(
            role="assistant",
            content="",
            function_call=FunctionCall(
                name="get_weather", arguments='{"city": "Beijing"}'
            ),
        )
    ]

    with patch.object(
        _StubChatModel, "_chat_no_stream", return_value=fake_response
    ):
        model.chat(
            messages=[Message(role="user", content="Weather?")],
            stream=False,
        )

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    attrs = dict(chat_spans[0].attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_calls",
    )
    output_messages = json.loads(attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    tool_call_parts = [
        part
        for message in output_messages
        for part in message["parts"]
        if part["type"] == "tool_call"
    ]
    assert tool_call_parts
    assert tool_call_parts[0]["name"] == "get_weather"
    assert tool_call_parts[0]["arguments"] == {"city": "Beijing"}


def test_chat_with_tool_definitions(span_exporter, instrument_with_content):
    """Function definitions passed to chat() land in gen_ai.tool.definitions."""
    model = _StubChatModel()
    fake_response = [Message(role="assistant", content="ok")]
    functions = [
        {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]

    with (
        patch.object(
            _StubChatModel, "_chat_with_functions", return_value=fake_response
        ),
        patch.object(
            _StubChatModel, "_chat_no_stream", return_value=fake_response
        ),
    ):
        model.chat(
            messages=[Message(role="user", content="Weather?")],
            functions=functions,
            stream=False,
        )

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    attrs = dict(chat_spans[0].attributes or {})
    tool_definitions = json.loads(
        attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    )
    assert tool_definitions[0]["name"] == "get_weather"
    assert tool_definitions[0]["type"] == "function"


def test_non_stream_chat_error(span_exporter, instrument_no_content):
    """An exception raised by chat() re-raises and marks the span as error."""
    model = _StubChatModel()

    with patch.object(
        _StubChatModel,
        "_chat_no_stream",
        side_effect=RuntimeError("API timeout"),
    ):
        with pytest.raises(RuntimeError, match="API timeout"):
            model.chat(
                messages=[Message(role="user", content="Hi")], stream=False
            )

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    span = chat_spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


def test_stream_chat_error_mid_iteration(span_exporter, instrument_no_content):
    """A stream-side error re-raises unchanged and finalizes the span."""
    model = _StubChatModel()

    def fake_stream(messages, **kwargs):
        yield [Message(role="assistant", content="partial")]
        raise ConnectionError("stream broken")

    with patch.object(_StubChatModel, "_chat_stream", side_effect=fake_stream):
        stream = model.chat(
            messages=[Message(role="user", content="Hi")], stream=True
        )
        with pytest.raises(ConnectionError, match="stream broken"):
            for _ in stream:
                pass

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    span = chat_spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


def test_stream_chat_caller_side_error(span_exporter, instrument_no_content):
    """A caller-side error inside ``with stream:`` re-raises unchanged and
    finalizes the span with the matching error.type."""
    model = _StubChatModel()

    def fake_stream(messages, **kwargs):
        yield [Message(role="assistant", content="partial")]
        yield [Message(role="assistant", content="partial more")]

    with patch.object(_StubChatModel, "_chat_stream", side_effect=fake_stream):
        stream = model.chat(
            messages=[Message(role="user", content="Hi")], stream=True
        )
        with pytest.raises(ValueError, match="caller aborted"):
            with stream:
                raise ValueError("caller aborted")

    chat_spans = _get_chat_spans(span_exporter)
    assert len(chat_spans) == 1
    span = chat_spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[ErrorAttributes.ERROR_TYPE] == "ValueError"
