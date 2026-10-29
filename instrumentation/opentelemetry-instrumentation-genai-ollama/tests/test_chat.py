# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Client.chat`` and the module-level ``ollama.chat``."""

from __future__ import annotations

import gc
import inspect
import json
from typing import Any

import ollama
import pytest

from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler

from .conftest import CHAT_MODEL, HOST, REASONING_MODEL

MESSAGES = [{"role": "user", "content": "Say hello in three words."}]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
      city: The city name
    """
    return "sunny"


def _messages(span: Any, key: str) -> list[dict[str, Any]]:
    return json.loads(span.attributes[key])


def test_chat_records_a_chat_span(instrument_with_content, span_exporter, vcr):
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        response = ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES
        )

    (span,) = span_exporter.get_finished_spans()
    assert span.name == f"chat {CHAT_MODEL}"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[GenAI.GEN_AI_PROVIDER_NAME] == "ollama"
    assert attrs[GenAI.GEN_AI_REQUEST_MODEL] == CHAT_MODEL
    assert attrs[GenAI.GEN_AI_RESPONSE_MODEL] == response.model
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == response.prompt_eval_count
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == response.eval_count
    assert attrs[server_attributes.SERVER_ADDRESS] == "localhost"
    assert attrs[server_attributes.SERVER_PORT] == 11434

    assert _messages(span, GenAI.GEN_AI_INPUT_MESSAGES) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": MESSAGES[0]["content"]}],
            "name": None,
        }
    ]
    (output,) = _messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert output["role"] == "assistant"
    assert output["parts"] == [
        {"type": "text", "content": response.message.content}
    ]


def test_chat_without_content_capture_records_no_messages(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        ollama.Client(host=HOST).chat(model=CHAT_MODEL, messages=MESSAGES)

    (span,) = span_exporter.get_finished_spans()
    assert GenAI.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert span.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] > 0


def test_chat_records_request_options(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_records_request_options.yaml"):
        ollama.Client(host=HOST).chat(
            model=CHAT_MODEL,
            messages=MESSAGES,
            format="json",
            options=ollama.Options(
                temperature=0.2,
                top_p=0.9,
                top_k=20,
                seed=42,
                num_predict=30,
                stop=["\n\n"],
            ),
        )

    (span,) = span_exporter.get_finished_spans()
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_REQUEST_TEMPERATURE] == pytest.approx(0.2)
    assert attrs[GenAI.GEN_AI_REQUEST_TOP_P] == pytest.approx(0.9)
    assert attrs[GenAI.GEN_AI_REQUEST_TOP_K] == 20
    assert attrs[GenAI.GEN_AI_REQUEST_SEED] == 42
    assert attrs[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 30
    assert attrs[GenAI.GEN_AI_REQUEST_STOP_SEQUENCES] == ("\n\n",)
    assert attrs[GenAI.GEN_AI_OUTPUT_TYPE] == "json"


def test_chat_stream_records_the_drained_response(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette(
        "test_chat_stream_records_the_drained_response.yaml"
    ):
        chunks = list(
            ollama.Client(host=HOST).chat(
                model=CHAT_MODEL, messages=MESSAGES, stream=True
            )
        )

    (span,) = span_exporter.get_finished_spans()
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_REQUEST_STREAM] is True
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert (
        attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == chunks[-1].prompt_eval_count
    )
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == chunks[-1].eval_count
    (output,) = _messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert output["parts"] == [
        {
            "type": "text",
            "content": "".join(chunk.message.content for chunk in chunks),
        }
    ]


def test_chat_stream_span_ends_only_after_the_stream_is_drained(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette(
        "test_chat_stream_records_the_drained_response.yaml"
    ):
        stream = ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES, stream=True
        )
        next(iter(stream))
        assert span_exporter.get_finished_spans() == ()
        for _ in stream:
            pass

    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.parametrize(
    "tool, cassette",
    [
        (WEATHER_TOOL, "test_chat_records_tool_calls.yaml"),
        (get_weather, "test_chat_records_tool_calls_from_a_function.yaml"),
    ],
    ids=["dict", "function"],
)
def test_chat_records_tool_calls(
    instrument_with_content, span_exporter, vcr, tool, cassette
):
    with vcr.use_cassette(cassette):
        ollama.Client(host=HOST).chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Use the tool.",
                }
            ],
            tools=[tool],
            options={"temperature": 0},
        )

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    (definition,) = json.loads(span.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS])
    assert definition["name"] == "get_weather"
    (output,) = _messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert output["parts"] == [
        {
            "type": "tool_call",
            "id": None,
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
    ]


def test_chat_records_tool_results_in_input_messages(
    instrument_with_content, span_exporter, vcr
):
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "sunny", "tool_name": "get_weather"},
    ]
    with vcr.use_cassette(
        "test_chat_records_tool_results_in_input_messages.yaml"
    ):
        ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=messages, tools=[WEATHER_TOOL]
        )

    (span,) = span_exporter.get_finished_spans()
    inputs = _messages(span, GenAI.GEN_AI_INPUT_MESSAGES)
    assert inputs[1]["parts"] == [
        {
            "type": "tool_call",
            "id": None,
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
    ]
    assert inputs[2] == {
        "role": "tool",
        "parts": [
            {"type": "tool_call_response", "id": None, "response": "sunny"}
        ],
        "name": None,
    }


@pytest.mark.skipif(
    "think" not in inspect.signature(ollama.Client.chat).parameters,
    reason="the think parameter is not available on this ollama version",
)
def test_chat_records_reasoning_and_a_length_finish(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette(
        "test_chat_records_reasoning_and_a_length_finish.yaml"
    ):
        ollama.Client(host=HOST).chat(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": "What is 17 * 23?"}],
            think=True,
            options={"num_predict": 40},
        )

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("length",)
    (output,) = _messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert output["parts"][0]["type"] == "reasoning"
    assert output["parts"][0]["content"]


def test_chat_error_is_recorded_and_reraised(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_error_is_recorded_and_reraised.yaml"):
        with pytest.raises(ollama.ResponseError):
            ollama.Client(host=HOST).chat(
                model="no-such-model:1b", messages=MESSAGES
            )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )


def test_chat_with_positional_arguments(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        ollama.Client(host=HOST).chat(CHAT_MODEL, MESSAGES)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_REQUEST_MODEL] == CHAT_MODEL
    assert len(_messages(span, GenAI.GEN_AI_INPUT_MESSAGES)) == 1


def test_chat_with_a_message_generator_sends_every_message(
    instrument_with_content, span_exporter, vcr
):
    def messages():
        yield from MESSAGES

    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        response = ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=messages()
        )

    assert response.message.content
    (span,) = span_exporter.get_finished_spans()
    assert len(_messages(span, GenAI.GEN_AI_INPUT_MESSAGES)) == 1


def test_module_level_chat_is_instrumented(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_module_level_chat_is_instrumented.yaml"):
        ollama.chat(model=CHAT_MODEL, messages=MESSAGES)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"


def test_negative_num_predict_is_not_recorded_as_max_tokens(
    instrument_no_content, span_exporter, vcr
):
    # -1 means no limit and -2 means fill the context.
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES, options={"num_predict": -1}
        )

    (span,) = span_exporter.get_finished_spans()
    assert GenAI.GEN_AI_REQUEST_MAX_TOKENS not in span.attributes


@pytest.mark.parametrize(
    "messages, keep_alive, cassette",
    [
        ([], None, "test_chat_model_load_records_no_span.yaml"),
        ([], 0, "test_chat_model_unload_records_no_span.yaml"),
    ],
    ids=["load", "unload"],
)
def test_chat_without_messages_records_no_span(
    instrument_no_content, span_exporter, vcr, messages, keep_alive, cassette
):
    # Ollama only loads or unloads the model for a chat without messages.
    with vcr.use_cassette(cassette):
        response = ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=messages, keep_alive=keep_alive
        )

    assert response.done_reason in ("load", "unload")
    assert span_exporter.get_finished_spans() == ()


def test_chat_inside_another_inference_adds_no_span(
    instrument_no_content,
    span_exporter,
    tracer_provider,
    meter_provider,
    logger_provider,
    vcr,
):
    # A framework instrumentation (LangChain's ChatOllama, for example) that
    # already records the inference must not get a second chat span.
    outer = TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        with outer.inference("langchain", request_model=CHAT_MODEL):
            ollama.Client(host=HOST).chat(model=CHAT_MODEL, messages=MESSAGES)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "langchain"


def test_messages_without_content_are_not_recorded(
    instrument_with_content, span_exporter, vcr
):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "assistant", "content": ""},
        *MESSAGES,
    ]
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        ollama.Client(host=HOST).chat(model=CHAT_MODEL, messages=messages)

    (span,) = span_exporter.get_finished_spans()
    roles = [m["role"] for m in _messages(span, GenAI.GEN_AI_INPUT_MESSAGES)]
    assert roles == ["system", "user"]


def test_client_created_before_instrument_is_traced(
    tracer_provider, logger_provider, meter_provider, span_exporter, vcr
):
    client = ollama.Client(host=HOST)
    instrumentor = OllamaInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
            client.chat(model=CHAT_MODEL, messages=MESSAGES)
    finally:
        instrumentor.uninstrument()

    assert len(span_exporter.get_finished_spans()) == 1


def test_chat_with_message_objects(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        ollama.Client(host=HOST).chat(
            model=CHAT_MODEL,
            messages=(ollama.Message(role="user", content="Hi"),),
        )

    (span,) = span_exporter.get_finished_spans()
    assert _messages(span, GenAI.GEN_AI_INPUT_MESSAGES)[0]["parts"] == [
        {"type": "text", "content": "Hi"}
    ]


def test_chat_with_a_tool_generator_sends_every_tool(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_records_tool_calls.yaml"):
        response = ollama.Client(host=HOST).chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Use the tool.",
                }
            ],
            tools=(tool for tool in [WEATHER_TOOL]),
            options={"temperature": 0},
        )

    assert response.message.tool_calls
    (span,) = span_exporter.get_finished_spans()
    assert len(json.loads(span.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS])) == 1


def test_chat_stream_error_is_recorded(
    instrument_no_content, span_exporter, vcr
):
    # With stream=True, Ollama raises on the first read, not on the call.
    with vcr.use_cassette("test_chat_stream_error_is_recorded.yaml"):
        stream = ollama.Client(host=HOST).chat(
            model="no-such-model:1b", messages=MESSAGES, stream=True
        )
        with pytest.raises(ollama.ResponseError):
            list(stream)

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )


def test_abandoned_stream_still_ends_its_span(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette(
        "test_chat_stream_records_the_drained_response.yaml"
    ):
        stream = ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES, stream=True
        )
        next(iter(stream))
        del stream
        gc.collect()

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
