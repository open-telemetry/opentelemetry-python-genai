# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming telemetry for the callback-based LangChain instrumentation.

The provider's stream methods are patched rather than replayed from a cassette:
these tests assert on chunk *timing*, which a cassette does not reproduce, and
the LangChain streaming machinery under test (``on_llm_new_token`` dispatch) is
the same either way.
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_openai import ChatOpenAI

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

_CHUNKS = ("Paris", " is", " the", " capital")


def _fake_stream(self, messages, stop=None, run_manager=None, **kwargs):
    for token in _CHUNKS:
        yield ChatGenerationChunk(message=AIMessageChunk(content=token))


async def _fake_astream(self, messages, stop=None, run_manager=None, **kwargs):
    for token in _CHUNKS:
        yield ChatGenerationChunk(message=AIMessageChunk(content=token))


def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content="".join(_CHUNKS)),
                generation_info={"finish_reason": "stop"},
            )
        ]
    )


def _metrics_by_name(metric_reader):
    resource_metrics = metric_reader.get_metrics_data().resource_metrics
    return {
        metric.name: metric
        for metric in resource_metrics[0].scope_metrics[0].metrics
    }


def _assert_stream_span(span):
    assert span.name == "chat gpt-3.5-turbo"
    assert span.attributes[gen_ai_attributes.GEN_AI_REQUEST_STREAM] is True

    ttfc = span.attributes[
        gen_ai_attributes.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK
    ]
    assert isinstance(ttfc, float)
    assert ttfc >= 0.0


def _assert_stream_metrics(metric_reader):
    metrics = _metrics_by_name(metric_reader)

    ttfc = metrics[gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK]
    (ttfc_point,) = ttfc.data.data_points
    assert ttfc_point.count == 1

    per_chunk = metrics[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK
    ]
    (per_chunk_point,) = per_chunk.data.data_points
    # One gap per chunk after the first. LangChain follows the provider's last
    # chunk with a contentless end-of-stream marker, which must not be counted.
    assert per_chunk_point.count == len(_CHUNKS) - 1

    for point in (ttfc_point, per_chunk_point):
        assert (
            point.attributes[gen_ai_attributes.GEN_AI_REQUEST_MODEL]
            == "gpt-3.5-turbo"
        )
        assert (
            point.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
            == gen_ai_attributes.GenAiOperationNameValues.CHAT.value
        )


def test_stream_records_stream_attributes_and_timing_metrics(
    span_exporter,
    metric_reader,
    start_instrumentation,
    chat_openai_gpt_3_5_turbo_model,
    monkeypatch,
):
    monkeypatch.setattr(ChatOpenAI, "_stream", _fake_stream)

    chunks = list(
        chat_openai_gpt_3_5_turbo_model.stream(
            [HumanMessage(content="What is the capital of France?")]
        )
    )
    assert "".join(chunk.content for chunk in chunks) == "".join(_CHUNKS)

    (span,) = span_exporter.get_finished_spans()
    _assert_stream_span(span)
    _assert_stream_metrics(metric_reader)


def test_astream_records_stream_attributes_and_timing_metrics(
    span_exporter,
    metric_reader,
    start_instrumentation,
    chat_openai_gpt_3_5_turbo_model,
    monkeypatch,
):
    monkeypatch.setattr(ChatOpenAI, "_astream", _fake_astream)

    async def drain():
        return [
            chunk
            async for chunk in chat_openai_gpt_3_5_turbo_model.astream(
                [HumanMessage(content="What is the capital of France?")]
            )
        ]

    chunks = asyncio.run(drain())
    assert "".join(chunk.content for chunk in chunks) == "".join(_CHUNKS)

    (span,) = span_exporter.get_finished_spans()
    _assert_stream_span(span)
    _assert_stream_metrics(metric_reader)


def test_non_streamed_call_omits_stream_telemetry(
    span_exporter,
    metric_reader,
    start_instrumentation,
    chat_openai_gpt_3_5_turbo_model,
    monkeypatch,
):
    monkeypatch.setattr(ChatOpenAI, "_generate", _fake_generate)

    chat_openai_gpt_3_5_turbo_model.invoke(
        [HumanMessage(content="What is the capital of France?")]
    )

    (span,) = span_exporter.get_finished_spans()
    assert gen_ai_attributes.GEN_AI_REQUEST_STREAM not in span.attributes
    assert (
        gen_ai_attributes.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK
        not in span.attributes
    )

    metrics = _metrics_by_name(metric_reader)
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK
        not in metrics
    )
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK
        not in metrics
    )


@pytest.mark.parametrize(
    "chunk_position,expected_stream_attribute",
    [("last", False), (None, True)],
)
def test_end_of_stream_marker_alone_does_not_mark_streaming(
    span_exporter,
    start_instrumentation,
    chat_openai_gpt_3_5_turbo_model,
    monkeypatch,
    chunk_position,
    expected_stream_attribute,
):
    """A stream whose only callback is the contentless marker is not streamed.

    Guards the marker check against being reduced to "empty token", which
    would also drop real content-free chunks (e.g. tool-call argument deltas).
    """

    def one_chunk_stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ):
        message = AIMessageChunk(content="")
        if chunk_position is not None:
            message.chunk_position = chunk_position
        yield ChatGenerationChunk(message=message)

    monkeypatch.setattr(ChatOpenAI, "_stream", one_chunk_stream)

    list(
        chat_openai_gpt_3_5_turbo_model.stream(
            [HumanMessage(content="What is the capital of France?")]
        )
    )

    (span,) = span_exporter.get_finished_spans()
    assert (
        gen_ai_attributes.GEN_AI_REQUEST_STREAM in span.attributes
    ) is expected_stream_attribute
