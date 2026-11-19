# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceData,
    RealtimeInferenceInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    Error,
    InputMessage,
    Modality,
    OutputMessage,
    TranscriptionPart,
)


def _make_handler():
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    return handler, span_exporter, log_exporter, metric_reader


def test_realtime_session_events_and_invocations():
    with patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_AND_EVENT"},
    ):
        handler, span_exporter, log_exporter, _ = _make_handler()

        handler.emit_realtime_session_started(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-1",
            server_address="generativelanguage.googleapis.com",
            server_port=443,
        )

        with handler.user_speech(
            provider="gcp.gemini",
            realtime_session_id="sess-1",
        ) as speech:
            speech.attributes["custom.vad"] = "manual"

        with handler.realtime_inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-1",
            server_address="generativelanguage.googleapis.com",
            server_port=443,
        ) as inv:
            assert isinstance(inv, RealtimeInferenceInvocation)
            inv.output_type = "speech"
            inv.finish_reasons = ["stop"]
            inv.input_messages = [
                InputMessage(
                    role="user",
                    parts=[
                        BlobPart(
                            mime_type="audio/pcm;rate=16000",
                            modality=Modality.AUDIO,
                            content=b"\x00\x01",
                        ),
                        TranscriptionPart(content="Hello model"),
                    ],
                )
            ]
            inv.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[
                        BlobPart(
                            mime_type="audio/pcm;rate=24000",
                            modality=Modality.AUDIO,
                            content=b"\x02\x03",
                        ),
                        TranscriptionPart(content="Hello human"),
                    ],
                    finish_reason="stop",
                )
            ]

        handler.emit_realtime_session_ended(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-1",
            server_address="generativelanguage.googleapis.com",
            server_port=443,
            error=ConnectionError("closed"),
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    speech_span = spans[0]
    assert speech_span.name == "user_speech"
    assert speech_span.kind == SpanKind.INTERNAL
    assert speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"
    assert speech_span.attributes["gen_ai.realtime_session.id"] == "sess-1"

    inf_span = spans[1]
    assert (
        inf_span.name
        == "realtime_inference gemini-2.5-flash-native-audio-preview-12-2025"
    )
    assert inf_span.kind == SpanKind.CLIENT
    assert (
        inf_span.attributes[GenAI.GEN_AI_OPERATION_NAME]
        == "realtime_inference"
    )
    assert inf_span.attributes["gen_ai.realtime_session.id"] == "sess-1"
    in_msgs = inf_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert in_msgs[0]["parts"][1] == {
        "type": "transcription",
        "content": "Hello model",
    }
    out_msgs = inf_span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert out_msgs[0]["parts"][1] == {
        "type": "transcription",
        "content": "Hello human",
    }

    logs = log_exporter.get_finished_logs()
    event_names = [l.log_record.event_name for l in logs]
    assert event_names == [
        "gen_ai.client.realtime_session.started",
        "gen_ai.client.realtime_session.ended",
    ]
    started = logs[0].log_record
    assert started.severity_number == SeverityNumber.INFO
    assert started.body is None
    assert (
        started.attributes[server_attributes.SERVER_ADDRESS]
        == "generativelanguage.googleapis.com"
    )

    ended = logs[1].log_record
    assert ended.severity_number == SeverityNumber.WARN
    assert ended.body is None
    assert ended.attributes[error_attributes.ERROR_TYPE] == "ConnectionError"


def test_realtime_session_events_respect_logger_enabled():
    handler, _, log_exporter, _ = _make_handler()
    with patch.object(handler._logger, "enabled", return_value=False):
        handler.emit_realtime_session_started(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-disabled",
        )
        handler.emit_realtime_session_ended(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-disabled",
            error=ConnectionError("closed"),
        )
    assert log_exporter.get_finished_logs() == ()


def test_mark_end_preserves_observed_end_timestamp():
    clock_s = 100.0
    clock_ns = 1_000_000_000

    with (
        patch(
            "opentelemetry.util.genai._invocation.timeit.default_timer",
            side_effect=lambda: clock_s,
        ),
        patch(
            "opentelemetry.util.genai._invocation.time.time_ns",
            side_effect=lambda: clock_ns,
        ),
    ):
        handler, span_exporter, _, metric_reader = _make_handler()
        inv = handler.realtime_inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        )
        tool_inv = handler.tool("get_weather")
        wf_inv = handler.workflow("voice_pipeline")
        agent_inv = handler.invoke_local_agent(agent_name="voice_agent")

        clock_s = 100.25
        clock_ns = 1_250_000_000
        for item in (inv, tool_inv, wf_inv, agent_inv):
            item.mark_end()

        clock_s = 105.0
        clock_ns = 6_000_000_000
        for item in (agent_inv, wf_inv, tool_inv, inv):
            item.stop()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    for span in spans:
        assert span.end_time == 1_250_000_000

    metrics_data = metric_reader.get_metrics_data()
    for metric_name in (
        "gen_ai.client.operation.duration",
        "gen_ai.execute_tool.duration",
        "gen_ai.invoke_workflow.duration",
        "gen_ai.invoke_agent.duration",
    ):
        duration_dp = [
            dp
            for rm in metrics_data.resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
            if m.name == metric_name
            for dp in m.data.data_points
        ]
        assert len(duration_dp) == 1
        assert duration_dp[0].sum == pytest.approx(0.25)


def test_realtime_inference_should_capture_content_event_only():
    with patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "EVENT_ONLY"},
    ):
        handler, _, _, _ = _make_handler()
        with handler.realtime_inference("gcp.gemini") as inv:
            assert inv.should_capture_content is False


def test_no_realtime_session_id_reverse_leakage_on_inference():
    handler, span_exporter, _, _ = _make_handler()
    assert not hasattr(InferenceData(), "realtime_session_id")
    with handler.inference("openai", request_model="gpt-4o") as inv:
        assert not hasattr(inv, "realtime_session_id")
        with handler.realtime_inference(
            "openai",
            request_model="gpt-4o-realtime",
            realtime_session_id="sess-inner",
        ) as inner:
            inner.output_tokens = 12

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "gen_ai.realtime_session.id" not in spans[0].attributes
    assert spans[0].attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 12


def test_nested_realtime_inference_suppression_and_session_id_propagation():
    handler, span_exporter, log_exporter, _ = _make_handler()
    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
    ) as outer:
        with handler.realtime_inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-from-inner",
        ) as inner:
            inner.input_tokens = 7
            inner.output_tokens = 19

    assert outer.realtime_session_id == "sess-from-inner"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes["gen_ai.realtime_session.id"] == "sess-from-inner"
    )
    assert spans[0].attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 7
    assert spans[0].attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 19
    assert log_exporter.get_finished_logs() == ()


def test_realtime_inference_and_user_speech_failure_handling():
    handler, span_exporter, log_exporter, _ = _make_handler()

    speech = handler.user_speech(
        provider="gcp.gemini",
        realtime_session_id="sess-err",
        error_type_resolver=lambda exc: f"custom.{type(exc).__name__}",
    )
    speech.fail(RuntimeError("mic disconnected"))
    speech.stop()

    inv = handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        realtime_session_id="sess-err",
        error_type_resolver=lambda exc: f"custom.{type(exc).__name__}",
    )
    inv.fail(ValueError("stream interrupted"))
    inv.fail(Error(message="ignored second fail", type="IgnoredError"))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    speech_span, inf_span = spans
    assert speech_span.status.status_code == StatusCode.ERROR
    assert (
        speech_span.attributes[error_attributes.ERROR_TYPE]
        == "custom.RuntimeError"
    )
    assert speech_span.attributes["gen_ai.realtime_session.id"] == "sess-err"

    assert inf_span.status.status_code == StatusCode.ERROR
    assert (
        inf_span.attributes[error_attributes.ERROR_TYPE] == "custom.ValueError"
    )
    assert inf_span.attributes["gen_ai.realtime_session.id"] == "sess-err"

    event_names = [
        l.log_record.event_name for l in log_exporter.get_finished_logs()
    ]
    assert "gen_ai.client.inference.operation.details" not in event_names
