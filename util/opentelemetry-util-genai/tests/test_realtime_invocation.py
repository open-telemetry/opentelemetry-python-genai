# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from unittest.mock import patch

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
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import RealtimeInferenceInvocation
from opentelemetry.util.genai.types import (
    BlobPart,
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
    in_msgs = json.loads(inf_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert in_msgs[0]["parts"][1] == {
        "type": "transcription",
        "content": "Hello model",
    }
    out_msgs = json.loads(inf_span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
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
    import time

    handler, span_exporter, _, metric_reader = _make_handler()
    inv = handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
    )
    inv.mark_end()
    marked_end_ns = inv._end_time_ns
    assert marked_end_ns is not None
    time.sleep(0.05)
    inv.stop()

    (span,) = span_exporter.get_finished_spans()
    assert span.end_time == marked_end_ns

    metrics_data = metric_reader.get_metrics_data()
    duration_dp = [
        dp
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "gen_ai.client.operation.duration"
        for dp in m.data.data_points
    ]
    assert len(duration_dp) == 1
    assert duration_dp[0].sum < 0.04
