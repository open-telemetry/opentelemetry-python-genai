# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from opentelemetry._logs import SeverityNumber
from opentelemetry.context import get_value
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
    OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    CLIENT_INFERENCE_CONTEXT_KEY,
    InferenceData,
    RealtimeInferenceInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    Error,
    InputMessage,
    Modality,
    OutputMessage,
    TextPart,
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
        {
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_AND_EVENT",
            OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true",
        },
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
    assert isinstance(speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME], str)
    assert speech_span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"
    assert isinstance(speech_span.attributes[GenAI.GEN_AI_PROVIDER_NAME], str)
    assert speech_span.attributes["gen_ai.realtime_session.id"] == "sess-1"
    assert isinstance(
        speech_span.attributes["gen_ai.realtime_session.id"], str
    )
    assert speech_span.attributes["custom.vad"] == "manual"

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
    assert isinstance(inf_span.attributes[GenAI.GEN_AI_OPERATION_NAME], str)
    assert inf_span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"
    assert inf_span.attributes["gen_ai.realtime_session.id"] == "sess-1"
    assert isinstance(inf_span.attributes["gen_ai.realtime_session.id"], str)
    assert (
        inf_span.attributes[server_attributes.SERVER_ADDRESS]
        == "generativelanguage.googleapis.com"
    )
    assert inf_span.attributes[server_attributes.SERVER_PORT] == 443
    assert isinstance(inf_span.attributes[server_attributes.SERVER_PORT], int)

    in_msgs = inf_span.attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert isinstance(in_msgs, tuple)
    assert in_msgs == (
        {
            "role": "user",
            "parts": (
                {
                    "type": "blob",
                    "mime_type": "audio/pcm;rate=16000",
                    "modality": "audio",
                    "content": b"\x00\x01",
                },
                {
                    "type": "transcription",
                    "content": "Hello model",
                },
            ),
        },
    )
    assert isinstance(in_msgs[0]["parts"][0]["content"], bytes)

    out_msgs = inf_span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert isinstance(out_msgs, tuple)
    assert out_msgs == (
        {
            "role": "assistant",
            "parts": (
                {
                    "type": "blob",
                    "mime_type": "audio/pcm;rate=24000",
                    "modality": "audio",
                    "content": b"\x02\x03",
                },
                {
                    "type": "transcription",
                    "content": "Hello human",
                },
            ),
            "finish_reason": "stop",
        },
    )
    assert isinstance(out_msgs[0]["parts"][0]["content"], bytes)

    logs = log_exporter.get_finished_logs()
    event_names = [l.log_record.event_name for l in logs]
    assert event_names == [
        "gen_ai.client.realtime_session.started",
        "gen_ai.client.realtime_session.ended",
    ]
    started = logs[0].log_record
    assert started.severity_number == SeverityNumber.INFO
    assert started.body is None
    assert started.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "gcp.gemini"
    assert started.attributes["gen_ai.realtime_session.id"] == "sess-1"
    assert (
        started.attributes[server_attributes.SERVER_ADDRESS]
        == "generativelanguage.googleapis.com"
    )
    assert started.attributes[server_attributes.SERVER_PORT] == 443
    assert isinstance(started.attributes[server_attributes.SERVER_PORT], int)

    ended = logs[1].log_record
    assert ended.severity_number == SeverityNumber.WARN
    assert ended.body is None
    assert ended.attributes[error_attributes.ERROR_TYPE] == "ConnectionError"
    assert isinstance(ended.attributes[error_attributes.ERROR_TYPE], str)
    assert ended.attributes["gen_ai.realtime_session.id"] == "sess-1"


def test_realtime_session_ended_normal_and_custom_error_resolver():
    handler, _, log_exporter, _ = _make_handler()

    handler.emit_realtime_session_ended(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        realtime_session_id="sess-ok",
    )
    handler.emit_realtime_session_ended(
        "gcp.gemini",
        realtime_session_id="sess-custom",
        error=TimeoutError("idle timeout"),
        error_type_resolver=lambda exc: f"custom.{type(exc).__name__}",
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    normal = logs[0].log_record
    assert normal.event_name == "gen_ai.client.realtime_session.ended"
    assert normal.severity_number == SeverityNumber.INFO
    assert error_attributes.ERROR_TYPE not in normal.attributes
    assert normal.attributes["gen_ai.realtime_session.id"] == "sess-ok"

    failed = logs[1].log_record
    assert failed.event_name == "gen_ai.client.realtime_session.ended"
    assert failed.severity_number == SeverityNumber.WARN
    assert (
        failed.attributes[error_attributes.ERROR_TYPE] == "custom.TimeoutError"
    )
    assert isinstance(failed.attributes[error_attributes.ERROR_TYPE], str)


def test_user_speech_omits_optional_attributes_when_unset():
    with patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true"},
    ):
        handler, span_exporter, _, _ = _make_handler()
        with handler.user_speech():
            pass

    (speech_span,) = span_exporter.get_finished_spans()
    assert speech_span.name == "user_speech"
    assert speech_span.kind == SpanKind.INTERNAL
    assert speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"
    assert GenAI.GEN_AI_PROVIDER_NAME not in speech_span.attributes
    assert "gen_ai.realtime_session.id" not in speech_span.attributes
    assert error_attributes.ERROR_TYPE not in speech_span.attributes


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


def test_suppressed_realtime_inference_preserves_event_only_capture_policy():
    with patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "EVENT_ONLY"},
    ):
        handler, _, log_exporter, _ = _make_handler()

        # Default outer capture_content=False keeps suppressed realtime capture off.
        with handler.inference("openai", request_model="gpt-4o"):
            with handler.realtime_inference(
                "openai", request_model="gpt-4o-realtime"
            ) as inner:
                assert inner.should_capture_content is False

        # When outer permits content capture, suppressed realtime honors EVENT_ONLY
        # so messages reach the outer inference's operation-details event.
        with handler.inference("openai", request_model="gpt-4o"):
            ctx_data = get_value(CLIENT_INFERENCE_CONTEXT_KEY)
            assert isinstance(ctx_data, InferenceData)
            ctx_data.capture_content = True
            with handler.realtime_inference(
                "openai", request_model="gpt-4o-realtime"
            ) as inner:
                assert inner.should_capture_content is True
                inner.input_messages = [
                    InputMessage(role="user", parts=[TextPart(content="hi")])
                ]
                inner.output_messages = [
                    OutputMessage(
                        role="assistant",
                        parts=[TextPart(content="hello")],
                        finish_reason="stop",
                    )
                ]

    detail_events = [
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name
        == "gen_ai.client.inference.operation.details"
    ]
    assert len(detail_events) == 2
    assert GenAI.GEN_AI_INPUT_MESSAGES not in detail_events[0].attributes
    assert detail_events[1].attributes[GenAI.GEN_AI_INPUT_MESSAGES] == (
        {"role": "user", "parts": ({"type": "text", "content": "hi"},)},
    )
    assert detail_events[1].attributes[GenAI.GEN_AI_OUTPUT_MESSAGES] == (
        {
            "role": "assistant",
            "parts": ({"type": "text", "content": "hello"},),
            "finish_reason": "stop",
        },
    )


def test_no_realtime_session_id_reverse_leakage_on_inference():
    handler, span_exporter, _, _ = _make_handler()
    assert not hasattr(InferenceData(), "realtime_session_id")
    assert not hasattr(InferenceData(), "_realtime_session_id")
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


def test_nested_realtime_inference_preserves_outer_session_id_and_inner_inference():
    handler, span_exporter, log_exporter, _ = _make_handler()

    # Outer realtime session ID takes precedence over inner realtime session ID.
    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        realtime_session_id="sess-outer",
    ) as outer:
        with handler.realtime_inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-inner",
        ) as inner_rt:
            inner_rt.input_tokens = 5

    assert outer.realtime_session_id == "sess-outer"

    # Regular InferenceInvocation nested inside RealtimeInferenceInvocation
    # merges cleanly into _RealtimeInferenceData without AttributeError.
    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        realtime_session_id="sess-outer-2",
    ) as outer2:
        with handler.inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        ) as inner_inf:
            inner_inf.output_tokens = 23

    assert outer2.realtime_session_id == "sess-outer-2"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].attributes["gen_ai.realtime_session.id"] == "sess-outer"
    assert spans[0].attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert spans[1].attributes["gen_ai.realtime_session.id"] == "sess-outer-2"
    assert spans[1].attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 23
    assert log_exporter.get_finished_logs() == ()


def test_user_speech_failure_default_and_custom_resolver():
    with patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "true"},
    ):
        handler, span_exporter, log_exporter, _ = _make_handler()

        # Default error_type resolution via context manager exception path.
        with pytest.raises(RuntimeError, match="mic lost"):
            with handler.user_speech(
                provider="gcp.gemini",
                realtime_session_id="sess-default",
            ):
                raise RuntimeError("mic lost")

        # Custom error_type_resolver + explicit fail() followed by repeated finalize calls.
        speech = handler.user_speech(
            provider="gcp.gemini",
            realtime_session_id="sess-custom",
            error_type_resolver=lambda exc: f"custom.{type(exc).__name__}",
        )
        speech.fail(RuntimeError("mic disconnected"))
        speech.stop()
        speech.fail(Error(message="ignored second fail", type="IgnoredError"))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    default_span, custom_span = spans
    assert default_span.status.status_code == StatusCode.ERROR
    assert (
        default_span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"
    )
    assert isinstance(
        default_span.attributes[error_attributes.ERROR_TYPE], str
    )
    assert (
        default_span.attributes["gen_ai.realtime_session.id"] == "sess-default"
    )
    assert isinstance(
        default_span.attributes["gen_ai.realtime_session.id"], str
    )

    assert custom_span.status.status_code == StatusCode.ERROR
    assert (
        custom_span.attributes[error_attributes.ERROR_TYPE]
        == "custom.RuntimeError"
    )
    assert isinstance(custom_span.attributes[error_attributes.ERROR_TYPE], str)
    assert (
        custom_span.attributes["gen_ai.realtime_session.id"] == "sess-custom"
    )
    assert isinstance(
        custom_span.attributes["gen_ai.realtime_session.id"], str
    )
    assert log_exporter.get_finished_logs() == ()


def test_realtime_inference_failure_default_and_custom_resolver():
    handler, span_exporter, log_exporter, metric_reader = _make_handler()

    # Default error_type resolution via context manager exception path.
    with pytest.raises(ValueError, match="bad frame"):
        with handler.realtime_inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
            realtime_session_id="sess-default",
        ):
            raise ValueError("bad frame")

    # Custom error_type_resolver + explicit fail() followed by repeated finalize calls.
    inv = handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        realtime_session_id="sess-custom",
        error_type_resolver=lambda exc: f"custom.{type(exc).__name__}",
    )
    inv.fail(ValueError("stream interrupted"))
    inv.stop()
    inv.fail(Error(message="ignored second fail", type="IgnoredError"))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    default_span, custom_span = spans
    assert default_span.status.status_code == StatusCode.ERROR
    assert default_span.attributes[error_attributes.ERROR_TYPE] == "ValueError"
    assert isinstance(
        default_span.attributes[error_attributes.ERROR_TYPE], str
    )
    assert (
        default_span.attributes["gen_ai.realtime_session.id"] == "sess-default"
    )
    assert isinstance(
        default_span.attributes["gen_ai.realtime_session.id"], str
    )

    assert custom_span.status.status_code == StatusCode.ERROR
    assert (
        custom_span.attributes[error_attributes.ERROR_TYPE]
        == "custom.ValueError"
    )
    assert isinstance(custom_span.attributes[error_attributes.ERROR_TYPE], str)
    assert (
        custom_span.attributes["gen_ai.realtime_session.id"] == "sess-custom"
    )
    assert isinstance(
        custom_span.attributes["gen_ai.realtime_session.id"], str
    )

    event_names = [
        l.log_record.event_name for l in log_exporter.get_finished_logs()
    ]
    assert "gen_ai.client.inference.operation.details" not in event_names

    metrics_data = metric_reader.get_metrics_data()
    duration_dps = [
        dp
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "gen_ai.client.operation.duration"
        for dp in m.data.data_points
    ]
    recorded_error_types = {
        dp.attributes[error_attributes.ERROR_TYPE] for dp in duration_dps
    }
    assert recorded_error_types == {"ValueError", "custom.ValueError"}


@pytest.mark.parametrize(
    "mode",
    ["NO_CONTENT", "SPAN_ONLY", "EVENT_ONLY", "SPAN_AND_EVENT"],
)
@pytest.mark.parametrize("hook_kind", ["none", "active", "safe_noop"])
def test_should_capture_realtime_content_matches_invocation_policy(
    mode: str, hook_kind: str
):
    from opentelemetry.context import get_current
    from opentelemetry.util.genai.completion_hook import (
        CompletionHook,
        _NoOpCompletionHook,
        _SafeCompletionHook,
    )

    class _RecordingHook(CompletionHook):
        def on_completion(self, **kwargs):
            return None

    hook: CompletionHook | None = (
        None
        if hook_kind == "none"
        else _RecordingHook()
        if hook_kind == "active"
        else _SafeCompletionHook(_NoOpCompletionHook())
    )
    with patch.dict(
        os.environ,
        {OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: mode},
    ):
        tracer_provider = TracerProvider()
        logger_provider = LoggerProvider()
        meter_provider = MeterProvider()
        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            completion_hook=hook,
        )

        # Unsuppressed realtime inference: pre-invocation query matches invocation.
        pre_unsuppressed = handler.should_capture_realtime_content()
        with handler.realtime_inference("gcp.gemini") as inv:
            assert pre_unsuppressed is inv.should_capture_content

        # Suppressed realtime inference across all (capture_content, upload_completion) flags.
        for outer_capture in (False, True):
            for outer_upload in (False, True):
                with handler.inference("gcp.gemini", request_model="outer"):
                    ctx_data = get_value(CLIENT_INFERENCE_CONTEXT_KEY)
                    assert isinstance(ctx_data, InferenceData)
                    ctx_data.capture_content = outer_capture
                    ctx_data.upload_completion = outer_upload
                    ctx = get_current()
                    pre_suppressed = handler.should_capture_realtime_content()
                    pre_suppressed_explicit = (
                        handler.should_capture_realtime_content(context=ctx)
                    )
                    with handler.realtime_inference("gcp.gemini") as inner:
                        assert pre_suppressed is inner.should_capture_content
                        assert (
                            pre_suppressed_explicit
                            is inner.should_capture_content
                        )


def test_user_speech_disabled_by_default_and_env_parsing(caplog, monkeypatch):
    from opentelemetry.trace import get_current_span
    from opentelemetry.util.genai.utils import get_emit_user_speech

    monkeypatch.delenv(
        OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH, raising=False
    )
    assert get_emit_user_speech() is False
    handler, span_exporter, _, _ = _make_handler()
    assert handler.should_emit_user_speech() is False

    with handler._tracer.start_as_current_span("caller") as caller_span:
        with handler.user_speech(
            provider="gcp.gemini", realtime_session_id="sess-noop"
        ) as speech:
            speech.attributes["custom"] = "ignored"
            assert get_current_span() is caller_span

        with pytest.raises(RuntimeError, match="boom"):
            with handler.user_speech(
                provider="gcp.gemini", realtime_session_id="sess-noop"
            ):
                raise RuntimeError("boom")

        assert caller_span.is_recording()
        assert get_current_span() is caller_span

    # Only the caller span was exported, and it was neither ended early nor
    # mutated with user_speech attributes or error status.
    (exported_caller,) = span_exporter.get_finished_spans()
    assert exported_caller.name == "caller"
    assert exported_caller.status.status_code == StatusCode.UNSET
    assert "gen_ai.operation.name" not in exported_caller.attributes
    assert "gen_ai.realtime_session.id" not in exported_caller.attributes
    assert error_attributes.ERROR_TYPE not in exported_caller.attributes

    for val, expected in (
        ("", False),
        ("false", False),
        ("FALSE", False),
        ("true", True),
        ("TRUE", True),
        (" True ", True),
    ):
        with patch.dict(
            os.environ, {OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: val}
        ):
            assert get_emit_user_speech() is expected
            h, _, _, _ = _make_handler()
            assert h.should_emit_user_speech() is expected

    caplog.clear()
    with patch.dict(
        os.environ, {OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH: "invalid"}
    ):
        assert get_emit_user_speech() is False
    assert OTEL_INSTRUMENTATION_GENAI_EMIT_USER_SPEECH in caplog.text
