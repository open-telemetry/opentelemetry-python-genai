# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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
from opentelemetry.sdk.trace.sampling import (
    Decision,
    Sampler,
    SamplingResult,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
    _SafeCompletionHook,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    GenAIInvocation,
    InferenceInvocation,
    RealtimeInferenceInvocation,
    UserSpeechInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    ContentCapturingMode,
    Error,
    FunctionToolDefinition,
    InputMessage,
    Modality,
    OutputMessage,
    TextPart,
    TranscriptionPart,
)


def _make_handler(
    *,
    sampler: Sampler | None = None,
    completion_hook: CompletionHook | None = None,
    content_capturing_mode: ContentCapturingMode | None = None,
):
    span_exporter = InMemorySpanExporter()
    tracer_provider = (
        TracerProvider(sampler=sampler)
        if sampler is not None
        else TracerProvider()
    )
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
        completion_hook=completion_hook,
        content_capturing_mode=content_capturing_mode,
    )
    return handler, span_exporter, log_exporter, metric_reader


def _collect_metric_names(metric_reader: InMemoryMetricReader) -> set[str]:
    metrics_data = metric_reader.get_metrics_data()
    if metrics_data is None:
        return set()
    return {
        m.name
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }


def test_realtime_session_events_and_invocations():
    handler, span_exporter, log_exporter, metric_reader = _make_handler(
        content_capturing_mode=ContentCapturingMode.SPAN_AND_EVENT
    )

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
        assert isinstance(speech, UserSpeechInvocation)
        assert isinstance(speech, GenAIInvocation)
        speech.attributes["custom.vad"] = "manual"

    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        realtime_session_id="sess-1",
        server_address="generativelanguage.googleapis.com",
        server_port=443,
        conversation_id="conv-1",
    ) as inv:
        assert isinstance(inv, RealtimeInferenceInvocation)
        assert isinstance(inv, GenAIInvocation)
        assert not isinstance(inv, InferenceInvocation)
        inv.output_type = "speech"
        inv.max_tokens = 256
        inv.reasoning_level = "medium"
        inv.response_id = "resp-1"
        inv.response_model_name = (
            "gemini-2.5-flash-native-audio-preview-12-2025"
        )
        inv.finish_reasons = ["stop"]
        inv.input_tokens = 30
        inv.output_tokens = 45
        inv.thinking_tokens = 10
        inv.cache_read_input_tokens = 5
        inv.cache_write_input_tokens = 2
        inv.set_input_tokens([(Modality.AUDIO, 20), (Modality.TEXT, 10)])
        inv.set_output_tokens([(Modality.AUDIO, 35), (Modality.TEXT, 10)])
        inv.set_cache_read_input_tokens([(Modality.TEXT, 5)])
        inv.system_instruction = [TextPart(content="Be concise.")]
        inv.tool_definitions = [
            FunctionToolDefinition(
                name="get_weather",
                description="Fetch weather",
                parameters={"type": "object", "properties": {}},
            )
        ]
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
    assert inf_span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "conv-1"
    assert inf_span.attributes[GenAI.GEN_AI_OUTPUT_TYPE] == "speech"
    assert inf_span.attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 256
    assert inf_span.attributes["gen_ai.request.reasoning.level"] == "medium"
    assert inf_span.attributes[GenAI.GEN_AI_RESPONSE_ID] == "resp-1"
    assert (
        inf_span.attributes[GenAI.GEN_AI_RESPONSE_MODEL]
        == "gemini-2.5-flash-native-audio-preview-12-2025"
    )
    assert inf_span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert inf_span.attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 30
    assert inf_span.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 45
    assert (
        inf_span.attributes[GenAI.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] == 10
    )
    assert inf_span.attributes[GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 5
    assert inf_span.attributes["gen_ai.usage.cache_write.input_tokens"] == 2
    assert inf_span.attributes["gen_ai.usage.audio.input_tokens"] == 20
    assert inf_span.attributes["gen_ai.usage.text.input_tokens"] == 10
    assert inf_span.attributes["gen_ai.usage.audio.output_tokens"] == 35
    assert inf_span.attributes["gen_ai.usage.text.output_tokens"] == 10
    assert (
        inf_span.attributes["gen_ai.usage.text.cache_read.input_tokens"] == 5
    )
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
    assert inf_span.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        {"type": "text", "content": "Be concise."},
    )
    assert inf_span.attributes[GenAI.GEN_AI_TOOL_DEFINITIONS] == (
        {
            "type": "function",
            "name": "get_weather",
            "description": "Fetch weather",
            "parameters": {"type": "object", "properties": {}},
        },
    )

    # Realtime semconv defines no client metrics on realtime_inference or user_speech.
    assert _collect_metric_names(metric_reader) == set()

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


def test_realtime_session_id_passed_to_head_sampler_and_supports_late_update():
    seen_start_attrs: list[dict[str, object]] = []

    class _SessionIdRecordingSampler(Sampler):
        def should_sample(
            self,
            parent_context,
            trace_id,
            name,
            kind=None,
            attributes=None,
            links=None,
            trace_state=None,
        ):
            attrs = dict(attributes or {})
            seen_start_attrs.append(attrs)
            decision = (
                Decision.DROP
                if attrs.get("gen_ai.realtime_session.id") == "sess-drop"
                else Decision.RECORD_AND_SAMPLE
            )
            return SamplingResult(decision, attributes)

        def get_description(self) -> str:
            return "SessionIdRecordingSampler"

    handler, span_exporter, _, _ = _make_handler(
        sampler=_SessionIdRecordingSampler()
    )

    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash",
        realtime_session_id="sess-keep",
    ):
        pass

    with handler.user_speech(
        provider="gcp.gemini",
        realtime_session_id="sess-keep",
    ):
        pass

    # Head sampler can drop spans by gen_ai.realtime_session.id at start time.
    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash",
        realtime_session_id="sess-drop",
    ):
        pass

    with handler.user_speech(
        provider="gcp.gemini",
        realtime_session_id="sess-drop",
    ):
        pass

    # Assigning realtime_session_id after span start still sets it on the span.
    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash",
    ) as late_rt:
        late_rt.realtime_session_id = "sess-late-rt"

    with handler.user_speech(provider="gcp.gemini") as late_speech:
        late_speech.realtime_session_id = "sess-late-speech"

    assert [a.get("gen_ai.realtime_session.id") for a in seen_start_attrs] == [
        "sess-keep",
        "sess-keep",
        "sess-drop",
        "sess-drop",
        None,
        None,
    ]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    assert [s.attributes.get("gen_ai.realtime_session.id") for s in spans] == [
        "sess-keep",
        "sess-keep",
        "sess-late-rt",
        "sess-late-speech",
    ]


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
    handler, span_exporter, _, metric_reader = _make_handler()
    with handler.user_speech():
        pass

    (speech_span,) = span_exporter.get_finished_spans()
    assert speech_span.name == "user_speech"
    assert speech_span.kind == SpanKind.INTERNAL
    assert speech_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "user_speech"
    assert GenAI.GEN_AI_PROVIDER_NAME not in speech_span.attributes
    assert "gen_ai.realtime_session.id" not in speech_span.attributes
    assert error_attributes.ERROR_TYPE not in speech_span.attributes
    assert _collect_metric_names(metric_reader) == set()


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


def test_mark_end_is_scoped_to_realtime_inference_and_preserves_end_timestamp():
    clock_ns = 1_000_000_000

    with patch(
        "opentelemetry.util.genai._realtime_invocation.time.time_ns",
        side_effect=lambda: clock_ns,
    ):
        handler, span_exporter, _, metric_reader = _make_handler()
        inv = handler.realtime_inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash-native-audio-preview-12-2025",
        )

        # mark_end is only exposed on RealtimeInferenceInvocation.
        for factory in (
            lambda: handler.inference(
                "gcp.gemini", request_model="gemini-2.5-flash"
            ),
            lambda: handler.tool("get_weather"),
            lambda: handler.workflow("voice_pipeline"),
            lambda: handler.invoke_local_agent(agent_name="voice_agent"),
            lambda: handler.user_speech(provider="gcp.gemini"),
        ):
            with factory() as other:
                assert not hasattr(other, "mark_end")

        clock_ns = 1_250_000_000
        inv.mark_end()

        # Repeated mark_end calls keep the first observed timestamp.
        clock_ns = 2_000_000_000
        inv.mark_end()

        clock_ns = 6_000_000_000
        inv.stop()

    rt_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
        == "realtime_inference"
    ]
    assert len(rt_spans) == 1
    assert rt_spans[0].end_time == 1_250_000_000
    assert "gen_ai.client.operation.duration" in _collect_metric_names(
        metric_reader
    )


def test_realtime_inference_completion_hook_and_content_capture():
    hook = MagicMock(spec=CompletionHook)
    handler, span_exporter, _, _ = _make_handler(
        completion_hook=hook,
        content_capturing_mode=ContentCapturingMode.NO_CONTENT,
    )

    with handler.realtime_inference(
        "gcp.gemini",
        request_model="gemini-2.5-flash",
    ) as inv:
        assert inv.should_capture_content is True
        inv.input_messages = [
            InputMessage(role="user", parts=[TextPart(content="hi")])
        ]
        inv.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content="hello")],
                finish_reason="stop",
            )
        ]

    # user_speech never uses the completion hook.
    with handler.user_speech(provider="gcp.gemini") as speech:
        assert speech.should_capture_content is False

    hook.on_completion.assert_called_once()
    kwargs = hook.on_completion.call_args.kwargs
    assert kwargs["log_record"] is None
    assert len(kwargs["inputs"]) == 1
    assert len(kwargs["outputs"]) == 1

    rt_span = span_exporter.get_finished_spans()[0]
    assert GenAI.GEN_AI_INPUT_MESSAGES not in rt_span.attributes
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in rt_span.attributes


def test_realtime_inference_is_independent_of_inference_context():
    handler, span_exporter, log_exporter, _ = _make_handler(
        content_capturing_mode=ContentCapturingMode.SPAN_ONLY
    )

    with handler.inference("openai", request_model="gpt-4o") as outer:
        outer.output_tokens = 10
        with handler.realtime_inference(
            "gcp.gemini",
            request_model="gemini-2.5-flash",
            realtime_session_id="sess-independent",
        ) as rt:
            assert isinstance(rt, RealtimeInferenceInvocation)
            rt.output_tokens = 25

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    rt_span, inf_span = spans
    assert (
        rt_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "realtime_inference"
    )
    assert rt_span.attributes["gen_ai.realtime_session.id"] == (
        "sess-independent"
    )
    assert rt_span.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 25
    assert inf_span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert "gen_ai.realtime_session.id" not in inf_span.attributes
    assert inf_span.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 10

    # Only the regular inference emits gen_ai.client.inference.operation.details.
    detail_events = [
        log.log_record
        for log in log_exporter.get_finished_logs()
        if log.log_record.event_name
        == "gen_ai.client.inference.operation.details"
    ]
    assert len(detail_events) == 1


def test_user_speech_failure_default_and_custom_resolver():
    handler, span_exporter, log_exporter, metric_reader = _make_handler()

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
    assert _collect_metric_names(metric_reader) == set()


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

    # Realtime inference emits neither inference details nor inference exception events or metrics.
    assert log_exporter.get_finished_logs() == ()
    assert _collect_metric_names(metric_reader) == set()


@pytest.mark.parametrize(
    "mode",
    ["NO_CONTENT", "SPAN_ONLY", "EVENT_ONLY", "SPAN_AND_EVENT"],
)
@pytest.mark.parametrize("hook_kind", ["none", "active", "safe_noop"])
def test_realtime_invocation_should_capture_content_policy(
    mode: str, hook_kind: str
):
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
        handler, _, _, _ = _make_handler(completion_hook=hook)
        expected = mode in ("SPAN_ONLY", "SPAN_AND_EVENT") or (
            hook_kind == "active"
        )
        with handler.realtime_inference("gcp.gemini") as inv:
            assert inv.should_capture_content is expected
        with handler.user_speech(
            provider="gcp.gemini", realtime_session_id="sess-speech"
        ) as speech:
            assert speech.should_capture_content is False
