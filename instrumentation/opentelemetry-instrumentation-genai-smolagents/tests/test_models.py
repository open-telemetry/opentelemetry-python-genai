# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Model (``chat``) instrumentation tests for the in-process runtimes:
provider resolution, request-parameter mapping, message conversion, streaming,
and errors.

The three instrumented classes run inference in the current process, so there is
no HTTP traffic to record; each test drives smolagents' real ``generate`` over
the stubbed runtime pieces built in ``test_utils``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Generator
from typing import Any

import pytest
from smolagents.models import ChatMessage, MessageRole

from opentelemetry.context import Context
from opentelemetry.instrumentation.genai.smolagents import (
    patch as patch_module,
)
from opentelemetry.instrumentation.genai.smolagents._messages import (
    to_input_messages,
    to_output_message,
    to_tool_definitions,
)
from opentelemetry.instrumentation.genai.smolagents.patch import (
    _merged_request_kwargs,
    _output_type,
)
from opentelemetry.instrumentation.genai.smolagents.provider import (
    resolve_provider,
)
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.types import Blob, Text, Uri

from .test_utils import (
    MESSAGES,
    GetWeatherTool,
    attr,
    data_point_attributes,
    metrics_by_name,
    mlx_model,
    parse_messages,
    spans_by_operation,
    transformers_model,
    vllm_model,
)

IMAGE_URL = (
    "https://fastly.picsum.photos/id/237/200/300.jpg"
    "?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
)


def test_transformers_generate_records_the_response(
    instrument_with_content, span_exporter, metric_reader
) -> None:
    model = transformers_model(
        prompt_ids=[1, 2, 3], generated_ids=[4, 5], text="In Paris"
    )

    output = model.generate(messages=MESSAGES)
    assert output.content == "In Paris"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.name == "chat HuggingFaceTB/SmolLM2-135M-Instruct"
    assert span.status.status_code == StatusCode.UNSET
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "huggingface"
    assert (
        attr(span, GenAI.GEN_AI_REQUEST_MODEL)
        == "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == 2
    # A runtime in this process returns no response envelope and listens on no
    # socket, so none of these have a value to report.
    assert attr(span, GenAI.GEN_AI_RESPONSE_ID) is None
    assert attr(span, GenAI.GEN_AI_RESPONSE_MODEL) is None
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None
    assert attr(span, server_attributes.SERVER_ADDRESS) is None
    assert attr(span, server_attributes.SERVER_PORT) is None

    inputs = parse_messages(span, GenAI.GEN_AI_INPUT_MESSAGES)
    assert inputs[0]["role"] == "user"
    assert inputs[0]["parts"] == [
        {"type": "text", "content": "Where is the Louvre?"}
    ]

    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["role"] == "assistant"
    assert outputs[0]["parts"] == [{"type": "text", "content": "In Paris"}]
    assert outputs[0]["finish_reason"] == ""

    metrics = metrics_by_name(metric_reader)
    duration = metrics[gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION]
    assert duration.unit == "s"
    assert data_point_attributes(duration) == [
        {
            GenAI.GEN_AI_OPERATION_NAME: "chat",
            GenAI.GEN_AI_PROVIDER_NAME: "huggingface",
            GenAI.GEN_AI_REQUEST_MODEL: "HuggingFaceTB/SmolLM2-135M-Instruct",
        }
    ]
    token_usage = metrics[gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE]
    assert {
        point.attributes[GenAI.GEN_AI_TOKEN_TYPE]: point.sum
        for point in token_usage.data.data_points
    } == {"input": 3, "output": 2}


@pytest.mark.parametrize(
    "runtime, provider, request_model, input_tokens, output_tokens",
    [
        ("mlx", "mlx", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", 3, 2),
        ("vllm", "vllm", "Qwen/Qwen2.5-0.5B-Instruct", 4, 2),
    ],
)
def test_other_runtimes_record_the_response(
    instrument_with_content,
    span_exporter,
    monkeypatch: pytest.MonkeyPatch,
    runtime: str,
    provider: str,
    request_model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    model = mlx_model() if runtime == "mlx" else vllm_model(monkeypatch)

    output = model.generate(messages=MESSAGES)
    assert output.content == "In Paris"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == provider
    assert attr(span, GenAI.GEN_AI_REQUEST_MODEL) == request_model
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == input_tokens
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == output_tokens
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None
    assert attr(span, server_attributes.SERVER_ADDRESS) is None
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"] == [{"type": "text", "content": "In Paris"}]


def test_no_content(instrument_no_content, span_exporter) -> None:
    transformers_model().generate(messages=MESSAGES)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "huggingface"
    assert attr(span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None
    # Token usage is metadata, so it survives without the content.
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3


def test_event_only_content_capture(
    instrument_event_only, span_exporter, log_exporter
) -> None:
    transformers_model().generate(messages=MESSAGES)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3

    (log,) = log_exporter.get_finished_logs()
    record = log.log_record
    assert record.event_name == "gen_ai.client.inference.operation.details"
    # Event attributes carry the messages as structured values, not JSON text.
    attributes = record.attributes or {}
    inputs = attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    outputs = attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert inputs[0]["parts"][0]["content"] == "Where is the Louvre?"
    assert outputs[0]["parts"][0]["content"] == "In Paris"


def test_runtime_error_is_recorded_and_reraised(
    instrument_with_content, span_exporter
) -> None:
    model = transformers_model(error=RuntimeError("CUDA out of memory"))

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        model.generate(messages=MESSAGES)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "RuntimeError"
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None


class _LifecycleRecorder(SpanProcessor):
    """Records span starts and ends.

    The exporter only sees a span once it ends, so an unfinished span reads
    there as no span at all.
    """

    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[str] = []

    def on_start(
        self, span: Span, parent_context: Context | None = None
    ) -> None:
        self.started.append(span.name)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.append(span.name)

    @property
    def leaked(self) -> list[str]:
        return self.started[len(self.ended) :]


@pytest.fixture
def lifecycle(tracer_provider) -> _LifecycleRecorder:
    recorder = _LifecycleRecorder()
    tracer_provider.add_span_processor(recorder)
    return recorder


def test_lifecycle_recorder_sees_the_happy_path(
    instrument_with_content, lifecycle
) -> None:
    transformers_model().generate(messages=MESSAGES)

    assert len(lifecycle.started) == 1
    assert lifecycle.leaked == []


def test_generate_bad_call_records_an_error_span(
    instrument_with_content, span_exporter, lifecycle
) -> None:
    model = transformers_model()

    # messages is required, so the call fails before it reaches the runtime.
    with pytest.raises(TypeError):
        model.generate()

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "TypeError"
    assert lifecycle.leaked == []


class _NotATool:
    """tools_to_call_from is a plain runtime kwarg; the annotation is a
    contract, not enforcement."""


def test_bad_tool_does_not_leak_a_span(
    instrument_with_content, span_exporter, lifecycle
) -> None:
    model = transformers_model()

    # The AttributeError is the runtime's own, from building the tool schemas.
    # Reading the tool for telemetry must not fail the call before that.
    with pytest.raises(AttributeError, match="has no attribute 'inputs'"):
        model.generate(messages=MESSAGES, tools_to_call_from=[_NotATool()])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "AttributeError"
    assert lifecycle.leaked == []


@pytest.mark.parametrize(
    "conversion", ["to_input_messages", "to_output_message"]
)
def test_a_failed_conversion_does_not_break_the_call(
    instrument_with_content,
    span_exporter,
    lifecycle,
    monkeypatch: pytest.MonkeyPatch,
    conversion: str,
) -> None:
    # A shape the conversion cannot read drops the content from the span, and
    # changes nothing else.
    def raise_error(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("unexpected message shape")

    monkeypatch.setattr(patch_module, conversion, raise_error)

    output = transformers_model().generate(messages=MESSAGES)
    assert output.content == "In Paris"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.UNSET
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3
    assert lifecycle.leaked == []


def test_tool_definitions_recorded(
    instrument_with_content, span_exporter
) -> None:
    transformers_model().generate(
        messages=MESSAGES, tools_to_call_from=[GetWeatherTool()]
    )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_DEFINITIONS)) == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city to get the weather for",
                    }
                },
                "required": ["location"],
            },
        }
    ]


def test_user_subclass_keeps_its_provider(
    instrument_with_content, span_exporter, metric_reader
) -> None:
    # A subclass inherits the patched generate, so it is instrumented; resolving
    # the provider by exact class name would report "unknown" on both the span
    # and the metrics.
    from smolagents.models import MLXModel

    class TenantMLXModel(MLXModel):
        pass

    model = mlx_model()
    # The factory builds the base class; rebinding __class__ makes the stub an
    # instance of the subclass without running __init__.
    model.__class__ = TenantMLXModel

    model.generate(messages=MESSAGES)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "mlx"
    duration = metrics_by_name(metric_reader)[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ]
    assert data_point_attributes(duration)[0][GenAI.GEN_AI_PROVIDER_NAME] == (
        "mlx"
    )


def _fake_model(class_name: str, **attrs: Any) -> Any:
    instance = type(class_name, (), {})()
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


@pytest.mark.parametrize(
    "class_name, expected",
    [
        # No GenAI registry value exists for these runtimes, so the product
        # name is used rather than the class name.
        ("TransformersModel", "huggingface"),
        ("VLLMModel", "vllm"),
        ("MLXModel", "mlx"),
        # gen_ai.provider.name is also a metric attribute. An unmapped model
        # must not fall back to a class name.
        ("CustomModel", "unknown"),
    ],
)
def test_resolve_provider(class_name: str, expected: str) -> None:
    assert resolve_provider(_fake_model(class_name)) == expected


@pytest.mark.parametrize(
    "model_class, expected",
    [
        ("TransformersModel", "huggingface"),
        ("VLLMModel", "vllm"),
        ("MLXModel", "mlx"),
    ],
)
def test_resolve_provider_covers_every_instrumented_class(
    model_class: str, expected: str
) -> None:
    # The mapping is keyed by class name, so pin it against the real classes
    # rather than only against synthetic stand-ins.
    import smolagents

    instance = object.__new__(getattr(smolagents, model_class))
    assert resolve_provider(instance) == expected


def test_request_parameters_recorded(
    instrument_with_content, span_exporter
) -> None:
    model = mlx_model(
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        frequency_penalty=0.25,
        presence_penalty=1,
        max_tokens=256,
        seed=7,
    )

    model.generate(messages=MESSAGES, stop_sequences=["<end>"])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_TEMPERATURE) == 0.5
    assert attr(span, GenAI.GEN_AI_REQUEST_TOP_P) == 0.9
    # top_k is a float attribute in the spec even though callers pass an int.
    assert attr(span, GenAI.GEN_AI_REQUEST_TOP_K) == 40.0
    assert isinstance(attr(span, GenAI.GEN_AI_REQUEST_TOP_K), float)
    assert attr(span, GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY) == 0.25
    assert attr(span, GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY) == 1.0
    assert attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS) == 256
    assert isinstance(attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS), int)
    assert attr(span, GenAI.GEN_AI_REQUEST_SEED) == 7
    assert attr(span, GenAI.GEN_AI_REQUEST_STOP_SEQUENCES) == ("<end>",)


def test_model_kwargs_win_over_call_kwargs(
    instrument_with_content, span_exporter
) -> None:
    # _prepare_completion_kwargs applies the call kwargs first and the model
    # kwargs on top, and drops any key whose model-level value is the
    # REMOVE_PARAMETER sentinel.
    from smolagents.models import REMOVE_PARAMETER

    model = mlx_model(temperature=0.1, max_tokens=REMOVE_PARAMETER)

    model.generate(
        messages=MESSAGES, temperature=0.9, max_tokens=512, top_p=0.5
    )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_TEMPERATURE) == 0.1
    assert attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS) is None
    assert attr(span, GenAI.GEN_AI_REQUEST_TOP_P) == 0.5


@pytest.mark.parametrize(
    "model_kwargs, expected",
    [
        ({}, ("<end>",)),
        # An explicit `stop` overrides the stop_sequences argument.
        ({"stop": ["STOP"]}, ("STOP",)),
        # The model-level sentinel pops the `stop` that
        # _prepare_completion_kwargs seeded from stop_sequences, leaving the
        # request with none.
        ({"stop": "REMOVE"}, None),
    ],
)
def test_stop_sequences_follow_what_is_sent(
    instrument_with_content,
    span_exporter,
    model_kwargs: dict[str, Any],
    expected: tuple[str, ...] | None,
) -> None:
    from smolagents.models import REMOVE_PARAMETER

    model_kwargs = {
        key: REMOVE_PARAMETER if value == "REMOVE" else value
        for key, value in model_kwargs.items()
    }
    model = mlx_model(**model_kwargs)

    model.generate(messages=MESSAGES, stop_sequences=["<end>"])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_STOP_SEQUENCES) == expected


def test_stop_sequences_a_model_cannot_send_are_not_recorded(
    instrument_with_content, span_exporter
) -> None:
    # supports_stop_parameter is False for the gpt-5 and o-series names, and
    # then _prepare_completion_kwargs never seeds `stop`. The runtime truncates
    # the generated text locally instead, so the request carries no stop
    # sequences.
    model = mlx_model()
    model.model_id = "gpt-5"
    assert model.supports_stop_parameter is False

    model.generate(messages=MESSAGES, stop_sequences=["<end>"])

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_STOP_SEQUENCES) is None


@pytest.mark.parametrize(
    "model_kwargs, call_kwargs, expected",
    [
        ({"max_tokens": 256}, {}, 256),
        ({}, {"max_tokens": 256}, 256),
        # max_new_tokens is the TransformersModel spelling of the same limit,
        # and max_tokens wins when a caller sets both.
        ({"max_new_tokens": 4096}, {}, 4096),
        ({"max_new_tokens": 4096, "max_tokens": 256}, {}, 256),
        ({}, {}, None),
    ],
)
def test_max_tokens_covers_both_spellings(
    instrument_with_content,
    span_exporter,
    model_kwargs: dict[str, Any],
    call_kwargs: dict[str, Any],
    expected: int | None,
) -> None:
    model = transformers_model(**model_kwargs)

    model.generate(messages=MESSAGES, **call_kwargs)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_REQUEST_MAX_TOKENS) == expected


def test_output_type_recorded_for_a_structured_request(
    instrument_with_content, span_exporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    # VLLMModel is the only instrumented runtime that accepts a
    # response_format; TransformersModel and MLXModel raise before generating.
    model = vllm_model(monkeypatch)

    model.generate(
        messages=MESSAGES,
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}},
        },
    )

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_OUTPUT_TYPE) == "json"


@pytest.mark.parametrize(
    "response_format, expected",
    [
        ({"type": "json_object"}, "json"),
        ({"type": "json_schema", "json_schema": {}}, "json"),
        ({"type": "text"}, "text"),
        (None, None),
        # smolagents forwards response_format unchanged, so its type is
        # whatever the runtime accepts. An unknown one is dropped rather than
        # recorded on an enum attribute.
        ({"type": "xml"}, None),
        ({}, None),
    ],
)
def test_output_type_mapping(
    response_format: dict[str, Any] | None, expected: str | None
) -> None:
    # A unit test rather than a generate() call: no instrumented runtime accepts
    # every response_format spelling, and the mapping table is what needs
    # testing.
    merged = (
        {} if response_format is None else {"response_format": response_format}
    )
    assert _output_type(merged) == expected


@pytest.mark.parametrize(
    "model_kwargs, response_format, expected",
    [
        # The model-level kwargs win over the argument, and the sentinel drops
        # the key from the request, the same as for every other parameter.
        (
            {"response_format": {"type": "text"}},
            {"type": "json_object"},
            "text",
        ),
        ({"response_format": "REMOVE"}, {"type": "json_object"}, None),
        ({}, {"type": "json_object"}, "json"),
    ],
)
def test_response_format_precedence(
    model_kwargs: dict[str, Any],
    response_format: dict[str, Any],
    expected: str | None,
) -> None:
    from smolagents.models import REMOVE_PARAMETER

    model_kwargs = {
        key: REMOVE_PARAMETER if value == "REMOVE" else value
        for key, value in model_kwargs.items()
    }
    merged = _merged_request_kwargs(
        transformers_model(**model_kwargs),
        {"response_format": response_format},
    )
    assert _output_type(merged) == expected


def _drain_stream(model: Any, **kwargs: Any) -> list[Any]:
    return list(model.generate_stream(messages=MESSAGES, **kwargs))


def _failing_streamer(
    chunks: list[str], error: Exception
) -> Generator[str, None, None]:
    yield from chunks
    raise error


def test_generate_stream_is_lazy_and_records_the_drained_response(
    instrument_with_content, span_exporter
) -> None:
    model = transformers_model(stream_chunks=["In ", "Paris"])

    stream = model.generate_stream(messages=MESSAGES)
    # A streamed response isn't finished until the caller drains it.
    assert spans_by_operation(span_exporter.get_finished_spans(), "chat") == []

    deltas = list(stream)
    assert "".join(delta.content or "" for delta in deltas) == "In Paris"

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_PROVIDER_NAME) == "huggingface"
    assert attr(span, GenAI.GEN_AI_REQUEST_STREAM) is True
    # TransformersModel reports the prompt tokens on the first delta only and
    # one output token per delta.
    assert attr(span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) == 3
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == 2
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"] == [{"type": "text", "content": "In Paris"}]
    # Deltas carry no finish reason, and "stop" would hide a truncation.
    assert attr(span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None


def test_generate_stream_stays_a_generator(
    instrument_with_content, span_exporter
) -> None:
    # Instrumentation observes; it must not change what generate_stream returns.
    model = transformers_model(stream_chunks=["In Paris"])

    stream = model.generate_stream(messages=MESSAGES)
    assert isinstance(stream, Generator)
    assert inspect.isgenerator(stream)
    list(stream)


def test_generate_stream_bad_call_records_an_error_span(
    instrument_with_content, span_exporter, lifecycle
) -> None:
    model = transformers_model(stream_chunks=["In Paris"])

    # messages is required, so the call fails before it reaches the runtime.
    # Nothing will drain the stream, so the span has to end here.
    with pytest.raises(TypeError):
        model.generate_stream()

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "TypeError"
    assert lifecycle.leaked == []


def test_generate_stream_error_mid_iteration_is_recorded_and_reraised(
    instrument_with_content, span_exporter
) -> None:
    model = transformers_model()
    model.streamer = _failing_streamer(["In "], RuntimeError("kernel crashed"))

    with pytest.raises(RuntimeError, match="kernel crashed"):
        _drain_stream(model)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "RuntimeError"
    # What was streamed before the failure is still recorded.
    outputs = parse_messages(span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"] == [{"type": "text", "content": "In "}]


def test_generate_stream_close_before_drain_finalizes_once(
    instrument_with_content, span_exporter
) -> None:
    model = transformers_model(stream_chunks=["In Paris"])

    stream = model.generate_stream(messages=MESSAGES)
    stream.close()
    stream.close()  # idempotent

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert span.status.status_code == StatusCode.UNSET
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None


def test_generate_stream_records_chunk_metrics(
    instrument_with_content, span_exporter, metric_reader
) -> None:
    model = transformers_model(stream_chunks=["In ", "Paris"])

    _drain_stream(model)

    metrics = metrics_by_name(metric_reader)
    assert gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION in metrics
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE in metrics
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK in metrics
    )
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK in metrics
    )


def test_generate_stream_no_content(instrument_no_content, span_exporter):
    model = transformers_model(stream_chunks=["In ", "Paris"])

    _drain_stream(model)

    (span,) = spans_by_operation(span_exporter.get_finished_spans(), "chat")
    assert attr(span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None
    assert attr(span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) == 2


def test_to_tool_definitions_uses_json_schema() -> None:
    (definition,) = to_tool_definitions([GetWeatherTool()])
    assert definition.name == "get_weather"
    assert definition.description == "Get the weather for a given city"
    # smolagents' raw ``inputs`` map is not a JSON Schema on its own.
    assert definition.parameters == {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city to get the weather for",
            }
        },
        "required": ["location"],
    }


def test_to_input_messages_maps_smolagents_only_roles() -> None:
    # smolagents converts these roles itself inside generate(), after the
    # wrapper has already read the messages.
    messages = to_input_messages(
        [
            {"role": MessageRole.TOOL_CALL, "content": "call"},
            {"role": MessageRole.TOOL_RESPONSE, "content": "response"},
            {"role": MessageRole.SYSTEM, "content": "sys"},
        ]
    )
    assert [message.role for message in messages] == [
        "assistant",
        "user",
        "system",
    ]


def test_to_input_messages_dict_and_chatmessage() -> None:
    messages = to_input_messages(
        [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
    )
    assert messages[0].role == "user"
    assert messages[0].parts == [Text(content="Hello")]
    assert messages[1].role == "assistant"
    assert messages[1].parts == [Text(content="Hi there")]


def test_to_input_messages_image_and_base64() -> None:
    # A vision-capable runtime (TransformersModel with a processor, VLLMModel
    # with _is_vlm) takes image parts alongside the text.
    messages = to_input_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "image", "image": "aVZCT1J3MEtHZ28="},
                ],
            }
        ]
    )
    parts = messages[0].parts
    assert parts[0] == Uri(mime_type=None, modality="image", uri=IMAGE_URL)
    assert isinstance(parts[1], Blob)
    assert parts[1].modality == "image"
    assert parts[1].mime_type == "image/png"
    assert isinstance(parts[1].content, bytes)


def test_to_input_messages_data_url_keeps_media_type() -> None:
    messages = to_input_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "data:image/jpeg;base64,aVZCT1J3MEtHZ28=",
                    }
                ],
            }
        ]
    )
    (part,) = messages[0].parts
    assert isinstance(part, Blob)
    assert part.mime_type == "image/jpeg"


@pytest.mark.parametrize(
    "image",
    [
        "not base64!!!!",
        # A path is not base64, but a non-validating decode silently turns this
        # one into 15 bytes of garbage after dropping the invalid characters.
        "/tmp/photos/my-cat.png",
    ],
)
def test_to_input_messages_drops_malformed_base64_image(image: str) -> None:
    messages = to_input_messages(
        [{"role": "user", "content": [{"type": "image", "image": image}]}]
    )
    assert messages[0].parts == []


def test_to_output_message_unwraps_the_role_enum() -> None:
    output = to_output_message(
        ChatMessage(role=MessageRole.ASSISTANT, content="done")
    )
    assert output.role == "assistant"
    assert output.parts == [Text(content="done")]
    # The in-process runtimes report no finish reason, and util-genai omits the
    # empty value.
    assert output.finish_reason == ""


def test_to_output_message_maps_image_content() -> None:
    # A response whose content is a list carries the same element shapes as a
    # request, so an image in it maps the same way.
    output = to_output_message(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=[
                {"type": "text", "text": "Here it is"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        )
    )
    assert output.parts[0] == Text(content="Here it is")
    assert output.parts[1] == Uri(
        mime_type=None, modality="image", uri=IMAGE_URL
    )
