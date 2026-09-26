# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from sys import float_info
from typing import Any

import httpx
import pytest
from google.genai import Client
from google.genai.types import HttpOptions

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
    AsyncInteractionsResource,
    InteractionsResource,
    _explicit_request_fields,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai.instrumentor import instrument

from .util import create_mock_completed_event, create_mock_interaction

if not _HAS_INTERACTIONS:
    pytest.skip("This SDK has no Interactions API", allow_module_level=True)

try:
    from google.genai._interactions._streaming import AsyncStream, Stream
    from google.genai._interactions.types.generation_config import (
        GenerationConfig,
    )
    from google.genai._interactions.types.text_response_format import (
        TextResponseFormat,
    )

    _HAS_REQUEST_BODY = False
except ImportError:
    from google.genai._gaos.interactions import AsyncStream, Stream
    from google.genai._gaos.models.createinteraction import (
        CreateInteractionRequest,
    )
    from google.genai._gaos.types.interactions import (
        CreateModelInteraction,
        GenerationConfig,
        TextResponseFormat,
    )

    _HAS_REQUEST_BODY = True


@dataclass
class _SDK:
    calls: list[dict[str, Any]] = field(default_factory=list)
    response: Any = field(default_factory=create_mock_interaction)
    error: Exception | None = None
    stream_error: Exception | None = None


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> _SDK:
    state = _SDK()

    def record(kwargs: dict[str, Any]) -> bool:
        state.calls.append(kwargs)
        if state.error is not None:
            raise state.error
        request = kwargs.get("request")
        body = (
            request.get("body")
            if isinstance(request, dict)
            else getattr(request, "body", None)
        )
        params = kwargs if body is None else body
        return (
            params.get("stream", False)
            if isinstance(params, dict)
            else params.stream
        )

    def events() -> Iterator[Any]:
        yield create_mock_completed_event(state.response)
        if state.stream_error is not None:
            raise state.stream_error

    def create(_self: object, **kwargs: Any) -> Any:
        return events() if record(kwargs) else state.response

    async def acreate(_self: object, **kwargs: Any) -> Any:
        async def async_events() -> AsyncIterator[Any]:
            for event in events():
                yield event

        return async_events() if record(kwargs) else state.response

    monkeypatch.setattr(InteractionsResource, "create", create)
    monkeypatch.setattr(AsyncInteractionsResource, "create", acreate)
    return state


@pytest.fixture
def instrumented(
    sdk: _SDK,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
) -> Iterator[None]:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", "true")
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="NO_CONTENT",
    ):
        yield


def _request(params: dict[str, Any], shape: str) -> dict[str, Any]:
    if shape == "kwargs":
        return params
    if not _HAS_REQUEST_BODY:
        pytest.skip("This SDK predates request.body")
    if shape == "request-dict":
        return {"request": {"body": params}}
    return {
        "request": CreateInteractionRequest(
            body=CreateModelInteraction(**params)
        )
    }


def _parameter_attributes(
    span_exporter: InMemorySpanExporter,
) -> dict[str, Any]:
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    return {
        key: value
        for key, value in span.attributes.items()
        if key
        in (
            "gen_ai.request.temperature",
            "gen_ai.request.top_p",
            "gen_ai.request.max_tokens",
            "gen_ai.request.seed",
            "gen_ai.request.stop_sequences",
            "gen_ai.output.type",
        )
    }


@pytest.mark.parametrize("value", [None, 1, "invalid", []])
def test_non_mapping_request_fields_are_normalized(value: object) -> None:
    assert _explicit_request_fields(value) == {}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("config_model", [False, True])
@pytest.mark.parametrize("shape", ["kwargs", "request-dict", "request-model"])
@pytest.mark.asyncio
async def test_generation_config(
    instrumented: None,
    sdk: _SDK,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    streaming: bool,
    config_model: bool,
    shape: str,
) -> None:
    values = {
        "max_output_tokens": 64,
        "seed": 0,
        "stop_sequences": ["DONE"],
    }
    expected = {
        "gen_ai.request.max_tokens": 64,
        "gen_ai.request.seed": 0,
        "gen_ai.request.stop_sequences": ("DONE",),
    }
    if "temperature" in GenerationConfig.model_fields:
        values.update(temperature=0, top_p=0)
        expected.update(
            {"gen_ai.request.temperature": 0.0, "gen_ai.request.top_p": 0.0}
        )
    config = GenerationConfig(**values) if config_model else values
    params = _request(
        {
            "model": "gemini-2.5-flash",
            "input": "hello",
            "generation_config": config,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
            },
            "stream": streaming,
        },
        shape,
    )
    client = Client(api_key="test-key", vertexai=False)
    result = (
        await client.aio.interactions.create(**params)
        if asynchronous
        else client.interactions.create(**params)
    )
    assert all(sdk.calls[0][key] is value for key, value in params.items())
    if streaming:
        assert not span_exporter.get_finished_spans()
        if asynchronous:
            chunks = [chunk async for chunk in result]
        else:
            chunks = list(result)
        assert len(chunks) == 1
    else:
        assert result is sdk.response
    expected["gen_ai.output.type"] = "json"
    attributes = _parameter_attributes(span_exporter)
    assert attributes == expected
    assert all(
        type(attributes[key]) is type(value) for key, value in expected.items()
    )


@pytest.mark.parametrize(
    "parameters,expected",
    [
        ({}, None),
        ({"response_mime_type": "application/json"}, "json"),
        ({"response_mime_type": "text/plain"}, "text"),
        ({"response_mime_type": "image/png"}, "image"),
        ({"response_mime_type": "audio/wav"}, "speech"),
        ({"response_mime_type": "application/octet-stream"}, None),
        ({"response_format": {"type": "text"}}, "text"),
        (
            {
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                }
            },
            "json",
        ),
        ({"response_format": {"type": "image"}}, "image"),
        ({"response_format": {"type": "audio"}}, "speech"),
        ({"response_format": {"type": "object", "properties": {}}}, "json"),
        ({"response_format": {"type": "json"}}, "json"),
        ({"response_format": {"type": "json_object"}}, "json"),
        ({"response_format": {"type": "json_schema"}}, "json"),
        (
            {
                "response_format": [
                    {"type": "json"},
                    {"type": "json_object"},
                    {"type": "json_schema"},
                ]
            },
            "json",
        ),
        ({"response_format": [{"type": "json"}, {"type": "text"}]}, None),
        ({"response_format": [{"type": "text"}]}, "text"),
        ({"response_format": [{"type": "text"}, {"type": "image"}]}, None),
        (
            {
                "response_format": [{"type": "text"}, {"type": "image"}],
                "response_mime_type": "text/plain",
            },
            None,
        ),
        ({"response_format": {"type": "unknown"}}, None),
        (
            {
                "response_format": TextResponseFormat(
                    type="text", mime_type="application/json"
                )
            },
            "json",
        ),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_output_format(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    parameters: dict[str, Any],
    expected: str | None,
    asynchronous: bool,
) -> None:
    client = Client(api_key="test-key", vertexai=False)
    params = {"model": "gemini-2.5-flash", "input": "hello", **parameters}
    if asynchronous:
        await client.aio.interactions.create(**params)
    else:
        client.interactions.create(**params)
    assert _parameter_attributes(span_exporter) == (
        {"gen_ai.output.type": expected} if expected else {}
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("name", ["temperature", "top_p"])
@pytest.mark.parametrize(
    "value", [0, 0.0, 0.5, 1, float_info.max, int(float_info.max)]
)
@pytest.mark.asyncio
async def test_sampling_parameters_are_recorded_as_floats(
    instrumented: None,
    sdk: _SDK,
    span_exporter: InMemorySpanExporter,
    name: str,
    value: float,
    asynchronous: bool,
) -> None:
    config = {name: value}
    client = Client(api_key="test-key", vertexai=False)
    params = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "generation_config": config,
    }
    result = (
        await client.aio.interactions.create(**params)
        if asynchronous
        else client.interactions.create(**params)
    )
    assert result is sdk.response
    assert sdk.calls[0]["generation_config"] is config
    attributes = _parameter_attributes(span_exporter)
    assert attributes == {f"gen_ai.request.{name}": float(value)}
    assert all(type(attribute) is float for attribute in attributes.values())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"temperature": None, "top_p": None, "seed": None},
        {"stop_sequences": []},
        {"stop_sequences": ()},
        {
            "temperature": True,
            "top_p": float("nan"),
            "max_output_tokens": False,
            "seed": "12",
            "stop_sequences": "DONE",
        },
        {"temperature": 10**400, "top_p": float("inf"), "stop_sequences": [1]},
        {"temperature": -0.5, "top_p": -1},
        {"temperature": float("-inf"), "top_p": float("nan")},
        {"temperature": -(10**400), "top_p": 10**400},
        {"temperature": "0.5", "top_p": True},
    ],
)
@pytest.mark.asyncio
async def test_absent_or_invalid_config_is_not_recorded(
    instrumented: None,
    sdk: _SDK,
    span_exporter: InMemorySpanExporter,
    config: object,
    asynchronous: bool,
) -> None:
    client = Client(api_key="test-key", vertexai=False)
    params = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "generation_config": config,
    }
    result = (
        await client.aio.interactions.create(**params)
        if asynchronous
        else client.interactions.create(**params)
    )
    assert result is sdk.response
    assert sdk.calls[0]["generation_config"] is config
    assert _parameter_attributes(span_exporter) == {}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("shape", ["kwargs", "request-dict", "request-model"])
@pytest.mark.parametrize("explicit_seed", [False, True])
@pytest.mark.asyncio
async def test_config_model_defaults_are_not_recorded(
    instrumented: None,
    sdk: _SDK,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    shape: str,
    explicit_seed: bool,
) -> None:
    class DefaultConfig(GenerationConfig):
        temperature: float = 0.75
        top_p: float = 0.95
        max_output_tokens: int = 512
        seed: int = 31
        stop_sequences: list[str] = ["DEFAULT"]

        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("Instrumentation must not serialize config")

    config = DefaultConfig(seed=0) if explicit_seed else DefaultConfig()
    params = _request(
        {
            "model": "gemini-2.5-flash",
            "input": "hello",
            "generation_config": config,
        },
        shape,
    )
    client = Client(api_key="test-key", vertexai=False)
    if asynchronous:
        result = await client.aio.interactions.create(**params)
    else:
        result = client.interactions.create(**params)

    assert result is sdk.response
    assert all(sdk.calls[0][key] is value for key, value in params.items())
    assert _parameter_attributes(span_exporter) == (
        {"gen_ai.request.seed": 0} if explicit_seed else {}
    )
    assert config.max_output_tokens == 512
    assert config.model_fields_set == ({"seed"} if explicit_seed else set())


@pytest.mark.skipif(not _HAS_REQUEST_BODY, reason="SDK predates request.body")
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit_config", [False, True])
@pytest.mark.asyncio
async def test_request_model_defaults_are_not_recorded(
    instrumented: None,
    sdk: _SDK,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    explicit_config: bool,
) -> None:
    class DefaultBody(CreateModelInteraction):
        generation_config: GenerationConfig = GenerationConfig(
            seed=31, max_output_tokens=512
        )
        response_format: TextResponseFormat = TextResponseFormat(
            type="text", mime_type="application/json"
        )

        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("Instrumentation must not serialize request")

    body = DefaultBody(
        model="gemini-2.5-flash",
        input="hello",
        **(
            {"generation_config": GenerationConfig(seed=0)}
            if explicit_config
            else {}
        ),
    )
    request = CreateInteractionRequest(body=body)
    client = Client(api_key="test-key", vertexai=False)
    if asynchronous:
        result = await client.aio.interactions.create(request=request)
    else:
        result = client.interactions.create(request=request)

    assert result is sdk.response
    assert sdk.calls[0]["request"] is request
    assert _parameter_attributes(span_exporter) == (
        {"gen_ai.request.seed": 0} if explicit_config else {}
    )
    assert body.model_fields_set == (
        {"model", "input", "generation_config"}
        if explicit_config
        else {"model", "input"}
    )


def test_explicit_config_model_extra_fields_are_recorded(
    instrumented: None, span_exporter: InMemorySpanExporter
) -> None:
    class ExtraConfig(GenerationConfig):
        model_config = {"extra": "allow"}

    config = ExtraConfig(temperature=0.0, top_p=0.5)
    Client(api_key="test-key", vertexai=False).interactions.create(
        model="gemini-2.5-flash", input="hello", generation_config=config
    )
    assert _parameter_attributes(span_exporter) == {
        "gen_ai.request.temperature": 0.0,
        "gen_ai.request.top_p": 0.5,
    }


def test_iterable_config_content_is_not_consumed(
    instrumented: None, span_exporter: InMemorySpanExporter
) -> None:
    stops = iter(["DONE"])
    formats = iter([{"type": "text"}])
    Client(api_key="test-key", vertexai=False).interactions.create(
        model="gemini-2.5-flash",
        input="hello",
        generation_config={"stop_sequences": stops},
        response_format=formats,
    )
    assert next(stops) == "DONE"
    assert next(formats) == {"type": "text"}
    assert _parameter_attributes(span_exporter) == {}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_request_errors_keep_configuration(
    instrumented: None,
    sdk: _SDK,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    streaming: bool,
) -> None:
    error = ValueError("SDK request failed")
    sdk.error = error
    client = Client(api_key="test-key", vertexai=False)
    params = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "generation_config": {"seed": 0, "max_output_tokens": 64},
        "stream": streaming,
    }
    with pytest.raises(ValueError) as raised:
        if asynchronous:
            await client.aio.interactions.create(**params)
        else:
            client.interactions.create(**params)
    assert raised.value is error
    assert _parameter_attributes(span_exporter) == {
        "gen_ai.request.seed": 0,
        "gen_ai.request.max_tokens": 64,
    }
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes["error.type"] == "ValueError"


def test_configuration_is_recorded_on_events_without_content_capture(
    instrumented: None, log_exporter: InMemoryLogRecordExporter
) -> None:
    Client(api_key="test-key", vertexai=False).interactions.create(
        model="gemini-2.5-flash",
        input="hello",
        generation_config={"seed": 0, "max_output_tokens": 64},
        response_mime_type="application/json",
    )
    (event,) = log_exporter.get_finished_logs()
    attributes = event.log_record.attributes
    assert attributes["gen_ai.request.seed"] == 0
    assert type(attributes["gen_ai.request.seed"]) is int
    assert attributes["gen_ai.request.max_tokens"] == 64
    assert attributes["gen_ai.output.type"] == "json"
    assert "gen_ai.input.messages" not in attributes
    assert "gen_ai.output.messages" not in attributes


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("shape", ["kwargs", "request-dict", "request-model"])
@pytest.mark.parametrize("error_source", ["sdk", "caller"])
@pytest.mark.asyncio
async def test_stream_errors_keep_configuration(
    instrumented: None,
    sdk: _SDK,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    shape: str,
    error_source: str,
) -> None:
    error = ConnectionError("stream interrupted")
    if error_source == "sdk":
        sdk.stream_error = error
    params = _request(
        {
            "model": "gemini-2.5-flash",
            "input": "hello",
            "generation_config": {"seed": 0, "max_output_tokens": 64},
            "stream": True,
        },
        shape,
    )
    client = Client(api_key="test-key", vertexai=False)
    stream = (
        await client.aio.interactions.create(**params)
        if asynchronous
        else client.interactions.create(**params)
    )
    assert not span_exporter.get_finished_spans()
    with pytest.raises(ConnectionError) as raised:
        if asynchronous:
            async with stream:
                async for _ in stream:
                    if error_source == "caller":
                        raise error
        else:
            with stream:
                for _ in stream:
                    if error_source == "caller":
                        raise error
    assert raised.value is error
    assert _parameter_attributes(span_exporter) == {
        "gen_ai.request.seed": 0,
        "gen_ai.request.max_tokens": 64,
    }
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes["error.type"] == "ConnectionError"


def test_agent_output_format_is_recorded(
    instrumented: None, span_exporter: InMemorySpanExporter
) -> None:
    Client(api_key="test-key", vertexai=False).interactions.create(
        agent="deep-research-preview-04-2026",
        input="hello",
        response_format={"type": "text", "mime_type": "application/json"},
    )
    assert _parameter_attributes(span_exporter) == {
        "gen_ai.output.type": "json",
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("shape", ["kwargs", "request-dict"])
@pytest.mark.parametrize("stream_value", [True, False, 1, 0, "true", "false"])
@pytest.mark.asyncio
async def test_real_sdk_stream_normalization(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    shape: str,
    stream_value: object,
) -> None:
    expected_streaming = (
        stream_value in (True, 1, "true")
        if _HAS_REQUEST_BODY
        else bool(stream_value)
    )
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert bool(body["stream"]) is expected_streaming
        if expected_streaming:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=b""
            )
        return httpx.Response(
            200,
            json={
                "id": "test-id",
                "status": "completed",
                "model": "gemini-2.5-flash",
                "steps": [],
                "usage": {"total_input_tokens": 1, "total_output_tokens": 1},
            },
        )

    params = _request(
        {
            "model": "gemini-2.5-flash",
            "input": "hello",
            "generation_config": {"seed": 0},
            "stream": stream_value,
        },
        shape,
    )
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="NO_CONTENT",
    ):
        if asynchronous:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            ) as transport:
                with Client(
                    api_key="test-key",
                    vertexai=False,
                    http_options=HttpOptions(httpx_async_client=transport),
                ) as client:
                    result = await client.aio.interactions.create(**params)
                    if expected_streaming:
                        assert isinstance(result, AsyncStream)
                        assert not span_exporter.get_finished_spans()
                        assert [item async for item in result] == []
                    else:
                        assert result.id == "test-id"
        else:
            with httpx.Client(
                transport=httpx.MockTransport(respond)
            ) as transport:
                with Client(
                    api_key="test-key",
                    vertexai=False,
                    http_options=HttpOptions(httpx_client=transport),
                ) as client:
                    result = client.interactions.create(**params)
                    if expected_streaming:
                        assert isinstance(result, Stream)
                        assert not span_exporter.get_finished_spans()
                        assert list(result) == []
                    else:
                        assert result.id == "test-id"
    assert len(requests) == 1
    assert _parameter_attributes(span_exporter) == {"gen_ai.request.seed": 0}
