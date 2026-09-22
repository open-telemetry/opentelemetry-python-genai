# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import yaml
from google.genai import Client
from google.genai.types import HttpOptions

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)

from ..common.otel_mocker import OTelMocker
from .util import create_request_parameters

pytestmark = pytest.mark.skipif(
    not _HAS_INTERACTIONS,
    reason="Interactions are not supported in this version of google-genai",
)

# Set GOOGLE_API_KEY (or GEMINI_API_KEY) to record these cassettes against the
# real API; playback uses the placeholder, so a real key never needs to be
# written into this file. Prefer GOOGLE_API_KEY: tox.ini's passenv forwards
# GOOGLE_* but not GEMINI_*.
_FAKE_API_KEY = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or "GEMINI_API_KEY"
)


class _LiteralBlockScalar(str):
    """Formats the string as a literal block scalar, preserving whitespace and
    without interpreting escape characters"""


def _literal_block_scalar_presenter(dumper, data):
    """Represents a scalar string as a literal block, via '|' syntax"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


@pytest.fixture(
    name="internal_setup_yaml_pretty_formatting", scope="module", autouse=True
)
def fixture_setup_yaml_pretty_formatting():
    yaml.add_representer(_LiteralBlockScalar, _literal_block_scalar_presenter)


def _process_string_value(string_value):
    """Pretty-prints JSON or returns long strings as a LiteralBlockScalar"""
    try:
        json_data = json.loads(string_value)
        return _LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return _LiteralBlockScalar(string_value)
    return string_value


def _convert_body_to_literal(data):
    """Searches the data for body strings, attempting to pretty-print JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle response body case (e.g., response.body.string)
            if key == "body" and isinstance(value, dict) and "string" in value:
                string_val = value["string"]
                if isinstance(string_val, bytes):
                    try:
                        string_val = string_val.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                if isinstance(string_val, str):
                    value["string"] = _process_string_value(string_val)

            # Handle request body case (e.g., request.body)
            elif key == "body" and isinstance(value, str):
                data[key] = _process_string_value(value)
            elif key == "body" and isinstance(value, bytes):
                try:
                    data[key] = _process_string_value(value.decode("utf-8"))
                except UnicodeDecodeError:
                    pass

            else:
                _convert_body_to_literal(value)

    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = _convert_body_to_literal(choice)

    return data


class _PrettyPrintJSONBody:
    """This makes request and response body recordings more readable."""

    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = _convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(name="fully_initialized_vcr", scope="module", autouse=True)
def setup_vcr(vcr):
    vcr.register_serializer("yaml", _PrettyPrintJSONBody)
    vcr.serializer = "yaml"
    return vcr


@pytest.fixture(name="vcr_config", scope="module")
def fixture_vcr_config():
    return {
        "filter_query_parameters": [
            "key",
            "apiKey",
            "quotaUser",
            "userProject",
            "token",
            "access_token",
            "accessToken",
            "refesh_token",
            "refreshToken",
            "authuser",
            "bearer",
            "bearer_token",
            "bearerToken",
            "userIp",
        ],
        "filter_headers": [
            "x-goog-api-key",
            "authorization",
            "server",
            "Server",
            "Server-Timing",
            "Date",
        ],
        "ignore_hosts": [
            "oauth2.googleapis.com",
            "iam.googleapis.com",
        ],
        "decode_compressed_response": True,
    }


@pytest.fixture(name="instrumentor")
def fixture_instrumentor():
    return GoogleGenAiSdkInstrumentor()


@pytest.fixture(name="setup_instrumentation", autouse=True)
def fixture_setup_instrumentation(instrumentor):
    instrumentor.instrument()
    yield
    instrumentor.uninstrument()


@pytest.fixture(name="otel_mocker", autouse=True)
def fixture_otel_mocker():
    result = OTelMocker()
    result.install()
    yield result
    result.uninstall()


@pytest.fixture(name="client")
def fixture_client():
    return Client(
        api_key=_FAKE_API_KEY,
        vertexai=False,
        http_options=HttpOptions(headers={"accept-encoding": "identity"}),
    )


@pytest.mark.vcr
def test_sync_interactions_create(
    client: Client,
    otel_mocker: OTelMocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )

    parameters, expected_attributes = create_request_parameters()
    response = client.interactions.create(
        model="gemini-2.5-flash",
        input="Hello, how can you help me today?",
        **parameters,
    )

    assert response is not None
    assert response.id is not None

    span = otel_mocker.get_span_named("interactions.create gemini-2.5-flash")
    assert span is not None
    assert span.attributes["gen_ai.provider.name"] == "gemini"
    assert span.attributes["gen_ai.request.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.response.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.operation.name"] == "interactions.create"
    for name, expected in expected_attributes.items():
        actual = span.attributes[name]
        if isinstance(expected, list):
            assert isinstance(actual, tuple)
            actual = list(actual)
        assert actual == expected, name
        assert type(actual) is type(expected), name


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_interactions_create(
    client: Client,
    otel_mocker: OTelMocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_AND_EVENT"
    )

    parameters, expected_attributes = create_request_parameters()
    response = await client.aio.interactions.create(
        model="gemini-2.5-flash",
        input="Hello, how can you help me today?",
        **parameters,
    )

    assert response is not None
    assert response.id is not None

    span = otel_mocker.get_span_named("interactions.create gemini-2.5-flash")
    assert span is not None
    assert span.attributes["gen_ai.provider.name"] == "gemini"
    assert span.attributes["gen_ai.request.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.response.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.operation.name"] == "interactions.create"
    for name, expected in expected_attributes.items():
        actual = span.attributes[name]
        if isinstance(expected, list):
            assert isinstance(actual, tuple)
            actual = list(actual)
        assert actual == expected, name
        assert type(actual) is type(expected), name


@pytest.fixture(name="capture_content")
def fixture_capture_content(instrumentor, monkeypatch: pytest.MonkeyPatch):
    """Re-instrument with content capture enabled.

    TelemetryHandler snapshots the content-capture setting when it is built, so
    the env var has to be set before instrument() runs — and the autouse
    instrumentation fixture has already run by the time a test body executes.
    """
    instrumentor.uninstrument()
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_ONLY"
    )
    instrumentor.instrument()
    yield


# `interactions.get` — the query string differs between the two SDK module
# layouts (2.8 sends none on a plain get, 2.9+ sends `stream=false`), so these
# match on the path alone and assert on the recorded request where the query
# matters.
_GET_MATCH_ON = ["method", "scheme", "host", "port", "path"]

# Background execution is what makes streaming retrieval possible, and it is
# supported only on certain models.
# https://ai.google.dev/gemini-api/docs/background-execution
_BACKGROUND_MODEL = "gemini-3.8-flash"

# A real interaction id, so these cassettes can be re-recorded against the API.
_INTERACTION_ID = (
    "v1_ChdVRVd2YXQ3UkJxX2Utc0FQbE1tVmtBdxIXVUVXdmF0N1JCcV9lLXNBUGxNbVZrQXc"
)


def _fetch_span(otel_mocker: OTelMocker):
    span = otel_mocker.get_span_named("fetch_response")
    assert span is not None
    assert span.attributes["gen_ai.operation.name"] == "fetch_response"
    assert span.attributes["gen_ai.provider.name"] == "gemini"
    # A fetch performs no inference, so the fetched interaction's token counts
    # must not be reported.
    assert not [
        name for name in span.attributes if name.startswith("gen_ai.usage.")
    ]
    return span


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get(
    client, otel_mocker: OTelMocker, capture_content
):
    response = client.interactions.get(_INTERACTION_ID)

    assert response.id == _INTERACTION_ID
    assert response.status == "completed"

    span = _fetch_span(otel_mocker)
    # Compare against the SDK's own parsed values: what matters is that the
    # instrumentation propagated them, not which model recorded the cassette.
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.response.model"] == response.model
    assert span.attributes["gen_ai.response.status"] == response.status
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert "gen_ai.request.stream" not in span.attributes
    # The model output round-trips from the real parsed interaction.
    expected_output = "".join(
        part.text
        for step in response.steps
        if step.type == "model_output"
        for part in step.content
        if part.type == "text"
    )
    assert expected_output
    messages = json.loads(span.attributes["gen_ai.output.messages"])
    texts = [
        part["content"]
        for part in messages[0]["parts"]
        if part["type"] == "text"
    ]
    assert texts == [expected_output]
    # A fetched response carries no record of the original request's input.
    assert "gen_ai.input.messages" not in span.attributes


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
@pytest.mark.asyncio
async def test_async_interactions_get(client, otel_mocker: OTelMocker):
    response = await client.aio.interactions.get(id=_INTERACTION_ID)

    assert response.id == _INTERACTION_ID

    span = _fetch_span(otel_mocker)
    assert span.attributes["gen_ai.response.id"] == response.id
    assert span.attributes["gen_ai.response.status"] == response.status


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get_with_raw_response(
    client, otel_mocker: OTelMocker, capture_content
):
    """with_raw_response routes through the patched get, unparsed.

    Instrumentation must not read the body -- the caller has not asked for it
    yet -- so the span describes the request only.
    """
    raw = client.interactions.with_raw_response.get(_INTERACTION_ID)

    # The SDK's own return contract is intact: still unparsed, still parseable.
    assert type(raw).__name__ == "APIResponse"
    parsed = raw.parse()
    assert parsed.id == _INTERACTION_ID

    span = _fetch_span(otel_mocker)
    assert span.attributes["gen_ai.response.id"] == _INTERACTION_ID
    assert "error.type" not in span.attributes
    # Nothing was read, so no response content or status is known.
    for attribute in (
        "gen_ai.response.status",
        "gen_ai.response.model",
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
    ):
        assert attribute not in span.attributes


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
@pytest.mark.asyncio
async def test_async_interactions_get_with_raw_response(
    client, otel_mocker: OTelMocker, capture_content
):
    raw = await client.aio.interactions.with_raw_response.get(_INTERACTION_ID)

    # The async variant's parse() is itself awaitable.
    parsed = await raw.parse()
    assert parsed.id == _INTERACTION_ID

    span = _fetch_span(otel_mocker)
    assert span.attributes["gen_ai.response.id"] == _INTERACTION_ID
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get_with_streaming_response(
    client, otel_mocker: OTelMocker, capture_content
):
    """with_streaming_response hands back a context manager, not an Interaction."""
    with client.interactions.with_streaming_response.get(
        _INTERACTION_ID
    ) as response:
        parsed = response.parse()

    assert parsed.id == _INTERACTION_ID

    span = _fetch_span(otel_mocker)
    assert span.attributes["gen_ai.response.id"] == _INTERACTION_ID
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
@pytest.mark.asyncio
async def test_async_interactions_get_with_streaming_response(
    client, otel_mocker: OTelMocker, capture_content
):
    """The async context manager is a different shape again."""
    async with client.aio.interactions.with_streaming_response.get(
        _INTERACTION_ID
    ) as response:
        parsed = await response.parse()

    assert parsed.id == _INTERACTION_ID

    span = _fetch_span(otel_mocker)
    assert span.attributes["gen_ai.response.id"] == _INTERACTION_ID
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get_failed_status(client, otel_mocker: OTelMocker):
    """A failed *generation* is not a failed fetch.

    The cassette is hand-maintained: the API only returns a failed interaction
    when a generation actually failed, so re-recording this test would capture a
    completed one instead. Exclude it from record runs
    (``-k "interactions_get and not failed_status"``).
    """
    response = client.interactions.get(_INTERACTION_ID)

    assert response.status == "failed"

    span = _fetch_span(otel_mocker)
    assert span.attributes["gen_ai.response.status"] == "failed"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("error",)
    assert "error.type" not in span.attributes


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get_invalid_id(client, otel_mocker: OTelMocker):
    # A malformed id is rejected as a bad request rather than a missing one.
    with pytest.raises(Exception) as exc_info:
        client.interactions.get("interaction-missing")

    span = otel_mocker.get_span_named("fetch_response")
    assert span is not None
    assert span.attributes["gen_ai.response.id"] == "interaction-missing"
    # The interactions API raises from its own error hierarchy, which carries
    # the status as `status_code`; resolve_error_type reports it like it does
    # for a generate_content error.
    assert exc_info.value.status_code == 400
    assert span.attributes["error.type"] == "400"


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get_streaming(
    client, otel_mocker: OTelMocker, capture_content
):
    """Stream-retrieve a background interaction.

    Streaming retrieval only applies to an interaction that is still running,
    so the interaction is created with background=True here rather than reusing
    a finished one -- which also keeps the cassette re-recordable.
    """
    created = client.interactions.create(
        model=_BACKGROUND_MODEL,
        input="Write a short guide on space exploration.",
        background=True,
    )

    stream = client.interactions.get(created.id, stream=True)
    events = list(stream)

    completed = [e for e in events if e.event_type == "interaction.completed"]
    assert len(completed) == 1
    interaction = completed[0].interaction

    span = _fetch_span(otel_mocker)
    assert "gen_ai.request.stream" not in span.attributes
    assert span.attributes["gen_ai.response.id"] == created.id
    # Proves the real SSE completion event is recognised and applied.
    assert span.attributes["gen_ai.response.status"] == interaction.status
    assert span.attributes["gen_ai.response.model"] == interaction.model
    # The create call is a separate operation with its own span.
    assert (
        otel_mocker.get_span_named(f"interactions.create {_BACKGROUND_MODEL}")
        is not None
    )
    # The completion event carries metadata only (`steps` comes back null), so
    # the captured content has to come from the step deltas.
    assert interaction.steps is None
    streamed = "".join(
        event.delta.text
        for event in events
        if event.event_type == "step.delta"
        and getattr(event.delta, "text", None)
    )
    assert streamed
    messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert [
        part["content"]
        for part in messages[0]["parts"]
        if part["type"] == "text"
    ] == [streamed]


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get_stream_error(client, otel_mocker: OTelMocker):
    """A real recording in which the generation failed mid-stream.

    The provider reports this in-band -- an `error` event over a 200 response,
    with no exception raised -- so the span has to be failed from the event.
    """
    created = client.interactions.create(
        model=_BACKGROUND_MODEL,
        input="Write a short guide on space exploration.",
        background=True,
    )

    events = list(client.interactions.get(created.id, stream=True))

    errors = [e for e in events if e.event_type == "error"]
    assert len(errors) == 1
    assert errors[0].error.code == "service_unavailable"

    span = _fetch_span(otel_mocker)
    assert span.attributes["error.type"] == "service_unavailable"
    # The generation never completed, so no response status was reported.
    assert "gen_ai.response.status" not in span.attributes


@pytest.mark.vcr(match_on=_GET_MATCH_ON)
def test_sync_interactions_get_resumed_stream(
    client, otel_mocker: OTelMocker, vcr_cassette
):
    """Resume a stream from an event_id carried by an earlier event."""
    created = client.interactions.create(
        model=_BACKGROUND_MODEL,
        input="Write a short guide on space exploration.",
        background=True,
    )

    # First pass: read events until one carries a resume token, then close the
    # stream. The `with` matters — abandoning a stream mid-iteration never
    # finalizes its span.
    cursor = None
    with client.interactions.get(created.id, stream=True) as first:
        for event in first:
            cursor = getattr(event, "event_id", None)
            if cursor:
                break
    assert cursor, "no event carried an event_id to resume from"

    resumed = client.interactions.get(
        created.id, stream=True, last_event_id=cursor
    )
    list(resumed)

    spans = [
        span
        for span in otel_mocker.get_finished_spans()
        if span.name == "fetch_response"
    ]
    assert len(spans) == 2
    assert "gen_ai.request.stream_cursor" not in spans[0].attributes
    assert spans[1].attributes["gen_ai.request.stream_cursor"] == cursor
    # The cursor really reached the wire, not just our kwargs.
    assert any(
        f"last_event_id={cursor}" in request.uri
        for request in vcr_cassette.requests
    )
