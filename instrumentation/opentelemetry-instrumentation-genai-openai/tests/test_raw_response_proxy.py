# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OpenAI ``with_raw_response`` proxy.

These exercise the routing ``parse()`` does — it wraps only SDK streams it can
drive and hands anything else back untouched — and the read/close fallbacks
that finalize the span for callers that never parse, so the span never leaks.
"""

import logging

import pytest
from openai import AsyncStream, Stream

from opentelemetry.instrumentation.genai.openai._raw_response import (
    wrap_chat_raw_response,
    wrap_responses_raw_response,
)
from opentelemetry.util.genai.handler import TelemetryHandler

try:
    from openai.types.responses import Response

    HAS_RESPONSES_API = True
except ImportError:
    Response = None
    HAS_RESPONSES_API = False

_requires_responses_api = pytest.mark.skipif(
    not HAS_RESPONSES_API, reason="Responses API requires a newer openai SDK"
)

_COMPLETION_BODY = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        }
    ],
}


class _HttpResponse:
    def __init__(self, body=None, is_closed=False):
        self.close_calls = 0
        self.read_calls = 0
        self.is_closed = is_closed
        self._body = body

    def close(self):
        self.close_calls += 1

    def read(self):
        self.read_calls += 1
        return b"{}"

    async def aread(self):
        return self.read()

    async def aclose(self):
        self.close_calls += 1

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _FakeStream(Stream):
    # Bypass Stream.__init__ (needs an httpx response + client); the proxy only
    # checks isinstance to decide whether it can wrap the parse result.
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _RawResponse:
    headers = {"openai-version": "2020-10-01"}
    request_id = "req_123"

    def __init__(self, parse_result, body=None, is_closed=False):
        self.http_response = _HttpResponse(body=body, is_closed=is_closed)
        self._parse_result = parse_result
        self.parse_calls = []

    def parse(self, *args, **kwargs):
        self.parse_calls.append((args, kwargs))
        if kwargs.get("to") is not None:
            return kwargs["to"]
        return self._parse_result


class _FakeStreamWrapper:
    """Stands in for a real stream wrapper, which needs a live SDK stream."""

    def __init__(self, stream, invocation, capture_content):
        self.stream = stream
        self.invocation = invocation
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _extract(invocation, payload, capture_content):
    invocation.response_id = payload.id


def _wrap(raw, invocation, **kwargs):
    options = {
        "extract": _extract,
        "stream_wrapper_cls": _FakeStreamWrapper,
    }
    options.update(kwargs)
    return wrap_chat_raw_response(raw, invocation, False, **options)


@pytest.fixture(autouse=True)
def fixture_vcr():
    """No VCR needed for these unit tests."""
    yield


@pytest.fixture(name="invocation")
def fixture_invocation(tracer_provider, meter_provider, logger_provider):
    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    return handler.inference("openai", request_model="gpt-4")


def test_metadata_resolves_natively(invocation):
    raw = _RawResponse(_FakeStream())
    proxy = _wrap(raw, invocation)

    assert proxy.headers == {"openai-version": "2020-10-01"}
    assert proxy.request_id == "req_123"


def test_parse_wraps_stream_and_replays_it(invocation, span_exporter):
    stream = _FakeStream()
    proxy = _wrap(_RawResponse(stream), invocation)

    parsed = proxy.parse()

    assert isinstance(parsed, _FakeStreamWrapper)
    assert parsed.stream is stream
    # Replayed, so a second parse cannot hand back the uninstrumented stream
    # over the same body.
    assert proxy.parse() is parsed
    assert span_exporter.get_finished_spans() == ()


def test_abandoned_parsed_stream_is_finalized_on_close(
    invocation, span_exporter
):
    """A caller that parses, stops reading, and closes must still get a span."""
    raw = _RawResponse(_FakeStream())
    proxy = _wrap(raw, invocation)

    stream_wrapper = proxy.parse()
    raw.http_response.close()

    # The close fallback drives the abandoned wrapper rather than standing down.
    assert stream_wrapper.close_calls == 1


def test_parse_with_cast_target_is_delegated(invocation):
    """``parse(to=...)`` must return the SDK's own object, not the memo."""
    proxy = _wrap(
        _RawResponse(_FakeStream(), body=_COMPLETION_BODY), invocation
    )

    first = proxy.parse()
    assert isinstance(first, _FakeStreamWrapper)

    sentinel = object()
    assert proxy.parse(to=sentinel) is sentinel


def test_parse_non_stream_returned_untouched(invocation, caplog):
    # parse() may return something we can't drive — e.g. a custom non-stream
    # cast target. The proxy hands it back untouched and logs.
    proxy = _wrap(_RawResponse("not-a-stream"), invocation)

    with caplog.at_level(
        logging.DEBUG, logger="opentelemetry.util.genai.raw_response"
    ):
        parsed = proxy.parse()

    assert parsed == "not-a-stream"
    assert "which we cannot instrument" in caplog.text


def test_close_without_parse_finalizes_once(invocation, span_exporter):
    # A caller can abandon the response without ever calling parse(). Closing
    # it must finalize the span exactly once, and still run the real close.
    raw = _RawResponse(_FakeStream())
    _wrap(raw, invocation)

    raw.http_response.close()
    assert raw.http_response.close_calls == 1
    assert len(span_exporter.get_finished_spans()) == 1

    raw.http_response.close()
    assert len(span_exporter.get_finished_spans()) == 1


def test_read_without_parse_records_response(invocation, span_exporter):
    """Reading the body without parsing still records response attributes."""
    raw = _RawResponse(_FakeStream(), body=_COMPLETION_BODY)
    _wrap(raw, invocation)

    raw.http_response.read()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.response.id"] == "chatcmpl-123"


def test_already_closed_response_is_not_proxied(invocation, span_exporter):
    """The non-streaming path is over on return, so nothing is deferred."""
    raw = _RawResponse(None, body=_COMPLETION_BODY, is_closed=True)

    result = _wrap(raw, invocation)

    assert result is raw
    assert (
        raw.parse_calls == []
    )  # instrumentation never runs the caller's parse
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.response.id"] == "chatcmpl-123"


class _ServedModelRawResponse(_RawResponse):
    headers = {"x-ms-served-model": "served-gpt-4.1"}


def _extract_model(invocation, payload, capture_content):
    invocation.response_model_name = payload.model


def test_served_model_header_wins_over_body_model(invocation, span_exporter):
    """The gateway's served model is only on the headers, not the payload."""
    raw = _ServedModelRawResponse(None, body=_COMPLETION_BODY, is_closed=True)

    _wrap(raw, invocation, extract=_extract_model)

    spans = span_exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.response.model"] == "served-gpt-4.1"


def test_body_model_used_when_served_model_header_absent(
    invocation, span_exporter
):
    raw = _RawResponse(None, body=_COMPLETION_BODY, is_closed=True)

    _wrap(raw, invocation, extract=_extract_model)

    spans = span_exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.response.model"] == "gpt-4"


_FAILED_RESPONSE_BODY = {
    "object": "response",
    "id": "resp-1",
    "model": "gpt-4",
    "status": "failed",
    "output": [],
    "error": {"code": "server_error", "message": "boom"},
}


@_requires_responses_api
def test_failed_payload_ends_the_invocation_as_an_error(
    invocation, span_exporter
):
    """A Responses payload can report its own failure; that is not a success."""
    raw = _RawResponse(None, body=_FAILED_RESPONSE_BODY, is_closed=True)

    wrap_responses_raw_response(
        raw,
        invocation,
        False,
        stream_wrapper_cls=_FakeStreamWrapper,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["error.type"] == "server_error"


@_requires_responses_api
def test_successful_payload_is_not_an_error(invocation, span_exporter):
    raw = _RawResponse(
        None,
        body={
            "object": "response",
            "id": "resp-1",
            "model": "gpt-4",
            "status": "completed",
            "output": [],
        },
        is_closed=True,
    )

    wrap_responses_raw_response(
        raw,
        invocation,
        False,
        stream_wrapper_cls=_FakeStreamWrapper,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "error.type" not in spans[0].attributes


class _FakeAsyncStream(AsyncStream):
    # Bypass AsyncStream.__init__ for the same reason as _FakeStream.
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class _AsyncRawResponse(_RawResponse):
    """The async client's ``parse`` is a coroutine function."""

    async def parse(self, *args, **kwargs):
        self.parse_calls.append((args, kwargs))
        if kwargs.get("to") is not None:
            return kwargs["to"]
        return self._parse_result


@pytest.mark.asyncio
async def test_async_parse_wraps_stream_and_replays_it(invocation):
    stream = _FakeAsyncStream()
    proxy = _wrap(_AsyncRawResponse(stream), invocation)

    parsed = await proxy.parse()

    assert isinstance(parsed, _FakeStreamWrapper)
    assert parsed.stream is stream
    # The replay must stay awaitable for an async client.
    assert await proxy.parse() is parsed


@pytest.mark.asyncio
async def test_async_abandoned_parsed_stream_is_finalized_on_aclose(
    invocation, span_exporter
):
    raw = _AsyncRawResponse(_FakeAsyncStream())
    proxy = _wrap(raw, invocation)

    stream_wrapper = await proxy.parse()
    await raw.http_response.aclose()

    assert stream_wrapper.close_calls == 1


@pytest.mark.asyncio
async def test_async_read_without_parse_records_response(
    invocation, span_exporter
):
    raw = _AsyncRawResponse(_FakeAsyncStream(), body=_COMPLETION_BODY)
    _wrap(raw, invocation)

    await raw.http_response.aread()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.response.id"] == "chatcmpl-123"


@pytest.mark.asyncio
async def test_async_parse_with_cast_target_is_delegated(invocation):
    proxy = _wrap(_AsyncRawResponse(_FakeAsyncStream()), invocation)

    sentinel = object()
    assert await proxy.parse(to=sentinel) is sentinel


_GATEWAY_BODY = {  # an OpenAI-compatible gateway that omits ``object``
    "id": "chatcmpl-gw",
    "created": 1,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        }
    ],
}

_ERROR_ENVELOPE = {"error": {"message": "nope", "type": "invalid_request"}}


def test_body_without_discriminator_is_still_recorded(
    invocation, span_exporter
):
    """A gateway may omit ``object``; the SDK parses such a body regardless."""
    raw = _RawResponse(None, body=_GATEWAY_BODY, is_closed=True)

    _wrap(raw, invocation)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.response.id"] == "chatcmpl-gw"


def test_error_envelope_is_not_recorded_as_a_response(
    invocation, span_exporter
):
    """An error envelope has no discriminator and no id, so it is not a payload."""
    raw = _RawResponse(None, body=_ERROR_ENVELOPE, is_closed=True)

    _wrap(raw, invocation)

    # The span still ends — it must not leak — but carries no response id.
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "gen_ai.response.id" not in spans[0].attributes


def test_body_with_wrong_discriminator_is_not_recorded(
    invocation, span_exporter
):
    raw = _RawResponse(
        None,
        body={"object": "list", "id": "batch-1"},
        is_closed=True,
    )

    _wrap(raw, invocation)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "gen_ai.response.id" not in spans[0].attributes
