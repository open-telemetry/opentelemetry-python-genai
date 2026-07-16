# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the streaming ``with_raw_response`` proxy.

These exercise ``RawResponseStreamProxy.parse()`` for today's synchronous
``parse()`` and for the graceful back-off when ``parse()`` returns an
unexpected shape (the async coroutine ``parse()`` the OpenAI SDK documents for
a future major version): the proxy hands the SDK result back untouched and
still finalizes the span so it does not leak.
"""

import inspect
import logging

import pytest

from opentelemetry.instrumentation.genai.openai._raw_response import (
    RawResponseStreamProxy,
)
from opentelemetry.instrumentation.genai.openai.chat_wrappers import (
    ChatStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler


class _SyncRawResponse:
    headers = {"openai-version": "2020-10-01"}
    request_id = "req_123"

    def parse(self, *, to=None):
        return "sync-stream"


class _AsyncRawResponse:
    headers = {"openai-version": "2020-10-01"}
    request_id = "req_123"

    async def parse(self, *, to=None):
        return "async-stream"


def _noop() -> None:
    pass


@pytest.fixture(autouse=True)
def fixture_vcr():
    """No VCR needed for these unit tests."""
    yield


def test_sync_parse_wraps_and_forwards_metadata():
    proxy = RawResponseStreamProxy(
        _SyncRawResponse(),
        wrap_stream=lambda s: ("wrapped", s),
        on_backoff=_noop,
    )

    # Metadata resolves natively off the raw response.
    assert proxy.headers == {"openai-version": "2020-10-01"}
    assert proxy.request_id == "req_123"

    parsed = proxy.parse()
    assert parsed == ("wrapped", "sync-stream")
    # Memoized so repeated calls share one wrapper / span.
    assert proxy.parse() is parsed


@pytest.mark.asyncio()
async def test_async_parse_backs_off_and_finalizes_once(caplog):
    # Simulates the SDK's documented future: async client parse() becomes a
    # coroutine. That is not the synchronous stream we expect, so the proxy
    # backs off: it hands the coroutine back untouched, logs a debug notice,
    # and finalizes the span exactly once (nothing else will close it).
    wrapped_calls = []
    backoff_calls = []
    proxy = RawResponseStreamProxy(
        _AsyncRawResponse(),
        wrap_stream=lambda s: wrapped_calls.append(s) or ("wrapped", s),
        on_backoff=lambda: backoff_calls.append(True),
    )

    with caplog.at_level(
        logging.DEBUG,
        logger="opentelemetry.instrumentation.genai.openai._raw_response",
    ):
        parsed = await proxy.parse()
    assert parsed == "async-stream"  # raw stream, not wrapped
    assert wrapped_calls == []  # wrap_stream never invoked on the coroutine
    assert backoff_calls == [True]  # span finalized
    assert "skipping stream instrumentation" in caplog.text

    # A second parse() must not finalize the span again.
    await proxy.parse()
    assert backoff_calls == [True]


def test_wrap_result_backoff_closes_span(
    tracer_provider, meter_provider, logger_provider, span_exporter
):
    # End-to-end: an unexpected raw-response shape must not leak the span the
    # instrumentation already started. Drive the real wrap_result path with a
    # raw response whose parse() returns a coroutine and assert the span ends.
    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    invocation = handler.inference("openai", request_model="gpt-4")

    result = ChatStreamWrapper.wrap_result(
        _AsyncRawResponse(), invocation, capture_content=False
    )

    assert span_exporter.get_finished_spans() == ()  # not closed yet

    parsed = result.parse()  # back-off path finalizes the span
    # Coroutine handed back untouched (not wrapped); close to avoid a warning.
    assert inspect.iscoroutine(parsed)
    parsed.close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time is not None
