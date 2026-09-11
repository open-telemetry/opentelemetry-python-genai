# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from openai import AsyncOpenAI, OpenAI

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.fixture(autouse=True)
def fixture_vcr() -> Iterator[None]:
    yield


@pytest.fixture(
    params=["instrument_no_content", "instrument_with_content"],
    autouse=True,
)
def instrumentation(
    request: pytest.FixtureRequest, content_mode: tuple[bool, str]
) -> None:
    request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        pytest.param(
            {
                "prompt_tokens_details": {
                    "cached_tokens": 5,
                    "cache_write_tokens": 10,
                    "text_tokens": 70,
                    "image_tokens": 20,
                    "audio_tokens": 10,
                },
                "completion_tokens_details": {
                    "text_tokens": 18,
                    "audio_tokens": 2,
                },
            },
            id="all-details",
        ),
        pytest.param(
            {"prompt_tokens_details": {"audio_tokens": 10}},
            id="partial-details",
        ),
        pytest.param({}, id="absent-details"),
        pytest.param(
            {
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            },
            id="null-details",
        ),
        pytest.param(
            {
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "text_tokens": 0,
                    "image_tokens": 0,
                    "audio_tokens": 0,
                },
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "audio_tokens": 0,
                },
            },
            id="zero-details",
        ),
    ],
)
def usage(request: pytest.FixtureRequest) -> dict[str, object]:
    return {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        **request.param,
    }


@pytest.fixture
def api_url(usage: dict[str, object]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            response: dict[str, object] = {
                "id": "chatcmpl-usage",
                "created": 1,
                "model": "gpt-4",
            }
            if body["stream"]:
                response["object"] = "chat.completion.chunk"
                chunks = [
                    {
                        **response,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": "hello",
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        **response,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                    {**response, "choices": [], "usage": usage},
                ]
                payload = (
                    "".join(
                        f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
                    )
                    + "data: [DONE]\n\n"
                ).encode()
                content_type = "text/event-stream"
            else:
                response.update(
                    object="chat.completion",
                    usage=usage,
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "hello",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                )
                payload = json.dumps(response).encode()
                content_type = "application/json"

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def assert_usage(
    span_exporter: InMemorySpanExporter, usage: dict[str, object]
) -> None:
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 100
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    expected: dict[str, int] = {
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 100,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 20,
    }
    for field, suffix, mapping in (
        (
            "prompt_tokens_details",
            "input_tokens",
            {
                "cached_tokens": "cache_read",
                "cache_write_tokens": "cache_write",
                "text_tokens": "text",
                "image_tokens": "image",
                "audio_tokens": "audio",
            },
        ),
        (
            "completion_tokens_details",
            "output_tokens",
            {"text_tokens": "text", "audio_tokens": "audio"},
        ),
    ):
        details = usage.get(field)
        if isinstance(details, dict):
            for name, attribute in mapping.items():
                if value := details.get(name):
                    expected[f"gen_ai.usage.{attribute}.{suffix}"] = value
    actual = {
        key: value
        for key, value in span.attributes.items()
        if key.startswith("gen_ai.usage.")
    }
    assert actual == expected


@pytest.mark.parametrize("streaming", [False, True])
def test_chat_detailed_token_usage(
    api_url: str,
    streaming: bool,
    usage: dict[str, object],
    span_exporter: InMemorySpanExporter,
) -> None:
    with OpenAI(base_url=api_url, api_key="test", max_retries=0) as client:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            stream=streaming,
            **(
                {"stream_options": {"include_usage": True}}
                if streaming
                else {}
            ),
        )
        if streaming:
            with response:
                chunks = list(response)
            assert chunks[-1].usage.prompt_tokens == 100
        else:
            assert response.usage.prompt_tokens == 100

    assert_usage(span_exporter=span_exporter, usage=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_chat_detailed_token_usage(
    api_url: str,
    streaming: bool,
    usage: dict[str, object],
    span_exporter: InMemorySpanExporter,
) -> None:
    async with AsyncOpenAI(
        base_url=api_url, api_key="test", max_retries=0
    ) as client:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            stream=streaming,
            **(
                {"stream_options": {"include_usage": True}}
                if streaming
                else {}
            ),
        )
        if streaming:
            async with response:
                chunks = [chunk async for chunk in response]
            assert chunks[-1].usage.prompt_tokens == 100
        else:
            assert response.usage.prompt_tokens == 100

    assert_usage(span_exporter=span_exporter, usage=usage)
