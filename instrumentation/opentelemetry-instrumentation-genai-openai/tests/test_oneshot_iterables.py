# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Content capture must not empty a one-shot messages or tools iterable."""

import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

MSGS = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi"},
]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]


def _completion():
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _handler(received):
    def handle(request):
        body = json.loads(request.content)
        received.append(
            {
                "messages": body.get("messages", []),
                "tools": body.get("tools", []),
            }
        )
        return httpx.Response(200, json=_completion())

    return handle


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        "span_only",
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    received = []
    try:
        yield received, exporter
    finally:
        instrumentor.uninstrument()
        exporter.clear()


def _captured_contents(exporter):
    spans = exporter.get_finished_spans()
    assert spans
    raw = spans[-1].attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    messages = json.loads(raw)
    return [
        part["content"]
        for message in messages
        for part in message["parts"]
    ]


def test_content_capture_does_not_empty_oneshot_messages_or_tools(captured):
    received, exporter = captured
    client = OpenAI(
        api_key="x",
        base_url="http://mock/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(_handler(received))
        ),
    )

    client.chat.completions.create(model="m", messages=(m for m in MSGS))
    assert [m["content"] for m in received[-1]["messages"]] == [
        "be brief",
        "hi",
    ]
    assert _captured_contents(exporter) == ["be brief", "hi"]

    client.chat.completions.create(
        model="m", messages=MSGS, tools=(t for t in TOOLS)
    )
    assert received[-1]["tools"][0]["function"]["name"] == "get_weather"
    assert len(received[-1]["messages"]) == 2

    client.chat.completions.create(model="m", messages=map(lambda m: m, MSGS))
    assert len(received[-1]["messages"]) == 2


def test_reused_mutable_message_keeps_each_yield(captured):
    received, exporter = captured
    message = {"role": "user", "content": "first"}

    def generate():
        yield message
        message["content"] = "second"
        yield message

    client = OpenAI(
        api_key="x",
        base_url="http://mock/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(_handler(received))
        ),
    )
    client.chat.completions.create(model="m", messages=generate())

    transmitted = [item["content"] for item in received[-1]["messages"]]
    assert len(transmitted) == 2
    assert _captured_contents(exporter) == ["first", "second"]


@pytest.mark.asyncio()
async def test_async_content_capture_does_not_empty_oneshot_messages(captured):
    received, exporter = captured
    client = AsyncOpenAI(
        api_key="x",
        base_url="http://mock/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_handler(received))
        ),
    )
    await client.chat.completions.create(model="m", messages=(m for m in MSGS))
    assert len(received[-1]["messages"]) == 2
    assert exporter.get_finished_spans()
