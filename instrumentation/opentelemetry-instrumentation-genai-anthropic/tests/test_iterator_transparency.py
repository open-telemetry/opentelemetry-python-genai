# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic import Anthropic, AsyncAnthropic

try:
    import httpx2 as httpx
except ImportError:
    import httpx

from opentelemetry.instrumentation.genai.anthropic import patch, wrappers
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

from .test_async_wrappers import (
    _FakeAsyncManager,
    _FakeAsyncStream,
    _FakeSyncManager,
    _FakeSyncStream,
)
from .test_sync_messages import (
    _STREAM_SSE_BODY,
    _build_message,
    _load_span_messages,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper_name",
    [
        "messages_create",
        "async_messages_create",
        "messages_stream",
        "async_messages_stream",
    ],
)
@pytest.mark.parametrize("sdk_failure", [False, True])
async def test_extraction_failure_preserves_sdk_outcome(
    monkeypatch: pytest.MonkeyPatch, wrapper_name: str, sdk_failure: bool
) -> None:
    extractor = MagicMock(side_effect=ValueError("extraction failed"))
    monkeypatch.setattr(patch, "get_input_messages", extractor)
    monkeypatch.setattr(wrappers, "get_input_messages", extractor)
    handler = MagicMock()
    handler.should_capture_content.return_value = True
    error = RuntimeError("SDK failed")
    asynchronous = wrapper_name.startswith("async")
    streaming = wrapper_name.endswith("stream")
    stream = _FakeAsyncStream() if asynchronous else _FakeSyncStream()
    manager_type = _FakeAsyncManager if asynchronous else _FakeSyncManager
    response = object()
    message = {"role": "user", "content": "Hi"}
    kwargs = {
        "messages": iter([message]),
        "system": [{"type": "text", "text": "Be brief"}],
    }

    def sdk_call(**kwargs: Any) -> Any:
        assert list(kwargs["messages"]) == [message]
        if streaming:
            return manager_type(
                stream, enter_error=error if sdk_failure else None
            )
        if sdk_failure:
            raise error
        return response

    async def async_sdk_call(**kwargs: Any) -> Any:
        return sdk_call(**kwargs)

    wrapper = getattr(patch, wrapper_name)(handler)

    async def invoke() -> Any:
        if wrapper_name == "async_messages_create":
            return await wrapper(async_sdk_call, MagicMock(), (), kwargs)
        result = wrapper(sdk_call, MagicMock(), (), kwargs)
        if streaming:
            if asynchronous:
                async with result as wrapped_stream:
                    assert wrapped_stream.__wrapped__ is stream
            else:
                with result as wrapped_stream:
                    assert wrapped_stream.__wrapped__ is stream
        return result

    invocation = handler.inference.return_value
    if sdk_failure:
        with pytest.raises(RuntimeError) as caught:
            await invoke()
        assert caught.value is error
        invocation.fail.assert_called_once_with(error)
    else:
        result = await invoke()
        if not streaming:
            assert result is response
        invocation.fail.assert_not_called()

    extractor.assert_called_once_with([message])
    assert invocation.input_messages == []
    assert invocation.system_instruction[0].content == "Be brief"


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_content", [False, True])
@pytest.mark.parametrize("iterator_failure", [False, True])
@pytest.mark.parametrize(
    "wrapper_name",
    [
        "messages_create",
        "async_messages_create",
        "messages_stream",
        "async_messages_stream",
    ],
)
async def test_wrappers_leave_iterators_for_sdk(
    wrapper_name: str, capture_content: bool, iterator_failure: bool
) -> None:
    handler = MagicMock()
    handler.should_capture_content.return_value = capture_content
    consumed: list[str] = []
    error = ValueError("SDK request failed")

    def generate(name: str, value: Any) -> Iterator[Any]:
        consumed.append(name)
        if iterator_failure:
            raise error
        yield value

    inputs = {
        "messages": generate("messages", {"role": "user", "content": "Hi"}),
        "system": generate("system", {"type": "text", "text": "Be brief"}),
        "tools": generate("tools", {"name": "weather", "input_schema": {}}),
    }
    other_kwargs = {
        "stop_sequences": iter(["END"]),
        "extra_body": {"custom": iter(["value"])},
    }

    def sdk_call(*args: Any, **kwargs: Any) -> Any:
        assert consumed == []
        for key, value in other_kwargs.items():
            assert kwargs[key] is value
        for key, value in inputs.items():
            if not capture_content:
                assert kwargs[key] is value
            assert isinstance(kwargs[key], type(value))
            assert next(kwargs[key])
        raise error

    async def async_sdk_call(*args: Any, **kwargs: Any) -> Any:
        return sdk_call(*args, **kwargs)

    wrapper = getattr(patch, wrapper_name)(handler)
    with pytest.raises(ValueError) as caught:
        if wrapper_name == "async_messages_create":
            await wrapper(
                async_sdk_call, MagicMock(), (), {**inputs, **other_kwargs}
            )
        else:
            wrapper(sdk_call, MagicMock(), (), {**inputs, **other_kwargs})
    assert caught.value is error
    assert consumed == (
        ["messages"] if iterator_failure else ["messages", "system", "tools"]
    )
    if wrapper_name.endswith("create"):
        handler.inference.return_value.fail.assert_called_once_with(error)
        if capture_content and not iterator_failure:
            invocation = handler.inference.return_value
            assert invocation.input_messages[0].parts[0].content == "Hi"
            assert invocation.system_instruction[0].content == "Be brief"
            assert invocation.tool_definitions[0].name == "weather"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("message_iterator", [False, True])
@pytest.mark.parametrize("block_type", ["text", "image", "document"])
async def test_sdk_consumes_and_captures_request_iterators(
    span_exporter: Any,
    instrument_with_content: Any,
    asynchronous: bool,
    streaming: bool,
    message_iterator: bool,
    block_type: str,
) -> None:
    seen: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = [{"type": "text", "text": "Hello"}]
    expected_blocks = blocks
    expected_parts = [{"type": "text", "content": "Hello"}]
    if block_type != "text":
        source = {
            "type": "base64",
            "media_type": (
                "image/png" if block_type == "image" else "application/pdf"
            ),
        }
        blocks = [
            {
                "type": block_type,
                "source": {**source, "data": BytesIO(b"content")},
            }
        ]
        expected_blocks = [
            {
                "type": block_type,
                "source": {**source, "data": "Y29udGVudA=="},
            }
        ]
        expected_parts = [{"type": "blob"}]
    system = [{"type": "text", "text": "Be brief"}]
    tools = [{"name": "weather", "input_schema": {"type": "object"}}]
    messages = [{"role": "user", "content": iter(blocks)}]
    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 32,
        "messages": iter(messages) if message_iterator else messages,
        "system": iter(system),
        "tools": iter(tools),
    }

    def respond(request: Any) -> Any:
        seen.append(json.loads(request.content))
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_STREAM_SSE_BODY,
            )
        return httpx.Response(200, json=_build_message().model_dump())

    if asynchronous:
        async with AsyncAnthropic(
            api_key="test",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(respond)
            ),
        ) as client:
            if streaming:
                async with client.messages.stream(**kwargs) as stream:
                    await stream.until_done()
            else:
                await client.messages.create(**kwargs)
    else:
        with Anthropic(
            api_key="test",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            if streaming:
                with client.messages.stream(**kwargs) as stream:
                    stream.until_done()
            else:
                client.messages.create(**kwargs)

    assert len(seen) == 1
    assert seen[0]["messages"] == [
        {"role": "user", "content": expected_blocks}
    ]
    assert seen[0]["system"] == system
    assert seen[0]["tools"] == tools
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    captured_messages = _load_span_messages(
        spans[0], GenAI.GEN_AI_INPUT_MESSAGES
    )
    assert captured_messages[0]["parts"] == expected_parts
    system_instructions = _load_span_messages(
        spans[0], GenAI.GEN_AI_SYSTEM_INSTRUCTIONS
    )
    assert system_instructions == [{"type": "text", "content": "Be brief"}]
    tool_definitions = _load_span_messages(
        spans[0], GenAI.GEN_AI_TOOL_DEFINITIONS
    )
    assert tool_definitions[0]["name"] == "weather"
