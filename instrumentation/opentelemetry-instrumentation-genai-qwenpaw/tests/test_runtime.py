# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive the installed QwenPaw 2.x runtime and its real envelope protocol."""

from __future__ import annotations

import asyncio
import gc
import json
from contextlib import suppress
from importlib.metadata import version
from types import AsyncGeneratorType
from typing import Any
from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind, StatusCode, get_current_span
from opentelemetry.util.genai.handler import TelemetryHandler

from .harness import make_runtime, runtime_request

pytestmark = pytest.mark.usefixtures("runtime_module")


async def drain(stream: Any) -> list[Any]:
    return [event async for event in stream]


def output_text(span: Any) -> list[str]:
    return [
        part["content"]
        for message in json.loads(
            span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
        )
        for part in message["parts"]
        if part["type"] == "text"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("keyword", [False, True])
async def test_runtime_turn(
    instrument_with_content: Any,
    span_exporter: Any,
    as_dict: Any,
    keyword: Any,
    metric_reader: Any,
) -> None:
    runtime = make_runtime()
    request = runtime_request()
    if as_dict:
        request = request.model_dump(mode="json")
    stream = runtime.run(request=request) if keyword else runtime.run(request)
    events = await drain(stream)
    assert events[-1].status == "completed"
    (span,) = span_exporter.get_finished_spans()
    assert span.name == "invoke_agent Friday"
    assert span.kind is SpanKind.INTERNAL
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert span.attributes[GenAI.GEN_AI_AGENT_NAME] == "Friday"
    assert span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "sess-2"
    assert GenAI.GEN_AI_AGENT_ID not in span.attributes
    assert isinstance(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES], str)
    assert isinstance(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES], str)
    assert json.loads(span.attributes[GenAI.GEN_AI_INPUT_MESSAGES])[0][
        "parts"
    ] == [{"type": "text", "content": "/test"}]
    assert output_text(span) == ["Runtime reply."]
    assert span.attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert GenAI.GEN_AI_REQUEST_STREAM not in span.attributes
    metrics = [
        metric
        for resource in metric_reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    ]
    assert [metric.name for metric in metrics] == [
        "gen_ai.invoke_agent.duration"
    ]
    (point,) = metrics[0].data.data_points
    assert point.count == 1
    assert point.sum >= 0
    assert point.attributes[GenAI.GEN_AI_AGENT_NAME] == "Friday"


@pytest.mark.asyncio
async def test_disabled_capture(
    instrument_no_content: Any, span_exporter: Any
) -> None:
    await drain(make_runtime().run(runtime_request()))
    (span,) = span_exporter.get_finished_spans()
    assert GenAI.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "sess-2"])
async def test_lazy_context_and_nested_invocations(
    instrument_with_content: Any,
    span_exporter: Any,
    tracer_provider: Any,
    session_id: str | None,
) -> None:
    runtime = make_runtime()
    from qwenpaw.runtime.slash_command_registry import CommandSpec

    handler = TelemetryHandler(tracer_provider=tracer_provider)

    async def nested(ctx: Any, args: Any) -> Any:
        from .harness import assistant_reply

        with handler.invoke_local_agent(agent_name="nested"):
            pass
        return assistant_reply("nested reply")

    runtime.workspace.plugins.slash_command_registry.register(
        CommandSpec(name="nested", handler=nested)
    )
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("parent") as parent:
        request = runtime_request("/nested", session_id=session_id)
        stream = runtime.run(request)
        assert request.session_id == session_id
        assert isinstance(stream, AsyncGeneratorType)
        assert stream.__class__ is AsyncGeneratorType
        assert span_exporter.get_finished_spans() == ()
        assert get_current_span() is parent
        async for _ in stream:
            assert get_current_span() is parent
        assert get_current_span() is parent
    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert spans["invoke_agent Friday"].parent == parent.get_span_context()
    assert (
        spans["invoke_agent nested"].parent
        == spans["invoke_agent Friday"].context
    )
    assert (
        spans["invoke_agent nested"].attributes[GenAI.GEN_AI_CONVERSATION_ID]
        == spans["invoke_agent Friday"].attributes[
            GenAI.GEN_AI_CONVERSATION_ID
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,error_type",
    [
        (ConnectionError("transport failed"), "ConnectionError"),
        (asyncio.CancelledError(), "asyncio.exceptions.CancelledError"),
    ],
)
async def test_runtime_stream_error(
    instrument_with_content: Any,
    span_exporter: Any,
    error: Any,
    error_type: Any,
) -> None:
    from qwenpaw.runtime.hooks import HookBase
    from qwenpaw.runtime.phases import Phase

    class FailAfterResponse(HookBase):
        phase = Phase.POST_RESPONSE
        name = "fail-after-response"

        async def run(self: Any, ctx: Any) -> None:
            raise error

    runtime = make_runtime("partial")
    runtime.workspace.plugins.hook_registry.register(FailAfterResponse())
    with pytest.raises(type(error)) as raised:
        await drain(runtime.run(runtime_request()))
    assert raised.value is error
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == error_type
    assert isinstance(span.attributes[ERROR_TYPE], str)
    assert GenAI.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    assert "partial" in output_text(span)


@pytest.mark.asyncio
async def test_caller_error(
    instrument_with_content: Any, span_exporter: Any
) -> None:
    error = ValueError("caller failed")
    stream = make_runtime().run(runtime_request())
    with pytest.raises(ValueError) as raised:
        async with stream:
            await anext(stream)
            await anext(stream)
            raise error
    assert raised.value is error
    if version("qwenpaw") == "2.0.0":
        with suppress(GeneratorExit):
            await anext(stream.__wrapped__)
    assert stream.ag_frame is None
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ERROR_TYPE] == "ValueError"
    assert GenAI.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    assert output_text(span) == ["Runtime reply."]


@pytest.mark.asyncio
@pytest.mark.parametrize("read", [False, True])
async def test_early_close(
    instrument_with_content: Any, span_exporter: Any, read: Any
) -> None:
    stream = make_runtime().run(runtime_request())
    if read:
        await anext(stream)
        await anext(stream)
    if read and version("qwenpaw") == "2.0.0":
        # QwenPaw 2.0.0 catches GeneratorExit and yields an error envelope.
        # Preserve its close failure rather than masking it in instrumentation.
        with pytest.raises(RuntimeError, match="ignored GeneratorExit"):
            await stream.aclose()
        with suppress(GeneratorExit):
            await anext(stream.__wrapped__)
    else:
        await stream.aclose()
    await stream.aclose()
    if not read:
        assert span_exporter.get_finished_spans() == ()
        return
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is (
        StatusCode.ERROR
        if read and version("qwenpaw") == "2.0.0"
        else StatusCode.UNSET
    )
    assert GenAI.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    assert stream.ag_frame is None
    if read:
        assert output_text(span) == ["Runtime reply."]


@pytest.mark.asyncio
async def test_abandoned_stream(
    instrument_with_content: Any, span_exporter: Any
) -> None:
    stream = make_runtime().run(runtime_request())
    await anext(stream)
    await anext(stream)
    underlying = stream.__wrapped__
    del stream
    gc.collect()
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ERROR_TYPE] == "_OTHER"
    assert output_text(span) == ["Runtime reply."]
    if version("qwenpaw") == "2.0.0":
        with pytest.raises(RuntimeError, match="ignored GeneratorExit"):
            await underlying.aclose()
    with suppress(GeneratorExit):
        await anext(underlying)


@pytest.mark.asyncio
async def test_runtime_generated_session(
    instrument_no_content: Any, span_exporter: Any
) -> None:
    events = await drain(make_runtime().run(runtime_request(session_id=None)))
    (span,) = span_exporter.get_finished_spans()
    assert isinstance(span.attributes[GenAI.GEN_AI_CONVERSATION_ID], str)
    assert (
        span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == events[-1].session_id
    )


@pytest.mark.asyncio
async def test_completion_hook(
    tracer_provider: Any, logger_provider: Any, meter_provider: Any
) -> None:
    hook = MagicMock()
    with instrument(
        QwenPawInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
        completion_hook=hook,
    ):
        await drain(make_runtime().run(runtime_request()))
    hook.on_completion.assert_called_once()
    assert (
        hook.on_completion.call_args.kwargs["inputs"][0].parts[0].content
        == "/test"
    )
    assert (
        hook.on_completion.call_args.kwargs["outputs"][0].parts[0].content
        == "Runtime reply."
    )


@pytest.mark.asyncio
async def test_partial_deltas_and_message_snapshots(
    tracer_provider: Any, span_exporter: Any, monkeypatch: Any
) -> None:
    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        make_runtime_wrapper,
    )

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)

    async def events(request: object) -> Any:
        yield {
            "object": "message",
            "id": "reply",
            "type": "message",
            "role": "assistant",
            "content": [],
        }
        yield {
            "object": "content",
            "msg_id": "reply",
            "index": 0,
            "type": "text",
            "text": "hello ",
            "delta": True,
        }
        yield {
            "object": "content",
            "msg_id": "reply",
            "index": 0,
            "type": "text",
            "text": "world",
            "delta": True,
        }
        yield {
            "object": "content",
            "msg_id": "reply",
            "index": 0,
            "type": "text",
            "text": "hello world",
            "delta": False,
        }

    stream = make_runtime_wrapper(handler)(
        events, make_runtime(), (runtime_request(),), {}
    )
    await anext(stream)
    await anext(stream)
    await anext(stream)
    await anext(stream)
    await stream.aclose()
    (span,) = span_exporter.get_finished_spans()
    assert output_text(span) == ["hello world"]
    assert GenAI.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes


def test_request_message_parts() -> None:
    from qwenpaw.schemas import (
        AgentRequest,
        FileContent,
        ImageContent,
        Message,
        Role,
        TextContent,
        VideoContent,
    )

    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        input_messages_from_request,
    )

    request = AgentRequest(
        input=[
            Message(
                role=Role.USER,
                content=[
                    TextContent(text="describe"),
                    ImageContent(
                        image_url="https://example.invalid/image.png"
                    ),
                    VideoContent(
                        video_url="https://example.invalid/video.mp4"
                    ),
                    FileContent(file_url="https://example.invalid/file.pdf"),
                ],
            )
        ]
    )
    (message,) = input_messages_from_request(request)
    assert message.role == "user"
    assert [
        (part.type, getattr(part, "modality", None)) for part in message.parts
    ] == [
        ("text", None),
        ("uri", "image"),
        ("uri", "video"),
        ("uri", "document"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_type,data,part_type",
    [
        ("reasoning", "thinking", "reasoning"),
        (
            "function_call",
            {
                "name": "weather",
                "call_id": "call-1",
                "arguments": '{"city":"Paris"}',
            },
            "tool_call",
        ),
        (
            "function_call_output",
            {"call_id": "call-1", "output": "sunny"},
            "tool_call_response",
        ),
    ],
)
async def test_output_message_parts(
    tracer_provider: Any,
    span_exporter: Any,
    monkeypatch: Any,
    message_type: str,
    data: Any,
    part_type: str,
) -> None:
    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        make_runtime_wrapper,
    )

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    block = (
        {"type": "text", "text": data}
        if message_type == "reasoning"
        else {"type": "data", "data": data}
    )

    async def events(request: object) -> Any:
        yield {
            "object": "message",
            "id": "reply",
            "type": message_type,
            "role": "assistant",
            "content": [],
        }
        yield {"object": "content", "msg_id": "reply", "index": 0, **block}

    stream = make_runtime_wrapper(handler)(
        events, make_runtime(), (runtime_request(),), {}
    )
    await drain(stream)
    (span,) = span_exporter.get_finished_spans()
    (message,) = json.loads(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert message["parts"][0]["type"] == part_type
    if message_type == "function_call":
        assert message["parts"][0]["arguments"] == {"city": "Paris"}
    if message_type == "function_call_output":
        assert message["parts"][0]["response"] == "sunny"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "part_kind", ["reasoning", "tool_call", "tool_result"]
)
async def test_real_envelope_parts(
    instrument_with_content: Any,
    tracer_provider: Any,
    span_exporter: Any,
    part_kind: str,
) -> None:
    from types import SimpleNamespace

    from agentscope.event import EventType
    from qwenpaw.runtime.envelope import Envelope

    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        make_runtime_wrapper,
    )

    if part_kind == "reasoning":
        event_fields = [
            (EventType.THINKING_BLOCK_START, {}),
            (EventType.THINKING_BLOCK_DELTA, {"delta": "hello "}),
            (EventType.THINKING_BLOCK_DELTA, {"delta": "world"}),
        ]
    else:
        event_fields = [
            (EventType.TOOL_CALL_START, {"tool_call_name": "weather"}),
            (EventType.TOOL_CALL_DELTA, {"delta": '{"city":'}),
            (EventType.TOOL_CALL_DELTA, {"delta": '"Paris"}'}),
        ]
        if part_kind == "tool_result":
            event_fields += [
                (EventType.TOOL_CALL_END, {}),
                (EventType.TOOL_RESULT_START, {"tool_call_name": "weather"}),
                (EventType.TOOL_RESULT_TEXT_DELTA, {"delta": "sunny"}),
                (EventType.TOOL_RESULT_END, {}),
            ]

    async def events(request: object) -> Any:
        envelope = Envelope("sess-2")
        for kind, extra in event_fields:
            event = SimpleNamespace(
                type=kind,
                tool_call_id="call-1",
                block_id="thinking-1",
                metadata={},
                **extra,
            )
            async for chunk in envelope.translate_event(event):
                yield chunk
        if part_kind == "tool_result":
            async for chunk in envelope.finalize():
                yield chunk

    handler = TelemetryHandler(tracer_provider=tracer_provider)
    stream = make_runtime_wrapper(handler)(
        events, make_runtime(), (runtime_request(),), {}
    )
    if part_kind == "tool_result":
        await drain(stream)
    else:
        for _ in range(3):
            await anext(stream)
        await stream.aclose()
    (span,) = span_exporter.get_finished_spans()
    messages = json.loads(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    parts = [part for msg in messages for part in msg["parts"]]
    if part_kind == "reasoning":
        assert parts == [{"type": "reasoning", "content": "hello world"}]
    else:
        assert parts[0] == {
            "type": "tool_call",
            "name": "weather",
            "id": "call-1",
            "arguments": {"city": "Paris"}
            if part_kind == "tool_result"
            else '{"city":',
        }
        if part_kind == "tool_result":
            assert messages[-1]["role"] == "tool"
            assert parts[-1] == {
                "type": "tool_call_response",
                "id": "call-1",
                "response": "sunny",
            }


@pytest.mark.asyncio
async def test_normalization_error_restores_context(
    instrument_with_content: Any, span_exporter: Any, tracer_provider: Any
) -> None:
    from pydantic import ValidationError

    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("parent") as parent:
        stream = make_runtime().run({"input": "invalid"})
        with pytest.raises(ValidationError):
            await anext(stream)
        assert get_current_span() is parent
        (span,) = span_exporter.get_finished_spans()
        assert span.parent == parent.get_span_context()
        assert (
            span.attributes[ERROR_TYPE]
            == "pydantic_core._pydantic_core.ValidationError"
        )


@pytest.mark.asyncio
async def test_unread_abandoned_stream_does_not_invoke_agent(
    instrument_no_content: Any, span_exporter: Any
) -> None:
    stream = make_runtime().run(runtime_request())
    del stream
    gc.collect()
    assert span_exporter.get_finished_spans() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_hook_context_survives_yields(
    instrument_with_content: Any,
    tracer_provider: Any,
    span_exporter: Any,
    fail: bool,
) -> None:
    from qwenpaw.runtime.hooks import HookBase, HookResult
    from qwenpaw.runtime.phases import Phase

    from opentelemetry import baggage
    from opentelemetry.context import attach, detach

    tracer = tracer_provider.get_tracer(__name__)
    error = ConnectionError("after response")

    class OpenScope(HookBase):
        phase = Phase.PRE_DISPATCH
        name = "open-scope"

        async def run(self, ctx: Any) -> HookResult:
            scope = tracer.start_as_current_span("hook")
            ctx.extras["scope"] = scope
            ctx.extras["span"] = scope.__enter__()
            ctx.extras["token"] = attach(baggage.set_baggage("hook", "active"))
            return HookResult()

    class AfterResponse(HookBase):
        phase = Phase.POST_RESPONSE
        name = "after-response"

        async def run(self, ctx: Any) -> HookResult:
            assert get_current_span() is ctx.extras["span"]
            assert baggage.get_baggage("hook") == "active"
            with tracer.start_as_current_span("after-response"):
                pass
            if fail:
                raise error
            return HookResult()

    class CloseScope(HookBase):
        phase = Phase.FINALLY
        name = "close-scope"

        async def run(self, ctx: Any) -> HookResult:
            assert get_current_span() is ctx.extras["span"]
            detach(ctx.extras["token"])
            ctx.extras["scope"].__exit__(None, None, None)
            return HookResult()

    runtime = make_runtime()
    for hook in (OpenScope(), AfterResponse(), CloseScope()):
        runtime.workspace.plugins.hook_registry.register(hook)
    with tracer.start_as_current_span("parent") as parent:
        try:
            async for _ in runtime.run(runtime_request()):
                assert get_current_span() is parent
                assert baggage.get_baggage("hook") is None
        except ConnectionError as caught:
            assert fail
            assert caught is error
        else:
            assert not fail
        assert get_current_span() is parent
        assert baggage.get_baggage("hook") is None
    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert spans["hook"].parent == spans["invoke_agent Friday"].context
    assert spans["after-response"].parent == spans["hook"].context
    assert spans["invoke_agent Friday"].parent == parent.get_span_context()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ConnectionError("failed"), asyncio.CancelledError()]
)
async def test_error_preserves_unfinished_second_block(
    instrument_with_content: Any,
    tracer_provider: Any,
    span_exporter: Any,
    error: BaseException,
) -> None:
    from types import SimpleNamespace

    from agentscope.event import EventType
    from qwenpaw.runtime.envelope import Envelope

    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        make_runtime_wrapper,
    )

    async def events(request: object) -> Any:
        envelope = Envelope("sess-2")
        for kind, block_id, delta in [
            (EventType.TEXT_BLOCK_START, "first", ""),
            (EventType.TEXT_BLOCK_DELTA, "first", "ready"),
            (EventType.TEXT_BLOCK_END, "first", ""),
            (EventType.TEXT_BLOCK_START, "second", ""),
            (EventType.TEXT_BLOCK_DELTA, "second", "unfinished"),
        ]:
            async for chunk in envelope.translate_event(
                SimpleNamespace(
                    type=kind,
                    block_id=block_id,
                    delta=delta,
                    metadata={},
                )
            ):
                yield chunk
        async for chunk in envelope.error_envelope(
            "failed", "ConnectionError"
        ):
            yield chunk
        raise error

    stream = make_runtime_wrapper(
        TelemetryHandler(tracer_provider=tracer_provider)
    )(
        events,
        make_runtime(),
        (runtime_request(),),
        {},
    )
    with pytest.raises(type(error)) as caught:
        await drain(stream)
    assert caught.value is error
    (span,) = span_exporter.get_finished_spans()
    assert output_text(span) == ["ready", "unfinished"]
    assert span.status.status_code is StatusCode.ERROR


@pytest.mark.parametrize("kind", ["image", "video", "file", "audio"])
def test_inline_request_media(kind: str) -> None:
    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        input_messages_from_request,
    )
    from opentelemetry.util.genai.types import BlobPart

    block = {"type": kind}
    if kind == "audio":
        block.update(data="YWJj", format="wav")
    else:
        block[f"{kind}_url"] = "data:application/octet-stream;base64,YWJj"
    (message,) = input_messages_from_request(
        {"input": [{"role": "user", "content": [block]}]}
    )
    (part,) = message.parts
    assert isinstance(part, BlobPart)
    assert part.content == b"abc"
    assert part.modality == ("document" if kind == "file" else kind)
    assert part.mime_type == (
        "audio/wav" if kind == "audio" else "application/octet-stream"
    )


@pytest.mark.parametrize("audio_format", [None, "ogg"])
def test_audio_only_request(audio_format: str | None) -> None:
    from qwenpaw.schemas import AgentRequest, AudioContent, Message, Role

    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        input_messages_from_request,
    )
    from opentelemetry.util.genai.types import UriPart

    url = "https://example.invalid/voice.ogg"
    (message,) = input_messages_from_request(
        AgentRequest(
            input=[
                Message(
                    role=Role.USER,
                    content=[AudioContent(data=url, format=audio_format)],
                )
            ]
        )
    )
    assert message.role == "user"
    assert message.parts == [
        UriPart(uri=url, modality="audio", mime_type=None)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mime_type", ["image/png", "video/mp4", "audio/wav"])
async def test_real_envelope_inline_media(
    tracer_provider: Any,
    span_exporter: Any,
    mime_type: str,
) -> None:
    from types import SimpleNamespace

    from agentscope.event import EventType
    from qwenpaw.runtime.envelope import Envelope

    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        make_runtime_wrapper,
    )
    from opentelemetry.util.genai.types import BlobPart

    async def events(request: object) -> Any:
        envelope = Envelope("sess-2")
        for kind in (
            EventType.DATA_BLOCK_START,
            EventType.DATA_BLOCK_DELTA,
            EventType.DATA_BLOCK_END,
        ):
            async for chunk in envelope.translate_event(
                SimpleNamespace(
                    type=kind,
                    block_id="media",
                    media_type=mime_type,
                    data="YWJj",
                    metadata={},
                )
            ):
                yield chunk
        async for chunk in envelope.finalize():
            yield chunk

    hook = MagicMock()
    handler = TelemetryHandler(
        tracer_provider=tracer_provider, completion_hook=hook
    )
    stream = make_runtime_wrapper(handler)(
        events, make_runtime(), (runtime_request(),), {}
    )
    await drain(stream)
    hook.on_completion.assert_called_once()
    (message,) = hook.on_completion.call_args.kwargs["outputs"]
    assert message.parts == [
        BlobPart(
            content=b"abc",
            mime_type=mime_type,
            modality=mime_type.split("/")[0],
        )
    ]
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.UNSET


@pytest.mark.asyncio
@pytest.mark.parametrize("mime_type", ["image/png", "video/mp4", "audio/wav"])
@pytest.mark.parametrize("ending", ["complete", "close", "error"])
async def test_real_envelope_mixed_content(
    tracer_provider: Any,
    span_exporter: Any,
    monkeypatch: Any,
    mime_type: str,
    ending: str,
) -> None:
    from types import SimpleNamespace

    from agentscope.event import EventType
    from qwenpaw.runtime.envelope import Envelope

    from opentelemetry.instrumentation.genai.qwenpaw.runtime import (
        make_runtime_wrapper,
    )
    from opentelemetry.util.genai.types import BlobPart, TextPart

    error = ConnectionError("mixed content interrupted")

    async def events(request: object) -> Any:
        envelope = Envelope("sess-2")
        for kind, block_id, fields in [
            (EventType.TEXT_BLOCK_START, "before", {}),
            (EventType.TEXT_BLOCK_DELTA, "before", {"delta": "before"}),
            (EventType.TEXT_BLOCK_END, "before", {}),
            (EventType.DATA_BLOCK_START, "media", {"media_type": mime_type}),
            (EventType.DATA_BLOCK_DELTA, "media", {"data": "YWJj"}),
            (EventType.DATA_BLOCK_END, "media", {}),
            (EventType.TEXT_BLOCK_START, "after", {}),
            (EventType.TEXT_BLOCK_DELTA, "after", {"delta": "af"}),
            (EventType.TEXT_BLOCK_DELTA, "after", {"delta": "ter"}),
        ]:
            async for chunk in envelope.translate_event(
                SimpleNamespace(
                    type=kind, block_id=block_id, metadata={}, **fields
                )
            ):
                yield chunk
        if ending == "error":
            async for chunk in envelope.error_envelope(
                "interrupted", "ConnectionError"
            ):
                yield chunk
            raise error
        async for chunk in envelope.translate_event(
            SimpleNamespace(
                type=EventType.TEXT_BLOCK_END, block_id="after", metadata={}
            )
        ):
            yield chunk
        async for chunk in envelope.finalize():
            yield chunk

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    hook = MagicMock()
    handler = TelemetryHandler(
        tracer_provider=tracer_provider, completion_hook=hook
    )
    stream = make_runtime_wrapper(handler)(
        events, make_runtime(), (runtime_request(),), {}
    )
    if ending == "close":
        async for chunk in stream:
            if getattr(chunk, "text", None) == "ter":
                break
        await stream.aclose()
    elif ending == "error":
        with pytest.raises(ConnectionError) as caught:
            await drain(stream)
        assert caught.value is error
    else:
        await drain(stream)
    expected = [
        TextPart(content="before"),
        BlobPart(
            content=b"abc",
            mime_type=mime_type,
            modality=mime_type.split("/")[0],
        ),
        TextPart(content="after"),
    ]
    hook.on_completion.assert_called_once()
    (message,) = hook.on_completion.call_args.kwargs["outputs"]
    assert message.parts == expected
    (span,) = span_exporter.get_finished_spans()
    (recorded,) = json.loads(span.attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert recorded["role"] == "assistant"
    assert recorded["parts"] == [
        {"type": "text", "content": "before"},
        {
            "type": "blob",
            "content": "YWJj",
            "mime_type": mime_type,
            "modality": mime_type.split("/")[0],
        },
        {"type": "text", "content": "after"},
    ]
    assert span.status.status_code is (
        StatusCode.ERROR if ending == "error" else StatusCode.UNSET
    )
