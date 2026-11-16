# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Span mapping tests for the Claude Agent SDK instrumentation.

The SDK is driven through a replay transport (see ``_replay.py``), so every test
runs in-process without the ``claude`` CLI.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import anyio
import claude_agent_sdk
import pytest
from claude_agent_sdk import types as sdk_types
from claude_agent_sdk.types import ClaudeAgentOptions

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.util.genai._conversation_context import (
    with_conversation_id,
)

from ._replay import (
    ReplayTransport,
    _message,
    assistant,
    result,
    single_turn,
)

AGENT_SPAN = "invoke_agent claude-agent-sdk"


class CallerError(Exception):
    """A caller-side failure raised while the stream is being consumed."""


class _RecordingHook:
    """A completion hook that is not the no-op, so content capture turns on."""

    def on_completion(self, **kwargs: Any) -> None:
        return None


def claude_agent_sdk_instrumentor():
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )

    return ClaudeAgentSDKInstrumentor()


@pytest.fixture(params=["NO_CONTENT", "SPAN_ONLY"])
def capture(request, tracer_provider, monkeypatch):
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", request.param
    )
    instrumentor = claude_agent_sdk_instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield request.param != "NO_CONTENT"
    instrumentor.uninstrument()


def open_stream(
    messages: list[dict[str, Any]],
    *,
    prompt: str = "Read /tmp/a.txt",
    options: Any = None,
    **transport_kwargs: Any,
):
    return claude_agent_sdk.query(
        prompt=prompt,
        options=options or ClaudeAgentOptions(),
        transport=ReplayTransport(messages, **transport_kwargs),
    )


def run_query(
    messages: list[dict[str, Any]],
    *,
    options: Any = None,
    **transport_kwargs: Any,
) -> None:
    async def _run() -> None:
        async for _ in open_stream(
            messages, options=options, **transport_kwargs
        ):
            pass

    anyio.run(_run)


def thinking_turn() -> list[dict[str, Any]]:
    return [
        _message(),
        assistant(
            [
                {
                    "type": "thinking",
                    "thinking": "Let me read the file first.",
                    "signature": "sig-1",
                },
                {"type": "text", "text": "Reading the file."},
            ],
            stop_reason="end_turn",
        ),
    ]


def agent_spans(span_exporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.name.startswith("invoke_agent")
    ]


def exception_events(span) -> list[Any]:
    return [
        event for event in span.events if event.name.endswith(".exception")
    ]


def _assistant_reports_usage() -> bool:
    return "usage" in inspect.signature(sdk_types.AssistantMessage).parameters


def _result_reports_stop_reason() -> bool:
    return (
        "stop_reason" in inspect.signature(sdk_types.ResultMessage).parameters
    )


def _result_reports_errors() -> bool:
    return "errors" in inspect.signature(sdk_types.ResultMessage).parameters


def test_single_turn_emits_one_agent_span(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(single_turn())

    spans = agent_spans(span_exporter)
    assert len(spans) == 1
    span = spans[0]
    assert span.name == AGENT_SPAN
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.agent.name"] == "claude-agent-sdk"


def test_native_tracing_skips_instrumentation(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    options = ClaudeAgentOptions(
        env={
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
            "OTEL_TRACES_EXPORTER": "otlp",
        }
    )
    run_query(single_turn(), options=options)

    assert agent_spans(span_exporter) == []


def test_native_tracing_without_exporter_still_instruments(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    options = ClaudeAgentOptions(
        env={
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
        }
    )
    run_query(single_turn(), options=options)

    assert len(agent_spans(span_exporter)) == 1


def test_session_id_reported_as_conversation_id(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(single_turn())

    span = agent_spans(span_exporter)[0]
    assert span.attributes["gen_ai.conversation.id"] == "session-1"


def test_conversation_id_falls_back_to_result_message(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(
        [
            assistant([{"type": "text", "text": "hi"}]),
            result(session_id="session-from-result"),
        ]
    )

    span = agent_spans(span_exporter)[0]
    assert span.attributes["gen_ai.conversation.id"] == "session-from-result"


def test_sdk_session_id_overrides_inherited_conversation_id(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    token = otel_context.attach(with_conversation_id("parent-conversation"))
    try:
        run_query(single_turn())
    finally:
        otel_context.detach(token)

    span = agent_spans(span_exporter)[0]
    assert span.attributes["gen_ai.conversation.id"] == "session-1"


def test_usage_comes_from_result_message(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    # Assistant messages add up to 20/38; the ResultMessage says 32/38. Counting
    # both would report 52/76, so this pins the authoritative source.
    run_query(single_turn(assistant_usage=(12, 30), result_usage=(32, 38)))

    span = agent_spans(span_exporter)[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == 32
    assert span.attributes["gen_ai.usage.output_tokens"] == 38


def test_cache_tokens_are_counted_as_input(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(
        [
            _message(),
            assistant(
                [{"type": "text", "text": "hi"}],
                usage=(10, 5),
                cache_read=1000,
                cache_creation=200,
                stop_reason="end_turn",
            ),
            result(usage=(10, 5), cache_read=1000, cache_creation=200),
        ]
    )

    span = agent_spans(span_exporter)[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == 1210
    assert span.attributes["gen_ai.usage.output_tokens"] == 5


@pytest.mark.skipif(
    not _result_reports_stop_reason(),
    reason="this claude-agent-sdk release does not report stop_reason",
)
@pytest.mark.parametrize(
    ("raw_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_call"),
        ("pause_turn", "pause_turn"),
    ],
)
def test_finish_reason_normalized_to_finish_reason_vocabulary(
    span_exporter, instrument_claude_agent_sdk, raw_reason, expected
) -> None:
    messages = single_turn()
    messages[-1] = result(stop_reason=raw_reason)

    run_query(messages)

    span = agent_spans(span_exporter)[0]
    assert span.attributes["gen_ai.response.finish_reasons"] == (expected,)


@pytest.mark.skipif(
    not _assistant_reports_usage(),
    reason="this claude-agent-sdk release reports usage only on ResultMessage",
)
@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"input_tokens": 7, "output_tokens": 9}, {"input": 7, "output": 9}),
        ({"input_tokens": 0, "output_tokens": 0}, {"input": 0, "output": 0}),
        ({"input_tokens": 7}, {"input": 7}),
    ],
)
def test_usage_falls_back_to_assistant_messages(
    span_exporter, instrument_claude_agent_sdk, usage, expected
) -> None:
    # No ResultMessage: the run ended early, so the per-message usage is the
    # only available total.
    message = assistant(
        [{"type": "text", "text": "hi"}], stop_reason="end_turn"
    )
    message["message"]["usage"] = usage
    run_query([_message(), message])

    span = agent_spans(span_exporter)[0]
    assert span.attributes.get("gen_ai.usage.input_tokens") == expected.get(
        "input"
    )
    assert span.attributes.get("gen_ai.usage.output_tokens") == expected.get(
        "output"
    )


def test_result_without_usage_exports_no_tokens(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    # Assistant messages report usage, but the terminal ResultMessage carries
    # none: the fallback must not turn unavailable totals into token counts.
    messages = single_turn()
    messages[-1] = result(usage=None)
    run_query(messages)

    span = agent_spans(span_exporter)[0]
    assert "gen_ai.usage.input_tokens" not in span.attributes
    assert "gen_ai.usage.output_tokens" not in span.attributes


def test_result_error_marks_agent_span_failed(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    # The SDK reports a failed run through ResultMessage.is_error without
    # raising, so the span must not stay UNSET.
    messages = single_turn()
    messages[-1] = result(
        usage=(20, 38),
        is_error=True,
        errors=["the agent run failed"],
    )
    run_query(messages)

    agent = agent_spans(span_exporter)[0]
    assert agent.status.status_code.name == "ERROR"
    assert agent.attributes["error.type"] == "error_during_execution"
    assert agent.attributes["gen_ai.usage.input_tokens"] == 20
    assert agent.attributes["gen_ai.usage.output_tokens"] == 38
    assert exception_events(agent) == []


@pytest.mark.skipif(
    not _result_reports_errors(),
    reason="this claude-agent-sdk release does not report errors",
)
def test_result_error_description_gated_by_capture(
    span_exporter, capture
) -> None:
    # The per-error text is user data, so it only reaches the status
    # description when content capture is on; the subtype stays otherwise.
    messages = single_turn()
    messages[-1] = result(
        usage=(20, 38),
        is_error=True,
        errors=["the agent run failed"],
    )
    run_query(messages)

    agent = agent_spans(span_exporter)[0]
    assert agent.status.status_code.name == "ERROR"
    assert agent.attributes["error.type"] == "error_during_execution"
    assert agent.status.description == (
        "the agent run failed" if capture else "error_during_execution"
    )


@pytest.mark.skipif(
    not _result_reports_errors(),
    reason="this claude-agent-sdk release does not report errors",
)
@pytest.mark.parametrize(
    ("mode", "hook"),
    [("EVENT_ONLY", False), ("NO_CONTENT", True)],
)
def test_result_error_description_stays_safe_without_span_capture(
    span_exporter, tracer_provider, monkeypatch, mode, hook
) -> None:
    # EVENT_ONLY and a completion hook both turn on should_capture_content
    # without allowing span content; the raw error text must stay out of the
    # status description.
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", mode
    )
    instrumentor = claude_agent_sdk_instrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        completion_hook=_RecordingHook() if hook else None,
    )
    try:
        messages = single_turn()
        messages[-1] = result(
            usage=(20, 38),
            is_error=True,
            errors=["the agent run failed"],
        )
        run_query(messages)
    finally:
        instrumentor.uninstrument()

    agent = agent_spans(span_exporter)[0]
    assert agent.status.status_code.name == "ERROR"
    assert agent.status.description == "error_during_execution"


@pytest.mark.skipif(
    not _result_reports_errors(),
    reason="this claude-agent-sdk release does not report errors",
)
def test_result_error_uses_capture_snapshot(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    # The span-content decision is snapshotted when the instrumentor starts;
    # an environment change afterwards must not put error text on the span.
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT"
    )
    instrumentor = claude_agent_sdk_instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        monkeypatch.setenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
        )
        messages = single_turn()
        messages[-1] = result(
            usage=(20, 38),
            is_error=True,
            errors=["the agent run failed"],
        )
        run_query(messages)
    finally:
        instrumentor.uninstrument()

    agent = agent_spans(span_exporter)[0]
    assert agent.status.status_code.name == "ERROR"
    assert agent.status.description == "error_during_execution"


def test_stream_error_marks_agent_span_failed(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    messages = single_turn()
    with pytest.raises(Exception, match="transport broke") as excinfo:
        run_query(messages, fail_after=2)

    agent = agent_spans(span_exporter)[0]
    assert agent.status.status_code.name == "ERROR"
    # Newer SDK releases surface this as RuntimeError, older ones raise a plain
    # Exception; the span must report whichever reached the caller.
    assert agent.attributes["error.type"] == type(excinfo.value).__name__


def test_query_argument_error_ends_span_and_restores_context(
    span_exporter, tracer_provider, instrument_claude_agent_sdk
) -> None:
    tracer = tracer_provider.get_tracer("test")
    with tracer.start_as_current_span("caller") as caller:
        with pytest.raises(TypeError):
            claude_agent_sdk.query()
        # The failed invocation must not stay current in the caller's context.
        assert trace.get_current_span() is caller

    (agent,) = agent_spans(span_exporter)
    assert agent.status.status_code.name == "ERROR"
    assert agent.attributes["error.type"] == "TypeError"


def test_caller_closing_early_ends_span(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    async def _run() -> None:
        stream = open_stream(single_turn())
        async for _ in stream:
            break
        await stream.aclose()

    anyio.run(_run)

    spans = agent_spans(span_exporter)
    assert len(spans) == 1
    # Early-close semantics are tracked upstream; pin today's behaviour only.
    assert spans[0].status.status_code.name == "UNSET"


def test_thinking_block_as_reasoning_part(span_exporter, capture) -> None:
    run_query(thinking_turn())

    span = agent_spans(span_exporter)[0]
    if not capture:
        assert "gen_ai.output.messages" not in span.attributes
        return
    messages = json.loads(span.attributes["gen_ai.output.messages"])
    assert messages[0]["parts"][0] == {
        "content": "Let me read the file first.",
        "type": "reasoning",
    }


def test_input_prompt_capture(span_exporter, capture) -> None:
    run_query(single_turn(), prompt="Read /tmp/a.txt")

    span = agent_spans(span_exporter)[0]
    if not capture:
        assert "gen_ai.input.messages" not in span.attributes
        return
    messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert messages[0]["role"] == "user"
    assert messages[0]["parts"] == [
        {"content": "Read /tmp/a.txt", "type": "text"}
    ]


def test_output_messages_capture(span_exporter, capture) -> None:
    run_query(single_turn())

    span = agent_spans(span_exporter)[0]
    if not capture:
        assert "gen_ai.output.messages" not in span.attributes
        return
    messages = json.loads(span.attributes["gen_ai.output.messages"])
    # One message per SDK message keeps the turn boundaries the model saw.
    assert [message["role"] for message in messages] == [
        "assistant",
        "tool",
        "assistant",
    ]
    text_part, call_part = messages[0]["parts"]
    assert text_part == {"content": "Reading the file.", "type": "text"}
    assert call_part == {
        "arguments": {"file_path": "/tmp/a.txt"},
        "name": "Read",
        "id": "toolu_1",
        "type": "tool_call",
    }
    assert messages[1]["parts"] == [
        {"response": "hello", "id": "toolu_1", "type": "tool_call_response"}
    ]
    assert messages[2]["parts"] == [{"content": "Done: hello", "type": "text"}]


def test_agent_span_starts_in_creator_context(
    span_exporter, tracer_provider, instrument_claude_agent_sdk
) -> None:
    tracer = tracer_provider.get_tracer("test")
    with tracer.start_as_current_span("creator") as creator:
        stream = open_stream(single_turn())

    async def _drain() -> None:
        async for _ in stream:
            pass

    anyio.run(_drain)

    (agent,) = agent_spans(span_exporter)
    assert agent.parent is not None
    assert agent.parent.span_id == creator.get_span_context().span_id


def test_caller_context_restored_between_reads(
    span_exporter, tracer_provider, instrument_claude_agent_sdk
) -> None:
    tracer = tracer_provider.get_tracer("test")
    seen: list[int] = []

    async def _run() -> int:
        with tracer.start_as_current_span("caller") as caller:
            async for _ in open_stream(single_turn()):
                span_context = trace.get_current_span().get_span_context()
                seen.append(span_context.span_id)
            return caller.get_span_context().span_id

    caller_id = anyio.run(_run)

    assert seen
    assert set(seen) == {caller_id}
    (agent,) = agent_spans(span_exporter)
    assert agent.parent.span_id == caller_id


def test_caller_exception_inside_context_finalizes_agent_span(
    span_exporter, tracer_provider, instrument_claude_agent_sdk
) -> None:
    tracer = tracer_provider.get_tracer("test")

    async def _run() -> tuple[int, int]:
        with tracer.start_as_current_span("caller") as caller:
            with pytest.raises(CallerError):
                async with open_stream(single_turn()) as stream:
                    seen = 0
                    async for _ in stream:
                        seen += 1
                        if seen == 2:
                            raise CallerError("caller failed")
            current = trace.get_current_span().get_span_context()
            return caller.get_span_context().span_id, current.span_id

    caller_id, current_id = anyio.run(_run)

    # __aexit__ finalizes telemetry and restores the caller's context.
    assert current_id == caller_id
    (agent,) = agent_spans(span_exporter)
    assert agent.status.status_code.name == "ERROR"
    assert agent.attributes["error.type"].endswith("CallerError")


def test_cancellation_marks_agent_span_failed(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    async def _run() -> None:
        stream = open_stream(single_turn(), delay=5)
        with anyio.move_on_after(0.05):
            async for _ in stream:
                pass

    anyio.run(_run)

    (agent,) = agent_spans(span_exporter)
    assert agent.status.status_code.name == "ERROR"
    assert agent.attributes["error.type"].endswith("CancelledError")


def test_closing_twice_does_not_duplicate_spans(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    async def _run() -> None:
        stream = open_stream(single_turn())
        async for _ in stream:
            break
        await stream.aclose()
        await stream.aclose()

    anyio.run(_run)

    assert len(agent_spans(span_exporter)) == 1


def test_close_before_first_read_ends_span(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    async def _run() -> None:
        stream = open_stream(single_turn())
        await stream.aclose()

    anyio.run(_run)

    assert len(agent_spans(span_exporter)) == 1
