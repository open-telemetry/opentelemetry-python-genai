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

from opentelemetry import trace

from ._replay import (
    ReplayTransport,
    _message,
    assistant,
    result,
    single_turn,
    two_tools_in_one_message,
)

AGENT_SPAN = "invoke_agent claude-agent-sdk"
OTHER_ERROR = "_OTHER"


def claude_agent_sdk_instrumentor():
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )

    return ClaudeAgentSDKInstrumentor()


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


def agent_spans(span_exporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.name.startswith("invoke_agent")
    ]


def tool_spans(span_exporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.name.startswith("execute_tool")
    ]


def exception_events(span) -> list[Any]:
    return [
        event for event in span.events if event.name.endswith(".exception")
    ]


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
    # GenAI input tokens include cache reads and cache writes; the local agent
    # span has no attribute of its own to report them separately.
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


def _assistant_reports_usage() -> bool:
    """True when this SDK release parses usage onto AssistantMessage."""
    return "usage" in inspect.signature(sdk_types.AssistantMessage).parameters


@pytest.mark.skipif(
    not _assistant_reports_usage(),
    reason="this claude-agent-sdk release reports usage only on ResultMessage",
)
def test_usage_falls_back_to_assistant_messages_without_result(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    # No ResultMessage: the run ended early, so the per-message usage is the
    # only available total.
    run_query(
        [
            _message(),
            assistant(
                [{"type": "text", "text": "hi"}],
                usage=(7, 9),
                stop_reason="end_turn",
            ),
        ]
    )

    span = agent_spans(span_exporter)[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 9


def test_builtin_tool_span_nests_under_agent(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(single_turn(tool_name="Read"))

    agent = agent_spans(span_exporter)[0]
    (tool,) = tool_spans(span_exporter)
    assert tool.name == "execute_tool Read"
    assert tool.attributes["gen_ai.tool.name"] == "Read"
    assert tool.attributes["gen_ai.tool.type"] == "function"
    assert tool.attributes["gen_ai.tool.call.id"] == "toolu_1"
    # The executing agent is known here, so the conditionally required
    # attribute must be reported on the tool span and its metrics.
    assert tool.attributes["gen_ai.agent.name"] == "claude-agent-sdk"
    assert tool.parent is not None
    assert tool.parent.span_id == agent.context.span_id


def test_two_tools_in_one_message_are_siblings(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(two_tools_in_one_message())

    agent = agent_spans(span_exporter)[0]
    first, second = sorted(
        tool_spans(span_exporter),
        key=lambda span: span.attributes["gen_ai.tool.call.id"],
    )
    for tool in (first, second):
        assert tool.parent is not None
        assert tool.parent.span_id == agent.context.span_id
    assert first.context.span_id != second.context.span_id
    assert first.parent.span_id != second.context.span_id


def test_tool_result_error_marks_tool_span_failed(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(single_turn(tool_error=True))

    (tool,) = tool_spans(span_exporter)
    assert tool.status.status_code.name == "ERROR"
    # A tool-result error flag is not a raised exception: report the semconv
    # fallback type and never fabricate an exception event.
    assert tool.attributes["error.type"] == OTHER_ERROR
    assert tool.status.description == "hello"
    assert exception_events(tool) == []


def test_stream_error_marks_agent_and_tool_failed(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    messages = single_turn()
    with pytest.raises(Exception, match="transport broke") as excinfo:
        run_query(messages, fail_after=2)

    agent = agent_spans(span_exporter)[0]
    (tool,) = tool_spans(span_exporter)
    assert agent.status.status_code.name == "ERROR"
    # Newer SDK releases surface this as RuntimeError, older ones raise a plain
    # Exception; the span must report whichever reached the caller.
    assert agent.attributes["error.type"] == type(excinfo.value).__name__
    assert tool.status.status_code.name == "ERROR"
    assert tool.attributes["error.type"] == OTHER_ERROR


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
    # Releases without per-error messages fall back to the subtype as the
    # description; either way it must not stay empty.
    assert agent.status.description in {
        "the agent run failed",
        "error_during_execution",
    }
    # Usage and output still land on the span.
    assert agent.attributes["gen_ai.usage.input_tokens"] == 20
    assert agent.attributes["gen_ai.usage.output_tokens"] == 38
    # No exception happened, so no exception event may be recorded.
    assert exception_events(agent) == []


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


def test_closing_with_pending_tool_fails_tool_span(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    messages = [
        _message(),
        assistant(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a.txt"},
                }
            ],
            stop_reason="tool_use",
        ),
    ]

    async def _run() -> None:
        stream = open_stream(messages)
        seen = 0
        async for _ in stream:
            seen += 1
            if seen == 2:
                break
        await stream.aclose()

    anyio.run(_run)

    (agent,) = agent_spans(span_exporter)
    (tool,) = tool_spans(span_exporter)
    assert len(agent_spans(span_exporter)) == 1
    assert len(tool_spans(span_exporter)) == 1
    # The tool_result never arrived, so the interval cannot claim success.
    assert tool.status.status_code.name == "ERROR"
    assert tool.attributes["error.type"] == OTHER_ERROR
    assert tool.status.description == "tool result was not observed"


def test_output_messages_captured_when_enabled(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    instrumentor = claude_agent_sdk_instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        run_query(single_turn())
    finally:
        instrumentor.uninstrument()

    span = agent_spans(span_exporter)[0]
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


def test_input_prompt_captured_when_enabled(
    span_exporter, tracer_provider, monkeypatch
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    instrumentor = claude_agent_sdk_instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        run_query(single_turn(), prompt="Read /tmp/a.txt")
    finally:
        instrumentor.uninstrument()

    span = agent_spans(span_exporter)[0]
    messages = json.loads(span.attributes["gen_ai.input.messages"])
    assert messages[0]["role"] == "user"
    assert messages[0]["parts"] == [
        {"content": "Read /tmp/a.txt", "type": "text"}
    ]


def test_no_content_captured_by_default(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(single_turn())

    span = agent_spans(span_exporter)[0]
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
    (tool,) = tool_spans(span_exporter)
    assert "gen_ai.tool.call.arguments" not in tool.attributes
    assert "gen_ai.tool.call.result" not in tool.attributes


def test_agent_span_starts_on_first_read(
    span_exporter, tracer_provider, instrument_claude_agent_sdk
) -> None:
    tracer = tracer_provider.get_tracer("test")
    # A stream created inside one context and read outside it belongs to the
    # context that reads it, not the one that created it.
    with tracer.start_as_current_span("creator"):
        stream = open_stream(single_turn())

    async def _drain() -> None:
        async for _ in stream:
            pass

    anyio.run(_drain)

    (agent,) = agent_spans(span_exporter)
    assert agent.parent is None


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
    assert len(tool_spans(span_exporter)) <= 1
