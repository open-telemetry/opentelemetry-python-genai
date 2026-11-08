# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Span mapping tests for the Claude Agent SDK instrumentation.

The SDK is driven through a replay transport (see ``_replay.py``), so every test
runs in-process without the ``claude`` CLI.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import claude_agent_sdk
import pytest
from claude_agent_sdk.types import ClaudeAgentOptions

from ._replay import ReplayTransport, single_turn

AGENT_SPAN = "invoke_agent claude-agent-sdk"


def claude_agent_sdk_instrumentor():
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )

    return ClaudeAgentSDKInstrumentor()


def run_query(
    messages: list[dict[str, Any]],
    *,
    options: Any = None,
    **transport_kwargs: Any,
) -> None:
    async def _run() -> None:
        async for _ in claude_agent_sdk.query(
            prompt="Read /tmp/a.txt",
            options=options or ClaudeAgentOptions(),
            transport=ReplayTransport(messages, **transport_kwargs),
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
    assert tool.parent is not None
    assert tool.parent.span_id == agent.context.span_id


def test_tool_result_error_marks_tool_span_failed(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    run_query(single_turn(tool_error=True))

    (tool,) = tool_spans(span_exporter)
    assert tool.status.status_code.name == "ERROR"


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


def test_caller_closing_early_ends_span(
    span_exporter, instrument_claude_agent_sdk
):
    async def _run() -> None:
        stream = claude_agent_sdk.query(
            prompt="x",
            options=ClaudeAgentOptions(),
            transport=ReplayTransport(single_turn()),
        )
        async for _ in stream:
            break
        await stream.aclose()

    anyio.run(_run)

    spans = agent_spans(span_exporter)
    assert len(spans) == 1
    # Early-close semantics are tracked upstream; pin today's behaviour only.
    assert spans[0].status.status_code.name == "UNSET"


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
    text = messages[0]["parts"][0]["content"]
    assert "Reading the file." in text
    assert "Done: hello" in text


def test_closing_twice_does_not_duplicate_spans(
    span_exporter, instrument_claude_agent_sdk
) -> None:
    async def _run() -> None:
        stream = claude_agent_sdk.query(
            prompt="x",
            options=ClaudeAgentOptions(),
            transport=ReplayTransport(single_turn()),
        )
        async for _ in stream:
            break
        await stream.aclose()
        await stream.aclose()

    anyio.run(_run)

    assert len(agent_spans(span_exporter)) == 1
