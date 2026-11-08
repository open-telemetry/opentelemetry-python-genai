# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay fixtures for the Claude Agent SDK tests.

The SDK normally talks to the ``claude`` CLI over a subprocess transport. These
helpers replace it with a fixed message stream so the instrumentation can be
exercised in-process: no CLI, no network, no API key.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import anyio
from claude_agent_sdk._internal.transport import Transport


def _message(
    *,
    session_id: str = "session-1",
    model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "init",
        "session_id": session_id,
        "tools": ["Read"],
        "model": model,
    }


def assistant(
    content: list[dict[str, Any]],
    *,
    usage: tuple[int, int] = (12, 30),
    session_id: str = "session-1",
    model: str = "claude-sonnet-4-5",
    stop_reason: str = "tool_use",
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "session_id": session_id,
        "message": {
            "id": "msg",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        },
    }


def tool_result(
    tool_use_id: str,
    content: str = "hello",
    *,
    is_error: bool = False,
    session_id: str = "session-1",
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return {
        "type": "user",
        "session_id": session_id,
        "message": {"role": "user", "content": [block]},
    }


def result(
    *,
    usage: tuple[int, int] = (20, 38),
    session_id: str = "session-1",
    is_error: bool = False,
) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "error_during_execution" if is_error else "success",
        "session_id": session_id,
        "duration_ms": 12,
        "duration_api_ms": 10,
        "is_error": is_error,
        "num_turns": 1,
        "result": "done",
        "total_cost_usd": 0.0,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


def single_turn(
    *,
    assistant_usage: tuple[int, int] = (12, 30),
    result_usage: tuple[int, int] = (20, 38),
    tool_name: str = "Read",
    tool_use_id: str = "toolu_1",
    tool_error: bool = False,
) -> list[dict[str, Any]]:
    """One agent turn: text, a tool call, its result, then the run result."""
    return [
        _message(),
        assistant(
            [
                {"type": "text", "text": "Reading the file."},
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": {"file_path": "/tmp/a.txt"},
                },
            ],
            usage=assistant_usage,
        ),
        tool_result(tool_use_id, is_error=tool_error),
        assistant(
            [{"type": "text", "text": "Done: hello"}],
            usage=(8, 8),
            stop_reason="end_turn",
        ),
        result(usage=result_usage),
    ]


def two_turns(
    *,
    first_usage: tuple[int, int] = (32, 38),
    second_usage: tuple[int, int] = (10, 9),
) -> list[dict[str, Any]]:
    """Two turns in one stream: each ends with its own ResultMessage."""
    return [
        _message(),
        assistant(
            [
                {"type": "text", "text": "Reading the file."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"file_path": "/tmp/a.txt"},
                },
            ],
            usage=(12, 30),
        ),
        tool_result("toolu_1"),
        assistant(
            [{"type": "text", "text": "Done: hello"}],
            usage=(8, 8),
            stop_reason="end_turn",
        ),
        result(usage=first_usage),
        assistant(
            [{"type": "text", "text": "Second turn"}],
            usage=(10, 9),
            stop_reason="end_turn",
        ),
        result(usage=second_usage),
    ]


class ReplayTransport(Transport):
    """Replays a fixed message stream; answers control requests in place."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        delay: float = 0,
        fail_after: int | None = None,
    ) -> None:
        self.messages = copy.deepcopy(messages)
        self.delay = delay
        self.fail_after = fail_after
        self.pending: list[dict[str, Any]] = []

    async def connect(self) -> None:
        return None

    async def write(self, data: str) -> None:
        payload = json.loads(data)
        if payload.get("type") == "control_request":
            self.pending.append(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": payload.get("request_id"),
                        "response": {"ok": True},
                    },
                }
            )

    def read_messages(self):
        async def gen():
            for index, message in enumerate(self.messages):
                if self.fail_after is not None and index >= self.fail_after:
                    raise RuntimeError("transport broke")
                while self.pending:
                    yield self.pending.pop(0)
                if self.delay:
                    await anyio.sleep(self.delay)
                yield message
            while self.pending:
                yield self.pending.pop(0)

        return gen()

    async def close(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    async def end_input(self) -> None:
        return None


def spans_by_name(span_exporter) -> dict[str, list[Any]]:
    """Group finished spans by name for assertions."""
    grouped: dict[str, list[Any]] = {}
    for span in span_exporter.get_finished_spans():
        grouped.setdefault(span.name, []).append(span)
    return grouped
