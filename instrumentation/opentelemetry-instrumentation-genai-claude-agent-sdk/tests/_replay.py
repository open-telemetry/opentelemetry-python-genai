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
    cache_read: int | None = None,
    cache_creation: int | None = None,
    session_id: str = "session-1",
    model: str = "claude-sonnet-4-5",
    stop_reason: str = "tool_use",
) -> dict[str, Any]:
    tokens: dict[str, Any] = {
        "input_tokens": usage[0],
        "output_tokens": usage[1],
    }
    if cache_read is not None:
        tokens["cache_read_input_tokens"] = cache_read
    if cache_creation is not None:
        tokens["cache_creation_input_tokens"] = cache_creation
    return {
        "type": "assistant",
        "session_id": session_id,
        "message": {
            "id": "msg",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "usage": tokens,
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
    cache_read: int | None = None,
    cache_creation: int | None = None,
    session_id: str = "session-1",
    is_error: bool = False,
    subtype: str | None = None,
    errors: list[str] | None = None,
    stop_reason: str | None = "end_turn",
) -> dict[str, Any]:
    tokens: dict[str, Any] = {
        "input_tokens": usage[0],
        "output_tokens": usage[1],
    }
    if cache_read is not None:
        tokens["cache_read_input_tokens"] = cache_read
    if cache_creation is not None:
        tokens["cache_creation_input_tokens"] = cache_creation
    message: dict[str, Any] = {
        "type": "result",
        "subtype": subtype
        or ("error_during_execution" if is_error else "success"),
        "session_id": session_id,
        "duration_ms": 12,
        "duration_api_ms": 10,
        "is_error": is_error,
        "num_turns": 1,
        "result": "done",
        "total_cost_usd": 0.0,
        "usage": tokens,
    }
    if stop_reason is not None:
        message["stop_reason"] = stop_reason
    if errors is not None:
        message["errors"] = errors
    return message


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
