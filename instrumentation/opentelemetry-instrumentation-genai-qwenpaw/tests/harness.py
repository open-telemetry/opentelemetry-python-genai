# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers to drive ``AgentRunner.query_handler`` without a real agent.

Tests route a slash-command turn (``/stop``) through ``query_handler`` and
replace the module-level ``run_command_path`` coroutine with a canned async
generator, so no model, tool, or network access is involved.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Callable, Iterator
from unittest import mock


def assistant_reply(text: str) -> Any:
    from agentscope.message import Msg, TextBlock  # noqa: PLC0415

    return Msg(
        name="Friday",
        role="assistant",
        content=[TextBlock(type="text", text=text)],
    )


def user_command_msgs(text: str = "/stop") -> list[Any]:
    from agentscope.message import Msg  # noqa: PLC0415

    return [Msg(name="user", role="user", content=text)]


def make_request(
    session_id: str = "sess-1",
    user_id: str = "user-2",
    channel: str = "console",
) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id, user_id=user_id, channel=channel
    )


def fake_run_command_path(
    response_text: str = "ok",
) -> Callable[..., AsyncGenerator[Any, None]]:
    """A ``run_command_path`` stand-in yielding one assistant reply."""

    async def _fake(request: Any, msgs: Any, runner: Any):
        del request, msgs, runner
        yield assistant_reply(response_text), True

    return _fake


def failing_run_command_path(
    error: BaseException,
    response_text: str = "partial",
) -> Callable[..., AsyncGenerator[Any, None]]:
    """A ``run_command_path`` stand-in yielding once, then raising *error*."""

    async def _fake(request: Any, msgs: Any, runner: Any):
        del request, msgs, runner
        yield assistant_reply(response_text), False
        raise error

    return _fake


@contextmanager
def patched_command_path(runner_module: Any, fake: Any) -> Iterator[None]:
    """Point the runner module's command path at *fake*.

    Older runtimes resolve pending approvals before dispatching commands;
    neutralize that step when present so the turn always reaches
    ``run_command_path``.
    """

    async def fake_resolve(self: Any, session_id: Any, query: Any):
        del self, session_id, query
        return (None, False, None)

    with ExitStack() as stack:
        if hasattr(runner_module.AgentRunner, "_resolve_pending_approval"):
            stack.enter_context(
                mock.patch.object(
                    runner_module.AgentRunner,
                    "_resolve_pending_approval",
                    fake_resolve,
                )
            )
        stack.enter_context(
            mock.patch.object(runner_module, "run_command_path", fake)
        )
        yield
