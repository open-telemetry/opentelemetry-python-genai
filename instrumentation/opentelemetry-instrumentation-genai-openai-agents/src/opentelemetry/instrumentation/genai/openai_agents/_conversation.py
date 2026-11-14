# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context propagation helpers for openai-agents Runner conversation_id and session."""

from __future__ import annotations

import contextvars
from typing import Any

from agents.memory.session import Session

try:
    from agents.run_state import RunState
except ImportError:
    RunState = None

__all__ = [
    "_current_conversation_id",
    "_current_session",
    "_extract_conversation_id",
    "_get_active_conversation_id",
]

_current_conversation_id: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("_current_conversation_id", default=None)
)
_current_session: contextvars.ContextVar[Session | None] = (
    contextvars.ContextVar("_current_session", default=None)
)


def _extract_conversation_id(arguments: dict[str, Any]) -> str | None:
    # 1. Explicit conversation_id parameter
    conversation_id = arguments.get("conversation_id")
    if not conversation_id and isinstance(arguments.get("kwargs"), dict):
        conversation_id = arguments["kwargs"].get("conversation_id")
    if conversation_id:
        return str(conversation_id)

    # 2. Resumed RunState passed as input
    input_arg = arguments.get("input")
    if not input_arg and isinstance(arguments.get("kwargs"), dict):
        input_arg = arguments["kwargs"].get("input")
    if (
        RunState is not None
        and isinstance(input_arg, RunState)
        and input_arg._conversation_id
    ):
        return str(input_arg._conversation_id)

    return None


def _get_active_conversation_id() -> str | None:
    conv_id = _current_conversation_id.get()
    if conv_id:
        return conv_id

    session = _current_session.get()
    if session is not None:
        try:
            return session.session_id
        except ValueError:
            return None

    return None
