# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry ``gen_ai.conversation.id`` on the OTel Context.

A framework instrumentation writes the id. Inference, invoke_agent and
invoke_workflow invocations read it when the caller set none explicitly.
That is how a ``chat`` span from another instrumentation package, which
only sees the HTTP call, ends up with the id.
"""

from __future__ import annotations

from opentelemetry.context import (
    Context,
    create_key,
    get_value,
    set_value,
)

CONVERSATION_ID_KEY = create_key("opentelemetry.util.genai.conversation_id")


def get_ambient_conversation_id(
    context: Context | None = None,
) -> str | None:
    value = get_value(CONVERSATION_ID_KEY, context=context)
    return value if isinstance(value, str) else None


def with_conversation_id(
    conversation_id: str,
    context: Context | None = None,
) -> Context:
    return set_value(CONVERSATION_ID_KEY, conversation_id, context=context)


__all__ = [
    "CONVERSATION_ID_KEY",
    "get_ambient_conversation_id",
    "with_conversation_id",
]
