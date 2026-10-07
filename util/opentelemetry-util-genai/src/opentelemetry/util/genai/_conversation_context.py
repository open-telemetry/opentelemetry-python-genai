# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry ``gen_ai.conversation.id`` on the OTel Context.

``GenAIInvocation._start()`` resolves the id and puts it back here, so a
``chat`` span owned by another instrumentation package -- which only sees
the HTTP call -- ends up with the id its enclosing agent run established.
Callers reach this through ``handler.*(conversation_id=...)``.
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
