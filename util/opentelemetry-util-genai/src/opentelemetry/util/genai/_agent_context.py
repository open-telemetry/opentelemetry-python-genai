# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry ``gen_ai.agent.name`` on the OTel Context.

``LocalAgentInvocation`` publishes the active agent's name here so a nested
client inference invocation owned by another instrumentation package can
attribute its metrics to the executing agent.
"""

from __future__ import annotations

from opentelemetry.context import (
    Context,
    create_key,
    get_value,
    set_value,
)

_AGENT_NAME_KEY = create_key("opentelemetry.util.genai.agent_name")


def get_ambient_agent_name(
    context: Context | None = None,
) -> str | None:
    value = get_value(_AGENT_NAME_KEY, context=context)
    return value if isinstance(value, str) else None


def with_agent_name(
    agent_name: str | None,
    context: Context | None = None,
) -> Context:
    return set_value(_AGENT_NAME_KEY, agent_name or None, context=context)
