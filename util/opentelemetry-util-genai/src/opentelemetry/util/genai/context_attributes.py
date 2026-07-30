# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach attributes to a context so GenAI telemetry emitted within it carries them.

This lets a caller that knows something the instrumentation cannot -- which
agent is running, which user made the request -- add it to telemetry emitted
further down the stack. Each attribute targets a signal, so data that is unsafe
on a span can still go on the event.

.. code-block:: python

    token = context.attach(
        set_context_scoped_attributes(
            span_attributes={"gen_ai.agent.name": "trip-planner"},
            log_attributes={"user.id": user_id},
        )
    )
    try:
        client.chat.completions.create(...)  # instrumented elsewhere
    finally:
        context.detach(token)

Only telemetry from this package is affected, and attributes never leave the
process. Metrics are not supported, to avoid unbounded cardinality.

The naming follows `OTEP 4931
<https://github.com/open-telemetry/opentelemetry-specification/blob/main/oteps/4931-context-scoped-attributes.md>`_,
but this is narrower: that OTEP has the SDK stamp all telemetry and gates it
per signal on the provider.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, NamedTuple

from opentelemetry.context import (
    Context,
    create_key,
    get_current,
    get_value,
    set_value,
)
from opentelemetry.util.types import AttributeValue

__all__ = ["set_context_scoped_attributes"]

_CONTEXT_SCOPED_ATTRIBUTES_KEY = create_key(
    "opentelemetry.util.genai.context_scoped_attributes"
)

_EMPTY: Mapping[str, AttributeValue] = MappingProxyType({})


class _ContextScopedAttributes(NamedTuple):
    """The per-signal attribute bags carried by a single context."""

    span: Mapping[str, AttributeValue] = _EMPTY
    log: Mapping[str, AttributeValue] = _EMPTY


_NO_ATTRIBUTES = _ContextScopedAttributes()


def set_context_scoped_attributes(
    *,
    span_attributes: Mapping[str, AttributeValue] | None = None,
    log_attributes: Mapping[str, AttributeValue] | None = None,
    context: Context | None = None,
) -> Context:
    """Return a new context carrying the given attributes. Does not attach it.

    Args:
        span_attributes: Added to GenAI spans on span start.
        log_attributes: Added to the GenAI logs emitted.
        context: Base context. Defaults to the current context.
    """
    if not span_attributes and not log_attributes:
        return context if context is not None else get_current()

    existing = _get_context_scoped_attributes(context)
    merged = _ContextScopedAttributes(
        span=MappingProxyType({**existing.span, **(span_attributes or {})}),
        log=MappingProxyType({**existing.log, **(log_attributes or {})}),
    )
    return set_value(_CONTEXT_SCOPED_ATTRIBUTES_KEY, merged, context)


def _get_context_scoped_attributes(
    context: Context | None = None,
) -> _ContextScopedAttributes:
    """Return the attribute bags on the given context, or empty ones."""
    value = get_value(_CONTEXT_SCOPED_ATTRIBUTES_KEY, context)
    if isinstance(value, _ContextScopedAttributes):
        return value
    return _NO_ATTRIBUTES
