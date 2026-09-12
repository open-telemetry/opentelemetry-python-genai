# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context helpers for GenAI inference attributes."""

from __future__ import annotations

from typing import Final, cast

from opentelemetry.context import Context, get_value, set_value
from opentelemetry.util.types import AttributeValue

INFERENCE_ATTRIBUTES_KEY: Final[str] = (
    "opentelemetry.genai.inference_attributes"
)
_INFERENCE_ATTRIBUTES_KEY = INFERENCE_ATTRIBUTES_KEY

__all__ = [
    "INFERENCE_ATTRIBUTES_KEY",
    "get_inference_attributes",
    "set_inference_attributes",
]


def set_inference_attributes(
    attributes: dict[str, AttributeValue],
    context: Context | None = None,
) -> Context:
    """Return a Context with the given inference attributes dictionary attached.

    Args:
        attributes: The mutable inference attributes dictionary.
        context: The context to attach to. Defaults to the current context.

    Returns:
        A new Context containing the inference attributes dictionary.
    """
    return set_value(_INFERENCE_ATTRIBUTES_KEY, attributes, context=context)


def get_inference_attributes(
    context: Context | None = None,
) -> dict[str, AttributeValue] | None:
    """Return the active inference attributes dictionary from context, if any.

    Args:
        context: The context to inspect. Defaults to the current context.

    Returns:
        The active inference attributes dictionary, or None if not set.
    """
    attrs = get_value(_INFERENCE_ATTRIBUTES_KEY, context=context)
    if isinstance(attrs, dict):
        return cast("dict[str, AttributeValue]", attrs)
    return None
