# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context helpers for GenAI inference attributes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from opentelemetry.context import Context, get_value, set_value

if TYPE_CHECKING:
    from opentelemetry.util.genai._inference_invocation import (
        InferenceNonContentCaptureData,
    )

INFERENCE_CONTEXT_KEY: Final[str] = "opentelemetry.genai.inference_context"
_INFERENCE_CONTEXT_KEY = INFERENCE_CONTEXT_KEY


def set_inference_context_data(
    data: InferenceNonContentCaptureData,
    context: Context | None = None,
) -> Context:
    """Return a Context with the given inference context data attached.

    Args:
        data: The mutable inference non-content capture data object.
        context: The context to attach to. Defaults to the current context.

    Returns:
        A new Context containing the inference context data.
    """
    return set_value(_INFERENCE_CONTEXT_KEY, data, context=context)


def get_inference_context_data(
    context: Context | None = None,
) -> InferenceNonContentCaptureData | None:
    """Return the active inference context data from context, if any.

    Args:
        context: The context to inspect. Defaults to the current context.

    Returns:
        The active inference context data, or None if not set.
    """
    from opentelemetry.util.genai._inference_invocation import (
        InferenceNonContentCaptureData,
    )

    data = get_value(_INFERENCE_CONTEXT_KEY, context=context)
    if isinstance(data, InferenceNonContentCaptureData):
        return data
    return None


from opentelemetry.util.genai._inference_invocation import (
    InferenceNonContentCaptureData,
)

__all__ = [
    "INFERENCE_CONTEXT_KEY",
    "InferenceNonContentCaptureData",
    "get_inference_context_data",
    "set_inference_context_data",
]
