# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a smolagents model instance to a ``gen_ai.provider.name`` value.

Only the in-process model classes are instrumented here, so the value comes
from the model class name.

``gen_ai.provider.name`` is a metric attribute as well as a span attribute, so
every value has to stay low cardinality. ``TelemetryHandler.inference`` requires
``provider`` as a string, so this always returns a value rather than ``None``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents.models import Model

_logger = logging.getLogger(__name__)

_UNKNOWN_PROVIDER = "unknown"

# Model class name -> provider value. The GenAI registry has no value for the
# Hugging Face, vLLM, and MLX runtimes, so those use the product name; a class
# name would look like a provider without being one.
_CLASS_NAME_TO_PROVIDER: dict[str, str] = {
    "TransformersModel": "huggingface",
    "VLLMModel": "vllm",
    "MLXModel": "mlx",
}


def resolve_provider(instance: Model) -> str:
    """Return the ``gen_ai.provider.name`` value for a smolagents model instance."""
    # An instrumented model can be a user subclass of a patched class. Matching
    # the exact class name alone would report ``unknown`` for every subclass, so
    # walk the hierarchy, most derived class first.
    class_names = [cls.__name__ for cls in type(instance).__mro__]

    for class_name in class_names:
        provider = _CLASS_NAME_TO_PROVIDER.get(class_name)
        if provider is not None:
            return provider

    _logger.debug(
        "No known gen_ai.provider.name for model class %s", class_names[0]
    )
    return _UNKNOWN_PROVIDER
