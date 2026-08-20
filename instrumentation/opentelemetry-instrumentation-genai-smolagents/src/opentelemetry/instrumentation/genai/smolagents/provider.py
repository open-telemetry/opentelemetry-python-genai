# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a smolagents model instance to a ``gen_ai.provider.name`` value.

``chat`` spans are emitted for the in-process model classes only, but
``gen_ai.client.operation.duration`` of an agent run needs the attribute too,
and an agent can be given any model class.

``gen_ai.provider.name`` is a metric attribute as well as a span attribute, so
every value has to stay low cardinality. ``TelemetryHandler.inference``
requires ``provider`` as a string, so this always returns a value rather than
``None``.

Resolution order:

1. For the LiteLLM model classes, the ``model_id`` vendor prefix
   (``anthropic/claude-...`` -> ``anthropic``).
2. The model class name (e.g. ``OpenAIModel`` -> ``openai``), looked up along the
   class hierarchy so a user subclass resolves to the provider of its base class.
3. ``unknown``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

if TYPE_CHECKING:
    from smolagents.models import Model

_logger = logging.getLogger(__name__)

_PROVIDER = GenAI.GenAiProviderNameValues

_UNKNOWN_PROVIDER = "unknown"

# Model class name -> provider value. The GenAI registry has no value for the
# Hugging Face, vLLM, and MLX runtimes, so those use the product name; a class
# name would look like a provider without being one.
_CLASS_NAME_TO_PROVIDER: dict[str, str] = {
    "OpenAIModel": _PROVIDER.OPENAI.value,
    "AzureOpenAIModel": _PROVIDER.AZURE_AI_OPENAI.value,
    "AmazonBedrockModel": _PROVIDER.AWS_BEDROCK.value,
    "InferenceClientModel": "huggingface",
    "TransformersModel": "huggingface",
    "VLLMModel": "vllm",
    "MLXModel": "mlx",
}

# LiteLLM model_id prefix -> semconv provider value, for the prefixes whose
# LiteLLM vendor slug differs from the semconv value. Every other prefix is
# passed through as-is (``ollama/llama3`` -> ``ollama``). LiteLLM's slugs are a
# closed vocabulary, which keeps the cardinality bounded.
_LITELLM_PREFIX_TO_PROVIDER: dict[str, str] = {
    "azure": _PROVIDER.AZURE_AI_OPENAI.value,
    "azure_ai": _PROVIDER.AZURE_AI_INFERENCE.value,
    "bedrock": _PROVIDER.AWS_BEDROCK.value,
    "gemini": _PROVIDER.GCP_GEMINI.value,
    "mistral": _PROVIDER.MISTRAL_AI.value,
    "vertex_ai": _PROVIDER.GCP_VERTEX_AI.value,
    "watsonx": _PROVIDER.IBM_WATSONX_AI.value,
    "xai": _PROVIDER.X_AI.value,
}

_LITELLM_CLASS_NAMES = frozenset({"LiteLLMModel", "LiteLLMRouterModel"})


def _provider_from_litellm(instance: Model) -> str | None:
    model_id = instance.model_id
    if model_id is None or "/" not in model_id:
        return None
    prefix = model_id.split("/", 1)[0].lower()
    return _LITELLM_PREFIX_TO_PROVIDER.get(prefix, prefix)


def resolve_provider(instance: Model) -> str:
    """Return the ``gen_ai.provider.name`` value for a smolagents model instance."""
    # An instrumented model can be a user subclass of a shipped class. Matching
    # the exact class name alone would report ``unknown`` for every subclass, so
    # walk the hierarchy, most derived class first.
    class_names = [cls.__name__ for cls in type(instance).__mro__]

    if not _LITELLM_CLASS_NAMES.isdisjoint(class_names):
        provider = _provider_from_litellm(instance)
        if provider is not None:
            return provider

    for class_name in class_names:
        provider = _CLASS_NAME_TO_PROVIDER.get(class_name)
        if provider is not None:
            return provider

    _logger.debug(
        "No known gen_ai.provider.name for model class %s", class_names[0]
    )
    return _UNKNOWN_PROVIDER
