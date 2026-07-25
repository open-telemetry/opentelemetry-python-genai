# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-effort ``gen_ai.provider.name`` inference from a component's class name.

Haystack generator/embedder components don't expose a normalized provider
identifier (unlike, say, the OpenAI or Anthropic SDKs, which this repo's
other packages instrument directly). This is a class-name heuristic,
narrowed to provider values that exist in the ``gen_ai.provider.name``
semconv registry today.
"""

from __future__ import annotations

from typing import Any, Optional

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

_OPENAI = GenAI.GenAiProviderNameValues.OPENAI.value
_AZURE_OPENAI = GenAI.GenAiProviderNameValues.AZURE_AI_OPENAI.value

_CLASS_NAME_TO_PROVIDER = {
    "OpenAIGenerator": _OPENAI,
    "OpenAIChatGenerator": _OPENAI,
    "OpenAITextEmbedder": _OPENAI,
    "OpenAIDocumentEmbedder": _OPENAI,
    "AzureOpenAIGenerator": _AZURE_OPENAI,
    "AzureOpenAIChatGenerator": _AZURE_OPENAI,
    "AzureOpenAITextEmbedder": _AZURE_OPENAI,
    "AzureOpenAIDocumentEmbedder": _AZURE_OPENAI,
}


def infer_provider(component: Any) -> Optional[str]:
    """Return the ``gen_ai.provider.name`` value for a generator/embedder component.

    Returns ``None`` for components with no known mapping (e.g. Hugging Face
    API generators, whose model string encodes the provider but has no
    corresponding ``gen_ai.provider.name`` enum value yet) — see
    MIGRATION_REPORT.md.
    """
    return _CLASS_NAME_TO_PROVIDER.get(component.__class__.__name__)
