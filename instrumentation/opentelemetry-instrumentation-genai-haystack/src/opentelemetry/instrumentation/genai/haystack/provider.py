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
_COHERE = GenAI.GenAiProviderNameValues.COHERE.value
_AWS_BEDROCK = GenAI.GenAiProviderNameValues.AWS_BEDROCK.value
_GCP_VERTEX_AI = GenAI.GenAiProviderNameValues.GCP_VERTEX_AI.value

_CLASS_NAME_TO_PROVIDER = {
    # haystack-ai core
    "OpenAIGenerator": _OPENAI,
    "OpenAIChatGenerator": _OPENAI,
    "OpenAITextEmbedder": _OPENAI,
    "OpenAIDocumentEmbedder": _OPENAI,
    "AzureOpenAIGenerator": _AZURE_OPENAI,
    "AzureOpenAIChatGenerator": _AZURE_OPENAI,
    "AzureOpenAITextEmbedder": _AZURE_OPENAI,
    "AzureOpenAIDocumentEmbedder": _AZURE_OPENAI,
    # cohere-haystack (haystack_integrations.components.{generators,embedders}.cohere)
    "CohereGenerator": _COHERE,
    "CohereChatGenerator": _COHERE,
    "CohereTextEmbedder": _COHERE,
    "CohereDocumentEmbedder": _COHERE,
    "CohereDocumentImageEmbedder": _COHERE,
    # amazon-bedrock-haystack (haystack_integrations.components.{generators,embedders}.amazon_bedrock)
    "AmazonBedrockGenerator": _AWS_BEDROCK,
    "AmazonBedrockChatGenerator": _AWS_BEDROCK,
    "AmazonBedrockTextEmbedder": _AWS_BEDROCK,
    "AmazonBedrockDocumentEmbedder": _AWS_BEDROCK,
    "AmazonBedrockDocumentImageEmbedder": _AWS_BEDROCK,
    # google-vertex-haystack (haystack_integrations.components.generators.google_vertex)
    "VertexAIGeminiGenerator": _GCP_VERTEX_AI,
    "VertexAIGeminiChatGenerator": _GCP_VERTEX_AI,
    "VertexAITextGenerator": _GCP_VERTEX_AI,
    "VertexAICodeGenerator": _GCP_VERTEX_AI,
}


def infer_provider(component: Any) -> Optional[str]:
    """Return the ``gen_ai.provider.name`` value for a generator/embedder component.

    Returns ``None`` for components with no known mapping (e.g. Hugging Face
    API generators, whose model string encodes the provider but has no
    corresponding ``gen_ai.provider.name`` enum value yet).
    """
    return _CLASS_NAME_TO_PROVIDER.get(component.__class__.__name__)
