# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for standalone Agno Embedder instrumentation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from agno.knowledge.embedder.azure_openai import AzureOpenAIEmbedder
from agno.knowledge.embedder.base import Embedder
from agno.knowledge.embedder.openai import OpenAIEmbedder

try:
    from agno.knowledge.embedder.openai_like import OpenAILikeEmbedder
except ImportError:

    class OpenAILikeEmbedder(OpenAIEmbedder):
        __module__ = "agno.knowledge.embedder.openai_like"


try:
    from agno.knowledge.embedder.google import GeminiEmbedder
except ImportError:
    GeminiEmbedder = None

from opentelemetry.instrumentation.genai.agno.utils import (
    resolve_embedder_provider,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_TYPE,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.trace.status import StatusCode


def _create_mock_response(
    embedding: list[float], usage: dict[str, Any] | None = None
) -> MagicMock:
    entry = MagicMock(embedding=embedding)
    resp = MagicMock(data=[entry])
    if usage is not None:
        mock_usage = MagicMock()
        mock_usage.model_dump.return_value = usage
        resp.usage = mock_usage
    else:
        resp.usage = None
    return resp


@dataclass
class SimpleEmbedder(Embedder):
    id: str = "text-embedding-3-small"
    provider: str = "openai"

    def get_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]


class SubSimpleEmbedder(SimpleEmbedder):
    """Subclass defined before instrumentation that inherits without overriding."""


@dataclass
class UsageEmbedder(Embedder):
    model: str = "text-embedding-usage"
    provider: str = "custom_provider"

    def get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        return [0.1, 0.2, 0.3], {"input_tokens": 12, "total_tokens": 12}


@dataclass
class AsyncEmbedder(Embedder):
    id: str = "async-embed-model"
    provider: str = "cohere"

    async def async_get_embedding(self, text: str) -> list[float]:
        return [0.7, 0.8]

    async def async_get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        return [0.7, 0.8], {"prompt_tokens": 5}


@dataclass
class NestedCallingEmbedder(Embedder):
    id: str = "nested-embedder"

    def get_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2]

    def get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        # Internally calls get_embedding
        vec = self.get_embedding(text)
        return vec, {"input_tokens": 4}

    async def async_get_embedding(self, text: str) -> list[float]:
        return [0.3, 0.4]

    async def async_get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, int]]:
        # Internally calls async_get_embedding
        vec = await self.async_get_embedding(text)
        return vec, {"input_tokens": 6}


@dataclass
class FailingEmbedder(Embedder):
    id: str = "fail-embedder"

    def get_embedding(self, text: str) -> list[float]:
        raise ValueError("Invalid embedding request")


def test_embedder_get_embedding_sync(
    instrument_agno,
    span_exporter,
) -> None:
    """Test sync get_embedding on a known embedder class."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _create_mock_response(
        [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    embedder = OpenAIEmbedder(
        api_key="fake",
        id="text-embedding-3-small",
        openai_client=mock_client,
    )
    res = embedder.get_embedding("test string for embedding")
    assert res == [0.1, 0.2, 0.3, 0.4, 0.5]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings text-embedding-3-small"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_PROVIDER_NAME) == "openai"
    assert (
        span.attributes.get(GEN_AI_REQUEST_MODEL) == "text-embedding-3-small"
    )
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 5
    assert span.status.status_code != StatusCode.ERROR


def test_embedder_get_embedding_and_usage_sync(
    instrument_agno,
    span_exporter,
) -> None:
    """Test sync get_embedding_and_usage records input_tokens attribute."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _create_mock_response(
        [0.1, 0.2, 0.3], {"input_tokens": 12, "total_tokens": 12}
    )
    embedder = OpenAIEmbedder(
        api_key="fake",
        id="text-embedding-usage",
        openai_client=mock_client,
    )
    vec, usage = embedder.get_embedding_and_usage("sample text")
    assert vec == [0.1, 0.2, 0.3]
    assert usage == {"input_tokens": 12, "total_tokens": 12}

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings text-embedding-usage"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_PROVIDER_NAME) == "openai"
    assert span.attributes.get(GEN_AI_REQUEST_MODEL) == "text-embedding-usage"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 3
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 12


def test_embedder_async_get_embedding(
    instrument_agno,
    span_exporter,
) -> None:
    """Test async async_get_embedding on a known embedder class."""
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=_create_mock_response([0.7, 0.8])
    )
    embedder = OpenAIEmbedder(
        api_key="fake",
        id="async-embed-model",
        async_client=mock_client,
    )

    async def _test() -> None:
        res = await embedder.async_get_embedding("async text")
        assert res == [0.7, 0.8]

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings async-embed-model"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_PROVIDER_NAME) == "openai"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2


def test_embedder_async_get_embedding_and_usage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test async async_get_embedding_and_usage records usage."""
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=_create_mock_response([0.7, 0.8], {"prompt_tokens": 5})
    )
    embedder = OpenAIEmbedder(
        api_key="fake",
        id="async-embed-model",
        async_client=mock_client,
    )

    async def _test() -> None:
        vec, usage = await embedder.async_get_embedding_and_usage("async text")
        assert vec == [0.7, 0.8]
        assert usage == {"prompt_tokens": 5}

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 5
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2


def test_embedder_nested_call_suppression_sync(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that when an embedder internally invokes another embedder method, only 1 span is emitted."""
    mock_client = MagicMock()

    embedder = OpenAIEmbedder(
        api_key="fake",
        id="nested-embedder",
        openai_client=mock_client,
    )

    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate inner embedder call
            embedder.get_embedding("inner")
        return _create_mock_response([0.1, 0.2], {"input_tokens": 4})

    mock_client.embeddings.create.side_effect = side_effect
    vec, usage = embedder.get_embedding_and_usage("nested test")
    assert vec == [0.1, 0.2]
    assert usage == {"input_tokens": 4}

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 4


def test_embedder_nested_call_suppression_async(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that when async embedder internally invokes another embedder method, only 1 span is emitted."""
    mock_client = MagicMock()

    embedder = OpenAIEmbedder(
        api_key="fake",
        id="nested-embedder",
        async_client=mock_client,
    )

    call_count = 0

    async def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await embedder.async_get_embedding("inner async")
        return _create_mock_response([0.3, 0.4], {"input_tokens": 6})

    mock_client.embeddings.create = AsyncMock(side_effect=side_effect)

    async def _test() -> None:
        vec, usage = await embedder.async_get_embedding_and_usage(
            "nested test"
        )
        assert vec == [0.3, 0.4]
        assert usage == {"input_tokens": 6}

    asyncio.run(_test())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert span.attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2
    assert span.attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 6


def test_embedder_error_handling(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an error raised by get_embedding marks the span as failed."""
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = ValueError(
        "Invalid embedding request"
    )
    embedder = OpenAIEmbedder(
        api_key="fake",
        id="fail-embedder",
        openai_client=mock_client,
    )

    with pytest.raises(Exception):
        embedder.get_embedding("failing input")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) in (
        "ValueError",
        "Exception",
        "agno.exceptions.EmbeddingError",
    )


def test_embedder_provider_extraction_azure(
    instrument_agno,
    span_exporter,
) -> None:
    """Test provider extraction for AzureOpenAIEmbedder."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _create_mock_response(
        [0.1, 0.2]
    )
    embedder = AzureOpenAIEmbedder(
        api_key="fake",
        azure_endpoint="https://test.openai.azure.com",
        openai_client=mock_client,
    )
    res = embedder.get_embedding("test")
    assert res == [0.1, 0.2]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "azure.ai.openai"


def test_embedder_uninstrument_restores_untraced_behavior(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that uninstrumenting restores clean behavior and stops emitting spans."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _create_mock_response([0.1])
    embedder = OpenAIEmbedder(
        api_key="fake",
        id="test-uninstrument",
        openai_client=mock_client,
    )
    embedder.get_embedding("traced")
    assert len(span_exporter.get_finished_spans()) == 1

    instrument_agno.uninstrument()
    span_exporter.clear()

    embedder.get_embedding("untraced")
    assert len(span_exporter.get_finished_spans()) == 0


def test_embedder_zero_input_tokens_usage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that 0 prompt_tokens is recorded as 0 and not overridden by total_tokens."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _create_mock_response(
        [0.1, 0.2], {"prompt_tokens": 0, "total_tokens": 12}
    )
    embedder = OpenAIEmbedder(
        api_key="fake",
        id="zero-tokens-model",
        openai_client=mock_client,
    )
    _, usage = embedder.get_embedding_and_usage("zero text")
    assert usage.get("prompt_tokens") == 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 0

    span_exporter.clear()

    mock_async_client = MagicMock()
    mock_async_client.embeddings.create = AsyncMock(
        return_value=_create_mock_response(
            [0.3, 0.4], {"prompt_tokens": 0, "total_tokens": 10}
        )
    )
    async_embedder = OpenAIEmbedder(
        api_key="fake",
        id="zero-tokens-model",
        async_client=mock_async_client,
    )

    async def _test() -> None:
        await async_embedder.async_get_embedding_and_usage("zero text async")

    asyncio.run(_test())
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_USAGE_INPUT_TOKENS) == 0


@pytest.mark.skipif(
    GeminiEmbedder is None, reason="google-genai not installed"
)
def test_embedder_provider_extraction_gemini(
    instrument_agno,
    span_exporter,
) -> None:
    """Test provider extraction for GeminiEmbedder."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings = [MagicMock(values=[0.1, 0.2])]
    mock_client.models.embed_content.return_value = mock_resp

    embedder = GeminiEmbedder(
        api_key="fake",
        gemini_client=mock_client,
    )
    res = embedder.get_embedding("test")
    assert res == [0.1, 0.2]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "gcp.gemini"

    span_exporter.clear()
    embedder_vertex = GeminiEmbedder(
        api_key="fake",
        gemini_client=mock_client,
        vertexai=True,
    )
    res = embedder_vertex.get_embedding("test")
    assert res == [0.1, 0.2]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "gcp.vertex_ai"


def test_embedder_provider_extraction_openai_like(
    instrument_agno,
    span_exporter,
) -> None:
    """Test provider extraction for OpenAILikeEmbedder falls back to unknown unless specified."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = _create_mock_response(
        [0.1, 0.2]
    )
    embedder = OpenAILikeEmbedder(
        api_key="fake",
        openai_client=mock_client,
    )
    res = embedder.get_embedding("test")
    assert res == [0.1, 0.2]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    # Must NOT fall back to "openai" or "agno"
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "unknown"

    span_exporter.clear()
    setattr(embedder, "provider", "mistral")
    res = embedder.get_embedding("test")
    assert res == [0.1, 0.2]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "mistral_ai"


def test_resolve_embedder_provider_unit() -> None:
    """Test resolve_embedder_provider maps known classes and falls back to unknown."""
    # Standard semconv values
    assert (
        resolve_embedder_provider(OpenAIEmbedder(api_key="fake")) == "openai"
    )
    assert (
        resolve_embedder_provider(AzureOpenAIEmbedder(api_key="fake"))
        == "azure.ai.openai"
    )
    if GeminiEmbedder is not None:
        assert (
            resolve_embedder_provider(GeminiEmbedder(api_key="fake"))
            == "gcp.gemini"
        )
        assert (
            resolve_embedder_provider(
                GeminiEmbedder(api_key="fake", vertexai=True)
            )
            == "gcp.vertex_ai"
        )
    else:
        gemini_stub = type("GeminiEmbedder", (Embedder,), {})
        assert resolve_embedder_provider(gemini_stub()) == "gcp.gemini"

    # Class hierarchy check for other embedders via stub subclasses of Embedder
    class AwsBedrockEmbedder(Embedder):
        pass

    class CohereEmbedder(Embedder):
        pass

    class MistralEmbedder(Embedder):
        pass

    class OllamaEmbedder(Embedder):
        pass

    class FireworksEmbedder(OpenAIEmbedder):
        pass

    class TogetherEmbedder(OpenAIEmbedder):
        pass

    class VoyageAIEmbedder(Embedder):
        pass

    class FastEmbedEmbedder(Embedder):
        pass

    class SentenceTransformerEmbedder(Embedder):
        pass

    class HuggingfaceCustomEmbedder(Embedder):
        pass

    class LangDBEmbedder(OpenAIEmbedder):
        pass

    class NebiusEmbedder(OpenAIEmbedder):
        pass

    class VLLMEmbedder(Embedder):
        pass

    class JinaEmbedder(Embedder):
        pass

    assert resolve_embedder_provider(AwsBedrockEmbedder()) == "aws.bedrock"
    assert resolve_embedder_provider(CohereEmbedder()) == "cohere"
    assert resolve_embedder_provider(MistralEmbedder()) == "mistral_ai"
    assert resolve_embedder_provider(OllamaEmbedder()) == "ollama"
    assert (
        resolve_embedder_provider(FireworksEmbedder(api_key="fake"))
        == "fireworks"
    )
    assert (
        resolve_embedder_provider(TogetherEmbedder(api_key="fake"))
        == "together"
    )
    assert resolve_embedder_provider(VoyageAIEmbedder()) == "voyageai"
    assert resolve_embedder_provider(FastEmbedEmbedder()) == "fastembed"
    assert (
        resolve_embedder_provider(SentenceTransformerEmbedder())
        == "sentence_transformer"
    )
    assert (
        resolve_embedder_provider(HuggingfaceCustomEmbedder()) == "huggingface"
    )
    assert (
        resolve_embedder_provider(LangDBEmbedder(api_key="fake")) == "langdb"
    )
    assert (
        resolve_embedder_provider(NebiusEmbedder(api_key="fake")) == "nebius"
    )
    assert resolve_embedder_provider(VLLMEmbedder()) == "vllm"
    assert resolve_embedder_provider(JinaEmbedder()) == "jina"

    # User subclass of a known embedder should resolve to that provider
    class CustomOpenAI(OpenAIEmbedder):
        pass

    assert resolve_embedder_provider(CustomOpenAI(api_key="fake")) == "openai"

    # Fallback to unknown: orchestrator (agno) and arbitrary suffixes must not be returned
    assert resolve_embedder_provider(Embedder()) == "unknown"
    assert (
        resolve_embedder_provider(OpenAILikeEmbedder(api_key="fake"))
        == "unknown"
    )

    class ArbitraryCustomEmbedder(Embedder):
        pass

    assert resolve_embedder_provider(ArbitraryCustomEmbedder()) == "unknown"

    class SimpleEmbedder:
        pass

    assert resolve_embedder_provider(SimpleEmbedder()) == "unknown"

    # Explicit provider attribute
    custom = ArbitraryCustomEmbedder()
    setattr(custom, "provider", "mistral")
    assert resolve_embedder_provider(custom) == "mistral_ai"

    setattr(custom, "provider", "azure_openai")
    assert resolve_embedder_provider(custom) == "azure.ai.openai"

    setattr(custom, "provider", "aws-bedrock")
    assert resolve_embedder_provider(custom) == "aws.bedrock"

    setattr(custom, "provider", "my_custom_provider")
    assert resolve_embedder_provider(custom) == "my_custom_provider"

    # Provider object with class name
    class GoogleProvider:
        pass

    setattr(custom, "provider", GoogleProvider())
    assert resolve_embedder_provider(custom) == "gcp.gemini"

    # Model prefix fallback
    custom_model = ArbitraryCustomEmbedder()
    setattr(custom_model, "id", "openai/text-embedding-3-small")
    assert resolve_embedder_provider(custom_model) == "openai"

    setattr(custom_model, "id", "cohere/embed-multilingual-v3.0")
    assert resolve_embedder_provider(custom_model) == "cohere"


def test_embedder_init_subclass_calls_original_and_unpatch_restores(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that custom base __init_subclass__ logic is called and restored on uninstrument."""
    called_with: list[tuple[type[Any], str | None]] = []

    class TrackedBaseEmbedder(Embedder):
        def __init_subclass__(
            cls, custom_arg: str | None = None, **kwargs: Any
        ) -> None:
            super().__init_subclass__(**kwargs)
            called_with.append((cls, custom_arg))

    class SubTrackedEmbedder(TrackedBaseEmbedder, custom_arg="value"):
        def get_embedding(self, text: str) -> list[float]:
            return [0.1]

    # Verify original __init_subclass__ was called with subclass and kwargs
    assert len(called_with) == 1
    assert called_with[0] == (SubTrackedEmbedder, "value")

    # Verify newly created subclass was instrumented
    sub = SubTrackedEmbedder()
    sub.get_embedding("traced")
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span_exporter.clear()

    # Uninstrument and verify restoration
    instrument_agno.uninstrument()
    called_with.clear()

    class SubTrackedEmbedderAfter(TrackedBaseEmbedder, custom_arg="after"):
        def get_embedding(self, text: str) -> list[float]:
            return [0.2]

    assert len(called_with) == 1
    assert called_with[0] == (SubTrackedEmbedderAfter, "after")

    sub_after = SubTrackedEmbedderAfter()
    sub_after.get_embedding("untraced")
    assert len(span_exporter.get_finished_spans()) == 0


def test_embedder_subclass_created_after_instrumentation(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that a new Embedder subclass defined after instrumentation is monkey-patched via __init_subclass__."""

    class DynamicEmbedder(Embedder):
        id: str = "dynamic-model"
        provider: str = "custom-provider"

        def get_embedding(self, text: str) -> list[float]:
            return [0.11, 0.22, 0.33]

        async def async_get_embedding(self, text: str) -> list[float]:
            return [0.44, 0.55]

    embedder = DynamicEmbedder()

    # Verify sync method emits embedding span
    vec = embedder.get_embedding("dynamic text")
    assert vec == [0.11, 0.22, 0.33]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embeddings dynamic-model"
    assert spans[0].attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert spans[0].attributes.get(GEN_AI_PROVIDER_NAME) == "custom-provider"
    assert spans[0].attributes.get(GEN_AI_REQUEST_MODEL) == "dynamic-model"
    assert spans[0].attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 3
    span_exporter.clear()

    # Verify async method emits embedding span
    async def _test() -> None:
        async_vec = await embedder.async_get_embedding("dynamic async text")
        assert async_vec == [0.44, 0.55]

    asyncio.run(_test())
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embeddings dynamic-model"
    assert spans[0].attributes.get(GEN_AI_OPERATION_NAME) == "embeddings"
    assert spans[0].attributes.get(GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2


def test_embedder_subclass_inheritance_prevents_double_instrumentation(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that a subclass inheriting methods without overriding them does not double-instrument."""
    # 1. Subclass defined BEFORE instrumentation (discovered via _wrap_subclasses)
    pre_embedder = SubSimpleEmbedder()
    vec1 = pre_embedder.get_embedding("hello pre")
    assert vec1 == [0.1, 0.2, 0.3, 0.4, 0.5]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embeddings text-embedding-3-small"
    assert "get_embedding" not in SubSimpleEmbedder.__dict__
    span_exporter.clear()

    # 2. Subclass defined AFTER instrumentation (hooked via __init_subclass__)
    class BaseCustomEmbedder(Embedder):
        id: str = "base-custom-model"

        def get_embedding(self, text: str) -> list[float]:
            return [0.1, 0.2]

    class DerivedCustomEmbedder(BaseCustomEmbedder):
        # Inherits get_embedding without overriding
        pass

    embedder = DerivedCustomEmbedder()
    vec2 = embedder.get_embedding("hello post")
    assert vec2 == [0.1, 0.2]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embeddings base-custom-model"
    assert "get_embedding" not in DerivedCustomEmbedder.__dict__
