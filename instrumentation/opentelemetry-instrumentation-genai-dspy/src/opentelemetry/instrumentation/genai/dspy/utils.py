# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for DSPy instrumentation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard, cast
from urllib.parse import urlsplit

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiProviderNameValues,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    ToolDefinition,
)

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.clients.embedding import Embedder
    from dspy.primitives.prediction import Prediction

SENTINEL_TOOL_NAMES: frozenset[str] = frozenset({"finish", "submit"})

_KNOWN_PROVIDERS: dict[str, str] = {
    "openai": GenAiProviderNameValues.OPENAI.value,
    "anthropic": GenAiProviderNameValues.ANTHROPIC.value,
    "cohere": GenAiProviderNameValues.COHERE.value,
    "bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "vertex_ai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "vertexai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "gemini": GenAiProviderNameValues.GCP_GEMINI.value,
    "google": GenAiProviderNameValues.GCP_GEMINI.value,
    "groq": GenAiProviderNameValues.GROQ.value,
    "mistral": GenAiProviderNameValues.MISTRAL_AI.value,
    "deepseek": GenAiProviderNameValues.DEEPSEEK.value,
    "azure": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "watsonx": GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "perplexity": GenAiProviderNameValues.PERPLEXITY.value,
    "xai": GenAiProviderNameValues.X_AI.value,
}


def is_mapping(val: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(val, Mapping)


def is_sequence(val: object) -> TypeGuard[Sequence[object]]:
    return isinstance(val, Sequence) and not isinstance(val, (str, bytes))


def is_tuple(val: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(val, tuple)


def safe_int(val: object) -> int | None:
    """Safely convert a value to int or return None."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float, str, bytes)):
        try:
            return int(val)
        except (ValueError, TypeError, OverflowError):
            return None
    return None


def parse_provider_and_model(
    model_str: str | None,
) -> tuple[str | None, str | None]:
    """Parse provider and model name from a LiteLLM-style model string."""
    if not isinstance(model_str, str) or not model_str:
        return None, None
    model_str = model_str.strip().rstrip("/")
    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
        provider = provider.strip().lower()
        model_name = model_name.strip()
        if "-" in provider:
            provider = provider.split("-")[-1]
        return provider, model_name or None
    return None, model_str.strip() or None


_MODEL_PREFIX_TO_PROVIDER: tuple[tuple[str, str], ...] = (
    ("text-embedding-", GenAiProviderNameValues.OPENAI.value),
    ("embed-", GenAiProviderNameValues.COHERE.value),
    ("amazon.titan-embed", GenAiProviderNameValues.AWS_BEDROCK.value),
    ("gemini-embedding", GenAiProviderNameValues.GCP_GEMINI.value),
    ("mistral-embed", GenAiProviderNameValues.MISTRAL_AI.value),
)


def resolve_embedder_provider_and_model(
    instance: Embedder,
) -> tuple[str, str | None]:
    """Resolve (gen_ai.provider.name, gen_ai.request.model) from a DSPy Embedder."""
    model = getattr(instance, "model", None)
    if isinstance(model, str):
        provider_raw, model_name = parse_provider_and_model(model)
        if provider_raw:
            return (
                _KNOWN_PROVIDERS.get(provider_raw) or provider_raw,
                model_name,
            )
        model_lower = (model_name or "").lower()
        for prefix, provider in _MODEL_PREFIX_TO_PROVIDER:
            if model_lower.startswith(prefix):
                return provider, model_name
        return "dspy", model_name

    if callable(model):
        bound_self = getattr(model, "__self__", None)
        model_name_attr = (
            getattr(model, "model_name", None)
            or getattr(bound_self, "model_name", None)
            or getattr(model, "__name__", None)
            or type(model).__name__
        )
        return "dspy", str(model_name_attr) if model_name_attr else None

    return "dspy", None


def extract_server_address_and_port(
    base_url: object,
) -> tuple[str | None, int | None]:
    """Extract server.address and server.port from an API base URL."""
    if not isinstance(base_url, str) or not base_url.strip():
        return None, None
    try:
        parsed = urlsplit(base_url.strip())
        return parsed.hostname, parsed.port
    except ValueError:
        return None, None


def extract_embedding_dimension(result: object) -> int | None:
    """Extract the embedding vector dimension count from an Embedder output."""
    shape = getattr(result, "shape", None)
    size = getattr(result, "size", None)
    size_int = safe_int(size)
    if (
        is_tuple(shape)
        and len(shape) >= 1
        and size_int is not None
        and size_int > 0
    ):
        dim = safe_int(shape[-1])
        if dim is not None and dim > 0:
            return dim

    if is_sequence(result) and len(result) > 0:
        first = result[0]
        if is_sequence(first):
            return len(first) if len(first) > 0 else None
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            return len(result)
    return None


def extract_input_content(input_args: dict[str, Any]) -> str:
    """Extract input content string from agent invocation arguments."""
    if not input_args:
        return ""
    if len(input_args) == 1:
        val: Any = next(iter(input_args.values()))
        if isinstance(val, str):
            return val
        if isinstance(val, (int, float, bool)):
            return str(val)
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    try:
        return json.dumps(input_args, ensure_ascii=False)
    except Exception:
        return str(input_args)


def extract_output_content(
    result: Prediction | None,
    signature: Any = None,
) -> str:
    """Extract output content string from a DSPy prediction result."""
    if result is None:
        return ""

    output_dict: dict[str, Any] = {}
    if hasattr(result, "items") and callable(getattr(result, "items")):
        try:
            items_func: Callable[[], Any] = getattr(result, "items")
            for k, v in cast(Iterable[tuple[Any, Any]], items_func()):
                output_dict[str(k)] = v
        except Exception:
            pass

    if output_dict:
        filtered_dict: dict[str, Any] = {}
        output_fields: Any = (
            getattr(signature, "output_fields", None) if signature else None
        )
        if isinstance(output_fields, (dict, list, set, tuple)):
            for key in cast(Iterable[Any], output_fields):
                key_str = str(key)
                if key_str in output_dict:
                    filtered_dict[key_str] = output_dict[key_str]
        if not filtered_dict:
            for k_str, v_val in output_dict.items():
                if k_str not in (
                    "trajectory",
                    "history",
                    "termination_reason",
                ):
                    filtered_dict[k_str] = v_val

        if filtered_dict:
            if len(filtered_dict) == 1:
                val: Any = next(iter(filtered_dict.values()))
                if isinstance(val, str):
                    return val
                if isinstance(val, (int, float, bool)):
                    return str(val)
                try:
                    return json.dumps(val, ensure_ascii=False)
                except Exception:
                    return str(val)
            try:
                return json.dumps(filtered_dict, ensure_ascii=False)
            except Exception:
                return str(filtered_dict)

    return str(result)


def prepare_tool_definitions(
    tools: Sequence[Tool | Callable[..., Any]]
    | Mapping[str, Tool | Callable[..., Any]]
    | None,
) -> list[ToolDefinition] | None:
    """Prepare FunctionToolDefinition instances from a tools collection."""
    if not tools:
        return None

    if isinstance(tools, Mapping):
        tool_items = list(tools.values())
    elif isinstance(tools, (list, tuple, set)):
        tool_items = list(tools)
    else:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tool_items:
        tool_name = getattr(tool, "name", None)
        if not tool_name:
            func = getattr(tool, "func", None)
            tool_name = (
                getattr(func, "__name__", None)
                if func
                else getattr(tool, "__name__", None)
            )
        if not tool_name:
            tool_name = "tool"

        name_str = str(tool_name)
        if name_str in SENTINEL_TOOL_NAMES:
            continue

        desc = (
            getattr(tool, "desc", None)
            or getattr(tool, "description", None)
            or getattr(tool, "__doc__", None)
        )
        args_schema = getattr(tool, "args", None)

        definitions.append(
            FunctionToolDefinition(
                name=name_str,
                description=str(desc).strip() if desc is not None else None,
                parameters=args_schema,
            )
        )

    return definitions or None
