# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Agno instrumentation."""

from __future__ import annotations

import mimetypes
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agno.knowledge.document.base import Document
    from agno.knowledge.embedder.base import Embedder
    from agno.media import Audio, File, Image, Video
    from agno.models.base import MessageData, Model
    from agno.models.message import Message
    from agno.models.response import ModelResponse

    MediaItem = Image | Audio | Video | File

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiProviderNameValues,
)
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    Modality,
    OutputMessage,
    ReasoningPart,
    RetrievalDocument,
    Role,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
    UriPart,
)
from opentelemetry.util.genai.utils import (
    decode_base64,
    gen_ai_json_dumps,
    get_argument,
    image_from_url,
    object_to_any_value,
    tool_arguments_to_any_value,
)


def safe_int(val: Any) -> int | None:
    """Safely convert a value to int or return None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val: Any) -> float | None:
    """Safely convert a value to float or return None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def extract_model_id(instance: Any) -> str | None:
    """Extract request model identifier from an Agno model or embedder instance."""
    model = (
        getattr(instance, "id", None)
        or getattr(instance, "model", None)
        or getattr(instance, "name", None)
    )
    return str(model) if model is not None else None


def populate_provider_data(
    invocation: InferenceInvocation,
    provider_data: Any,
) -> None:
    """Populate response_id and response_model_name from provider_data dict."""
    if not isinstance(provider_data, dict):
        return
    if (
        resp_id := _get_property_value(provider_data, "id")
    ) is not None and not invocation.response_id:
        invocation.response_id = str(resp_id)
    if (
        resp_model := _get_property_value(provider_data, "model")
    ) is not None and not invocation.response_model_name:
        invocation.response_model_name = str(resp_model)


_GOOGLE_PROVIDERS = (
    GenAiProviderNameValues.GCP_GEMINI.value,
    GenAiProviderNameValues.GCP_VERTEX_AI.value,
)

_UNCACHED_INPUT_PROVIDERS = (
    GenAiProviderNameValues.ANTHROPIC.value,
    GenAiProviderNameValues.AWS_BEDROCK.value,
)

_UNCACHED_INPUT_CLASSES = frozenset(
    {"Claude", "Anthropic", "AnthropicClaude", "AwsBedrock", "Bedrock"}
)

_TOKEN_USAGE_ATTRS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "audio_input_tokens",
    "audio_output_tokens",
)


def has_model_token_usage(metrics: Any) -> bool:
    """Return True if an Agno metrics object has any positive token usage counts."""
    if metrics is None:
        return False
    return any(
        (val := safe_int(getattr(metrics, attr, None))) is not None and val > 0
        for attr in _TOKEN_USAGE_ATTRS
    )


def _uses_uncached_input_tokens(
    invocation: InferenceInvocation,
    model: Model | None = None,
) -> bool:
    if invocation.provider in _UNCACHED_INPUT_PROVIDERS:
        return True
    if model is not None:
        for cls in type(model).__mro__:
            if cls.__name__ in _UNCACHED_INPUT_CLASSES:
                return True
    return False


def record_model_token_usage(
    invocation: InferenceInvocation,
    metrics: Any,
    model: Model | None = None,
) -> None:
    """Populate token usage fields on an InferenceInvocation from Agno metrics."""
    if metrics is None:
        return
    input_tokens = safe_int(getattr(metrics, "input_tokens", None))
    cache_read_tokens = safe_int(getattr(metrics, "cache_read_tokens", None))
    cache_write_tokens = safe_int(getattr(metrics, "cache_write_tokens", None))
    if cache_read_tokens is not None and cache_read_tokens > 0:
        invocation.cache_read_input_tokens = cache_read_tokens
    if cache_write_tokens is not None and cache_write_tokens > 0:
        invocation.cache_write_input_tokens = cache_write_tokens

    if _uses_uncached_input_tokens(invocation, model):
        cache_read_count = (
            cache_read_tokens
            if cache_read_tokens is not None and cache_read_tokens > 0
            else 0
        )
        cache_write_count = (
            cache_write_tokens
            if cache_write_tokens is not None and cache_write_tokens > 0
            else 0
        )
        if (
            input_tokens is not None
            or cache_read_count > 0
            or cache_write_count > 0
        ):
            input_tokens = (
                (input_tokens or 0) + cache_read_count + cache_write_count
            )
    if input_tokens is not None:
        invocation.input_tokens = input_tokens

    output_tokens = safe_int(getattr(metrics, "output_tokens", None))
    reasoning_tokens = safe_int(getattr(metrics, "reasoning_tokens", None))
    if reasoning_tokens is not None and reasoning_tokens > 0:
        invocation.thinking_tokens = reasoning_tokens
        if invocation.provider in _GOOGLE_PROVIDERS:
            output_tokens = (output_tokens or 0) + reasoning_tokens
    if output_tokens is not None:
        invocation.output_tokens = output_tokens
    if (
        tok := safe_int(getattr(metrics, "audio_input_tokens", None))
    ) is not None and tok > 0:
        invocation.audio_input_tokens = tok
    if (
        tok := safe_int(getattr(metrics, "audio_output_tokens", None))
    ) is not None and tok > 0:
        invocation.audio_output_tokens = tok


def format_retrieval_document(doc: Document) -> RetrievalDocument:
    """Format an Agno Document into a RetrievalDocument."""
    return RetrievalDocument(
        id=str(doc.id) if doc.id is not None else None,
        score=safe_float(doc.reranking_score),
    )


def format_content(val: object) -> str:
    """Format content into a string, converting structured objects to JSON."""
    if isinstance(val, str):
        return val
    any_value = object_to_any_value(val)
    if any_value is None:
        return str(val)
    return gen_ai_json_dumps(any_value, ensure_ascii=False)


def _get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)

    return getattr(obj, property_name, None)


def _extract_desc(tool: Any) -> str | None:
    desc = _get_property_value(tool, "description")
    if not desc:
        entrypoint = _get_property_value(tool, "entrypoint")
        if entrypoint:
            desc = _get_property_value(entrypoint, "__doc__")
    return str(desc).strip() if desc else None


def prepare_tool_definitions(
    tools: Iterable[Any] | None,
) -> list[ToolDefinition] | None:
    """Extract tool definitions from Agno Agent tools."""
    if not tools:
        return None

    seen_names: set[str] = set()
    definitions: list[ToolDefinition] = []

    def _add_def(name: str, desc: str | None, params: Any) -> None:
        if not name or name in seen_names:
            return
        seen_names.add(name)
        definitions.append(
            FunctionToolDefinition(
                name=name,
                description=desc,
                parameters=params,
            )
        )

    for tool in tools:
        if isinstance(tool, dict):
            if (
                "type" in tool
                and _get_property_value(tool, "type") == "function"
                and isinstance(_get_property_value(tool, "function"), dict)
            ):
                func_dict = _get_property_value(tool, "function")
                _add_def(
                    str(_get_property_value(func_dict, "name") or ""),
                    str(_get_property_value(func_dict, "description"))
                    if _get_property_value(func_dict, "description")
                    is not None
                    else None,
                    _get_property_value(func_dict, "parameters"),
                )
            elif "name" in tool:
                _add_def(
                    str(_get_property_value(tool, "name") or ""),
                    str(_get_property_value(tool, "description"))
                    if _get_property_value(tool, "description") is not None
                    else None,
                    _get_property_value(tool, "parameters"),
                )
        elif hasattr(tool, "functions") or hasattr(tool, "get_functions"):
            try:
                funcs = None
                if hasattr(tool, "get_functions") and callable(
                    _get_property_value(tool, "get_functions")
                ):
                    funcs_fn = _get_property_value(tool, "get_functions")
                    if callable(funcs_fn):
                        funcs = funcs_fn()
                else:
                    funcs = _get_property_value(tool, "functions")
                if isinstance(funcs, dict):
                    sub_defs = prepare_tool_definitions(
                        list(cast(dict[str, Any], funcs).values())
                    )
                    if sub_defs:
                        for defn in sub_defs:
                            _add_def(
                                _get_property_value(defn, "name") or "",
                                _get_property_value(defn, "description"),
                                _get_property_value(defn, "parameters"),
                            )
            except Exception:
                pass
        elif hasattr(tool, "name") and hasattr(tool, "parameters"):
            name = _get_property_value(tool, "name") or ""
            desc = _extract_desc(tool)
            params = _get_property_value(tool, "parameters")
            _add_def(
                str(name),
                desc,
                params,
            )
        elif callable(tool):
            try:
                import agno.tools.function  # pylint: disable=import-outside-toplevel

                fn_cls = _get_property_value(agno.tools.function, "Function")
                func = fn_cls.from_callable(tool)
                name = _get_property_value(func, "name") or ""
                desc = _extract_desc(func)
                params = _get_property_value(func, "parameters")
                _add_def(
                    str(name),
                    desc,
                    params,
                )
            except Exception:
                name = _get_property_value(tool, "__name__") or str(tool)
                desc = _get_property_value(tool, "__doc__")
                _add_def(
                    str(name),
                    str(desc).strip() if desc is not None else None,
                    None,
                )

    return definitions or None


def extract_session_id(
    instance: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_response: Any = None,
    wrapped: Callable[..., Any] | None = None,
    *,
    is_continue: bool = False,
) -> str | None:
    """Extract session_id from call arguments, instance, or response."""
    if is_continue:
        if run_response is None:
            if wrapped is not None and (args or kwargs):
                run_response = get_argument(
                    "run_response", wrapped, args or (), kwargs or {}
                )
            elif kwargs:
                run_response = kwargs.get("run_response")

        if run_response is not None:
            if (
                session_id := getattr(run_response, "session_id", None)
            ) is not None:
                return str(session_id)
            if instance is not None and (
                (session_id := getattr(instance, "session_id", None))
                is not None
            ):
                return str(session_id)
            return None

    if wrapped is not None and (args or kwargs):
        session_id = get_argument(
            "session_id", wrapped, args or (), kwargs or {}
        )
        if session_id is not None:
            return str(session_id)
    else:
        if kwargs and (session_id := kwargs.get("session_id")) is not None:
            return str(session_id)
        if args and len(args) > 4 and args[4] is not None:
            return str(args[4])
    if instance is not None and (
        (session_id := getattr(instance, "session_id", None)) is not None
    ):
        return str(session_id)
    if run_response is not None and (
        (session_id := getattr(run_response, "session_id", None)) is not None
    ):
        return str(session_id)
    return None


_UNKNOWN_PROVIDER = "unknown"

# Mapping of raw provider identifiers to GenAI semantic conventions standard values.
_KNOWN_PROVIDERS: dict[str, str] = {
    "openai": GenAiProviderNameValues.OPENAI.value,
    "azure": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure_openai": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure-openai": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure_ai": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "azure_ai_inference": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "azure-ai-inference": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "aws": GenAiProviderNameValues.AWS_BEDROCK.value,
    "awsbedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "aws_bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "aws-bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "amazon_bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "anthropic": GenAiProviderNameValues.ANTHROPIC.value,
    "cohere": GenAiProviderNameValues.COHERE.value,
    "google": GenAiProviderNameValues.GCP_GEMINI.value,
    "gemini": GenAiProviderNameValues.GCP_GEMINI.value,
    "google_generativeai": GenAiProviderNameValues.GCP_GEMINI.value,
    "vertex_ai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "vertexai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google_vertexai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "gcp_vertex_ai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "mistral": GenAiProviderNameValues.MISTRAL_AI.value,
    "mistralai": GenAiProviderNameValues.MISTRAL_AI.value,
    "mistral_ai": GenAiProviderNameValues.MISTRAL_AI.value,
    "groq": GenAiProviderNameValues.GROQ.value,
    "deepseek": GenAiProviderNameValues.DEEPSEEK.value,
    "watsonx": GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "ibm_watsonx_ai": GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "perplexity": GenAiProviderNameValues.PERPLEXITY.value,
    "xai": GenAiProviderNameValues.X_AI.value,
    "x_ai": GenAiProviderNameValues.X_AI.value,
    "ollama": "ollama",
    "fireworks": "fireworks",
    "together": "together",
    "huggingface": "huggingface",
    "langdb": "langdb",
    "nebius": "nebius",
    "vllm": "vllm",
    "cerebras": "cerebras",
    "cloudflare": "cloudflare",
    "dashscope": "dashscope",
    "deepinfra": "deepinfra",
    "internlm": "internlm",
    "litellm": "litellm",
    "llama_cpp": "llama_cpp",
    "lmstudio": "lmstudio",
    "minimax": "minimax",
    "moonshot": "moonshot",
    "openrouter": "openrouter",
    "sambanova": "sambanova",
    "azureaifoundry": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "azure_ai_foundry": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "azure-ai-foundry": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
}

# Mapping of known embedder class names to provider values.
_CLASS_NAME_TO_PROVIDER: dict[str, str] = {
    "OpenAIEmbedder": GenAiProviderNameValues.OPENAI.value,
    "AzureOpenAIEmbedder": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "AwsBedrockEmbedder": GenAiProviderNameValues.AWS_BEDROCK.value,
    "CohereEmbedder": GenAiProviderNameValues.COHERE.value,
    "MistralEmbedder": GenAiProviderNameValues.MISTRAL_AI.value,
    "OllamaEmbedder": "ollama",
    "FireworksEmbedder": "fireworks",
    "TogetherEmbedder": "together",
    "VoyageAIEmbedder": "voyageai",
    "FastEmbedEmbedder": "fastembed",
    "SentenceTransformerEmbedder": "sentence_transformer",
    "HuggingfaceCustomEmbedder": "huggingface",
    "LangDBEmbedder": "langdb",
    "NebiusEmbedder": "nebius",
    "VLLMEmbedder": "vllm",
    "JinaEmbedder": "jina",
}


def _resolve_provider(
    instance: Model | Embedder,
    *,
    class_name_to_provider: dict[str, str],
    google_classes: tuple[str, ...],
    stop_classes: tuple[str, ...],
    allow_provider_attr: bool = True,
) -> str:
    google_provider = (
        GenAiProviderNameValues.GCP_VERTEX_AI.value
        if getattr(instance, "vertexai", False)
        or (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true")
        else GenAiProviderNameValues.GCP_GEMINI.value
    )

    # 1. Check the class hierarchy (most derived first)
    for cls in type(instance).__mro__:
        cls_name = cls.__name__
        cls_mod = getattr(cls, "__module__", "")
        if cls_name == "Claude":
            if cls_mod == "agno.models.aws.claude":
                return GenAiProviderNameValues.AWS_BEDROCK.value
            if cls_mod == "agno.models.vertexai.claude":
                return GenAiProviderNameValues.GCP_VERTEX_AI.value
            if cls_mod == "agno.models.azure.claude":
                return GenAiProviderNameValues.AZURE_AI_INFERENCE.value
        if cls_name in google_classes:
            return google_provider
        if cls_name in stop_classes:
            if getattr(instance, "provider", None) == getattr(
                cls, "provider", None
            ):
                return _UNKNOWN_PROVIDER
            break
        if cls_name in class_name_to_provider:
            return class_name_to_provider[cls_name]

    # 2. Explicit provider attribute on the instance
    if allow_provider_attr:
        provider_attr = getattr(instance, "provider", None)
        if isinstance(provider_attr, str) and provider_attr.strip():
            p_name = provider_attr.strip().lower()
            if p_name in ("google", "gemini"):
                return google_provider
            p_clean = "".join(c for c in p_name if c.isalnum())
            return (
                _KNOWN_PROVIDERS.get(p_name)
                or _KNOWN_PROVIDERS.get(p_clean)
                or p_name
            )

    return _UNKNOWN_PROVIDER


def resolve_embedder_provider(embedder: Embedder) -> str:
    """Resolve the ``gen_ai.provider.name`` value for an Agno embedder instance."""
    return _resolve_provider(
        embedder,
        class_name_to_provider=_CLASS_NAME_TO_PROVIDER,
        google_classes=("GeminiEmbedder", "GoogleEmbedder"),
        stop_classes=("OpenAILikeEmbedder", "Embedder"),
        allow_provider_attr=False,
    )


# Mapping of known model class names to provider values.
_MODEL_CLASS_NAME_TO_PROVIDER: dict[str, str] = {
    "OpenAIChat": GenAiProviderNameValues.OPENAI.value,
    "OpenAIResponses": GenAiProviderNameValues.OPENAI.value,
    "OpenAI": GenAiProviderNameValues.OPENAI.value,
    "Claude": GenAiProviderNameValues.ANTHROPIC.value,
    "Anthropic": GenAiProviderNameValues.ANTHROPIC.value,
    "AnthropicClaude": GenAiProviderNameValues.ANTHROPIC.value,
    "VertexAI": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "AwsBedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "Bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "AzureOpenAI": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "AzureOpenAIResponses": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "AzureAIFoundry": GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "MistralChat": GenAiProviderNameValues.MISTRAL_AI.value,
    "Mistral": GenAiProviderNameValues.MISTRAL_AI.value,
    "Groq": GenAiProviderNameValues.GROQ.value,
    "Cohere": GenAiProviderNameValues.COHERE.value,
    "DeepSeek": GenAiProviderNameValues.DEEPSEEK.value,
    "WatsonX": GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "Perplexity": GenAiProviderNameValues.PERPLEXITY.value,
    "xAI": GenAiProviderNameValues.X_AI.value,
    "Ollama": "ollama",
    "Fireworks": "fireworks",
    "Together": "together",
    "VLLM": "vllm",
    "HuggingFace": "huggingface",
    "Cerebras": "cerebras",
    "CerebrasOpenAI": "cerebras",
    "LiteLLM": "litellm",
    "LiteLLMOpenAI": "litellm",
    "OpenRouter": "openrouter",
    "Nebius": "nebius",
    "LangDB": "langdb",
    "LMStudio": "lmstudio",
    "DashScope": "dashscope",
    "DeepInfra": "deepinfra",
    "Sambanova": "sambanova",
    "Cloudflare": "cloudflare",
    "AIMLAPI": "aimlapi",
    "MiniMax": "minimax",
    "MoonShot": "moonshot",
    "InternLM": "internlm",
    "Siliconflow": "siliconflow",
    "LlamaCpp": "llama_cpp",
    "Llama": "meta",
    "LlamaOpenAI": "meta",
}


def resolve_model_provider(model: Model) -> str:
    """Resolve the ``gen_ai.provider.name`` value for an Agno model instance."""
    return _resolve_provider(
        model,
        class_name_to_provider=_MODEL_CLASS_NAME_TO_PROVIDER,
        google_classes=("Gemini", "Google", "GeminiInteractions"),
        stop_classes=("OpenAILike", "OpenResponses", "Model"),
    )


def extract_model_finish_reasons(
    assistant_message: Message | None = None,
    stream_data: MessageData | None = None,
) -> list[str]:
    """Derive gen_ai.response.finish_reasons from assistant message or stream data."""
    provider_data: Any = None
    if assistant_message is not None:
        provider_data = _get_property_value(assistant_message, "provider_data")
    if not provider_data and stream_data is not None:
        provider_data = stream_data.response_provider_data

    if isinstance(provider_data, dict):
        raw_reason = _get_property_value(
            provider_data, "finish_reason"
        ) or _get_property_value(provider_data, "stop_reason")
        if raw_reason is not None:
            r = str(raw_reason).lower()
            if r in ("stop", "end_turn", "stop_sequence"):
                return ["stop"]
            if r in ("tool_calls", "tool_use", "function_call"):
                return ["tool_call"]
            if r in ("length", "max_tokens"):
                return ["length"]
            if r in ("content_filter", "safety"):
                return ["content_filter"]
            return [r]

    return []


_FORMAT_MIME_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "pdf": "application/pdf",
    "txt": "text/plain",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "text/xml",
    "html": "text/html",
    "md": "text/markdown",
}


def _infer_mime_type(media: MediaItem, modality: Modality) -> str | None:
    if media.mime_type:
        return str(media.mime_type)
    if media.format:
        fmt = str(media.format).lower().lstrip(".")
        if fmt in _FORMAT_MIME_TYPES:
            return _FORMAT_MIME_TYPES[fmt]
        guessed, _ = mimetypes.guess_type(f"file.{fmt}")
        if guessed:
            return guessed
        if modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO):
            return f"{modality.value}/{fmt}"
        return f"application/{fmt}"
    for path_attr in ("filepath", "filename", "url"):
        path_val = getattr(media, path_attr, None)
        if path_val and not str(path_val).startswith("data:"):
            guessed, _ = mimetypes.guess_type(str(path_val))
            if guessed:
                return guessed
    return None


def _media_item_to_part(
    media: MediaItem,
    modality: Modality,
) -> MessagePart | None:
    mime_type = _infer_mime_type(media, modality)

    if media.content is not None:
        if isinstance(media.content, bytes):
            return BlobPart(
                mime_type=mime_type,
                modality=modality,
                content=media.content,
            )
        if isinstance(media.content, str):
            return BlobPart(
                mime_type=mime_type,
                modality=modality,
                content=media.content.encode("utf-8"),
            )

    url = media.url
    media_ref = getattr(media, "media_reference", None)
    if not url and media_ref is not None:
        url = getattr(media_ref, "url", None)
    if url:
        url_str = str(url)
        if url_str.startswith("data:"):
            part = image_from_url(url_str, modality=modality)
            if isinstance(part, BlobPart) and part.mime_type is None:
                part.mime_type = mime_type
            return part
        return UriPart(
            mime_type=mime_type,
            modality=modality,
            uri=url_str,
        )

    if media.filepath is not None:
        return UriPart(
            mime_type=mime_type,
            modality=modality,
            uri=str(media.filepath),
        )

    external = getattr(media, "external", None)
    if external is not None:
        ext_uri = _get_property_value(external, "uri") or _get_property_value(
            external, "url"
        )
        if ext_uri:
            return UriPart(
                mime_type=mime_type,
                modality=modality,
                uri=str(ext_uri),
            )
        ext_id = _get_property_value(external, "id") or _get_property_value(
            external, "name"
        )
        if ext_id:
            return FilePart(
                mime_type=mime_type,
                modality=modality,
                file_id=str(ext_id),
            )

    if media.id:
        return FilePart(
            mime_type=mime_type,
            modality=modality,
            file_id=str(media.id),
        )

    return None


def _append_media_items(
    parts: list[MessagePart],
    items: Sequence[MediaItem] | MediaItem | None,
    modality: Modality,
) -> None:
    if items is None:
        return
    if isinstance(items, Sequence):
        for item in items:
            if (part := _media_item_to_part(item, modality)) is not None:
                parts.append(part)
    elif (part := _media_item_to_part(items, modality)) is not None:
        parts.append(part)


def _extract_media_parts(
    obj: Message | ModelResponse,
    stream_data: MessageData | None = None,
) -> list[MessagePart]:
    parts: list[MessagePart] = []
    _append_media_items(parts, obj.images, Modality.IMAGE)
    image_output = getattr(obj, "image_output", None) or (
        getattr(stream_data, "response_image", None)
        if stream_data is not None
        else None
    )
    _append_media_items(parts, image_output, Modality.IMAGE)

    _append_media_items(parts, obj.audio, Modality.AUDIO)
    _append_media_items(parts, getattr(obj, "audios", None), Modality.AUDIO)
    audio_output = getattr(obj, "audio_output", None) or (
        getattr(stream_data, "response_audio", None)
        if stream_data is not None
        else None
    )
    _append_media_items(parts, audio_output, Modality.AUDIO)

    _append_media_items(parts, obj.videos, Modality.VIDEO)
    video_output = getattr(obj, "video_output", None) or (
        getattr(stream_data, "response_video", None)
        if stream_data is not None
        else None
    )
    _append_media_items(parts, video_output, Modality.VIDEO)

    _append_media_items(parts, obj.files, Modality.DOCUMENT)
    file_output = getattr(obj, "file_output", None) or (
        getattr(stream_data, "response_file", None)
        if stream_data is not None
        else None
    )
    _append_media_items(parts, file_output, Modality.DOCUMENT)
    return parts


def has_stream_data_content(stream_data: MessageData | None) -> bool:
    """Return True if MessageData has accumulated text, reasoning, tool calls, or media."""
    if stream_data is None:
        return False
    return bool(
        stream_data.response_content
        or stream_data.response_reasoning_content
        or stream_data.response_tool_calls
        or stream_data.response_audio
        or stream_data.response_image
        or stream_data.response_video
        or getattr(stream_data, "response_file", None)
    )


def has_model_output_content(obj: Message | ModelResponse | None) -> bool:
    """Return True if a Message or ModelResponse contains text, tool calls, reasoning, or media."""
    if obj is None:
        return False
    return bool(
        obj.content
        or obj.tool_calls
        or obj.reasoning_content
        or obj.images
        or obj.audio
        or obj.videos
        or obj.files
        or getattr(obj, "audios", None)
        or getattr(obj, "image_output", None)
        or getattr(obj, "audio_output", None)
        or getattr(obj, "video_output", None)
        or getattr(obj, "file_output", None)
    )


def _normalize_stream_tool_calls(
    raw_tool_calls: Sequence[Any],
    model: Model | None = None,
) -> list[Any]:
    tool_calls: list[Any] = list(raw_tool_calls)
    if model is not None:
        parse_fn = getattr(model, "parse_tool_calls", None)
        if callable(parse_fn):
            try:
                parsed = parse_fn(tool_calls)
                if isinstance(parsed, list):
                    tool_calls = cast(list[Any], parsed)
            except Exception:
                pass
    if any(not isinstance(tc, Mapping) for tc in tool_calls):
        merged: list[dict[str, Any]] = []
        for tc in tool_calls:
            if isinstance(tc, Mapping):
                merged.append(dict(cast(Mapping[str, Any], tc)))
                continue
            idx = safe_int(getattr(tc, "index", None)) or 0
            tc_id = getattr(tc, "id", None)
            tc_type = getattr(tc, "type", None)
            fn = getattr(tc, "function", None)
            fn_name = (
                _get_property_value(fn, "name") if fn is not None else None
            )
            fn_args = (
                _get_property_value(fn, "arguments")
                if fn is not None
                else None
            )
            if len(merged) <= idx:
                merged.extend({} for _ in range(idx - len(merged) + 1))
            entry = merged[idx]
            if not entry:
                entry["id"] = tc_id
                entry["type"] = tc_type
                entry["function"] = {
                    "name": str(fn_name) if fn_name else "",
                    "arguments": str(fn_args) if fn_args else "",
                }
            else:
                if fn_name:
                    entry["function"]["name"] += str(fn_name)
                if fn_args:
                    entry["function"]["arguments"] += str(fn_args)
                if tc_id:
                    entry["id"] = tc_id
                if tc_type:
                    entry["type"] = tc_type
        return [e for e in merged if e]
    return tool_calls


def _extract_tool_call_parts(
    tool_calls: Sequence[Any] | None,
) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if not tool_calls:
        return parts
    for tc in tool_calls:
        tc_id = _get_property_value(tc, "id")
        fn = _get_property_value(tc, "function")
        target = fn if fn is not None else tc
        fn_name_val = _get_property_value(target, "name")
        fn_name = str(fn_name_val) if fn_name_val else ""
        fn_args = tool_arguments_to_any_value(
            _get_property_value(target, "arguments")
        )
        parts.append(
            ToolCallRequestPart(
                id=str(tc_id) if tc_id is not None else None,
                name=fn_name,
                arguments=fn_args,
            )
        )
    return parts


def _content_block_to_part(item: Any) -> MessagePart | None:
    if isinstance(item, str):
        return TextPart(content=item) if item else None

    part_type = _get_property_value(item, "type")
    text = _get_property_value(item, "text")
    if part_type in ("text", "input_text", "output_text") or (
        part_type is None and isinstance(text, str)
    ):
        if isinstance(text, str) and text:
            return TextPart(content=text)
        return None

    if part_type in ("thinking", "reasoning"):
        reasoning = (
            _get_property_value(item, "thinking")
            or _get_property_value(item, "reasoning")
            or text
        )
        if isinstance(reasoning, str) and reasoning:
            return ReasoningPart(content=reasoning)
        return None

    if part_type == "refusal":
        refusal = _get_property_value(item, "refusal")
        if isinstance(refusal, str) and refusal:
            return TextPart(content=refusal)
        return None

    if part_type in ("image_url", "input_image"):
        image_url = _get_property_value(item, "image_url")
        if not isinstance(image_url, str):
            image_url = _get_property_value(image_url, "url")
        if isinstance(image_url, str) and image_url:
            return image_from_url(image_url)
        file_id = _get_property_value(item, "file_id")
        if isinstance(file_id, str) and file_id:
            mime = _get_property_value(item, "mime_type")
            return FilePart(
                mime_type=mime if isinstance(mime, str) else None,
                modality=Modality.IMAGE,
                file_id=file_id,
            )
        return None

    if part_type == "image":
        source = _get_property_value(item, "source")
        if isinstance(source, Mapping):
            source_type = _get_property_value(source, "type")
            if source_type == "base64":
                data = _get_property_value(source, "data")
                media_type = _get_property_value(source, "media_type")
                if (
                    isinstance(data, str)
                    and (decoded := decode_base64(data)) is not None
                ):
                    return BlobPart(
                        mime_type=media_type
                        if isinstance(media_type, str)
                        else None,
                        modality=Modality.IMAGE,
                        content=decoded,
                    )
                return None
            if source_type == "url":
                src_url = _get_property_value(source, "url")
                if isinstance(src_url, str) and src_url:
                    return image_from_url(src_url)
                return None
            if source_type == "file":
                src_file_id = _get_property_value(source, "file_id")
                media_type = _get_property_value(source, "media_type")
                if isinstance(src_file_id, str) and src_file_id:
                    return FilePart(
                        mime_type=media_type
                        if isinstance(media_type, str)
                        else None,
                        modality=Modality.IMAGE,
                        file_id=src_file_id,
                    )
                return None
        b64_data = _get_property_value(item, "base64") or _get_property_value(
            item, "data"
        )
        mime = _get_property_value(item, "mime_type") or _get_property_value(
            item, "media_type"
        )
        if (
            isinstance(b64_data, str)
            and (decoded := decode_base64(b64_data)) is not None
        ):
            return BlobPart(
                mime_type=mime if isinstance(mime, str) else None,
                modality=Modality.IMAGE,
                content=decoded,
            )
        img_url = _get_property_value(item, "url")
        if isinstance(img_url, str) and img_url:
            return image_from_url(img_url)
        img_file_id = _get_property_value(
            item, "file_id"
        ) or _get_property_value(item, "id")
        if isinstance(img_file_id, str) and img_file_id:
            return FilePart(
                mime_type=mime if isinstance(mime, str) else None,
                modality=Modality.IMAGE,
                file_id=img_file_id,
            )
        return None

    if part_type in ("input_audio", "audio"):
        audio_obj = _get_property_value(item, "input_audio") or item
        data = _get_property_value(audio_obj, "data")
        if (
            isinstance(data, str)
            and (decoded := decode_base64(data)) is not None
        ):
            fmt = _get_property_value(audio_obj, "format")
            mime = (
                _FORMAT_MIME_TYPES.get(fmt.lower().lstrip("."))
                if isinstance(fmt, str)
                else None
            )
            return BlobPart(
                mime_type=mime,
                modality=Modality.AUDIO,
                content=decoded,
            )
        audio_url = _get_property_value(audio_obj, "url")
        if isinstance(audio_url, str) and audio_url:
            return image_from_url(audio_url, modality=Modality.AUDIO)
        return None

    if part_type in ("file", "input_file", "document"):
        file_obj = (
            _get_property_value(item, "file")
            or _get_property_value(item, "source")
            or item
        )
        filename = _get_property_value(file_obj, "filename")
        mime_type = _get_property_value(
            file_obj, "mime_type"
        ) or _get_property_value(file_obj, "media_type")
        if not isinstance(mime_type, str):
            mime_type = (
                mimetypes.guess_type(filename)[0]
                if isinstance(filename, str) and filename
                else None
            )
        if _get_property_value(file_obj, "type") == "text":
            text_data = _get_property_value(
                file_obj, "data"
            ) or _get_property_value(file_obj, "text")
            if isinstance(text_data, str):
                return BlobPart(
                    mime_type=mime_type or "text/plain",
                    modality=Modality.DOCUMENT,
                    content=text_data.encode("utf-8"),
                )
            return None
        file_id = _get_property_value(
            file_obj, "file_id"
        ) or _get_property_value(file_obj, "id")
        if isinstance(file_id, str) and file_id:
            return FilePart(
                mime_type=mime_type,
                modality=Modality.DOCUMENT,
                file_id=file_id,
            )
        file_data = _get_property_value(
            file_obj, "file_data"
        ) or _get_property_value(file_obj, "data")
        if isinstance(file_data, str) and file_data:
            if file_data.startswith("data:"):
                return image_from_url(file_data, modality=Modality.DOCUMENT)
            if (decoded := decode_base64(file_data)) is not None:
                return BlobPart(
                    mime_type=mime_type,
                    modality=Modality.DOCUMENT,
                    content=decoded,
                )
        file_url = _get_property_value(file_obj, "url") or _get_property_value(
            file_obj, "uri"
        )
        if isinstance(file_url, str) and file_url:
            return UriPart(
                mime_type=mime_type,
                modality=Modality.DOCUMENT,
                uri=file_url,
            )
        return None

    if part_type == "tool_use":
        tc_id = _get_property_value(item, "id")
        tc_name = _get_property_value(item, "name")
        tc_input = _get_property_value(item, "input")
        return ToolCallRequestPart(
            id=str(tc_id) if tc_id is not None else None,
            name=str(tc_name) if tc_name else "",
            arguments=tool_arguments_to_any_value(tc_input),
        )

    if part_type == "tool_result":
        tr_id = _get_property_value(
            item, "tool_use_id"
        ) or _get_property_value(item, "id")
        tr_content = _get_property_value(item, "content")
        return ToolCallResponsePart(
            id=str(tr_id) if tr_id is not None else None,
            response=object_to_any_value(tr_content),
        )

    return None


def _content_to_parts(
    content: Any,
    *,
    allow_empty_text: bool = False,
) -> list[MessagePart]:
    if content is None:
        return []
    if isinstance(content, str):
        if content or allow_empty_text:
            return [TextPart(content=content)]
        return []
    if isinstance(content, Sequence) and not isinstance(
        content, (bytes, bytearray)
    ):
        parts: list[MessagePart] = []
        for item in cast(Sequence[Any], content):
            if (part := _content_block_to_part(item)) is not None:
                parts.append(part)
        if not parts and allow_empty_text:
            return [TextPart(content="")]
        return parts
    formatted = format_content(content)
    if formatted or allow_empty_text:
        return [TextPart(content=formatted)]
    return []


_ROLE_ALIASES: dict[str, str] = {
    "model": Role.ASSISTANT.value,
    "ai": Role.ASSISTANT.value,
    "human": Role.USER.value,
    "function": Role.TOOL.value,
}


def format_model_input_messages(
    messages: Iterable[Message],
) -> list[InputMessage]:
    """Format an iterable of Agno Message objects into InputMessage list."""
    result: list[InputMessage] = []
    for msg in messages:
        raw_role = str(msg.role).strip().lower() if msg.role else ""
        normalized_role = (
            _ROLE_ALIASES.get(raw_role, raw_role)
            if raw_role
            else Role.USER.value
        )
        name_str = str(msg.name) if msg.name is not None else None

        parts: list[MessagePart] = []

        # Tool response message
        if normalized_role == Role.TOOL.value or msg.tool_call_id is not None:
            parts.append(
                ToolCallResponsePart(
                    id=str(msg.tool_call_id)
                    if msg.tool_call_id is not None
                    else None,
                    response=object_to_any_value(msg.content),
                )
            )
            parts.extend(_extract_media_parts(msg))
            result.append(
                InputMessage(role=Role.TOOL.value, parts=parts, name=name_str)
            )
            continue

        # Reasoning content
        if msg.reasoning_content:
            parts.append(ReasoningPart(content=str(msg.reasoning_content)))

        # Tool calls requested by assistant in history
        parts.extend(_extract_tool_call_parts(msg.tool_calls))

        media_parts = _extract_media_parts(msg)

        # Main content
        if msg.content is not None:
            parts.extend(
                _content_to_parts(
                    msg.content,
                    allow_empty_text=not parts and not media_parts,
                )
            )

        parts.extend(media_parts)

        if parts:
            result.append(
                InputMessage(role=normalized_role, parts=parts, name=name_str)
            )

    return result


def format_model_output_message(
    assistant_message: Message | ModelResponse,
    stream_data: MessageData | None = None,
    model: Model | None = None,
) -> OutputMessage:
    """Format an Agno assistant message into an OutputMessage."""
    parts: list[MessagePart] = []

    reasoning = (
        stream_data.response_reasoning_content
        if stream_data and stream_data.response_reasoning_content
        else None
    ) or assistant_message.reasoning_content
    if reasoning:
        parts.append(ReasoningPart(content=str(reasoning)))

    content = (
        stream_data.response_content
        if stream_data and stream_data.response_content
        else assistant_message.content
    )
    if content is not None:
        parts.extend(_content_to_parts(content))

    parts.extend(
        _extract_media_parts(assistant_message, stream_data=stream_data)
    )
    if assistant_message.tool_calls:
        tool_calls: Sequence[Any] | None = assistant_message.tool_calls
    elif stream_data and stream_data.response_tool_calls:
        tool_calls = _normalize_stream_tool_calls(
            stream_data.response_tool_calls, model=model
        )
    else:
        tool_calls = None
    parts.extend(_extract_tool_call_parts(tool_calls))

    if not parts:
        parts.append(TextPart(content=""))

    name = getattr(assistant_message, "name", None)
    name_str = str(name) if name is not None else None

    return OutputMessage(
        role=Role.ASSISTANT.value,
        parts=parts,
        name=name_str,
    )
