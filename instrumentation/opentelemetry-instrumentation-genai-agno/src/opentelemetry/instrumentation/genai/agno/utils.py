# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Agno instrumentation."""

from __future__ import annotations

import json
import mimetypes
import os
from collections.abc import Callable, Iterable, Sequence
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
    gen_ai_json_dumps,
    get_argument,
    image_from_url,
    object_to_any_value,
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


def extract_user_id(
    instance: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_response: Any = None,
    wrapped: Callable[..., Any] | None = None,
) -> str | None:
    """Extract user_id from call arguments, instance, or response."""
    if wrapped is not None and (args or kwargs):
        user_id = get_argument("user_id", wrapped, args or (), kwargs or {})
        if user_id is not None:
            return str(user_id)
    else:
        if kwargs and (user_id := kwargs.get("user_id")) is not None:
            return str(user_id)
        if args and len(args) > 2 and args[2] is not None:
            return str(args[2])
    if instance:
        if (user_id := getattr(instance, "user_id", None)) is not None:
            return str(user_id)
        if (user := getattr(instance, "user", None)) is not None:
            return str(user)
    if run_response and (
        (user_id := getattr(run_response, "user_id", None)) is not None
    ):
        return str(user_id)
    return None


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


def set_invocation_user_id(
    invocation: Any,
    instance: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    run_response: Any = None,
    wrapped: Callable[..., Any] | None = None,
) -> None:
    """Extract and set user.id on the invocation attributes if present."""
    from opentelemetry.semconv._incubating.attributes.user_attributes import (  # pylint: disable=import-outside-toplevel
        USER_ID,
    )

    user_id = extract_user_id(
        instance, args, kwargs, run_response, wrapped=wrapped
    )
    if user_id is not None:
        invocation.attributes[USER_ID] = user_id


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
    "voyage": "voyageai",
    "voyageai": "voyageai",
    "voyage_ai": "voyageai",
    "fastembed": "fastembed",
    "sentence_transformer": "sentence_transformer",
    "sentence_transformers": "sentence_transformer",
    "sentence-transformers": "sentence_transformer",
    "huggingface": "huggingface",
    "langdb": "langdb",
    "nebius": "nebius",
    "vllm": "vllm",
    "jina": "jina",
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
    module_prefix: str,
    ignored_submodules: tuple[str, ...],
) -> str:
    google_provider = (
        GenAiProviderNameValues.GCP_VERTEX_AI.value
        if getattr(instance, "vertexai", False)
        or (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true")
        else GenAiProviderNameValues.GCP_GEMINI.value
    )

    # 1. Explicit provider attribute on the instance
    provider_attr = getattr(instance, "provider", None)
    if provider_attr is not None:
        if isinstance(provider_attr, str):
            p_name = provider_attr.strip().lower()
            if p_name in ("google", "gemini"):
                return google_provider
            if p_name in _KNOWN_PROVIDERS:
                return _KNOWN_PROVIDERS[p_name]
            p_clean = "".join(c for c in p_name if c.isalnum())
            if p_clean in _KNOWN_PROVIDERS:
                return _KNOWN_PROVIDERS[p_clean]
            if p_name and p_name != "none":
                return p_name
        else:
            cls_name = provider_attr.__class__.__name__.lower()
            if "provider" in cls_name and cls_name != "provider":
                p_name = cls_name.removesuffix("provider")
                if p_name in ("google", "gemini"):
                    return google_provider
                if p_name in _KNOWN_PROVIDERS:
                    return _KNOWN_PROVIDERS[p_name]
                if p_name:
                    return p_name

    # 2. Check the class hierarchy (most derived first)
    for cls in type(instance).__mro__:
        cls_name = cls.__name__
        if cls_name in google_classes:
            return google_provider
        if cls_name in stop_classes:
            break
        if cls_name in class_name_to_provider:
            return class_name_to_provider[cls_name]

    # 3. Check module name if in <module_prefix><submodule>
    module = getattr(instance, "__module__", "")
    if module_prefix in module:
        sub = module.split(module_prefix)[-1].split(".")[0]
        if sub not in ignored_submodules:
            if sub in ("google", "gemini", "vertexai"):
                if sub == "vertexai":
                    return GenAiProviderNameValues.GCP_VERTEX_AI.value
                return google_provider
            if sub in _KNOWN_PROVIDERS:
                return _KNOWN_PROVIDERS[sub]
            return sub

    # 4. Check model/id prefix if it has provider/model format
    model_id = (
        getattr(instance, "id", None)
        or getattr(instance, "model", None)
        or getattr(instance, "name", None)
    )
    if model_id is not None and isinstance(model_id, str):
        model_str = model_id.strip()
        if "/" in model_str:
            prefix = model_str.split("/")[0].strip().lower()
            if prefix in _KNOWN_PROVIDERS:
                return _KNOWN_PROVIDERS[prefix]

    # 5. Unresolved - fallback to unknown
    return _UNKNOWN_PROVIDER


def resolve_embedder_provider(embedder: Embedder) -> str:
    """Resolve the ``gen_ai.provider.name`` value for an Agno embedder instance."""
    return _resolve_provider(
        embedder,
        class_name_to_provider=_CLASS_NAME_TO_PROVIDER,
        google_classes=("GeminiEmbedder", "GoogleEmbedder"),
        stop_classes=("OpenAILikeEmbedder", "Embedder"),
        module_prefix="agno.knowledge.embedder.",
        ignored_submodules=("base", "openai_like"),
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
        stop_classes=("OpenAILike", "Model"),
        module_prefix="agno.models.",
        ignored_submodules=(
            "base",
            "openai_like",
            "message",
            "response",
            "utils",
            "fallback",
        ),
    )


def extract_model_finish_reasons(
    assistant_message: Message | None = None,
    model_response: ModelResponse | MessageData | None = None,
) -> list[str]:
    """Derive gen_ai.response.finish_reasons from assistant message or model response."""
    provider_data: Any = None
    if assistant_message is not None:
        provider_data = _get_property_value(assistant_message, "provider_data")
    if not provider_data and model_response is not None:
        provider_data = _get_property_value(model_response, "provider_data")

    if isinstance(provider_data, dict):
        raw_reason = _get_property_value(
            provider_data, "finish_reason"
        ) or _get_property_value(provider_data, "stop_reason")
        if raw_reason is not None:
            r = str(raw_reason).lower()
            if r in ("stop", "end_turn"):
                return ["stop"]
            if r in ("tool_calls", "tool_use", "function_call"):
                return ["tool_calls"]
            if r in ("length", "max_tokens"):
                return ["length"]
            if r in ("content_filter", "safety"):
                return ["content_filter"]
            return [r]

    tool_calls = (
        _get_property_value(assistant_message, "tool_calls")
        if assistant_message is not None
        else None
    )
    if tool_calls:
        return ["tool_calls"]
    return ["stop"]


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


def _extract_media_parts(obj: Message | ModelResponse) -> list[MessagePart]:
    parts: list[MessagePart] = []
    _append_media_items(parts, obj.images, Modality.IMAGE)
    _append_media_items(
        parts, getattr(obj, "image_output", None), Modality.IMAGE
    )
    _append_media_items(parts, obj.audio, Modality.AUDIO)
    _append_media_items(parts, getattr(obj, "audios", None), Modality.AUDIO)
    _append_media_items(
        parts, getattr(obj, "audio_output", None), Modality.AUDIO
    )
    _append_media_items(parts, obj.videos, Modality.VIDEO)
    _append_media_items(
        parts, getattr(obj, "video_output", None), Modality.VIDEO
    )
    _append_media_items(parts, obj.files, Modality.DOCUMENT)
    _append_media_items(
        parts, getattr(obj, "file_output", None), Modality.DOCUMENT
    )
    return parts


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


def _extract_tool_call_parts(
    tool_calls: list[dict[str, Any]] | None,
) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if not tool_calls:
        return parts
    for tc in tool_calls:
        tc_id = tc.get("id")
        fn: dict[str, Any] | None = tc.get("function")
        target = fn or tc
        fn_name_val = target.get("name")
        fn_name = str(fn_name_val) if fn_name_val else ""
        fn_args = target.get("arguments")
        if isinstance(fn_args, str):
            try:
                fn_args = json.loads(fn_args)
            except Exception:
                pass
        parts.append(
            ToolCallRequestPart(
                id=str(tc_id) if tc_id is not None else None,
                name=fn_name,
                arguments=fn_args,
            )
        )
    return parts


def format_model_input_messages(
    messages: Iterable[Message],
) -> list[InputMessage]:
    """Format an iterable of Agno Message objects into InputMessage list."""
    result: list[InputMessage] = []
    for msg in messages:
        role_str = msg.role.lower() if msg.role else "user"
        name_str = str(msg.name) if msg.name is not None else None

        parts: list[MessagePart] = []

        # Tool response message
        if role_str == "tool" or msg.tool_call_id is not None:
            parts.append(
                ToolCallResponsePart(
                    id=str(msg.tool_call_id)
                    if msg.tool_call_id is not None
                    else None,
                    response=format_content(msg.content)
                    if msg.content is not None
                    else "",
                )
            )
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
            formatted_content = format_content(msg.content)
            if formatted_content or (not parts and not media_parts):
                parts.append(TextPart(content=formatted_content))

        parts.extend(media_parts)

        # Normalize role
        if role_str in ("system", "user", "assistant"):
            normalized_role = role_str
        else:
            normalized_role = Role.USER.value

        if parts:
            result.append(
                InputMessage(role=normalized_role, parts=parts, name=name_str)
            )

    return result


def format_model_output_message(
    assistant_message: Message | ModelResponse,
    finish_reason: str = "stop",
    stream_data: MessageData | None = None,
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
        formatted = format_content(content)
        if formatted:
            parts.append(TextPart(content=formatted))

    parts.extend(_extract_media_parts(assistant_message))
    tool_calls = assistant_message.tool_calls or (
        stream_data.response_tool_calls if stream_data else None
    )
    parts.extend(_extract_tool_call_parts(tool_calls))

    if not parts:
        parts.append(TextPart(content=""))

    name = getattr(assistant_message, "name", None)
    name_str = str(name) if name is not None else None

    return OutputMessage(
        role=Role.ASSISTANT.value,
        parts=parts,
        finish_reason=finish_reason,
        name=name_str,
    )
