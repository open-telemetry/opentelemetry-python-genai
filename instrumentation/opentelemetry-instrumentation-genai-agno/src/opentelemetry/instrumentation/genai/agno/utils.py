# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Agno instrumentation."""

from __future__ import annotations

import mimetypes
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast

if TYPE_CHECKING:
    from agno.knowledge.document.base import Document
    from agno.knowledge.embedder.base import Embedder
    from agno.media import File
    from agno.media.reference import MediaReference
    from agno.models.message import Message
    from agno.workflow.types import WorkflowExecutionInput

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiProviderNameValues,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    FunctionToolDefinition,
    MessagePart,
    Modality,
    RetrievalDocument,
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
    if isinstance(obj, Mapping):
        return cast("Mapping[str, Any]", obj).get(property_name)

    return getattr(obj, property_name, None)


def _get_media_reference(media: Any) -> MediaReference | None:
    # media_reference was added in Agno 3.x.
    return cast(
        "MediaReference | None", _get_property_value(media, "media_reference")
    )


# "audio/mp3" is not a registered media type, and mimetypes.guess_type returns
# "audio/x-wav" on some Linux systems.
_FORMAT_MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}


def _resolve_media_mime_type(
    media: Any,
    file_cls: type[File],
    media_ref: MediaReference | None,
) -> str | None:
    mime_type = _get_property_value(media, "mime_type")
    if mime_type:
        return str(mime_type)
    format_val = _get_property_value(media, "format")
    if format_val:
        format_str = str(format_val)
        if "/" in format_str:
            return format_str
        ext = format_str.lstrip(".").lower()
        if ext in _FORMAT_MIME_TYPES:
            return _FORMAT_MIME_TYPES[ext]
        guessed, _ = mimetypes.guess_type(f"file.{ext}")
        if guessed:
            return guessed
    filename = _get_property_value(media, "filename")
    if filename and (
        isinstance(media, file_cls)
        or _get_property_value(media, "type") == "file"
        or (isinstance(media, Mapping) and "id" in media)
    ):
        guessed, _ = mimetypes.guess_type(str(filename))
        if guessed:
            return guessed
    if media_ref is not None:
        ref_mime = _get_property_value(media_ref, "mime_type")
        if ref_mime:
            return str(ref_mime)
    return None


def _url_to_media_part(
    url: str,
    *,
    modality: Modality,
    mime_type: str | None,
) -> MessagePart | None:
    part = image_from_url(url, modality=modality)
    if (
        isinstance(part, (BlobPart, UriPart))
        and part.mime_type is None
        and mime_type is not None
    ):
        part.mime_type = mime_type
    return part


def _media_to_part(
    media: Any,
    modality: Modality,
    file_cls: type[File],
) -> MessagePart | None:
    media_ref = _get_media_reference(media)
    mime_type = _resolve_media_mime_type(media, file_cls, media_ref)

    url = _get_property_value(media, "url")
    if url:
        return _url_to_media_part(
            str(url), modality=modality, mime_type=mime_type
        )

    content = _get_property_value(media, "content")
    if isinstance(content, (bytes, bytearray)):
        return BlobPart(
            mime_type=mime_type,
            modality=modality,
            content=bytes(content),
        )
    if (
        isinstance(media, file_cls)
        or (isinstance(media, Mapping) and modality == Modality.DOCUMENT)
    ) and isinstance(content, str):
        return BlobPart(
            mime_type=mime_type,
            modality=modality,
            content=content.encode("utf-8"),
        )

    filepath = _get_property_value(media, "filepath")
    if filepath:
        return UriPart(
            mime_type=mime_type,
            modality=modality,
            uri=Path(str(filepath)).absolute().as_uri(),
        )

    if media_ref is not None:
        ref_url = _get_property_value(media_ref, "url")
        if ref_url:
            return _url_to_media_part(
                str(ref_url), modality=modality, mime_type=mime_type
            )
        return None

    # File.id is a provider file reference only when no other source was supplied;
    # Agno 3.x auto-generates a UUID id whenever another source is present.
    external = _get_property_value(media, "external")
    media_id = _get_property_value(media, "id")
    if (
        (
            isinstance(media, file_cls)
            or (isinstance(media, Mapping) and modality == Modality.DOCUMENT)
        )
        and external is None
        and media_id
    ):
        return FilePart(
            mime_type=mime_type,
            modality=modality,
            file_id=str(media_id),
        )

    return None


def extract_media_parts(
    *,
    images: Sequence[Any] | None = None,
    audio: Sequence[Any] | None = None,
    videos: Sequence[Any] | None = None,
    files: Sequence[Any] | None = None,
) -> list[MessagePart]:
    """Convert Agno media sequences into GenAI message parts."""
    if not (images or audio or videos or files):
        return []

    from agno.media import File  # pylint: disable=import-outside-toplevel

    parts: list[MessagePart] = []
    media_groups: tuple[tuple[Sequence[Any] | None, Modality], ...] = (
        (images, Modality.IMAGE),
        (audio, Modality.AUDIO),
        (videos, Modality.VIDEO),
        (files, Modality.DOCUMENT),
    )
    for items, modality in media_groups:
        for item in items or ():
            part = _media_to_part(item, modality, File)
            if part is not None:
                parts.append(part)
    return parts


def is_agno_message(val: object) -> TypeGuard[Message]:
    """Return True if val is an Agno Message instance."""
    if val is None:
        return False
    from agno.models.message import (  # pylint: disable=import-outside-toplevel
        Message,
    )

    return isinstance(val, Message)


def is_workflow_execution_input(
    val: object,
) -> TypeGuard[WorkflowExecutionInput]:
    """Return True if val is an Agno WorkflowExecutionInput instance."""
    if val is None:
        return False
    from agno.workflow.types import (  # pylint: disable=import-outside-toplevel
        WorkflowExecutionInput,
    )

    return isinstance(val, WorkflowExecutionInput)


def is_message_like(val: object) -> bool:
    """Return True if val is an Agno Message instance or a message-shaped mapping."""
    if val is None:
        return False
    if is_agno_message(val):
        return True
    return isinstance(val, Mapping) and "role" in val


def is_message_sequence(val: object) -> bool:
    """Return True if val is a non-empty sequence containing message-like objects."""
    if (
        isinstance(val, Sequence)
        and not isinstance(val, (str, bytes, bytearray, Mapping))
        and len(cast("Sequence[object]", val)) > 0
    ):
        return any(
            is_message_like(item) for item in cast("Sequence[object]", val)
        )
    return False


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


def resolve_embedder_provider(embedder: Embedder) -> str:
    """Resolve the ``gen_ai.provider.name`` value for an Agno embedder instance."""
    for cls in type(embedder).__mro__:
        name = cls.__name__
        if name in ("GeminiEmbedder", "GoogleEmbedder"):
            vertexai = (
                getattr(embedder, "vertexai", False)
                or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower()
                == "true"
            )
            return (
                GenAiProviderNameValues.GCP_VERTEX_AI.value
                if vertexai
                else GenAiProviderNameValues.GCP_GEMINI.value
            )
        if name in ("OpenAILikeEmbedder", "Embedder"):
            break
        if name in _CLASS_NAME_TO_PROVIDER:
            return _CLASS_NAME_TO_PROVIDER[name]
    return _UNKNOWN_PROVIDER
