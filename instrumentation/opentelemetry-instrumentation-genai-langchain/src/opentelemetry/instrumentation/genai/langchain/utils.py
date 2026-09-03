# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import decode_base64, image_from_url

# Mapping from LangChain ``ls_provider`` metadata values to the well-known
# ``gen_ai.provider.name`` values defined by the GenAI semantic conventions.
_PROVIDER_NAME_OVERRIDES: dict[str, str] = {
    "amazon_bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "bedrock_converse": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "azure_openai": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "vertexai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google_vertexai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google_genai": GenAIAttributes.GenAiProviderNameValues.GCP_GEN_AI.value,
    "google_generativeai": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "mistralai": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "mistral": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
}


def is_stream_end_marker(
    token: str | list[str | dict[str, Any]],
    chunk: GenerationChunk | ChatGenerationChunk | None,
) -> bool:
    """Whether an ``on_llm_new_token`` call carries LangChain's end-of-stream marker."""
    if token:
        return False
    message = getattr(chunk, "message", None)
    return getattr(message, "chunk_position", None) == "last"


def normalize_provider(metadata: dict[str, Any] | None) -> str | None:
    """Return the spec ``gen_ai.provider.name`` value derived from metadata.

    Returns ``None`` when no provider can be determined; callers decide how
    to handle that (typically by skipping the span).
    """
    if not metadata:
        return None
    raw = metadata.get("ls_provider")
    if not isinstance(raw, str) or not raw:
        return None
    return _PROVIDER_NAME_OVERRIDES.get(raw, raw)


_ROLE_BY_CLASS: tuple[tuple[type[BaseMessage], Role], ...] = (
    (ToolMessage, Role.TOOL),
    (FunctionMessage, Role.TOOL),
    (AIMessage, Role.ASSISTANT),
    (HumanMessage, Role.USER),
    (SystemMessage, Role.SYSTEM),
)


def _normalize_role(message: BaseMessage) -> str | None:
    # ChatMessage is not binding to a specific role
    # but it carries a role in the `role` field.
    if isinstance(message, ChatMessage):
        return message.role or None
    for message_class, role in _ROLE_BY_CLASS:
        if isinstance(message, message_class):
            return role.value
    return None


def _blob_from_base64(data: Any, mime_type: Any) -> MessagePart | None:
    if not isinstance(data, str):
        return None
    decoded = decode_base64(data)
    if decoded is None:
        return None
    return BlobPart(
        mime_type=mime_type if isinstance(mime_type, str) else None,
        modality="image",
        content=decoded,
    )


def _file_from_id(file_id: Any, mime_type: Any) -> MessagePart | None:
    """Build a :class:`FilePart` for a provider-hosted image reference."""
    if not isinstance(file_id, str) or not file_id:
        return None
    return FilePart(
        mime_type=mime_type if isinstance(mime_type, str) else None,
        modality="image",
        file_id=file_id,
    )


def _media_part(item: dict[str, Any]) -> MessagePart | None:
    """Convert a LangChain multimodal image content block into a media part.

    Handles every image block shape LangChain chat models accept:

    - OpenAI style ``{"type": "image_url", "image_url": {"url": ...}}`` (or a
      bare ``"image_url": "..."`` string). A ``data:<mime>;base64,<payload>``
      URL becomes a :class:`BlobPart`; any other URL becomes a :class:`UriPart`.
    - OpenAI Responses API ``{"type": "input_image", "image_url": "..."}``,
      which langchain-openai passes through verbatim.
    - Anthropic style ``{"type": "image", "source": {...}}`` where ``source``
      is either ``{"type": "base64", "media_type": ..., "data": ...}`` (→
      :class:`BlobPart`) or ``{"type": "url", "url": ...}`` (→ :class:`UriPart`).
    - langchain-core 0.3 standard blocks ``{"type": "image", "source_type":
      "base64"|"url"|"id", "data"/"url"/"id": ..., "mime_type": ...}``.
    - langchain-core 1.x standard blocks ``{"type": "image",
      "base64"|"url"|"id": ..., "mime_type": ...}``.

    A provider-hosted image, referenced by file id rather than carrying bytes
    or a URL, becomes a :class:`FilePart`. Returns ``None`` only when the
    block carries no recordable reference at all.
    """
    block_type = item.get("type")
    # OpenAI style: {"type": "image_url", "image_url": {"url": ...}}
    if block_type in ("image_url", "input_image"):
        image_url = item.get("image_url")
        url: str | None = None
        if isinstance(image_url, str):
            url = image_url
        elif isinstance(image_url, dict):
            image_url_dict = cast(dict[str, Any], image_url)
            raw_url = image_url_dict.get("url")
            url = raw_url if isinstance(raw_url, str) else None
        if not url:
            return _file_from_id(item.get("file_id"), item.get("mime_type"))
        return image_from_url(url)
    if block_type != "image":
        return None

    # Anthropic style: {"type": "image", "source": {...}}
    source = item.get("source")
    if isinstance(source, dict):
        source_dict = cast(dict[str, Any], source)
        source_type = source_dict.get("type")
        if source_type == "base64":
            return _blob_from_base64(
                source_dict.get("data"), source_dict.get("media_type")
            )
        if source_type == "url":
            source_url = source_dict.get("url")
            if isinstance(source_url, str) and source_url:
                return image_from_url(source_url)
            return None
        if source_type == "file":
            return _file_from_id(
                source_dict.get("file_id"), source_dict.get("media_type")
            )
        return None

    # langchain-core 0.3 standard block: tagged with "source_type". Standard
    # blocks name the MIME key ``mime_type``, not Anthropic's ``media_type``.
    source_type = item.get("source_type")
    if source_type == "base64":
        return _blob_from_base64(item.get("data"), item.get("mime_type"))
    if source_type == "url":
        standard_url = item.get("url")
        if isinstance(standard_url, str) and standard_url:
            return image_from_url(standard_url)
        return None
    if source_type == "id":
        return _file_from_id(item.get("id"), item.get("mime_type"))

    # langchain-core 1.x standard block: payload keys sit at the top level.
    base64_data = item.get("base64")
    if isinstance(base64_data, str):
        return _blob_from_base64(base64_data, item.get("mime_type"))
    standard_url = item.get("url")
    if isinstance(standard_url, str) and standard_url:
        return image_from_url(standard_url)
    return _file_from_id(
        item.get("id") or item.get("file_id"), item.get("mime_type")
    )


def _content_to_parts(
    content: str | list[str | dict[str, Any]],
) -> list[MessagePart]:
    """Convert a LangChain message ``content`` payload into ``MessagePart`` s.

    Content may be a plain string or a list of provider-specific block dicts
    (e.g. Anthropic structured content). We extract :class:`TextPart` and
    :class:`ReasoningPart` parts; ``tool_use`` blocks are intentionally ignored
    here because LangChain consolidates them into ``message.tool_calls`` which
    is read separately.
    """
    parts: list[MessagePart] = []
    if isinstance(content, str):
        if content:
            parts.append(TextPart(content=content))
        return parts
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(TextPart(content=item))
            continue
        block_type = item.get("type")
        if block_type in ("text", "input_text", "output_text"):
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value:
                parts.append(TextPart(content=text_value))
        elif block_type in ("thinking", "reasoning"):
            reasoning_value = (
                item.get("thinking")
                or item.get("reasoning")
                or item.get("text")
            )
            if isinstance(reasoning_value, str) and reasoning_value:
                parts.append(ReasoningPart(content=reasoning_value))
        elif block_type in ("image_url", "image", "input_image"):
            media = _media_part(item)
            if media is not None:
                parts.append(media)
    return parts


def _has_content(message: BaseMessage) -> bool:
    # Whether the message carried content we failed to convert.
    content = message.content
    if isinstance(content, str):
        return bool(content)
    return any(bool(block) for block in content)


def _legacy_function_call_request(
    message: AIMessage,
) -> ToolCallRequestPart | None:
    """Extract a legacy OpenAI ``function_call`` as a :class:`ToolCallRequestPart`.

    Pre-tools OpenAI models return a single call under
    ``additional_kwargs['function_call']`` (``{"name", "arguments"}``) rather
    than ``message.tool_calls``. ``arguments`` may be a JSON string or an
    already-decoded mapping. Returns ``None`` when absent or unnamed.
    """
    function_call = message.additional_kwargs.get("function_call")
    if not isinstance(function_call, dict):
        return None
    function_call = cast(dict[str, Any], function_call)
    name = function_call.get("name")
    if not name:
        return None
    raw_arguments = function_call.get("arguments")
    arguments: Any = raw_arguments
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = raw_arguments
    return ToolCallRequestPart(arguments=arguments, name=name, id=None)


def _ai_message_parts(message: AIMessage) -> list[MessagePart]:
    """Build :class:`MessagePart` s for an :class:`AIMessage`.

    Includes any text/reasoning content followed by a
    :class:`ToolCallRequestPart` for each entry in ``message.tool_calls``, plus a
    legacy ``additional_kwargs['function_call']`` when present.
    """
    parts: list[MessagePart] = _content_to_parts(message.content)
    for call in message.tool_calls:
        name = call["name"]
        if not name:
            continue
        parts.append(
            ToolCallRequestPart(
                arguments=call["args"],
                name=name,
                id=call["id"],
            )
        )
    if not message.tool_calls:
        legacy_call = _legacy_function_call_request(message)
        if legacy_call is not None:
            parts.append(legacy_call)
    return parts


def _tool_message_parts(message: ToolMessage) -> list[MessagePart]:
    """Build :class:`MessagePart` s for a :class:`ToolMessage` (tool result)."""
    tool_call_id = getattr(message, "tool_call_id", None)
    return [
        ToolCallResponsePart(
            response=message.content,
            id=tool_call_id if isinstance(tool_call_id, str) else None,
        )
    ]


def _message_parts(message: BaseMessage) -> list[MessagePart]:
    if isinstance(message, ToolMessage):
        return _tool_message_parts(message)
    if isinstance(message, AIMessage):
        return _ai_message_parts(message)
    return _content_to_parts(message.content)


def to_input_messages(
    messages: Iterable[Any],
) -> list[InputMessage]:
    """Convert LangChain messages into spec-conformant ``InputMessage`` s.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    try:
        normalized_messages: Iterable[BaseMessage] = convert_to_messages(
            list(messages)
        )
    except Exception:  # pylint: disable=broad-except
        normalized_messages = [
            m for m in messages if isinstance(m, BaseMessage)
        ]
    result: list[InputMessage] = []
    for message in normalized_messages:
        parts = _message_parts(message)
        if not parts and not _has_content(message):
            continue
        result.append(
            InputMessage(
                role=_normalize_role(message) or Role.USER.value,
                parts=parts,
            )
        )
    return result


def to_output_messages(
    messages: Iterable[BaseMessage],
    *,
    finish_reason: str = "",
) -> list[OutputMessage]:
    """Convert LangChain ``AIMessage`` instances into ``OutputMessage`` s.

    Non-``AIMessage`` entries are skipped: only assistant turns are recorded
    as ``gen_ai.output.messages``. Tool execution results belong on the
    *input* side of the next inference call, not the output side of the
    previous one.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    result: list[OutputMessage] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        parts = _ai_message_parts(message)
        if not parts and not _has_content(message):
            continue
        result.append(
            OutputMessage(
                role=_normalize_role(message) or Role.ASSISTANT.value,
                parts=parts,
                finish_reason=finish_reason,
            )
        )
    return result


def _get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)

    return getattr(obj, property_name, None)


def prepare_tool_definitions(tools: list[Any]) -> list[ToolDefinition] | None:
    if not tools:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tools:
        tool_type = _get_property_value(tool, "type")
        if tool_type == "function":
            func = _get_property_value(tool, "function")
            if func:
                func_name = _get_property_value(func, "name")
                func_description = _get_property_value(func, "description")
                definitions.append(
                    FunctionToolDefinition(
                        name=str(func_name) if func_name is not None else "",
                        description=str(func_description)
                        if func_description is not None
                        else None,
                        parameters=_get_property_value(func, "parameters"),
                    )
                )
        elif (
            tool_type is None and _get_property_value(tool, "name") is not None
        ):
            # Pre-tools OpenAI ``functions`` entries are flat mappings
            # (``{"name", "description", "parameters"}``) with no ``type`` or
            # nested ``function`` wrapper. Surface them so the legacy
            # function-calling path still populates ``gen_ai.tool.definitions``.
            func_name = _get_property_value(tool, "name")
            func_description = _get_property_value(tool, "description")
            definitions.append(
                FunctionToolDefinition(
                    name=str(func_name),
                    description=str(func_description)
                    if func_description is not None
                    else None,
                    parameters=_get_property_value(tool, "parameters"),
                )
            )
    return definitions or None


def make_input_message(data: Any) -> list[InputMessage]:
    """Build ``InputMessage`` s from a workflow/agent input mapping.

    When ``data['messages']`` is present, every LangChain ``BaseMessage`` in it
    is converted via :func:`to_input_messages` (which preserves the original
    role: a prior ``AIMessage`` becomes ``role='assistant'``, a
    ``SystemMessage`` becomes ``role='system'``, and so on) and includes
    tool-call structure.

    When no ``messages`` key exists (common in LangGraph state dicts), the
    remaining state fields are serialized as JSON and emitted as a single
    user-role :class:`TextPart` part.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    messages: Any = data_dict.get("messages")
    if messages is not None:
        if isinstance(messages, (str, bytes)) or not isinstance(
            messages, Iterable
        ):
            return []
        return to_input_messages(cast(Iterable[BaseMessage], messages))
    # Fallback: serialize non-message state fields as input.
    # Common in LangGraph where nodes use structured state fields
    # (e.g., user_query) rather than a message list.
    exclude_keys = {"messages", "intermediate_steps"}
    input_data: dict[str, Any] = {
        k: v
        for k, v in data_dict.items()
        if k not in exclude_keys and v is not None
    }
    if input_data:
        serialized = serialize(input_data)
        if serialized:
            return [
                InputMessage(
                    role=Role.USER.value, parts=[TextPart(serialized)]
                )
            ]
    return []


def make_output_message(data: Any) -> list[OutputMessage]:
    """Build ``OutputMessage`` s from a workflow/agent output mapping.

    Only ``AIMessage`` entries become outputs. ``finish_reason`` is left
    empty: the underlying per-LLM-call finish reasons are recorded on child
    inference spans, and util-genai filters empty values out of
    ``gen_ai.response.finish_reasons``.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    messages: Any = data_dict.get("messages")
    if (
        messages is None
        or isinstance(messages, (str, bytes))
        or not isinstance(messages, Iterable)
    ):
        return []
    return to_output_messages(cast(Iterable[BaseMessage], messages))


def make_last_output_message(data: Any) -> list[OutputMessage]:
    """Extract only the last AI message as the output.

    For Workflow and AgentInvocation spans, the final AI message best represents
    the actual output. Intermediate AI messages (e.g., tool-call decisions) are
    already captured in child LLM invocation spans.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    all_messages = make_output_message(data)
    if all_messages:
        return [all_messages[-1]]
    return []


def serialize(obj: Any) -> str | None:
    """Serialize object to JSON string.

    Uses default=str to handle non-JSON-serializable objects (like LangChain
    message objects) by converting them to their string representation while
    keeping the overall structure as valid JSON.
    """
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def response_fields_from_generation(
    chat_generation: Any,
) -> tuple[str | None, str | None]:
    """Return the ``(response_model, response_id)`` a generation carries."""
    message = getattr(chat_generation, "message", None)
    response_metadata: Mapping[str, Any] = (
        getattr(message, "response_metadata", None) or {}
    )
    generation_info: Mapping[str, Any] = (
        getattr(chat_generation, "generation_info", None) or {}
    )

    response_model = _first_string(
        response_metadata.get("model_name"),
        response_metadata.get("model"),
        generation_info.get("model_name"),
        generation_info.get("model"),
    )
    response_id = _first_string(
        response_metadata.get("id"),
        generation_info.get("id"),
    )
    return response_model, response_id


def resolve_response_model_and_id(
    *,
    llm_output: Mapping[str, Any] | None,
    served_model: str | None,
    generation_model: str | None,
    generation_response_id: str | None,
) -> tuple[str | None, str | None]:
    """Return the ``gen_ai.response.model`` and ``gen_ai.response.id`` to record.

    The model comes from the Responses API header if there is one, else
    ``llm_output``, else the generation. The id comes from ``llm_output``, else
    the generation.
    """
    response_model: str | None = None
    response_id: str | None = None

    if llm_output is not None:
        raw_model = llm_output.get("model_name") or llm_output.get("model")
        if raw_model is not None:
            response_model = str(raw_model)
        raw_id = llm_output.get("id")
        if raw_id is not None:
            response_id = str(raw_id)

    if response_model is None:
        response_model = generation_model
    if response_id is None:
        response_id = generation_response_id
    if served_model:
        response_model = served_model

    return response_model, response_id


def extract_token_details(usage_metadata: dict[str, Any]) -> dict[str, int]:
    """Extract cache/reasoning token break-downs from LangChain usage metadata."""

    token_details: dict[str, int] = {}
    raw_input_details = usage_metadata.get("input_token_details")
    input_details: dict[str, Any] = (
        cast(dict[str, Any], raw_input_details)
        if isinstance(raw_input_details, dict)
        else {}
    )
    raw_output_details = usage_metadata.get("output_token_details")
    output_details: dict[str, Any] = (
        cast(dict[str, Any], raw_output_details)
        if isinstance(raw_output_details, dict)
        else {}
    )

    cache_creation = input_details.get("cache_creation")
    if isinstance(cache_creation, int) and cache_creation:
        token_details["cache_creation_input_tokens"] = cache_creation

    cache_read = input_details.get("cache_read")
    if isinstance(cache_read, int) and cache_read:
        token_details["cache_read_input_tokens"] = cache_read

    reasoning = output_details.get("reasoning")
    if isinstance(reasoning, int) and reasoning:
        token_details["reasoning_tokens"] = reasoning

    return token_details
