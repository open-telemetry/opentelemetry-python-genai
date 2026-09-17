# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert smolagents message, tool, and agent-run values to GenAI types.

smolagents passes ``generate(messages=...)`` a list of ``ChatMessage`` objects
or plain dicts, and returns a ``ChatMessage``. ``MultiStepAgent.run`` takes a
task string and optional images, and returns a final answer. This module maps
those values, and the ``tools_to_call_from`` tool objects, onto the types in
``opentelemetry.util.genai.types``. util-genai then serializes them into
``gen_ai.input.messages``, ``gen_ai.output.messages``, and
``gen_ai.tool.definitions``.
"""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from PIL.Image import Image
from smolagents.models import get_tool_json_schema
from smolagents.utils import encode_image_base64

from opentelemetry.util.genai.types import (
    BlobPart,
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Role,
    TextPart,
    ToolDefinition,
    UriPart,
)

if TYPE_CHECKING:
    from smolagents.agents import MultiStepAgent
    from smolagents.models import ChatMessage, MessageRole
    from smolagents.tools import Tool

_logger = logging.getLogger(__name__)

# One element of a message ``content`` list, typed as smolagents types it:
# ``ChatMessage.content`` is ``str | list[dict[str, Any]]``. The values within an
# element have mixed types, a ``str`` under ``text``, a nested dict under
# ``image_url``, and a PIL image under ``image``.
_ContentElement: TypeAlias = dict[str, Any]

# smolagents passes managed agents in ``tools_to_call_from`` alongside tools.
_ModelCallable: TypeAlias = "Tool | MultiStepAgent"

_DEFAULT_IMAGE_MIME_TYPE = "image/png"
_DATA_URL_PREFIX = "data:"

# smolagents-internal roles -> semconv ``gen_ai`` message roles. smolagents
# applies the same mapping (``models.tool_role_conversions``) inside
# ``generate``, after the wrapper has already read ``messages``, so the wrapper
# has to apply it too. Without this map the input messages would carry roles
# the spec doesn't define. A model configured with ``custom_role_conversions``
# overrides the mapping and can send different roles. Roles that are already
# spec values pass through unchanged.
_ROLE_MAP: dict[str, str] = {
    "tool-call": Role.ASSISTANT.value,
    "tool-response": Role.USER.value,
}


def _unwrap_role(role: MessageRole | str | None) -> str | None:
    if role is None:
        return None
    if isinstance(role, Enum):
        role = role.value
    name = str(role)
    return _ROLE_MAP.get(name, name)


# TODO leverage util helpers
def _decode_base64_image(image: str) -> tuple[bytes, str] | None:
    """Decode a base64 payload or data URL into ``(bytes, mime_type)``."""
    mime_type = _DEFAULT_IMAGE_MIME_TYPE
    if image.startswith(_DATA_URL_PREFIX):
        header, _, image = image.partition(",")
        media_type = header[len(_DATA_URL_PREFIX) :].split(";")[0]
        if media_type:
            mime_type = media_type
    try:
        return base64.b64decode(image, validate=True), mime_type
    except (binascii.Error, ValueError):
        _logger.debug("Failed to decode a base64 image", exc_info=True)
        return None


def _encode_base64_image(image: Image) -> str | None:
    try:
        encoded = encode_image_base64(image)
    except BaseException:  # pylint: disable=broad-except
        _logger.debug(
            "Failed to encode image of type %s, dropping it from telemetry",
            type(image).__name__,
            exc_info=True,
        )
        return None
    return encoded if isinstance(encoded, str) else None


def _image_blob(image: Image | str) -> BlobPart | None:
    """Build a ``BlobPart`` part from a base64 string, data URL, or PIL image."""
    if isinstance(image, str):
        decoded = _decode_base64_image(image)
    else:
        encoded = _encode_base64_image(image)
        decoded = (
            _decode_base64_image(encoded) if encoded is not None else None
        )
    if decoded is None:
        return None
    content, mime_type = decoded
    return BlobPart(mime_type=mime_type, modality="image", content=content)


def _image_part_from_element(
    element: _ContentElement,
) -> UriPart | BlobPart | None:
    content_type = element.get("type")
    if content_type == "image_url":
        image_url = element.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else None
        if isinstance(url, str) and url:
            return UriPart(mime_type=None, modality="image", uri=url)
        return None
    if content_type == "image":
        image = element.get("image")
        if image is not None:
            return _image_blob(image)
    return None


def _parts_from_content(
    content: str | list[_ContentElement] | None,
) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if isinstance(content, str):
        parts.append(TextPart(content=content))
        return parts
    if isinstance(content, list):
        for element in content:
            if not isinstance(element, dict):
                _logger.debug(
                    "Unknown message content dropped from telemetry: %s",
                    type(element).__name__,
                )
                continue
            if element.get("type") == "text" and (text := element.get("text")):
                parts.append(TextPart(content=text))
                continue
            if image_part := _image_part_from_element(element):
                parts.append(image_part)
            else:
                _logger.debug(
                    "Unknown message part dropped from telemetry: %s",
                    element.get("type"),
                )
    return parts


def _get_role_and_content(
    message: ChatMessage | dict[str, Any],
) -> tuple[MessageRole | str | None, str | list[_ContentElement] | None]:
    # smolagents reads a message the same way: a dict goes through
    # ChatMessage.from_dict, anything else has its attributes read directly.
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    return message.role, message.content


def to_input_messages(
    messages: list[ChatMessage | dict[str, Any]] | None,
) -> list[InputMessage]:
    """Map smolagents ``generate`` input messages to ``InputMessage`` objects."""
    result: list[InputMessage] = []
    if not isinstance(messages, list):
        return result
    for message in messages:
        raw_role, content = _get_role_and_content(message)
        role = _unwrap_role(raw_role)
        if not role:
            continue
        result.append(
            InputMessage(role=role, parts=_parts_from_content(content))
        )
    return result


def to_output_message(output_message: ChatMessage) -> OutputMessage:
    """Map a smolagents ``ChatMessage`` response to an ``OutputMessage``.

    The in-process runtimes return the generated text and nothing else: no tool
    calls (the agent parses those out of the text afterwards), no reasoning
    content, and no finish reason. ``OutputMessage`` requires
    ``finish_reason``, and util-genai drops an empty value when it emits
    ``gen_ai.response.finish_reasons``. Defaulting it to ``"stop"`` instead
    would make a generation cut short by ``max_new_tokens`` look like a natural
    stop.
    """
    role = _unwrap_role(output_message.role) or Role.ASSISTANT.value
    parts = _parts_from_content(output_message.content)
    return OutputMessage(role=role, parts=parts, finish_reason="")


def _tool_parameters(tool: _ModelCallable) -> dict[str, Any] | None:
    """Return the JSON Schema ``parameters`` object for a smolagents tool.

    A tool's ``inputs`` map is not a JSON Schema on its own: smolagents wraps it
    in an object schema, derives ``required`` from ``nullable``, and rewrites its
    non-JSON-Schema ``"any"`` type. ``get_tool_json_schema`` builds exactly the
    schema the provider receives.

    ``get_tool_json_schema`` also accepts managed agents despite its ``Tool``
    annotation.
    """
    try:
        schema = get_tool_json_schema(cast("Tool", tool))
        parameters = schema["function"]["parameters"]
    except BaseException:  # pylint: disable=broad-except
        _logger.debug(
            "Failed to build a JSON Schema for %s",
            type(tool).__name__,
            exc_info=True,
        )
        return None
    return parameters if isinstance(parameters, dict) else None


def to_tool_definitions(
    tools: Sequence[_ModelCallable] | None,
) -> list[ToolDefinition] | None:
    """Map values accepted by smolagents' schema builder."""
    if not tools:
        return None
    definitions: list[ToolDefinition] = []
    for tool in tools:
        parameters = _tool_parameters(tool)
        if parameters is None:
            continue
        definitions.append(
            FunctionToolDefinition(
                name=cast(str, tool.name),
                description=tool.description,
                parameters=parameters,
            )
        )
    return definitions or None


def final_answer_parts(output: object) -> list[MessagePart]:
    """Convert image and string answers without file-writing ``__str__`` calls."""
    if isinstance(output, Image):
        if blob := _image_blob(output):
            return [blob]
        return []
    text = str.__str__(output) if isinstance(output, str) else str(output)
    return [TextPart(content=text)]


def task_to_input_messages(
    task: str | None, images: list[Image | str] | None
) -> list[InputMessage]:
    parts: list[MessagePart] = []
    if task:
        parts.append(TextPart(content=task))
    if isinstance(images, list):
        for image in images:
            if blob := _image_blob(image):
                parts.append(blob)
    if not parts:
        return []
    return [InputMessage(role="user", parts=parts)]
