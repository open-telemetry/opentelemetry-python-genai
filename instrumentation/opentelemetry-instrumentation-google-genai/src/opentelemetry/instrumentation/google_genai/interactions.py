# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections.abc import (
    AsyncIterable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from sys import float_info
from typing import Any, cast

try:
    # Google GenAI < 2.9.0
    from google.genai._interactions._streaming import AsyncStream, Stream
    from google.genai._interactions.resources.interactions import (
        AsyncInteractionsResource,
        InteractionsResource,
    )
    from google.genai._interactions.types.interaction import Interaction, Usage
    from google.genai._interactions.types.interaction_create_params import (
        Input,
    )
    from google.genai._interactions.types.interaction_sse_event import (
        InteractionSSEEvent,
    )
    from google.genai._interactions.types.step import Step

    _HAS_INTERACTIONS = True
except ImportError:
    try:
        # Google GenAI >= 2.9.0
        from google.genai._gaos.interactions import (
            AsyncInteractions as AsyncInteractionsResource,
        )
        from google.genai._gaos.interactions import (
            AsyncStream,
            Stream,
        )
        from google.genai._gaos.interactions import (
            Interactions as InteractionsResource,
        )
        from google.genai._gaos.types.interactions import (
            Interaction,
            InteractionSSEEvent,
            Step,
            Usage,
        )
        from google.genai._gaos.types.interactions import (
            InteractionsInput as Input,
        )

        _HAS_INTERACTIONS = True
    except ImportError:
        _HAS_INTERACTIONS = False

        # Placeholders for older versions where interactions are not supported
        class InteractionsResource:
            create = None
            get = None

        class AsyncInteractionsResource:
            create = None
            get = None

        class Interaction:
            model = None
            usage = None

        class Usage:
            total_input_tokens = None
            total_output_tokens = None
            total_thought_tokens = None

        class Input:
            pass

        class InteractionSSEEvent:
            pass

        class Step:
            pass

        class Stream:
            pass

        class AsyncStream:
            pass


from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.google_genai._error_type import (
    resolve_error_type,
)
from opentelemetry.instrumentation.google_genai.client_info import (
    get_client_info as _get_client_info,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    FetchResponseInvocation,
    InferenceInvocation,
    RemoteAgentInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
    Error,
    FunctionToolDefinition,
    GenericPart,
    GenericToolDefinition,
    InputMessage,
    MessagePart,
    Modality,
    ModalityTokens,
    OutputMessage,
    Role,
    ServerToolCallPart,
    ServerToolCallResponsePart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
    UriPart,
)

# Interaction.status values that differ from the gen_ai.response.status value
# set. `in_progress`, `completed`, `failed`, `cancelled` and `incomplete` are
# already spelled identically; `budget_exceeded` maps onto `incomplete`, which
# semconv describes as generation stopping short of completion. semconv has no
# member for `requires_action` -- none covers waiting on the caller to supply
# tool results -- so it is reported as a provider-specific value, which semconv
# permits when no member applies.
_INTERACTION_STATUS_TO_SEMCONV: dict[str, str] = {
    "budget_exceeded": "incomplete",
}


# Interaction.status describes the lifecycle of the *original* generation, so it
# maps onto the semconv finish reason value set. A still-running interaction has
# no finish reason yet.
_INTERACTION_STATUS_FINISH_REASONS: dict[str, str] = {
    "budget_exceeded": "length",
    "cancelled": "error",
    "completed": "stop",
    "failed": "error",
    "incomplete": "length",
    "requires_action": "tool_calls",
}


class _InteractionsMethodsSnapshot:
    def __init__(self) -> None:
        self._original_create = InteractionsResource.create
        self._original_create_code = InteractionsResource.create.__code__
        self._original_async_create = AsyncInteractionsResource.create
        self._original_async_create_code = (
            AsyncInteractionsResource.create.__code__
        )
        self._original_get = InteractionsResource.get
        self._original_get_code = InteractionsResource.get.__code__
        self._original_async_get = AsyncInteractionsResource.get
        self._original_async_get_code = AsyncInteractionsResource.get.__code__

    def restore(self) -> None:
        self._original_create.__code__ = self._original_create_code
        self._original_async_create.__code__ = self._original_async_create_code

        InteractionsResource.create = self._original_create
        AsyncInteractionsResource.create = self._original_async_create

        self._original_get.__code__ = self._original_get_code
        self._original_async_get.__code__ = self._original_async_get_code

        InteractionsResource.get = self._original_get
        AsyncInteractionsResource.get = self._original_async_get


# Magic incantation used by native Google ADK instrumentation to identify
# instrumented functions and suppress its own internal tracing when OTel is active.
def _set_co_filename(wrapped: object) -> None:
    wrapped.__wrapped__.__code__ = wrapped.__wrapped__.__code__.replace(
        co_filename=__file__.replace("\\", "/")
    )


def _apply_interaction_response_attributes(
    response: Interaction,
    invocation: InferenceInvocation | RemoteAgentInvocation,
    telemetry_handler: TelemetryHandler,
) -> None:
    if isinstance(invocation, InferenceInvocation):
        invocation.response_model_name = response.model
        if getattr(response, "id", None):
            invocation.response_id = response.id

    usage = response.usage or Usage()

    invocation.input_tokens = usage.total_input_tokens
    invocation.output_tokens = usage.total_output_tokens
    invocation.cache_read_input_tokens = usage.total_cached_tokens

    if isinstance(invocation, InferenceInvocation):
        invocation.thinking_tokens = usage.total_thought_tokens

        invocation.set_input_tokens(
            _modality_tokens(usage, "input_tokens_by_modality")
        )
        invocation.set_output_tokens(
            _modality_tokens(usage, "output_tokens_by_modality")
        )
        invocation.set_cache_read_input_tokens(
            _modality_tokens(usage, "cached_tokens_by_modality")
        )

    if telemetry_handler.should_capture_content():
        invocation.output_messages = _interactions_response_to_messages(
            response
        )


def _modality_tokens(usage: Any, name: str) -> ModalityTokens | None:
    entries = _get_field(usage, name)
    if entries is None:
        return None
    return [
        (_get_field(entry, "modality") or "", _get_field(entry, "tokens"))
        for entry in entries
    ]


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


_SERVER_TOOL_CALL_NAMES = {
    "code_execution_call": "code_execution",
    "file_search_call": "file_search",
    "google_maps_call": "google_maps",
    "google_search_call": "google_search",
    "mcp_server_tool_call": "mcp",
    "processing_call": "processing",
    "retrieval_call": "retrieval",
    "url_context_call": "url_context",
}

_SERVER_TOOL_RESPONSE_NAMES = {
    "code_execution_result": "code_execution",
    "file_search_result": "file_search",
    "google_maps_result": "google_maps",
    "google_search_result": "google_search",
    "mcp_server_tool_result": "mcp",
    "processing_result": "processing",
    "retrieval_result": "retrieval",
    "url_context_result": "url_context",
}

_TOOL_STEP_TYPES = {
    "function_call",
    "function_result",
    *_SERVER_TOOL_CALL_NAMES,
    *_SERVER_TOOL_RESPONSE_NAMES,
}


def _to_server_tool_part(
    item: Step | dict[str, object],
) -> MessagePart | None:
    item_type = item.get("type") if isinstance(item, dict) else item.type
    tool_name = _SERVER_TOOL_CALL_NAMES.get(item_type)
    response_name = _SERVER_TOOL_RESPONSE_NAMES.get(item_type)
    if tool_name is None and response_name is None:
        return None

    if isinstance(item, dict):
        payload = dict(item)
    else:
        payload = item.model_dump(exclude_none=True, mode="json")

    item_id = payload.pop("id", None)
    call_id = payload.pop("call_id", None)
    payload.pop("type", None)
    name = payload.pop("name", None)
    canonical_name = tool_name or response_name
    if canonical_name is None:
        return None
    payload["type"] = canonical_name

    if response_name is not None:
        return ServerToolCallResponsePart(
            id=call_id if isinstance(call_id, str) else None,
            server_tool_call_response=payload,
        )
    return ServerToolCallPart(
        id=item_id if isinstance(item_id, str) else None,
        name=name if isinstance(name, str) else canonical_name,
        server_tool_call=payload,
    )


def _interaction_item_to_part(
    item: Step | dict[str, object],
) -> MessagePart | None:
    if isinstance(item, dict):
        item_type = item.get("type")
        item_id = item.get("id")
        name = item.get("name")
        arguments = item.get("arguments")
        call_id = item.get("call_id")
        result = item.get("result")
    else:
        item_type = item.type
        item_id = item.id if item_type == "function_call" else None
        name = item.name if item_type == "function_call" else None
        arguments = item.arguments if item_type == "function_call" else None
        call_id = item.call_id if item_type == "function_result" else None
        result = item.result if item_type == "function_result" else None

    if item_type == "function_call":
        return ToolCallRequestPart(
            id=item_id if isinstance(item_id, str) else None,
            name=name if isinstance(name, str) else "",
            arguments=arguments,
        )
    if item_type == "function_result":
        return ToolCallResponsePart(
            id=call_id if isinstance(call_id, str) else None,
            response=result,
        )
    return _to_server_tool_part(item)


# Logic for parsing Input is tricky:
# https://github.com/open-telemetry/donation-openinference/blob/6cdd644d79fccf50aedcb614187f924ddfcafb7b/python/instrumentation/openinference-instrumentation-google-genai/src/openinference/instrumentation/google_genai/interactions_attributes.py#L103
# It doesn't make sense for this to be a List[InputMessage] (per semconv),
# because this API doesn't take conversation history as input (unlike the generate_content API).
# Conversation history is stored server-side and referenced via a interaction ID parameter.
def _interactions_input_to_messages(
    input_data: Input | None,
) -> list[InputMessage]:
    # None will end up raising an exception by the SDK
    if input_data is None:
        return []
    if isinstance(input_data, str):
        return [
            InputMessage(
                role=Role.USER.value, parts=[TextPart(content=input_data)]
            )
        ]

    if not isinstance(input_data, Sequence):
        input_data = [input_data]

    parts: list[MessagePart] = []
    for item in input_data:
        if isinstance(item, str):
            parts.append(TextPart(content=item))
            continue

        item_type = _get_field(item, "type")
        message_part = None
        if isinstance(item, dict):
            message_part = _interaction_item_to_part(item)
        elif item_type in _TOOL_STEP_TYPES:
            message_part = _interaction_item_to_part(cast(Step, item))

        if message_part is not None:
            parts.append(message_part)
        elif item_type == "text":
            part = TextPart(content=_get_field(item, "text") or "")
            parts.append(part)
        elif item_type == "document":
            part = UriPart(
                mime_type=_get_field(item, "mime_type"),
                modality=Modality.DOCUMENT,
                uri=_get_field(item, "uri") or "",
            )
            parts.append(part)
        elif item_type is not None:
            part = GenericPart(type=item_type)
            parts.append(part)

    return [InputMessage(role=Role.USER.value, parts=parts)]


def _get_interaction_output_text(interaction: Interaction) -> str:
    if getattr(interaction, "output_text", None):
        return interaction.output_text

    texts = []
    if interaction.steps:
        for step in interaction.steps:
            if getattr(step, "type", None) == "model_output":
                content = getattr(step, "content", None)
                if content:
                    for item in content:
                        if getattr(item, "type", None) == "text" and hasattr(
                            item, "text"
                        ):
                            texts.append(item.text)
    return "".join(texts)


# It doesn't make sense for this to be a list of OutputMessage (per semconv),
# because this API doesn't return conversation history as output (unlike the generate_content API).
# Model's response is returned as a list of steps:
# https://ai.google.dev/gemini-api/docs/migrate-to-interactions#basic-input-output
# https://ai.google.dev/api/interactions-api#Resource:Step
def _interactions_response_to_messages(
    interaction: Interaction,
    finish_reason: str | None = "stop",
) -> list[OutputMessage]:
    parts: list[MessagePart] = []
    for step in interaction.steps or []:
        if part := _interaction_item_to_part(step):
            parts.append(part)
            continue
        if step.type != "model_output":
            continue
        for item in step.content or []:
            if item.type == "text" and isinstance(item.text, str):
                text = item.text
                parts.append(TextPart(content=text))

    if not any(isinstance(part, TextPart) for part in parts):
        parts.append(
            TextPart(content=_get_interaction_output_text(interaction))
        )
    return [
        OutputMessage(
            role=Role.ASSISTANT.value,
            parts=parts,
            finish_reason=finish_reason,
        )
    ]


def _interaction_response_status(status: str | None) -> str | None:
    """Return the gen_ai.response.status value for an interaction status.

    Args:
        status: The fetched ``Interaction.status``, or None when absent.

    Returns:
        The matching member of the semconv value set, the status unchanged when
        it is already one (or has no closest member), or None when no status was
        reported.
    """
    if not status:
        return None
    return _INTERACTION_STATUS_TO_SEMCONV.get(status, status)


def _interaction_finish_reason(status: str | None) -> str | None:
    """Return the finish reason implied by an interaction status.

    Args:
        status: The fetched ``Interaction.status``, or None when absent.

    Returns:
        The matching semconv finish reason, or None when the generation has not
        finished (or no status was reported).
    """
    if not status:
        return None
    return _INTERACTION_STATUS_FINISH_REASONS.get(status)


def _interaction_tool_definitions(
    interaction: Interaction,
) -> list[ToolDefinition] | None:
    """Extract the tool definitions carried by a fetched interaction.

    Args:
        interaction: The fetched interaction.

    Returns:
        The tool definitions declared on the interaction, or None when it
        declares none that map to a semconv tool definition.
    """
    tools = _get_field(interaction, "tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return None
    dumped: list[Any] = []
    for tool in tools:
        if isinstance(tool, dict):
            dumped.append(tool)
        elif hasattr(tool, "model_dump"):
            dumped.append(tool.model_dump(exclude_none=True, mode="json"))
    return _maybe_get_tool_definitions(dumped)


def _apply_fetched_interaction_attributes(
    response: Interaction,
    invocation: FetchResponseInvocation,
    telemetry_handler: TelemetryHandler,
    streamed_parts: list[MessagePart] | None = None,
) -> None:
    """Record the attributes of a *fetched* interaction on its invocation.

    Deliberately separate from _apply_interaction_response_attributes: the token
    counts carried by a fetched interaction belong to the original generation
    and must not be reported again here.

    Args:
        response: The fetched interaction.
        invocation: The in-flight fetch invocation to record onto.
        telemetry_handler: Handler consulted for whether to capture content.
        streamed_parts: Parts accumulated from stream events, used when the
            interaction itself carries no content.
    """
    invocation.response_model_name = _get_field(response, "model")
    status = _get_field(response, "status")
    invocation.response_status = _interaction_response_status(status)
    finish_reason = _interaction_finish_reason(status)
    invocation.finish_reasons = [finish_reason] if finish_reason else None

    if telemetry_handler.should_capture_content():
        # gen_ai.tool.definitions is opt-in, so dumping the tools is wasted
        # work unless content is being captured.
        invocation.tool_definitions = _interaction_tool_definitions(response)
        messages = _interactions_response_to_messages(
            response, finish_reason=None
        )
        if streamed_parts and not _has_content(messages):
            messages = _streamed_output_messages(streamed_parts)
        invocation.output_messages = messages
        if system_instruction := _get_field(response, "system_instruction"):
            invocation.system_instruction = [
                TextPart(content=system_instruction)
            ]


class InteractionsStreamWrapper(SyncStreamWrapper[InteractionSSEEvent]):
    def __init__(
        self,
        stream: Iterable[InteractionSSEEvent],
        invocation: InferenceInvocation | RemoteAgentInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        event_type = _get_field(chunk, "event_type")
        if event_type == "interaction_completed":
            interaction = _get_field(chunk, "interaction")
            if interaction:
                self._self_last_interaction = interaction

    def _on_stream_end(self) -> None:
        if self._self_last_interaction:
            _apply_interaction_response_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
            )
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class AsyncInteractionsStreamWrapper(AsyncStreamWrapper[InteractionSSEEvent]):
    def __init__(
        self,
        stream: AsyncIterable[InteractionSSEEvent],
        invocation: InferenceInvocation | RemoteAgentInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        event_type = _get_field(chunk, "event_type")
        if event_type == "interaction_completed":
            interaction = _get_field(chunk, "interaction")
            if interaction:
                self._self_last_interaction = interaction

    def _on_stream_end(self) -> None:
        if self._self_last_interaction:
            _apply_interaction_response_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
            )
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


def _maybe_json(value: str) -> Any:
    """Parse streamed tool arguments, keeping the raw string if incomplete."""
    try:
        return json.loads(value)
    except ValueError:
        return value


def _as_json_value(value: Any) -> Any:
    """A streamed payload value as JSON-compatible data.

    Step deltas carry SDK models, including nested inside lists -- a search
    result set, for instance. Leaving one in place makes the span's content
    attribute unserializable, and the resulting error would surface inside the
    caller's own call rather than in telemetry. ``bytes`` is left alone because
    the util's encoder base64s it.
    """
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    dumped = _as_payload(value)
    return dumped if dumped is not None else str(value)


def _as_payload(value: Any) -> dict[str, Any] | None:
    """A step or delta payload as a JSON-compatible dict, or None if neither."""
    if isinstance(value, dict):
        return {key: _as_json_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True, mode="json")
        return dumped if isinstance(dumped, dict) else None
    return None


def _step_content_text(payload: dict[str, Any] | None) -> str:
    """Text carried by a model output step's own ``content``.

    A ``step.start`` may already contain output, with the deltas that follow
    continuing it. Only text items are read, matching the non-streamed mapping.
    """
    if payload is None:
        return ""
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    )


class _StreamedContent:
    """Accumulates output message parts from an interaction stream.

    A streamed fetch's completion event carries metadata only -- ``steps`` comes
    back null -- so every part has to be rebuilt from the step events.
    ``step.start`` carries the whole step; ``step.delta`` then fills it in, as a
    model output's text, a function call's arguments in string fragments, or a
    server tool's structured arguments and results. Each step's payload is
    merged as it arrives and converted once at the end, so the same mapping
    that handles a non-streamed interaction produces the parts.
    """

    def __init__(self) -> None:
        # Keyed by step index, in the order the steps were first seen.
        self._steps: dict[object, dict[str, Any]] = {}

    def add(self, chunk: InteractionSSEEvent) -> None:
        """Accumulate one ``step.start`` or ``step.delta`` event."""
        entry = self._steps.setdefault(
            _get_field(chunk, "index"),
            {"type": None, "payload": None, "text": [], "arguments": []},
        )
        if _get_field(chunk, "event_type") == "step.start":
            step = _get_field(chunk, "step")
            entry["type"] = _get_field(step, "type")
            entry["payload"] = _as_payload(step)
            return
        # A resumed stream picks up mid-interaction, so an index whose
        # step.start went to the dropped connection has no type yet and stays
        # eligible. A thought delta carries an opaque signature rather than
        # text, so it contributes nothing either way.
        if entry["type"] == "thought":
            return
        delta = _get_field(chunk, "delta")
        if delta is None:
            return
        text = _get_field(delta, "text")
        if isinstance(text, str):
            entry["text"].append(text)
        self._merge_delta(entry, delta)

    @staticmethod
    def _merge_delta(entry: dict[str, Any], delta: Any) -> None:
        """Fold a delta's structured fields into the step's payload."""
        delta_type = _get_field(delta, "type")
        if entry["payload"] is None and delta_type in _TOOL_STEP_TYPES:
            # The step.start was delivered to a connection that dropped.
            entry["payload"] = {"type": delta_type}
        payload = entry["payload"]

        arguments = _get_field(delta, "arguments")
        if isinstance(arguments, str):
            # A function call streams its arguments as JSON fragments.
            entry["arguments"].append(arguments)
        elif arguments is not None and payload is not None:
            # A server tool call carries them structured instead.
            merged = _as_payload(arguments)
            if merged is not None:
                existing = payload.get("arguments")
                if isinstance(existing, dict):
                    existing.update(merged)
                else:
                    payload["arguments"] = merged

        if payload is None:
            return
        result = _get_field(delta, "result")
        if result is not None:
            existing_result = payload.get("result")
            converted = _as_json_value(result)
            if isinstance(existing_result, str) and isinstance(converted, str):
                payload["result"] = existing_result + converted
            else:
                payload["result"] = converted
        is_error = _get_field(delta, "is_error")
        if is_error is not None:
            payload["is_error"] = is_error

    def parts(self) -> list[MessagePart]:
        """The accumulated parts, in step order."""
        parts: list[MessagePart] = []
        for entry in self._steps.values():
            payload = entry["payload"]
            if payload is not None and payload.get("type"):
                if entry["arguments"]:
                    payload = {
                        **payload,
                        "arguments": _maybe_json("".join(entry["arguments"])),
                    }
                if part := _interaction_item_to_part(payload):
                    parts.append(part)
                    continue
            # Only a model output step carries output text. A resumed stream may
            # not know the type -- its step.start went to the connection that
            # dropped -- so an unknown one stays eligible; a user input or a
            # thought is not output and must not be reported as one.
            if entry["type"] not in (None, "model_output"):
                continue
            # The step may have arrived with content already, which the
            # deltas then continued.
            if text := _step_content_text(payload) + "".join(entry["text"]):
                parts.append(TextPart(content=text))
        return parts


def _streamed_output_messages(
    parts: list[MessagePart],
) -> list[OutputMessage]:
    """Wrap parts accumulated from stream events as an output message."""
    return [OutputMessage(role=Role.ASSISTANT.value, parts=parts)]


def _has_content(messages: list[OutputMessage]) -> bool:
    """Whether a message carries content, ignoring empty text placeholders."""
    return any(
        not isinstance(part, TextPart) or part.content
        for message in messages
        for part in message.parts
    )


def _interaction_stream_error(chunk: InteractionSSEEvent) -> Error | None:
    """Return the provider error an ``error`` stream event reports, if any.

    An interaction stream reports a generation failure in-band, over a
    successful HTTP response, rather than by raising.

    Args:
        chunk: One event from an interaction stream.

    Returns:
        The reported error, carrying the provider's own error code as
        ``error.type``, or None for every other event.
    """
    if _get_field(chunk, "event_type") != "error":
        return None
    error = _get_field(chunk, "error")
    code = _get_field(error, "code")
    return Error(
        message=_get_field(error, "message"),
        type=code if isinstance(code, str) and code else "error",
    )


def _interaction_from_event(chunk: InteractionSSEEvent) -> Interaction | None:
    """Return the completed interaction a stream event carries, if any.

    Args:
        chunk: One event from an interaction stream.

    Returns:
        The interaction attached to a completion event, or None for every other
        event.
    """
    # The SSE discriminator is dot-separated on the wire ("interaction.completed"),
    # not underscored.
    if _get_field(chunk, "event_type") != "interaction.completed":
        return None
    return _get_field(chunk, "interaction")


class FetchInteractionStreamWrapper(SyncStreamWrapper[InteractionSSEEvent]):
    """Instruments a streamed ``interactions.get``.

    Keeps the fetch invocation open until the caller drains, closes, or fails
    the stream, then records the interaction from the completion event.
    """

    def __init__(
        self,
        stream: Iterable[InteractionSSEEvent],
        invocation: FetchResponseInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None
        self._self_stream_error: Error | None = None
        # Snapshotted so no response text is buffered when content capture is
        # off; the handler settles the mode at construction anyway.
        self._self_capture_content = telemetry_handler.should_capture_content()
        self._self_content = _StreamedContent()

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        event_type = _get_field(chunk, "event_type")
        if event_type in ("step.start", "step.delta"):
            if self._self_capture_content:
                self._self_content.add(chunk)
            return
        if interaction := _interaction_from_event(chunk):
            self._self_last_interaction = interaction
        elif error := _interaction_stream_error(chunk):
            self._self_stream_error = error

    def _on_stream_end(self) -> None:
        streamed_parts = self._self_content.parts()
        if self._self_last_interaction:
            _apply_fetched_interaction_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
                streamed_parts=streamed_parts,
            )
        elif streamed_parts and self._self_capture_content:
            # No completion event arrived, but report what was received.
            self._self_invocation.output_messages = _streamed_output_messages(
                streamed_parts
            )
        if self._self_stream_error is not None:
            # The provider reported the failure in-band; the caller never
            # receives the response, so the retrieval itself failed.
            self._self_invocation.fail(self._self_stream_error)
            return
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class AsyncFetchInteractionStreamWrapper(
    AsyncStreamWrapper[InteractionSSEEvent]
):
    """Async counterpart of FetchInteractionStreamWrapper."""

    def __init__(
        self,
        stream: AsyncIterable[InteractionSSEEvent],
        invocation: FetchResponseInvocation,
        telemetry_handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_telemetry_handler = telemetry_handler
        self._self_last_interaction: Interaction | None = None
        self._self_stream_error: Error | None = None
        # Snapshotted so no response text is buffered when content capture is
        # off; the handler settles the mode at construction anyway.
        self._self_capture_content = telemetry_handler.should_capture_content()
        self._self_content = _StreamedContent()

    def _process_chunk(self, chunk: InteractionSSEEvent) -> None:
        event_type = _get_field(chunk, "event_type")
        if event_type in ("step.start", "step.delta"):
            if self._self_capture_content:
                self._self_content.add(chunk)
            return
        if interaction := _interaction_from_event(chunk):
            self._self_last_interaction = interaction
        elif error := _interaction_stream_error(chunk):
            self._self_stream_error = error

    def _on_stream_end(self) -> None:
        streamed_parts = self._self_content.parts()
        if self._self_last_interaction:
            _apply_fetched_interaction_attributes(
                self._self_last_interaction,
                self._self_invocation,
                self._self_telemetry_handler,
                streamed_parts=streamed_parts,
            )
        elif streamed_parts and self._self_capture_content:
            # No completion event arrived, but report what was received.
            self._self_invocation.output_messages = _streamed_output_messages(
                streamed_parts
            )
        if self._self_stream_error is not None:
            # The provider reported the failure in-band; the caller never
            # receives the response, so the retrieval itself failed.
            self._self_invocation.fail(self._self_stream_error)
            return
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


# See https://ai.google.dev/gemini-api/docs/function-calling
def _maybe_get_tool_definitions(
    tools: Any | None,
) -> list[ToolDefinition] | None:
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return None
    definitions: list[ToolDefinition] = []
    for tool in tools:
        # Tool must be a list of dictionaries
        if not isinstance(tool, dict):
            continue
        # This is currently a required field and the SDK will raise an error if it isn't present or
        # takes on a type field that isn't supported.
        tool_type = tool.get("type")
        if tool_type in (
            "bash",
            "code_execution",
            "filesystem",
            "file_search",
            "google_maps",
            "google_search",
            "url_context",
            "computer_use",
        ):
            definitions.append(
                GenericToolDefinition(
                    name=tool.get("name") or tool_type,
                    type=tool_type,
                )
            )

        elif tool_type == "mcp_server":
            # Name and uri are optional.
            name = tool.get("name") or tool.get("url") or "mcp_server"
            # GenericToolDefinition only has 2 fields (name and type). It'd be useful to
            # add support for extra fields somehow, so we can put the URI in here.
            definitions.append(
                GenericToolDefinition(
                    name=name,
                    type="mcp_server",
                )
            )
        elif tool_type == "function":
            definitions.append(
                FunctionToolDefinition(
                    name=tool.get("name") or "function",
                    description=tool.get("description"),
                    parameters=tool.get("parameters"),
                )
            )
    return definitions if definitions else None


def _explicit_request_fields(value: object) -> Mapping[str, object]:
    stored_fields = getattr(value, "__dict__", value)
    if not isinstance(stored_fields, dict):
        return {}
    fields: dict[str, object] = stored_fields
    supplied = getattr(value, "model_fields_set", None)
    if isinstance(supplied, set):
        # Model dumps can serialize lazy content; stored fields also avoid
        # invoking the SDK's deprecated property accessors.
        extra = getattr(value, "model_extra", None)
        if isinstance(extra, dict):
            fields = fields | extra
        return {name: fields[name] for name in supplied if name in fields}
    return fields


def _interaction_request(kwargs: dict[str, Any]) -> Mapping[str, object]:
    body = _get_field(kwargs.get("request"), "body")
    return _explicit_request_fields(body) if body is not None else kwargs


def _is_interaction_stream(
    response: object, request: Mapping[str, object]
) -> bool:
    if isinstance(response, (Stream, AsyncStream)):
        return True
    if isinstance(response, Interaction):
        return False
    return bool(_get_field(request, "stream"))


def _output_type_from_mime_type(mime_type: object) -> str | None:
    if not isinstance(mime_type, str):
        return None
    mime_type = mime_type.partition(";")[0].strip().lower()
    output_types = GenAIAttributes.GenAiOutputTypeValues
    if mime_type == "application/json" or mime_type.endswith("+json"):
        return output_types.JSON.value
    if mime_type.startswith("text/"):
        return output_types.TEXT.value
    if mime_type.startswith("image/"):
        return output_types.IMAGE.value
    if mime_type.startswith("audio/"):
        return output_types.SPEECH.value
    return None


def _output_type_from_format(response_format: object) -> str | None:
    if isinstance(response_format, (list, tuple)):
        output_types = {
            _output_type_from_format(item) for item in response_format
        }
        return output_types.pop() if len(output_types) == 1 else None

    output_type = _output_type_from_mime_type(
        _get_field(response_format, "mime_type")
    )
    if output_type is not None:
        return output_type

    format_type = _get_field(response_format, "type")
    if not isinstance(format_type, str):
        return None
    output_types = GenAIAttributes.GenAiOutputTypeValues
    if format_type == "text":
        return output_types.TEXT.value
    if format_type == "image":
        return output_types.IMAGE.value
    if format_type == "audio":
        return output_types.SPEECH.value
    if format_type in (
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
        "json",
        "json_object",
        "json_schema",
    ):
        return output_types.JSON.value
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > float_info.max:
        return None
    if isinstance(value, (int, float)) and value >= 0 and math.isfinite(value):
        return float(value)
    return None


def _apply_interaction_request_attributes(
    invocation: InferenceInvocation | RemoteAgentInvocation,
    request: Mapping[str, object],
) -> None:
    config = _explicit_request_fields(_get_field(request, "generation_config"))
    invocation.temperature = _coerce_float(_get_field(config, "temperature"))
    invocation.top_p = _coerce_float(_get_field(config, "top_p"))
    max_tokens = _get_field(config, "max_output_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
        invocation.max_tokens = max_tokens
    seed = _get_field(config, "seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        invocation.seed = seed
    stop_sequences = _get_field(config, "stop_sequences")
    if (
        isinstance(stop_sequences, (list, tuple))
        and stop_sequences
        and all(isinstance(item, str) for item in stop_sequences)
    ):
        invocation.stop_sequences = list(stop_sequences)

    response_format = _get_field(request, "response_format")
    invocation.output_type = _output_type_from_format(response_format)
    if invocation.output_type is None and not isinstance(
        response_format, (list, tuple)
    ):
        invocation.output_type = _output_type_from_mime_type(
            _get_field(request, "response_mime_type")
        )


def _start_interactions_invocation(
    telemetry_handler: TelemetryHandler,
    instance: InteractionsResource | AsyncInteractionsResource,
    request: Mapping[str, object],
) -> InferenceInvocation | RemoteAgentInvocation:
    # Vertex AI does not support the interactions API yet, but eventually will.
    # SDK will raise an exception if model or agent is not passed or if input data is not passed.
    is_vertex, server_address = _get_client_info(instance)
    provider = (
        GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
        if is_vertex
        else GenAIAttributes.GenAiSystemValues.GEMINI.value
    )
    if agent := _get_field(request, "agent"):
        invocation: InferenceInvocation | RemoteAgentInvocation = (
            telemetry_handler.invoke_remote_agent(
                provider=provider,
                request_model=_get_field(request, "model"),
                server_address=server_address,
                agent_name=agent,
            )
        )
    else:
        invocation = telemetry_handler.inference(
            provider=provider,
            request_model=_get_field(request, "model"),
            operation_name="interactions.create",
            server_address=server_address,
            error_type_resolver=resolve_error_type,
        )
    invocation.tool_definitions = _maybe_get_tool_definitions(
        _get_field(request, "tools")
    )
    _apply_interaction_request_attributes(invocation, request)

    if telemetry_handler.should_capture_content():
        invocation.input_messages = _interactions_input_to_messages(
            _get_field(request, "input")
        )
        if system_instruction := _get_field(request, "system_instruction"):
            invocation.system_instruction = [
                TextPart(content=system_instruction)
            ]

    return invocation


def _create_instrumented_interactions_create(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Interaction | Stream[InteractionSSEEvent]],
        InteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Interaction | InteractionsStreamWrapper,
]:
    def instrumented_interactions_create(
        wrapped: Callable[..., Interaction | Stream[InteractionSSEEvent]],
        instance: InteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | InteractionsStreamWrapper:
        request = _interaction_request(kwargs)
        invocation = _start_interactions_invocation(
            telemetry_handler, instance, request
        )

        try:
            response = wrapped(*args, **kwargs)
            if _is_interaction_stream(response, request):
                return InteractionsStreamWrapper(
                    response,
                    invocation,
                    telemetry_handler,
                )
            _apply_interaction_response_attributes(
                response, invocation, telemetry_handler
            )
            invocation.stop()
            return response
        except BaseException as exc:
            invocation.fail(exc)
            raise

    return instrumented_interactions_create


def _create_instrumented_async_interactions_create(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Any],
        AsyncInteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Any,
]:
    async def instrumented_interactions_create(
        wrapped: Callable[..., Any],
        instance: AsyncInteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | AsyncInteractionsStreamWrapper:
        request = _interaction_request(kwargs)
        invocation = _start_interactions_invocation(
            telemetry_handler, instance, request
        )

        try:
            response = await wrapped(*args, **kwargs)
            if _is_interaction_stream(response, request):
                return AsyncInteractionsStreamWrapper(
                    response,
                    invocation,
                    telemetry_handler,
                )
            response = cast(
                Interaction,
                response,
            )
            _apply_interaction_response_attributes(
                response, invocation, telemetry_handler
            )
            invocation.stop()
            return response
        except BaseException as exc:
            invocation.fail(exc)
            raise

    return instrumented_interactions_create


def _is_fetched_interaction(result: object) -> bool:
    """Whether a ``get`` result is a parsed interaction.

    Checked before any stream test because an Interaction is itself iterable.
    """
    return hasattr(result, "steps")


def _is_unparsed_response(result: object) -> bool:
    """Whether a ``get`` result is a response whose body is still unread.

    ``with_raw_response`` and ``with_streaming_response`` route through this
    same method but hand back an unparsed response (which exposes ``parse``) or
    a context manager. Reading either here would consume a body the caller has
    not asked for yet, so they are recorded as request-only spans.
    """
    return hasattr(result, "parse")


def _get_interaction_id(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    """Return the interaction id a ``get`` call was made with.

    Args:
        args: Positional arguments the caller passed, excluding ``self``.
        kwargs: Keyword arguments the caller passed.

    Returns:
        The id, passed either positionally or as ``id``, or None when it is
        missing or not a non-empty string.
    """
    interaction_id = args[0] if args else kwargs.get("id")
    if isinstance(interaction_id, str) and interaction_id:
        return interaction_id
    return None


def _start_fetch_invocation(
    telemetry_handler: TelemetryHandler,
    instance: InteractionsResource | AsyncInteractionsResource,
    interaction_id: str,
    kwargs: dict[str, Any],
) -> FetchResponseInvocation:
    """Start a fetch invocation for an ``interactions.get`` call.

    Args:
        telemetry_handler: Handler that creates the invocation.
        instance: The interactions resource the call was made on, used to
            resolve the provider and server address.
        interaction_id: The id being fetched.
        kwargs: Keyword arguments the caller passed, read for ``stream`` and
            ``last_event_id``.

    Returns:
        A started fetch invocation; the caller must stop or fail it.
    """
    is_vertex, server_address = _get_client_info(instance)
    provider = (
        GenAIAttributes.GenAiSystemValues.VERTEX_AI.value
        if is_vertex
        else GenAIAttributes.GenAiSystemValues.GEMINI.value
    )
    streaming = bool(kwargs.get("stream"))
    # gen_ai.request.stream is not part of the fetch_response span's attributes;
    # a resumed fetch is identified by gen_ai.request.stream_cursor instead.
    invocation = telemetry_handler.fetch_response(
        provider=provider,
        response_id=interaction_id,
        server_address=server_address,
        error_type_resolver=resolve_error_type,
    )
    last_event_id = kwargs.get("last_event_id")
    if streaming and isinstance(last_event_id, str) and last_event_id:
        invocation.stream_cursor = last_event_id
    return invocation


def _create_instrumented_interactions_get(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Interaction | Stream[InteractionSSEEvent]],
        InteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Interaction | FetchInteractionStreamWrapper,
]:
    """Build the wrapt wrapper for the sync ``Interactions.get``.

    Args:
        telemetry_handler: Handler the wrapper records telemetry through.

    Returns:
        A wrapper that emits a fetch_response span around the call and returns
        the SDK's own interaction, or a stream wrapper when streaming.
    """

    def instrumented_interactions_get(
        wrapped: Callable[..., Interaction | Stream[InteractionSSEEvent]],
        instance: InteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | FetchInteractionStreamWrapper:
        interaction_id = _get_interaction_id(args, kwargs)
        if interaction_id is None:
            # Without an id the SDK raises before issuing a request; there is
            # no response to describe.
            return cast(Interaction, wrapped(*args, **kwargs))

        invocation = _start_fetch_invocation(
            telemetry_handler, instance, interaction_id, kwargs
        )

        try:
            result = wrapped(*args, **kwargs)
            if _is_fetched_interaction(result):
                _apply_fetched_interaction_attributes(
                    cast(Interaction, result), invocation, telemetry_handler
                )
            elif not _is_unparsed_response(result) and hasattr(
                result, "__iter__"
            ):
                return FetchInteractionStreamWrapper(
                    cast("Stream[InteractionSSEEvent]", result),
                    invocation,
                    telemetry_handler,
                )
            invocation.stop()
            return result
        except BaseException as exc:
            invocation.fail(exc)
            raise

    return instrumented_interactions_get


def _create_instrumented_async_interactions_get(
    telemetry_handler: TelemetryHandler,
) -> Callable[
    [
        Callable[..., Any],
        AsyncInteractionsResource,
        tuple[Any, ...],
        dict[str, Any],
    ],
    Any,
]:
    """Build the wrapt wrapper for the async ``Interactions.get``.

    Args:
        telemetry_handler: Handler the wrapper records telemetry through.

    Returns:
        A coroutine wrapper mirroring the sync one.
    """

    async def instrumented_interactions_get(
        wrapped: Callable[..., Any],
        instance: AsyncInteractionsResource,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Interaction | AsyncFetchInteractionStreamWrapper:
        interaction_id = _get_interaction_id(args, kwargs)
        if interaction_id is None:
            return cast(Interaction, await wrapped(*args, **kwargs))

        invocation = _start_fetch_invocation(
            telemetry_handler, instance, interaction_id, kwargs
        )

        try:
            result = await wrapped(*args, **kwargs)
            if _is_fetched_interaction(result):
                _apply_fetched_interaction_attributes(
                    cast(Interaction, result), invocation, telemetry_handler
                )
            elif not _is_unparsed_response(result) and hasattr(
                result, "__aiter__"
            ):
                return AsyncFetchInteractionStreamWrapper(
                    result, invocation, telemetry_handler
                )
            invocation.stop()
            return result
        except BaseException as exc:
            invocation.fail(exc)
            raise

    return instrumented_interactions_get


def uninstrument_interactions(snapshot: object) -> None:
    if snapshot is None:
        return
    assert isinstance(snapshot, _InteractionsMethodsSnapshot)
    snapshot.restore()


def instrument_interactions(
    telemetry_handler: TelemetryHandler,
) -> object | None:
    if not _HAS_INTERACTIONS:
        return None

    snapshot = _InteractionsMethodsSnapshot()

    try:
        import google.genai._interactions.resources.interactions  # noqa: F401

        module_path = "google.genai._interactions.resources.interactions"
        sync_class = "InteractionsResource"
        async_class = "AsyncInteractionsResource"
    except ImportError:
        # In version 2.9 of google-genai these were moved.
        module_path = "google.genai._gaos.interactions"
        sync_class = "Interactions"
        async_class = "AsyncInteractions"

    wrapped = wrap_function_wrapper(
        module_path,
        f"{sync_class}.create",
        _create_instrumented_interactions_create(telemetry_handler),
    )
    _set_co_filename(wrapped)
    wrapped2 = wrap_function_wrapper(
        module_path,
        f"{async_class}.create",
        _create_instrumented_async_interactions_create(telemetry_handler),
    )
    _set_co_filename(wrapped2)

    wrapped3 = wrap_function_wrapper(
        module_path,
        f"{sync_class}.get",
        _create_instrumented_interactions_get(telemetry_handler),
    )
    _set_co_filename(wrapped3)
    wrapped4 = wrap_function_wrapper(
        module_path,
        f"{async_class}.get",
        _create_instrumented_async_interactions_get(telemetry_handler),
    )
    _set_co_filename(wrapped4)
    return snapshot
