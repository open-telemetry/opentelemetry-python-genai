# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

        class AsyncInteractionsResource:
            create = None

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
    InferenceInvocation,
    RemoteAgentInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
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


class _InteractionsMethodsSnapshot:
    def __init__(self) -> None:
        self._original_create = InteractionsResource.create
        self._original_create_code = InteractionsResource.create.__code__
        self._original_async_create = AsyncInteractionsResource.create
        self._original_async_create_code = (
            AsyncInteractionsResource.create.__code__
        )

    def restore(self) -> None:
        self._original_create.__code__ = self._original_create_code
        self._original_async_create.__code__ = self._original_async_create_code

        InteractionsResource.create = self._original_create
        AsyncInteractionsResource.create = self._original_async_create


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
            finish_reason="stop",
        )
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

    def _on_stream_error(self, error: Exception) -> None:
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

    def _on_stream_error(self, error: Exception) -> None:
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
    return snapshot
