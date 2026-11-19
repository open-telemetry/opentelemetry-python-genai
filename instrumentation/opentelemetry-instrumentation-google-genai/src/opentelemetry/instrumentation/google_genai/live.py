# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Final, cast
from urllib.parse import urlsplit

from google.genai import _transformers as genai_transformers
from google.genai import live as genai_live
from google.genai import types as genai_types
from google.genai.live import AsyncLive, AsyncSession
from google.genai.types import (
    BlobOrDict,
    Content,
    ContentDict,
    ContentListUnion,
    ContentListUnionDict,
    ContentUnion,
    ContentUnionDict,
    FunctionResponseOrDict,
    GenerateContentResponseUsageMetadata,
    LiveConnectConfigOrDict,
    LiveServerContent,
    LiveServerMessage,
    Tool,
    ToolListUnion,
    ToolListUnionDict,
    ToolUnionDict,
)

from opentelemetry.context import Context, get_current
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes.error_attributes import ErrorTypeValues
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    Error,
    RealtimeInferenceInvocation,
    UserSpeechInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    InputMessage,
    MessagePart,
    Modality,
    OutputMessage,
    ReasoningPart,
    SystemInstructionPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
    TranscriptionPart,
)
from opentelemetry.util.genai.utils import (
    decode_base64,
    get_argument,
    object_to_any_value,
)

from ._error_type import resolve_error_type
from .client_info import get_client_info
from .generate_content import _to_tool_definition
from .message import _to_part, to_input_messages, to_system_instructions

_ORIGINAL_ASYNC_LIVE_CONNECT = AsyncLive.connect
_MAX_BUFFERED_MEDIA_BYTES: Final = 4 * 1024 * 1024
_ACTIVE_SEND_CALLBACK: contextvars.ContextVar[_StagedSend | None] = (
    contextvars.ContextVar(
        "_otel_google_genai_live_active_send_callback", default=None
    )
)


def _strip_model_prefix(model: str | None) -> str | None:
    if not model:
        return None
    model_str = str(model)
    if model_str.startswith("models/"):
        return model_str[len("models/") :]
    return model_str


def _snake_to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


def _get_config_field(config: object, name: str) -> object:
    if config is None:
        return None
    if isinstance(config, Mapping):
        mapping = cast(Mapping[str, object], config)
        if name in mapping and mapping[name] is not None:
            return mapping[name]
        camel = _snake_to_camel(name)
        if camel in mapping:
            return mapping[camel]
        return mapping.get(name)
    return getattr(config, name, None)


def _get_live_config_field(
    config: LiveConnectConfigOrDict | None, name: str
) -> object:
    val = _get_config_field(config, name)
    if val is not None:
        return val
    gen_config = _get_config_field(config, "generation_config")
    return _get_config_field(gen_config, name)


def _extract_live_endpoint_info(
    live_instance: AsyncLive,
    config: LiveConnectConfigOrDict | None = None,
) -> tuple[str, str | None, int | None]:
    """Return ``(provider, server_address, server_port)`` for an ``AsyncLive`` instance."""
    is_vertex, server_address = get_client_info(live_instance)
    provider = (
        GenAI.GenAiProviderNameValues.GCP_VERTEX_AI.value
        if is_vertex
        else GenAI.GenAiProviderNameValues.GCP_GEMINI.value
    )
    api_client: object = getattr(live_instance, "_api_client", None)
    raw_url = (
        _get_config_field(
            _get_config_field(config, "http_options"), "base_url"
        )
        or _get_config_field(
            getattr(api_client, "_http_options", None), "base_url"
        )
        or getattr(api_client, "custom_base_url", None)
    )
    if raw_url:
        url_str = str(raw_url)
        parsed = urlsplit(url_str if "://" in url_str else f"wss://{url_str}")
        server_address = parsed.hostname or server_address
        server_port = parsed.port or (
            80 if parsed.scheme in ("ws", "http") else 443
        )
        return provider, server_address, server_port
    return provider, server_address, 443 if server_address else None


def _resolve_output_type(
    config: LiveConnectConfigOrDict | None, is_vertex: bool
) -> str | None:
    modalities = _get_live_config_field(config, "response_modalities")
    if isinstance(modalities, Sequence) and not isinstance(modalities, str):
        seq_modalities = cast(Sequence[object], modalities)
        if seq_modalities:
            upper_modalities = [
                str(getattr(m, "value", m)).upper() for m in seq_modalities
            ]
            if any(m.endswith("AUDIO") for m in upper_modalities):
                return GenAI.GenAiOutputTypeValues.SPEECH.value
            if any(m.endswith("TEXT") for m in upper_modalities):
                return GenAI.GenAiOutputTypeValues.TEXT.value
            if any(m.endswith("IMAGE") for m in upper_modalities):
                return GenAI.GenAiOutputTypeValues.IMAGE.value
    if is_vertex:
        # Vertex AI Live defaults responseModalities to ["AUDIO"] when omitted.
        return GenAI.GenAiOutputTypeValues.SPEECH.value
    return None


def _extract_system_instruction(
    sys_inst: ContentUnion | ContentUnionDict | None,
) -> list[SystemInstructionPart]:
    if sys_inst is None:
        return []
    try:
        t_content = cast(
            Callable[[object], Content], genai_transformers.t_content
        )
        return to_system_instructions(content=t_content(sys_inst))
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def _extract_tool_definitions(
    tools: ToolListUnion | ToolListUnionDict | None,
) -> list[ToolDefinition]:
    if not tools or not isinstance(tools, Sequence):
        return []
    definitions: list[ToolDefinition] = []
    for item in tools:
        tool_item: object = item
        if isinstance(tool_item, Mapping):
            try:
                tool_item = genai_types.Tool.model_validate(tool_item)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        if isinstance(tool_item, Tool) or callable(tool_item):
            try:
                definitions.extend(
                    _to_tool_definition(cast(ToolUnionDict, tool_item))
                )
            except Exception:  # pylint: disable=broad-exception-caught
                continue
    return definitions


def _extract_blob_bytes_and_mime(
    blob_or_dict: BlobOrDict | Mapping[str, object] | None,
    default_mime: str = "audio/pcm",
) -> tuple[bytes, str] | None:
    if blob_or_dict is None:
        return None
    try:
        t_blob = cast(
            Callable[[object], genai_types.Blob], genai_transformers.t_blob
        )
        blob = t_blob(blob_or_dict)
        if blob.data is not None:
            return bytes(blob.data), str(blob.mime_type or default_mime)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    raw_data = _get_config_field(blob_or_dict, "data")
    mime_type = _get_config_field(blob_or_dict, "mime_type") or default_mime
    if isinstance(raw_data, (bytes, bytearray)):
        return bytes(raw_data), str(mime_type)
    if isinstance(raw_data, str):
        return (
            decode_base64(raw_data) or raw_data.encode("utf-8"),
            str(mime_type),
        )
    return None


def _build_tool_response_input_message(
    function_responses: FunctionResponseOrDict
    | Sequence[FunctionResponseOrDict]
    | None,
) -> InputMessage | None:
    if function_responses is None:
        return None
    items: Sequence[object]
    try:
        tool_resp = genai_transformers.t_tool_response(function_responses)
        items = tool_resp.function_responses or []
    except Exception:  # pylint: disable=broad-exception-caught
        items = (
            [function_responses]
            if isinstance(function_responses, Mapping)
            or not isinstance(function_responses, Sequence)
            else function_responses
        )

    parts: list[MessagePart] = [
        ToolCallResponsePart(
            id=str(call_id)
            if (call_id := _get_config_field(item, "id")) is not None
            else None,
            response=object_to_any_value(_get_config_field(item, "response")),
        )
        for item in items
    ]
    return InputMessage(role="tool", parts=parts) if parts else None


def _turn_complete_reason(
    server_content: LiveServerContent | None,
) -> str | None:
    if server_content is None:
        return None
    reason: object = getattr(server_content, "turn_complete_reason", None)
    if reason is None:
        return None
    reason_val = str(getattr(reason, "value", reason))
    if (
        not reason_val
        or reason_val.upper() == "TURN_COMPLETE_REASON_UNSPECIFIED"
    ):
        return None
    return reason_val.lower()


def _is_interaction_complete(
    server_content: LiveServerContent | None,
) -> bool:
    if server_content is None:
        return False
    checker = getattr(genai_live, "_is_interaction_complete", None)
    if callable(checker):
        try:
            return bool(checker(server_content))
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return bool(getattr(server_content, "turn_complete", False))


def _extract_tool_response_keys(
    function_responses: FunctionResponseOrDict
    | Sequence[FunctionResponseOrDict]
    | None,
) -> tuple[tuple[str, ...], int]:
    """Return ``(call_ids, unidentified_count)`` for a tool-response payload."""
    if function_responses is None:
        return (), 0
    items: Sequence[object]
    try:
        tool_resp = genai_transformers.t_tool_response(function_responses)
        items = tool_resp.function_responses or []
    except Exception:  # pylint: disable=broad-exception-caught
        items = (
            [function_responses]
            if isinstance(function_responses, Mapping)
            or not isinstance(function_responses, Sequence)
            else function_responses
        )
    ids: list[str] = []
    unidentified = 0
    for item in items:
        call_id = _get_config_field(item, "id")
        if call_id is not None and str(call_id):
            ids.append(str(call_id))
        else:
            unidentified += 1
    return tuple(ids), unidentified


@dataclass(eq=False)
class _PendingToolCallBatch:
    """Tracks unanswered function calls from a Gemini 2.x ``tool_call`` batch."""

    all_ids: set[str] = field(default_factory=set)
    remaining_ids: set[str] = field(default_factory=set)
    cancelled_ids: set[str] = field(default_factory=set)
    remaining_unidentified: int = 0
    turn: _OutboundTurn | None = None

    @property
    def is_complete(self) -> bool:
        return not self.remaining_ids and self.remaining_unidentified <= 0


@dataclass(eq=False)
class _StagedSend:
    """Represents an in-flight WebSocket send awaiting transport confirmation."""

    commit_fn: Callable[[], None]
    context: contextvars.Context
    completes_turn: bool = False
    completes_turn_if_nonempty: bool = False
    has_input_payload: bool = False
    omit_tool_response: bool = False
    starts_new_turn: bool = False
    is_tool_response: bool = False
    tool_response_ids: tuple[str, ...] = ()
    tool_response_unidentified: int = 0
    consumed_tool_batch: _PendingToolCallBatch | None = None
    consumed_tool_ids: list[str] = field(default_factory=list)
    consumed_tool_unidentified: int = 0
    popped_tool_batch: bool = False
    updates_system_instruction: bool = False
    updates_tool_definitions: bool = False
    dependent_gens: list[_ActiveGeneration] = field(default_factory=list)
    seq: int = 0
    turn: _OutboundTurn | None = None
    status: str = "pending"
    applied: bool = False


@dataclass(eq=False)
class _OutboundTurn:
    """Buffers outbound inputs and ordered in-flight sends for one logical turn."""

    pending_audio_chunks: deque[bytes] = field(default_factory=deque)
    pending_audio_bytes: int = 0
    pending_audio_mime_type: str | None = None
    input_messages: list[InputMessage] = field(default_factory=list)
    media_bytes: int = 0
    system_instruction: list[SystemInstructionPart] | None = None
    system_instruction_rev: int = 0
    tool_definitions: list[ToolDefinition] | None = None
    tool_definitions_rev: int = 0
    staged_sends: deque[_StagedSend] = field(default_factory=deque)
    sealed: bool = False
    has_confirmed_turn_complete: bool = False
    has_confirmed_input_payload: bool = False
    tool_batch: _PendingToolCallBatch | None = None
    bound_gen: _ActiveGeneration | None = None


def _classify_legacy_send(
    input_val: object,
    end_of_turn: bool,
    *,
    omit_tool_response: bool,
) -> tuple[bool, bool, bool, tuple[str, ...], int]:
    """Return ``(completes_turn, omit_from_queue, is_tool_response, tool_ids, unidentified)``."""
    if not input_val:
        # AsyncSession._parse_client_message treats any falsy input as
        # client_content(turn_complete=True).
        return True, False, False, (), 0
    if isinstance(input_val, genai_types.LiveClientRealtimeInput) or (
        isinstance(input_val, Mapping)
        and ("media_chunks" in input_val or "mediaChunks" in input_val)
    ):
        return False, False, False, (), 0
    if isinstance(input_val, genai_types.Blob) or (
        isinstance(input_val, Mapping) and "data" in input_val
    ):
        return False, False, False, (), 0
    if isinstance(input_val, genai_types.LiveClientToolResponse) or (
        isinstance(input_val, Mapping)
        and (
            "function_responses" in input_val
            or "functionResponses" in input_val
        )
    ):
        fn_resps = cast(
            FunctionResponseOrDict | Sequence[FunctionResponseOrDict] | None,
            _get_config_field(input_val, "function_responses"),
        )
        ids, unidentified = _extract_tool_response_keys(fn_resps)
        return (
            (not omit_tool_response),
            omit_tool_response,
            True,
            ids,
            unidentified,
        )
    if isinstance(input_val, genai_types.FunctionResponse) or (
        isinstance(input_val, Mapping)
        and "name" in input_val
        and "response" in input_val
    ):
        ids, unidentified = _extract_tool_response_keys(
            cast(FunctionResponseOrDict, input_val)
        )
        return (
            (not omit_tool_response),
            omit_tool_response,
            True,
            ids,
            unidentified,
        )
    if isinstance(input_val, genai_types.LiveClientContent) or (
        isinstance(input_val, Mapping)
        and ("turns" in input_val or "content" in input_val)
    ):
        tc = _get_config_field(input_val, "turn_complete")
        return bool(end_of_turn if tc is None else tc), False, False, (), 0
    if isinstance(input_val, Sequence) and not isinstance(
        input_val, (str, bytes, bytearray)
    ):
        seq_val = cast(Sequence[object], input_val)
        if seq_val and any(
            isinstance(item, genai_types.FunctionResponse)
            or (
                isinstance(item, Mapping)
                and "name" in item
                and "response" in item
            )
            for item in seq_val
        ):
            ids, unidentified = _extract_tool_response_keys(
                cast(Sequence[FunctionResponseOrDict], seq_val)
            )
            return (
                (not omit_tool_response),
                omit_tool_response,
                True,
                ids,
                unidentified,
            )
        if seq_val and any(
            isinstance(item, genai_types.Blob)
            or (isinstance(item, Mapping) and "data" in item)
            for item in seq_val
        ):
            return False, False, False, (), 0
    return bool(end_of_turn), False, False, (), 0


def _completed_finish_reason(active_gen: _ActiveGeneration) -> str | None:
    if active_gen.turn_complete_reason:
        return active_gen.turn_complete_reason
    if active_gen.interrupted:
        return "interrupted"
    if active_gen.generation_complete or active_gen.turn_complete:
        return "stop"
    return None


@dataclass(eq=False)
class _ActiveGeneration:
    invocation: RealtimeInferenceInvocation
    input_messages: list[InputMessage] = field(default_factory=list)
    input_media_bytes: int = 0
    input_audio_chunks: deque[bytes] = field(default_factory=deque)
    input_audio_bytes: int = 0
    input_audio_mime_type: str | None = None
    system_instruction: list[SystemInstructionPart] = field(
        default_factory=list
    )
    system_instruction_rev: int = 0
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
    tool_definitions_rev: int = 0
    completed_output_messages: list[OutputMessage] = field(
        default_factory=list
    )
    input_transcript_deltas: list[str] = field(default_factory=list)
    output_transcript_deltas: list[str] = field(default_factory=list)
    output_audio_chunks: list[bytes] = field(default_factory=list)
    output_audio_bytes: int = 0
    output_audio_mime_type: str | None = None
    output_text_deltas: list[str] = field(default_factory=list)
    output_thought_deltas: list[str] = field(default_factory=list)
    output_other_parts: list[MessagePart] = field(default_factory=list)
    usage_metadata: GenerateContentResponseUsageMetadata | None = None
    turn_complete_reason: str | None = None
    interrupted: bool = False
    generation_complete: bool = False
    turn_complete: bool = False
    awaiting_tool_response: bool = False
    accepting_input_transcription: bool = True
    bound_turn: _OutboundTurn | None = None
    pending_context_deps: list[_StagedSend] = field(default_factory=list)
    deferred_finish_reason: str | None = None
    deferred_error: Error | BaseException | None = None

    def _enforce_input_media_budget(self) -> None:
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        excess = self.input_media_bytes + self.input_audio_bytes - max_bytes
        if excess <= 0:
            return

        # Evict oldest buffered media from input_messages first so the most
        # recent media preceding/during this generation is retained.
        msg_idx = 0
        while excess > 0 and msg_idx < len(self.input_messages):
            msg = self.input_messages[msg_idx]
            new_parts: list[MessagePart] = []
            for part in msg.parts:
                if (
                    excess > 0
                    and isinstance(part, BlobPart)
                    and isinstance(part.content, bytes)
                ):
                    part_len = len(part.content)
                    if part.modality == Modality.AUDIO and part_len > excess:
                        part.content = part.content[excess:]
                        self.input_media_bytes -= excess
                        excess = 0
                        new_parts.append(part)
                    else:
                        self.input_media_bytes -= part_len
                        excess -= part_len
                else:
                    new_parts.append(part)
            if new_parts:
                msg.parts = new_parts
                msg_idx += 1
            else:
                self.input_messages.pop(msg_idx)

        while excess > 0 and self.input_audio_chunks:
            oldest = self.input_audio_chunks[0]
            if len(oldest) <= excess:
                self.input_audio_chunks.popleft()
                self.input_audio_bytes -= len(oldest)
                excess -= len(oldest)
            else:
                self.input_audio_chunks[0] = oldest[excess:]
                self.input_audio_bytes -= excess
                excess = 0

        if not self.input_audio_chunks:
            self.input_audio_mime_type = None

    def append_input_audio(self, chunk: bytes, mime_type: str) -> None:
        if not chunk:
            return
        if (
            self.input_audio_chunks
            and self.input_audio_mime_type
            and mime_type != self.input_audio_mime_type
        ):
            self.flush_input_audio()
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        if len(chunk) >= max_bytes:
            trimmed = chunk[-max_bytes:]
            self.input_audio_chunks.clear()
            self.input_audio_chunks.append(trimmed)
            self.input_audio_bytes = len(trimmed)
        else:
            self.input_audio_chunks.append(chunk)
            self.input_audio_bytes += len(chunk)
        self.input_audio_mime_type = mime_type
        self._enforce_input_media_budget()

    def flush_input_audio(self) -> None:
        if not self.input_audio_chunks:
            return
        combined = b"".join(self.input_audio_chunks)
        self.input_audio_chunks.clear()
        self.input_audio_bytes = 0
        mime_type = self.input_audio_mime_type or "audio/pcm"
        self.input_audio_mime_type = None
        if len(combined) > _MAX_BUFFERED_MEDIA_BYTES:
            combined = combined[-_MAX_BUFFERED_MEDIA_BYTES:]
        self.input_messages.append(
            InputMessage(
                role="user",
                parts=[
                    BlobPart(
                        mime_type=mime_type,
                        modality=Modality.AUDIO,
                        content=combined,
                    )
                ],
            )
        )
        self.input_media_bytes += len(combined)
        self._enforce_input_media_budget()

    def append_input_messages(self, messages: Sequence[InputMessage]) -> None:
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        for msg in messages:
            if (
                msg.role == "user"
                and len(msg.parts) == 1
                and isinstance(part := msg.parts[0], BlobPart)
                and part.modality == Modality.AUDIO
                and isinstance(part.content, bytes)
            ):
                self.append_input_audio(
                    part.content, part.mime_type or "audio/pcm"
                )
            else:
                self.flush_input_audio()
                added_bytes = 0
                for part in msg.parts:
                    if isinstance(part, BlobPart) and isinstance(
                        part.content, bytes
                    ):
                        if len(part.content) > max_bytes:
                            part.content = (
                                part.content[-max_bytes:]
                                if part.modality == Modality.AUDIO
                                else part.content[:max_bytes]
                            )
                        added_bytes += len(part.content)
                self.input_messages.append(msg)
                if added_bytes:
                    self.input_media_bytes += added_bytes
                    self._enforce_input_media_budget()

    def flush_current_turn_output_message(
        self, finish_reason: str | None
    ) -> OutputMessage | None:
        parts: list[MessagePart] = []
        if self.output_thought_deltas:
            parts.append(
                ReasoningPart(content="".join(self.output_thought_deltas))
            )
            self.output_thought_deltas.clear()
        if self.output_audio_chunks:
            parts.append(
                BlobPart(
                    mime_type=self.output_audio_mime_type or "audio/pcm",
                    modality=Modality.AUDIO,
                    content=b"".join(self.output_audio_chunks),
                )
            )
            self.output_audio_chunks.clear()
            self.output_audio_bytes = 0
            self.output_audio_mime_type = None
        if self.output_text_deltas:
            parts.append(TextPart(content="".join(self.output_text_deltas)))
            self.output_text_deltas.clear()
        if self.output_transcript_deltas:
            parts.append(
                TranscriptionPart(
                    content="".join(self.output_transcript_deltas)
                )
            )
            self.output_transcript_deltas.clear()
        if self.output_other_parts:
            parts.extend(self.output_other_parts)
            self.output_other_parts.clear()

        if not parts:
            return None
        msg = OutputMessage(
            role="assistant",
            parts=parts,
            finish_reason=finish_reason,
        )
        self.completed_output_messages.append(msg)
        return msg


class _LiveSessionTracker:
    """Tracks state for a single ``AsyncSession`` established by ``AsyncLive.connect``."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        telemetry_handler: TelemetryHandler,
        provider: str,
        is_vertex: bool,
        request_model: str | None,
        realtime_session_id: str | None,
        server_address: str | None,
        server_port: int | None,
        config: LiveConnectConfigOrDict | None,
        caller_context: Context,
    ) -> None:
        self.session = session
        self._telemetry_handler = telemetry_handler
        self.provider = provider
        self.is_vertex = is_vertex
        self.request_model = request_model
        self.realtime_session_id = realtime_session_id
        self.server_address = server_address
        self.server_port = server_port
        self.config = config
        self.caller_context = caller_context

        self.output_type = _resolve_output_type(config, is_vertex)
        raw_max_tokens = _get_live_config_field(config, "max_output_tokens")
        self.max_tokens = (
            int(raw_max_tokens)
            if isinstance(raw_max_tokens, (int, float))
            and not isinstance(raw_max_tokens, bool)
            else None
        )
        thinking_config = _get_live_config_field(config, "thinking_config")
        thinking_level = _get_config_field(thinking_config, "thinking_level")
        self.reasoning_level = (
            str(getattr(thinking_level, "value", thinking_level))
            if thinking_level is not None
            else None
        )

        # Gemini Live 3.x models report the tool request and spoken answer
        # within a single generation; 2.x models report them as two sibling
        # generations.
        self.uses_single_generation_tool_exchange: bool = "gemini-3" in (
            (request_model or "").lower()
        )

        self._outbound_turns: deque[_OutboundTurn] = deque()
        self._pending_tool_call_batches: deque[_PendingToolCallBatch] = deque()
        self._pending_context_updates: list[_StagedSend] = []
        self._write_turn: _OutboundTurn | None = None
        self._next_send_seq: int = 0
        self._current_send_seq: int = 0
        self._pending_input_transcript_deltas: list[str] = []
        self._deferred_generations: list[_ActiveGeneration] = []
        self._capture_content_override: bool | None = None
        self._active_gen: _ActiveGeneration | None = None
        self._speech_invocation: UserSpeechInvocation | None = None
        self._session_finalized: bool = False
        self.terminal_error: BaseException | None = None

        self.system_instruction: list[SystemInstructionPart] = []
        self._system_instruction_rev: int = 0
        self.tool_definitions: list[ToolDefinition] = []
        self._tool_definitions_rev: int = 0
        self._config_content_extracted: bool = False
        if self._should_capture_content():
            self._ensure_config_content_extracted()

    @property
    def _pending_audio_chunks(self) -> deque[bytes]:
        chunks: deque[bytes] = deque()
        for turn in self._outbound_turns:
            chunks.extend(turn.pending_audio_chunks)
        return chunks

    @property
    def _pending_input_messages(self) -> list[InputMessage]:
        messages: list[InputMessage] = []
        for turn in self._outbound_turns:
            messages.extend(turn.input_messages)
        return messages

    def _current_open_turn(self) -> _OutboundTurn:
        if not self._outbound_turns or self._outbound_turns[-1].sealed:
            turn = _OutboundTurn()
            self._outbound_turns.append(turn)
            return turn
        return self._outbound_turns[-1]

    def _target_turn(self) -> _OutboundTurn:
        if self._write_turn is not None:
            return self._write_turn
        return self._current_open_turn()

    def _ensure_config_content_extracted(self) -> None:
        if self._config_content_extracted:
            return
        self._config_content_extracted = True
        sys_inst = _get_live_config_field(self.config, "system_instruction")
        if sys_inst is not None:
            self.system_instruction = _extract_system_instruction(
                cast(ContentUnion | ContentUnionDict, sys_inst)
            )
        tools = _get_live_config_field(self.config, "tools")
        if tools is not None:
            self.tool_definitions = _extract_tool_definitions(
                cast(ToolListUnion | ToolListUnionDict, tools)
            )

    def _should_capture_content(self) -> bool:
        if self._capture_content_override is not None:
            return self._capture_content_override
        if self._active_gen is not None:
            return self._active_gen.invocation.should_capture_content
        return self._telemetry_handler.should_capture_realtime_content()

    def register_inflight_send(self, staged: _StagedSend) -> None:
        if (
            self._session_finalized
            or staged.status != "pending"
            or staged.turn is not None
            or staged.omit_tool_response
        ):
            return
        self._next_send_seq += 1
        staged.seq = self._next_send_seq
        if (
            staged.updates_system_instruction
            or staged.updates_tool_definitions
        ) and staged not in self._pending_context_updates:
            self._pending_context_updates.append(staged)
        matched_batch: _PendingToolCallBatch | None = None
        if staged.is_tool_response:
            if self._pending_tool_call_batches:
                batch = self._pending_tool_call_batches[0]
                matched_batch = batch
                staged.consumed_tool_batch = batch
                for call_id in staged.tool_response_ids:
                    if call_id in batch.remaining_ids:
                        batch.remaining_ids.remove(call_id)
                        staged.consumed_tool_ids.append(call_id)
                unidentified_left = staged.tool_response_unidentified
                while (
                    unidentified_left > 0 and batch.remaining_unidentified > 0
                ):
                    batch.remaining_unidentified -= 1
                    staged.consumed_tool_unidentified += 1
                    unidentified_left -= 1
                while unidentified_left > 0 and batch.remaining_ids:
                    popped_id = next(iter(batch.remaining_ids))
                    batch.remaining_ids.remove(popped_id)
                    staged.consumed_tool_ids.append(popped_id)
                    unidentified_left -= 1
                if batch.is_complete:
                    self._pending_tool_call_batches.popleft()
                    staged.popped_tool_batch = True
                    staged.completes_turn = True
                else:
                    staged.completes_turn = False
            else:
                staged.completes_turn = True
        if (
            staged.starts_new_turn
            and self._outbound_turns
            and not self._outbound_turns[-1].sealed
        ):
            last = self._outbound_turns[-1]
            if (
                last.input_messages
                or last.pending_audio_chunks
                or any(
                    other.has_input_payload and other.status != "aborted"
                    for other in last.staged_sends
                )
            ):
                last.sealed = True
        if (
            matched_batch is not None
            and matched_batch.turn is not None
            and matched_batch.turn in self._outbound_turns
            and matched_batch.turn.bound_gen is None
        ):
            turn = matched_batch.turn
        else:
            turn = self._current_open_turn()
        if matched_batch is not None:
            matched_batch.turn = turn
            turn.tool_batch = matched_batch
        if staged.completes_turn_if_nonempty and not staged.completes_turn:
            has_prior_input = (
                turn.has_confirmed_input_payload
                or bool(turn.input_messages)
                or bool(turn.pending_audio_chunks)
                or any(
                    other.has_input_payload and other.status != "aborted"
                    for other in turn.staged_sends
                )
            )
            if has_prior_input:
                staged.completes_turn = True
        staged.turn = turn
        turn.staged_sends.append(staged)
        if staged.completes_turn:
            turn.sealed = True

    def _apply_session_context_update(self, turn: _OutboundTurn) -> None:
        if (
            turn.system_instruction is not None
            and turn.system_instruction_rev >= self._system_instruction_rev
        ):
            self.system_instruction = list(turn.system_instruction)
            self._system_instruction_rev = turn.system_instruction_rev
        if (
            turn.tool_definitions is not None
            and turn.tool_definitions_rev >= self._tool_definitions_rev
        ):
            self.tool_definitions = list(turn.tool_definitions)
            self._tool_definitions_rev = turn.tool_definitions_rev

    def _apply_context_update_to_generation(
        self, gen: _ActiveGeneration, source: _OutboundTurn
    ) -> None:
        if not gen.invocation.should_capture_content:
            return
        if (
            source.system_instruction is not None
            and source.system_instruction_rev >= gen.system_instruction_rev
        ):
            gen.system_instruction = list(source.system_instruction)
            gen.system_instruction_rev = source.system_instruction_rev
        if (
            source.tool_definitions is not None
            and source.tool_definitions_rev >= gen.tool_definitions_rev
        ):
            gen.tool_definitions = list(source.tool_definitions)
            gen.tool_definitions_rev = source.tool_definitions_rev

    def _release_pending_context_update(
        self,
        staged: _StagedSend,
        source: _OutboundTurn | None = None,
    ) -> None:
        if staged in self._pending_context_updates:
            self._pending_context_updates.remove(staged)
        if not staged.dependent_gens:
            return
        dependents = list(staged.dependent_gens)
        staged.dependent_gens.clear()
        for dep_gen in dependents:
            if staged in dep_gen.pending_context_deps:
                dep_gen.pending_context_deps.remove(staged)
            if source is not None:
                self._apply_context_update_to_generation(dep_gen, source)
            self._maybe_finish_deferred_generation(dep_gen)

    def _commit_to_bound_generation(
        self, staged: _StagedSend, target_gen: _ActiveGeneration
    ) -> None:
        # Execute the confirmed send callback against an isolated scratch turn
        # so subsequent turns' pending state is unaffected, then attach any
        # resulting inputs/context updates to ``target_gen``.
        scratch_turn = _OutboundTurn()
        saved_write_turn = self._write_turn
        saved_send_seq = self._current_send_seq
        saved_active_gen = self._active_gen
        saved_override = self._capture_content_override

        target_capture = target_gen.invocation.should_capture_content
        any_dep_capture = any(
            dep.invocation.should_capture_content
            for dep in staged.dependent_gens
        )
        effective_capture = target_capture or any_dep_capture
        self._write_turn = scratch_turn
        self._current_send_seq = staged.seq
        self._active_gen = None
        self._capture_content_override = effective_capture
        try:
            staged.context.run(staged.commit_fn)
        finally:
            self._write_turn = saved_write_turn
            self._current_send_seq = saved_send_seq
            self._active_gen = saved_active_gen
            self._capture_content_override = saved_override

        if target_capture:
            if scratch_turn.input_messages:
                target_gen.append_input_messages(scratch_turn.input_messages)
            if scratch_turn.pending_audio_chunks:
                mime_type = scratch_turn.pending_audio_mime_type or "audio/pcm"
                for chunk in scratch_turn.pending_audio_chunks:
                    target_gen.append_input_audio(chunk, mime_type)
            self._apply_context_update_to_generation(target_gen, scratch_turn)
        if effective_capture:
            self._apply_session_context_update(scratch_turn)
        self._release_pending_context_update(
            staged, source=scratch_turn if effective_capture else None
        )

    def _drain_turn(self, turn: _OutboundTurn) -> None:
        while turn.staged_sends and turn.staged_sends[0].status != "pending":
            staged = turn.staged_sends.popleft()
            if staged.status != "confirmed" or staged.applied:
                continue
            staged.applied = True
            if staged.has_input_payload:
                turn.has_confirmed_input_payload = True
            if staged.completes_turn and staged.consumed_tool_batch is None:
                turn.has_confirmed_turn_complete = True
            if turn.bound_gen is not None:
                self._commit_to_bound_generation(staged, turn.bound_gen)
            else:
                detached_turn = turn not in self._outbound_turns
                saved_write_turn = self._write_turn
                saved_send_seq = self._current_send_seq
                saved_active_gen = self._active_gen
                saved_override = self._capture_content_override
                self._write_turn = turn
                self._current_send_seq = staged.seq
                if detached_turn:
                    self._active_gen = None
                    any_dep_capture = any(
                        dep.invocation.should_capture_content
                        for dep in staged.dependent_gens
                    )
                    self._capture_content_override = (
                        any_dep_capture
                        or self._telemetry_handler.should_capture_realtime_content()
                    )
                try:
                    staged.context.run(staged.commit_fn)
                finally:
                    self._write_turn = saved_write_turn
                    self._current_send_seq = saved_send_seq
                    self._active_gen = saved_active_gen
                    self._capture_content_override = saved_override
                if detached_turn:
                    self._apply_session_context_update(turn)
                self._release_pending_context_update(staged, source=turn)

        if turn.bound_gen is not None:
            if not turn.staged_sends:
                target_gen = turn.bound_gen
                turn.bound_gen = None
                if target_gen.bound_turn is turn:
                    target_gen.bound_turn = None
                self._maybe_finish_deferred_generation(target_gen)
        elif (
            not turn.staged_sends
            and not turn.sealed
            and not turn.input_messages
            and not turn.pending_audio_chunks
            and turn.system_instruction is None
            and turn.tool_definitions is None
            and turn in self._outbound_turns
        ):
            self._outbound_turns.remove(turn)

    def _detach_bound_turn(self, active_gen: _ActiveGeneration) -> None:
        if active_gen.pending_context_deps:
            for ctx_staged in list(active_gen.pending_context_deps):
                if active_gen in ctx_staged.dependent_gens:
                    ctx_staged.dependent_gens.remove(active_gen)
            active_gen.pending_context_deps.clear()
        turn = active_gen.bound_turn
        active_gen.bound_turn = None
        if turn is None:
            return
        remaining_staged = list(turn.staged_sends)
        turn.staged_sends.clear()
        turn.bound_gen = None
        for staged in remaining_staged:
            if staged.status == "confirmed" and not staged.applied:
                staged.applied = True
                self._commit_to_bound_generation(staged, active_gen)
            elif (
                staged.status == "pending"
                and not self._session_finalized
                and (
                    staged.updates_system_instruction
                    or staged.updates_tool_definitions
                )
            ):
                turn.staged_sends.append(staged)
            elif not staged.applied:
                self._release_pending_context_update(staged)

    def _has_pending_sends(self, active_gen: _ActiveGeneration) -> bool:
        return bool(
            (
                active_gen.bound_turn is not None
                and active_gen.bound_turn.staged_sends
            )
            or active_gen.pending_context_deps
        )

    def _maybe_finish_deferred_generation(
        self, active_gen: _ActiveGeneration
    ) -> None:
        if (
            active_gen.deferred_finish_reason is not None
            and not self._has_pending_sends(active_gen)
            and active_gen in self._deferred_generations
        ):
            self._deferred_generations.remove(active_gen)
            finish_reason = active_gen.deferred_finish_reason
            error = active_gen.deferred_error
            active_gen.deferred_finish_reason = None
            active_gen.deferred_error = None
            self._populate_and_finish_generation(
                active_gen, finish_reason=finish_reason, error=error
            )

    def _finish_or_defer_generation(
        self,
        active_gen: _ActiveGeneration,
        finish_reason: str,
        error: Error | BaseException | None = None,
    ) -> None:
        if self._has_pending_sends(active_gen) and error is None:
            active_gen.invocation.mark_end()
            active_gen.deferred_finish_reason = finish_reason
            active_gen.deferred_error = None
            if active_gen not in self._deferred_generations:
                self._deferred_generations.append(active_gen)
            return
        self._detach_bound_turn(active_gen)
        self._populate_and_finish_generation(
            active_gen, finish_reason=finish_reason, error=error
        )

    def _flush_deferred_generations(self) -> None:
        while self._deferred_generations:
            gen = self._deferred_generations.pop(0)
            self._detach_bound_turn(gen)
            finish_reason = gen.deferred_finish_reason or "stop"
            error = gen.deferred_error
            gen.deferred_finish_reason = None
            gen.deferred_error = None
            self._populate_and_finish_generation(
                gen, finish_reason=finish_reason, error=error
            )

    def confirm_inflight_send(self, staged: _StagedSend) -> None:
        if staged.status != "pending":
            return
        if self._session_finalized or staged.omit_tool_response:
            staged.status = "confirmed"
            staged.applied = True
            self._release_pending_context_update(staged)
            return
        if staged.turn is None:
            self.register_inflight_send(staged)
        staged.status = "confirmed"
        turn = staged.turn
        if turn is not None:
            self._drain_turn(turn)

    def abort_inflight_send(self, staged: _StagedSend) -> None:
        if staged.status != "pending":
            return
        staged.status = "aborted"
        self._release_pending_context_update(staged)
        if self._session_finalized:
            return
        reopened_batch: _PendingToolCallBatch | None = None
        if staged.consumed_tool_batch is not None:
            batch = staged.consumed_tool_batch
            restored_ids = [
                cid
                for cid in staged.consumed_tool_ids
                if cid not in batch.cancelled_ids
            ]
            batch.remaining_ids.update(restored_ids)
            batch.remaining_unidentified += staged.consumed_tool_unidentified
            if not batch.is_complete:
                reopened_batch = batch
                if batch not in self._pending_tool_call_batches:
                    self._pending_tool_call_batches.appendleft(batch)
            staged.consumed_tool_batch = None
            staged.consumed_tool_ids.clear()
            staged.consumed_tool_unidentified = 0
            staged.popped_tool_batch = False
        turn = staged.turn
        if turn is None:
            return
        if reopened_batch is not None:
            for other in turn.staged_sends:
                if other.consumed_tool_batch is reopened_batch:
                    other.completes_turn = False
        if (
            staged.completes_turn
            or reopened_batch is not None
            or turn.tool_batch is not None
        ):
            batch_has_active_response = (
                turn.has_confirmed_input_payload
                or bool(turn.input_messages)
                or any(
                    other.has_input_payload and other.status != "aborted"
                    for other in turn.staged_sends
                )
            )
            batch_completed = bool(
                turn.tool_batch is not None
                and turn.tool_batch.is_complete
                and batch_has_active_response
            )
            turn.sealed = (
                turn.has_confirmed_turn_complete
                or batch_completed
                or any(
                    other.completes_turn and other.status != "aborted"
                    for other in turn.staged_sends
                )
            )
        self._drain_turn(turn)

    def _mark_new_user_utterance(self) -> None:
        if self._active_gen is not None:
            self._active_gen.accepting_input_transcription = False

    def _enforce_pending_media_budget(self) -> None:
        if (
            self._write_turn is not None
            and self._write_turn not in self._outbound_turns
        ):
            return
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        excess = (
            sum(
                turn.media_bytes + turn.pending_audio_bytes
                for turn in self._outbound_turns
            )
            - max_bytes
        )
        if excess <= 0:
            return

        # Evict oldest buffered media across queued turns from oldest to newest
        # so the most recent audio/media preceding the next generation is kept.
        turn_idx = 0
        while excess > 0 and turn_idx < len(self._outbound_turns):
            turn = self._outbound_turns[turn_idx]
            msg_idx = 0
            while excess > 0 and msg_idx < len(turn.input_messages):
                msg = turn.input_messages[msg_idx]
                new_parts: list[MessagePart] = []
                for part in msg.parts:
                    if (
                        excess > 0
                        and isinstance(part, BlobPart)
                        and isinstance(part.content, bytes)
                    ):
                        part_len = len(part.content)
                        if (
                            part.modality == Modality.AUDIO
                            and part_len > excess
                        ):
                            part.content = part.content[excess:]
                            turn.media_bytes -= excess
                            excess = 0
                            new_parts.append(part)
                        else:
                            turn.media_bytes -= part_len
                            excess -= part_len
                    else:
                        new_parts.append(part)
                if new_parts:
                    msg.parts = new_parts
                    msg_idx += 1
                else:
                    turn.input_messages.pop(msg_idx)

            while excess > 0 and turn.pending_audio_chunks:
                oldest = turn.pending_audio_chunks[0]
                if len(oldest) <= excess:
                    turn.pending_audio_chunks.popleft()
                    turn.pending_audio_bytes -= len(oldest)
                    excess -= len(oldest)
                else:
                    turn.pending_audio_chunks[0] = oldest[excess:]
                    turn.pending_audio_bytes -= excess
                    excess = 0

            if not turn.pending_audio_chunks:
                turn.pending_audio_mime_type = None

            if (
                not turn.input_messages
                and not turn.pending_audio_chunks
                and not turn.staged_sends
                and not turn.sealed
                and turn.system_instruction is None
                and turn.tool_definitions is None
            ):
                del self._outbound_turns[turn_idx]
            else:
                turn_idx += 1

    def _track_input_message_media(
        self, turn: _OutboundTurn, messages: Sequence[InputMessage]
    ) -> None:
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        added_bytes = 0
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, BlobPart) and isinstance(
                    part.content, bytes
                ):
                    if len(part.content) > max_bytes:
                        part.content = (
                            part.content[-max_bytes:]
                            if part.modality == Modality.AUDIO
                            else part.content[:max_bytes]
                        )
                    added_bytes += len(part.content)
        if added_bytes:
            turn.media_bytes += added_bytes
            self._enforce_pending_media_budget()

    def _flush_pending_audio(
        self,
        *,
        turn: _OutboundTurn | None = None,
        capture_content: bool | None = None,
    ) -> None:
        if turn is None:
            if self._write_turn is not None:
                turn = self._write_turn
            elif self._outbound_turns and not self._outbound_turns[-1].sealed:
                turn = self._outbound_turns[-1]
            else:
                return
        if not turn.pending_audio_chunks:
            return
        should_capture = (
            self._should_capture_content()
            if capture_content is None
            else capture_content
        )
        if should_capture:
            combined = b"".join(turn.pending_audio_chunks)
            mime_type = turn.pending_audio_mime_type or "audio/pcm"
            if len(combined) > _MAX_BUFFERED_MEDIA_BYTES:
                combined = combined[-_MAX_BUFFERED_MEDIA_BYTES:]
            turn.input_messages.append(
                InputMessage(
                    role="user",
                    parts=[
                        BlobPart(
                            mime_type=mime_type,
                            modality=Modality.AUDIO,
                            content=combined,
                        )
                    ],
                )
            )
            turn.media_bytes += len(combined)
        turn.pending_audio_chunks.clear()
        turn.pending_audio_bytes = 0
        turn.pending_audio_mime_type = None
        self._enforce_pending_media_budget()

    def _start_user_speech(self) -> None:
        if self._session_finalized:
            return
        self._mark_new_user_utterance()
        if (
            self._speech_invocation is not None
            or not self._telemetry_handler.should_emit_user_speech()
        ):
            return
        self._speech_invocation = self._telemetry_handler.user_speech(
            provider=self.provider,
            realtime_session_id=self.realtime_session_id,
            error_type_resolver=resolve_error_type,
            context=get_current(),
            _attach_to_context=False,
        )

    def _stop_user_speech(self, error: BaseException | None = None) -> None:
        if self._speech_invocation is None:
            return
        speech = self._speech_invocation
        self._speech_invocation = None
        speech.realtime_session_id = self.realtime_session_id
        if error is not None:
            speech.fail(error)
        else:
            speech.stop()

    def _record_blob_input(
        self,
        blob_or_dict: BlobOrDict | Mapping[str, object] | None,
        default_mime: str,
        force_non_audio: bool = False,
    ) -> None:
        extracted = _extract_blob_bytes_and_mime(
            blob_or_dict, default_mime=default_mime
        )
        if extracted is None:
            return
        turn = self._target_turn()
        data_bytes, mime_type = extracted
        lower_mime = mime_type.lower()
        if not force_non_audio and "audio" in lower_mime:
            if not data_bytes:
                return
            if (
                turn.pending_audio_chunks
                and turn.pending_audio_mime_type
                and mime_type != turn.pending_audio_mime_type
            ):
                self._flush_pending_audio(turn=turn)
            if len(data_bytes) >= _MAX_BUFFERED_MEDIA_BYTES:
                trimmed = data_bytes[-_MAX_BUFFERED_MEDIA_BYTES:]
                turn.pending_audio_chunks.clear()
                turn.pending_audio_chunks.append(trimmed)
                turn.pending_audio_bytes = len(trimmed)
            else:
                turn.pending_audio_chunks.append(data_bytes)
                turn.pending_audio_bytes += len(data_bytes)
            if not turn.pending_audio_mime_type:
                turn.pending_audio_mime_type = mime_type
            self._enforce_pending_media_budget()
            return
        self._flush_pending_audio(turn=turn)
        if len(data_bytes) > _MAX_BUFFERED_MEDIA_BYTES:
            data_bytes = data_bytes[:_MAX_BUFFERED_MEDIA_BYTES]
        modality = (
            Modality.VIDEO
            if force_non_audio or "video" in lower_mime
            else Modality.IMAGE
            if "image" in lower_mime
            else Modality.DOCUMENT
        )
        turn.input_messages.append(
            InputMessage(
                role="user",
                parts=[
                    BlobPart(
                        mime_type=mime_type,
                        modality=modality,
                        content=data_bytes,
                    )
                ],
            )
        )
        turn.media_bytes += len(data_bytes)
        self._enforce_pending_media_budget()

    def on_send_realtime_input(self, kwargs: Mapping[str, object]) -> None:
        if kwargs.get("activity_start") is not None:
            self._start_user_speech()

        audio = kwargs.get("audio")
        media = kwargs.get("media")
        video = kwargs.get("video")
        text = kwargs.get("text")
        if text is not None:
            self._mark_new_user_utterance()

        capture_content = self._should_capture_content()

        if audio is not None:
            if capture_content:
                self._record_blob_input(
                    cast(BlobOrDict | Mapping[str, object], audio),
                    default_mime="audio/pcm",
                )
        elif media is not None:
            if capture_content:
                self._record_blob_input(
                    cast(BlobOrDict | Mapping[str, object], media),
                    default_mime="audio/pcm",
                )
        elif video is not None:
            self._flush_pending_audio()
            if capture_content:
                self._record_blob_input(
                    cast(BlobOrDict | Mapping[str, object], video),
                    default_mime="image/jpeg",
                    force_non_audio=True,
                )
        elif text is not None:
            self._flush_pending_audio()
            if capture_content:
                self._target_turn().input_messages.append(
                    InputMessage(
                        role="user",
                        parts=[TextPart(content=str(text))],
                    )
                )

        if kwargs.get("audio_stream_end"):
            self._mark_new_user_utterance()
            self._flush_pending_audio()
        if kwargs.get("activity_end") is not None:
            self._mark_new_user_utterance()
            self._flush_pending_audio()
            self._stop_user_speech()

    def on_send_client_content(
        self,
        turns: Content
        | ContentDict
        | list[Content | ContentDict]
        | ContentListUnion
        | ContentListUnionDict
        | None,
        turn_complete: bool = True,
    ) -> None:
        self._flush_pending_audio()
        if turns is not None:
            self._mark_new_user_utterance()
        if turns is not None and self._should_capture_content():
            normalized_turns: list[Content] | None = None
            t_client_content = getattr(
                genai_transformers, "t_client_content", None
            )
            if callable(t_client_content):
                try:
                    client_content = t_client_content(turns, turn_complete)
                    raw_turns: object = getattr(client_content, "turns", None)
                    if isinstance(raw_turns, list):
                        normalized_turns = cast(list[Content], raw_turns)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            if normalized_turns is None:
                t_contents = getattr(genai_transformers, "t_contents", None)
                if callable(t_contents):
                    try:
                        raw_contents = t_contents(turns)
                        if isinstance(raw_contents, list):
                            normalized_turns = cast(
                                list[Content], raw_contents
                            )
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
            if normalized_turns is None:
                if isinstance(turns, Content):
                    normalized_turns = [turns]
                elif isinstance(turns, Mapping):
                    try:
                        normalized_turns = [Content.model_validate(turns)]
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                elif isinstance(turns, Sequence) and not isinstance(
                    turns, str
                ):
                    try:
                        seq_turns = cast(Sequence[object], turns)
                        normalized_turns = [
                            Content.model_validate(item)
                            if isinstance(item, Mapping)
                            else cast(Content, item)
                            for item in seq_turns
                        ]
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
            if normalized_turns is not None:
                try:
                    messages = to_input_messages(contents=normalized_turns)
                    turn = self._target_turn()
                    turn.input_messages.extend(messages)
                    self._track_input_message_media(turn, messages)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

    def on_send_tool_response(
        self,
        function_responses: FunctionResponseOrDict
        | Sequence[FunctionResponseOrDict]
        | None,
        *,
        omit_tool_response: bool = False,
    ) -> None:
        if omit_tool_response or (
            self._active_gen is not None
            and self._active_gen.awaiting_tool_response
        ):
            return
        self._flush_pending_audio()
        if not self._should_capture_content():
            return
        tool_msg = _build_tool_response_input_message(function_responses)
        if tool_msg is not None:
            self._target_turn().input_messages.append(tool_msg)

    def on_send_context_update(
        self,
        system_instruction: ContentUnion | ContentUnionDict | None = None,
        tools: ToolListUnion | ToolListUnionDict | None = None,
    ) -> None:
        if not self._should_capture_content():
            return
        self._ensure_config_content_extracted()
        turn = self._target_turn()
        rev = self._current_send_seq
        if rev == 0:
            self._next_send_seq += 1
            rev = self._next_send_seq
        if system_instruction is not None:
            turn.system_instruction = _extract_system_instruction(
                system_instruction
            )
            turn.system_instruction_rev = rev
        if tools is not None:
            turn.tool_definitions = _extract_tool_definitions(tools)
            turn.tool_definitions_rev = rev

    def on_send(
        self,
        input_val: object = None,
        end_of_turn: bool = False,
        *,
        omit_tool_response: bool = False,
    ) -> None:
        if not input_val:
            self._mark_new_user_utterance()
            self._flush_pending_audio()
            return

        if isinstance(input_val, genai_types.LiveClientRealtimeInput) or (
            isinstance(input_val, Mapping)
            and ("media_chunks" in input_val or "mediaChunks" in input_val)
        ):
            if self._should_capture_content():
                chunks = _get_config_field(input_val, "media_chunks")
                if isinstance(chunks, Sequence):
                    for chunk in cast(Sequence[object], chunks):
                        self._record_blob_input(
                            cast(BlobOrDict | Mapping[str, object], chunk),
                            default_mime="audio/pcm",
                        )
            return

        if isinstance(input_val, genai_types.Blob) or (
            isinstance(input_val, Mapping) and "data" in input_val
        ):
            if self._should_capture_content():
                self._record_blob_input(
                    cast(BlobOrDict | Mapping[str, object], input_val),
                    default_mime="audio/pcm",
                )
            return

        if isinstance(input_val, genai_types.LiveClientToolResponse) or (
            isinstance(input_val, Mapping)
            and (
                "function_responses" in input_val
                or "functionResponses" in input_val
            )
        ):
            fn_resps = cast(
                FunctionResponseOrDict
                | Sequence[FunctionResponseOrDict]
                | None,
                _get_config_field(input_val, "function_responses"),
            )
            self.on_send_tool_response(
                fn_resps, omit_tool_response=omit_tool_response
            )
            return

        if isinstance(input_val, genai_types.FunctionResponse) or (
            isinstance(input_val, Mapping)
            and "name" in input_val
            and "response" in input_val
        ):
            self.on_send_tool_response(
                cast(FunctionResponseOrDict, input_val),
                omit_tool_response=omit_tool_response,
            )
            return

        if isinstance(input_val, genai_types.LiveClientContent) or (
            isinstance(input_val, Mapping)
            and ("turns" in input_val or "content" in input_val)
        ):
            turns = _get_config_field(input_val, "turns")
            if turns is None:
                turns = _get_config_field(input_val, "content")
            tc = _get_config_field(input_val, "turn_complete")
            self.on_send_client_content(
                cast(ContentListUnion | ContentListUnionDict | None, turns),
                turn_complete=bool(end_of_turn if tc is None else tc),
            )
            return

        if isinstance(input_val, Sequence) and not isinstance(
            input_val, (str, bytes, bytearray)
        ):
            seq_val = cast(Sequence[object], input_val)
            if seq_val and any(
                isinstance(item, genai_types.FunctionResponse)
                or (
                    isinstance(item, Mapping)
                    and "name" in item
                    and "response" in item
                )
                for item in seq_val
            ):
                self.on_send_tool_response(
                    cast(Sequence[FunctionResponseOrDict], seq_val),
                    omit_tool_response=omit_tool_response,
                )
                return
            if seq_val and any(
                isinstance(item, genai_types.Blob)
                or (isinstance(item, Mapping) and "data" in item)
                for item in seq_val
            ):
                if self._should_capture_content():
                    for item in seq_val:
                        self._record_blob_input(
                            cast(BlobOrDict | Mapping[str, object], item),
                            default_mime="audio/pcm",
                        )
                return

        self.on_send_client_content(
            cast(ContentListUnion | ContentListUnionDict | None, input_val),
            turn_complete=bool(end_of_turn),
        )

    def _consume_turn(
        self,
        turn: _OutboundTurn,
        active_gen: _ActiveGeneration,
        capture_content: bool,
    ) -> None:
        if capture_content:
            self._apply_session_context_update(turn)
            if turn.input_messages:
                active_gen.append_input_messages(turn.input_messages)
            if turn.pending_audio_chunks:
                mime_type = turn.pending_audio_mime_type or "audio/pcm"
                for chunk in turn.pending_audio_chunks:
                    active_gen.append_input_audio(chunk, mime_type)
        turn.input_messages.clear()
        turn.pending_audio_chunks.clear()
        turn.pending_audio_bytes = 0
        turn.pending_audio_mime_type = None
        turn.media_bytes = 0

    def _claim_outbound_inputs(
        self, active_gen: _ActiveGeneration, capture_content: bool
    ) -> _OutboundTurn | None:
        bound_turn: _OutboundTurn | None = None
        claimed_any_input = False

        while self._outbound_turns:
            head = self._outbound_turns[0]
            if head.staged_sends:
                if claimed_any_input:
                    # Do not bind a later turn's unconfirmed sends to a
                    # generation that already claimed prior input.
                    break
                turn = self._outbound_turns.popleft()
                self._consume_turn(turn, active_gen, capture_content)
                bound_turn = turn
                break
            turn = self._outbound_turns.popleft()
            if turn.input_messages or turn.pending_audio_chunks:
                claimed_any_input = True
            self._consume_turn(turn, active_gen, capture_content)
            if turn.sealed:
                break

        return bound_turn

    def ensure_active_generation(self) -> _ActiveGeneration:
        if self._active_gen is None:
            invocation = self._telemetry_handler.realtime_inference(
                self.provider,
                request_model=self.request_model,
                server_address=self.server_address,
                server_port=self.server_port,
                realtime_session_id=self.realtime_session_id,
                error_type_resolver=resolve_error_type,
                context=get_current(),
                _attach_to_context=False,
            )
            capture_content = invocation.should_capture_content
            if capture_content:
                self._ensure_config_content_extracted()
            invocation.output_type = self.output_type
            if self.max_tokens is not None:
                invocation.max_tokens = self.max_tokens
            if self.reasoning_level is not None:
                invocation.reasoning_level = self.reasoning_level

            input_transcript_deltas = list(
                self._pending_input_transcript_deltas
            )
            self._pending_input_transcript_deltas.clear()
            active_gen = _ActiveGeneration(
                invocation=invocation,
                input_transcript_deltas=input_transcript_deltas,
            )
            bound_turn = self._claim_outbound_inputs(
                active_gen, capture_content
            )
            if capture_content:
                active_gen.system_instruction = list(self.system_instruction)
                active_gen.tool_definitions = list(self.tool_definitions)
            active_gen.system_instruction_rev = self._system_instruction_rev
            active_gen.tool_definitions_rev = self._tool_definitions_rev
            active_gen.bound_turn = bound_turn
            if bound_turn is not None:
                bound_turn.bound_gen = active_gen
            if capture_content and self._pending_context_updates:
                for ctx_staged in self._pending_context_updates:
                    if (
                        ctx_staged.turn is None
                        or ctx_staged.turn is bound_turn
                        or ctx_staged.turn in self._outbound_turns
                    ):
                        continue
                    needs_sys = (
                        ctx_staged.updates_system_instruction
                        and active_gen.system_instruction_rev < ctx_staged.seq
                    )
                    needs_tools = (
                        ctx_staged.updates_tool_definitions
                        and active_gen.tool_definitions_rev < ctx_staged.seq
                    )
                    if needs_sys or needs_tools:
                        active_gen.pending_context_deps.append(ctx_staged)
                        ctx_staged.dependent_gens.append(active_gen)
            self._active_gen = active_gen
        return self._active_gen

    def _populate_and_finish_generation(
        self,
        active_gen: _ActiveGeneration,
        finish_reason: str,
        error: Error | BaseException | None = None,
    ) -> None:
        invocation = active_gen.invocation
        invocation.realtime_session_id = self.realtime_session_id
        capture_content = invocation.should_capture_content

        if capture_content:
            active_gen.flush_input_audio()

        # Attach any server-reported input audio transcription to the user message.
        if active_gen.input_transcript_deltas and capture_content:
            input_transcript = "".join(active_gen.input_transcript_deltas)
            active_gen.input_transcript_deltas.clear()
            user_msg: InputMessage | None = None
            for msg in reversed(active_gen.input_messages):
                if msg.role == "user":
                    user_msg = msg
                    break
            if user_msg is not None:
                user_msg.parts.append(
                    TranscriptionPart(content=input_transcript)
                )
            else:
                active_gen.input_messages.append(
                    InputMessage(
                        role="user",
                        parts=[TranscriptionPart(content=input_transcript)],
                    )
                )

        if capture_content:
            active_gen.flush_current_turn_output_message(finish_reason)

        invocation.finish_reasons = [finish_reason]
        invocation.output_type = self.output_type

        if capture_content:
            invocation.input_messages = active_gen.input_messages
            invocation.output_messages = active_gen.completed_output_messages
            if active_gen.system_instruction:
                invocation.system_instruction = active_gen.system_instruction
            if active_gen.tool_definitions:
                invocation.tool_definitions = active_gen.tool_definitions

        usage = active_gen.usage_metadata
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            if prompt_tokens is not None:
                invocation.input_tokens = prompt_tokens

            thoughts_tokens = getattr(usage, "thoughts_token_count", None)
            if thoughts_tokens is not None:
                invocation.thinking_tokens = thoughts_tokens

            response_tokens = getattr(usage, "response_token_count", None)
            if response_tokens is not None:
                # On Vertex Live, google-genai maps wire `candidatesTokenCount`
                # (which excludes reasoning tokens) to `response_token_count`.
                invocation.output_tokens = (
                    response_tokens + (thoughts_tokens or 0)
                    if self.is_vertex
                    else response_tokens
                )
            else:
                candidates_tokens = getattr(
                    usage, "candidates_token_count", None
                )
                if candidates_tokens is not None:
                    invocation.output_tokens = candidates_tokens + (
                        thoughts_tokens or 0
                    )
                elif thoughts_tokens is not None:
                    invocation.output_tokens = thoughts_tokens

            cached_tokens = getattr(usage, "cached_content_token_count", None)
            if cached_tokens is not None:
                invocation.cache_read_input_tokens = cached_tokens

            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                invocation.set_input_tokens(
                    (detail.modality, detail.token_count)
                    for detail in prompt_details
                )

            response_details = getattr(
                usage, "response_tokens_details", None
            ) or getattr(usage, "candidates_tokens_details", None)
            if response_details is not None:
                invocation.set_output_tokens(
                    (detail.modality, detail.token_count)
                    for detail in response_details
                )

            cache_details = getattr(usage, "cache_tokens_details", None)
            if cache_details is not None:
                invocation.set_cache_read_input_tokens(
                    (detail.modality, detail.token_count)
                    for detail in cache_details
                )

        if error is not None:
            invocation.fail(error)
        else:
            invocation.stop()

    def on_server_message(self, message: LiveServerMessage | None) -> None:
        if self._session_finalized:
            return
        self.terminal_error = None
        if message is None:
            return

        # Server-side VAD is reported via voice_activity or
        # voice_activity_detection_signal (both added after google-genai==1.32.0).
        voice_activity: object = getattr(message, "voice_activity", None)
        vad_signal: object = getattr(
            message, "voice_activity_detection_signal", None
        )
        server_content = getattr(message, "server_content", None)
        tool_call = getattr(message, "tool_call", None)
        tool_call_cancellation = getattr(
            message, "tool_call_cancellation", None
        )
        usage_metadata = getattr(message, "usage_metadata", None)

        if tool_call_cancellation is not None:
            cancelled_ids = getattr(tool_call_cancellation, "ids", None) or ()
            cancelled_set = {str(cid) for cid in cancelled_ids if cid}
            if cancelled_set:
                seen_batches: set[int] = set()
                live_batches: list[_PendingToolCallBatch] = []

                def _add_batch(batch: _PendingToolCallBatch | None) -> None:
                    if batch is not None and id(batch) not in seen_batches:
                        seen_batches.add(id(batch))
                        live_batches.append(batch)

                for batch in self._pending_tool_call_batches:
                    _add_batch(batch)
                live_turns: list[_OutboundTurn] = list(self._outbound_turns)
                if (
                    self._active_gen is not None
                    and self._active_gen.bound_turn is not None
                ):
                    live_turns.append(self._active_gen.bound_turn)
                for deferred_gen in self._deferred_generations:
                    if deferred_gen.bound_turn is not None:
                        live_turns.append(deferred_gen.bound_turn)
                for turn in live_turns:
                    _add_batch(turn.tool_batch)
                    for staged in turn.staged_sends:
                        _add_batch(staged.consumed_tool_batch)

                for batch in live_batches:
                    matched_cancelled = cancelled_set & batch.all_ids
                    if matched_cancelled:
                        batch.cancelled_ids.update(matched_cancelled)
                    batch.remaining_ids.difference_update(cancelled_set)

                remaining_batches: deque[_PendingToolCallBatch] = deque()
                completed_batches: list[_PendingToolCallBatch] = []
                for batch in self._pending_tool_call_batches:
                    if batch.is_complete:
                        completed_batches.append(batch)
                    else:
                        remaining_batches.append(batch)
                self._pending_tool_call_batches = remaining_batches

                for completed_batch in completed_batches:
                    batch_turn = completed_batch.turn
                    if (
                        batch_turn is not None
                        and batch_turn in self._outbound_turns
                        and not batch_turn.sealed
                    ):
                        batch_has_active_response = (
                            batch_turn.has_confirmed_input_payload
                            or bool(batch_turn.input_messages)
                            or any(
                                other.has_input_payload
                                and other.status != "aborted"
                                for other in batch_turn.staged_sends
                            )
                        )
                        if batch_has_active_response:
                            batch_turn.sealed = True

        raw_vad: object = (
            getattr(voice_activity, "voice_activity_type", None)
            if voice_activity is not None
            else getattr(vad_signal, "vad_signal_type", None)
            if vad_signal is not None
            else None
        )
        if raw_vad is not None:
            vad_str = str(getattr(raw_vad, "value", raw_vad)).upper()
            if "START" in vad_str or "BOS" in vad_str or "SOS" in vad_str:
                self._start_user_speech()
            elif "END" in vad_str or "EOS" in vad_str:
                if (
                    self._outbound_turns
                    and not self._outbound_turns[-1].sealed
                ):
                    open_turn = self._outbound_turns[-1]
                    self._flush_pending_audio(turn=open_turn)
                    if (
                        open_turn.input_messages
                        or open_turn.pending_audio_chunks
                        or any(
                            other.has_input_payload
                            and other.status != "aborted"
                            for other in open_turn.staged_sends
                        )
                    ):
                        open_turn.sealed = True
                self._stop_user_speech()

        if server_content is not None and getattr(
            server_content, "interrupted", False
        ):
            self._mark_new_user_utterance()

        capture_content = self._should_capture_content()

        # Accumulate input transcription deltas without starting the generation span.
        if server_content is not None and capture_content:
            input_tx = getattr(server_content, "input_transcription", None)
            if input_tx is not None and getattr(input_tx, "text", None):
                if (
                    self._active_gen is not None
                    and self._active_gen.accepting_input_transcription
                ):
                    self._active_gen.input_transcript_deltas.append(
                        str(input_tx.text)
                    )
                else:
                    self._pending_input_transcript_deltas.append(
                        str(input_tx.text)
                    )

        provider_reason = _turn_complete_reason(server_content)
        interaction_complete = _is_interaction_complete(server_content)
        has_model_output = bool(
            server_content is not None
            and (
                (
                    getattr(server_content, "output_transcription", None)
                    and getattr(
                        server_content.output_transcription, "text", None
                    )
                )
                or (
                    getattr(server_content, "model_turn", None)
                    and getattr(server_content.model_turn, "parts", None)
                )
            )
        )
        has_completion_signal = bool(
            server_content is not None
            and (
                getattr(server_content, "interrupted", False)
                or getattr(server_content, "generation_complete", False)
                or interaction_complete
                or provider_reason is not None
            )
        )
        has_tool_call = bool(
            tool_call is not None
            and getattr(tool_call, "function_calls", None)
        )

        if (
            self._active_gen is None
            and not has_model_output
            and not has_tool_call
        ):
            return

        active_gen = self.ensure_active_generation()
        capture_content = active_gen.invocation.should_capture_content

        if usage_metadata is not None:
            active_gen.usage_metadata = usage_metadata

        if server_content is not None:
            if provider_reason is not None:
                active_gen.turn_complete_reason = provider_reason

            if capture_content:
                output_tx = getattr(
                    server_content, "output_transcription", None
                )
                if output_tx is not None and getattr(output_tx, "text", None):
                    active_gen.output_transcript_deltas.append(
                        str(output_tx.text)
                    )

                model_turn = getattr(server_content, "model_turn", None)
                if model_turn is not None and getattr(
                    model_turn, "parts", None
                ):
                    for part in model_turn.parts or []:
                        converted = _to_part(
                            part, len(active_gen.output_other_parts)
                        )
                        if converted is None:
                            continue
                        if isinstance(converted, ReasoningPart):
                            active_gen.output_thought_deltas.append(
                                converted.content
                            )
                        elif isinstance(converted, TextPart):
                            active_gen.output_text_deltas.append(
                                converted.content
                            )
                        elif isinstance(converted, BlobPart) and (
                            converted.modality == Modality.AUDIO
                            or "audio" in (converted.mime_type or "").lower()
                            or not converted.mime_type
                        ):
                            raw_data: bytes | str = converted.content
                            if isinstance(raw_data, str):
                                raw_data = decode_base64(
                                    raw_data
                                ) or raw_data.encode("utf-8")
                            raw_bytes = bytes(raw_data)
                            remaining = (
                                _MAX_BUFFERED_MEDIA_BYTES
                                - active_gen.output_audio_bytes
                            )
                            if remaining > 0:
                                if len(raw_bytes) > remaining:
                                    raw_bytes = raw_bytes[:remaining]
                                active_gen.output_audio_chunks.append(
                                    raw_bytes
                                )
                                active_gen.output_audio_bytes += len(raw_bytes)
                                if not active_gen.output_audio_mime_type:
                                    active_gen.output_audio_mime_type = (
                                        converted.mime_type or "audio/pcm"
                                    )
                        else:
                            active_gen.output_other_parts.append(converted)

            if getattr(server_content, "interrupted", False):
                active_gen.interrupted = True
            if getattr(server_content, "generation_complete", False):
                active_gen.generation_complete = True
            if interaction_complete:
                active_gen.turn_complete = True
            if has_completion_signal:
                active_gen.invocation.mark_end()

        if has_tool_call and tool_call is not None:
            if self.uses_single_generation_tool_exchange:
                # Per PR #394, gen_ai.output.messages represents the output
                # produced by the completed generation and SHOULD NOT include
                # intermediate events that occur while the generation is active,
                # such as a tool call handled before the generation completes.
                active_gen.awaiting_tool_response = True
            else:
                batch_ids: set[str] = set()
                batch_unidentified = 0
                for fc in tool_call.function_calls or []:
                    fc_id = getattr(fc, "id", None)
                    if fc_id is not None and str(fc_id):
                        batch_ids.add(str(fc_id))
                    else:
                        batch_unidentified += 1
                if batch_ids or batch_unidentified:
                    self._pending_tool_call_batches.append(
                        _PendingToolCallBatch(
                            all_ids=set(batch_ids),
                            remaining_ids=batch_ids,
                            remaining_unidentified=batch_unidentified,
                        )
                    )
                if capture_content:
                    tool_parts: list[MessagePart] = [
                        ToolCallRequestPart(
                            id=getattr(fc, "id", None),
                            name=getattr(fc, "name", None) or "",
                            arguments=dict(fc.args)
                            if getattr(fc, "args", None) is not None
                            and fc.args is not None
                            else {},
                        )
                        for fc in (tool_call.function_calls or [])
                    ]
                    active_gen.flush_current_turn_output_message(
                        finish_reason=None
                    )
                    active_gen.completed_output_messages.append(
                        OutputMessage(
                            role="assistant",
                            parts=tool_parts,
                            finish_reason="tool_call",
                        )
                    )
                self._active_gen = None
                self._finish_or_defer_generation(
                    active_gen, finish_reason="tool_call"
                )
                return

        if interaction_complete:
            finish_reason = active_gen.turn_complete_reason or (
                "interrupted" if active_gen.interrupted else "stop"
            )
            self._active_gen = None
            self._finish_or_defer_generation(
                active_gen, finish_reason=finish_reason
            )

    def _fail_active_operations(self, exc: BaseException) -> None:
        self.terminal_error = exc
        self._stop_user_speech(error=exc)
        if self._active_gen is not None:
            active_gen = self._active_gen
            self._active_gen = None
            self._detach_bound_turn(active_gen)
            completed_reason = _completed_finish_reason(active_gen)
            if completed_reason is not None:
                self._populate_and_finish_generation(
                    active_gen, finish_reason=completed_reason, error=None
                )
            else:
                self._populate_and_finish_generation(
                    active_gen, finish_reason="error", error=exc
                )

    def on_send_error(self, exc: BaseException) -> None:
        if self._session_finalized:
            return
        self._fail_active_operations(exc)

    def on_receive_error(self, exc: BaseException) -> None:
        if self._session_finalized:
            return
        self._fail_active_operations(exc)

    def on_session_error(self, exc: BaseException) -> None:
        self._flush_deferred_generations()
        self._fail_active_operations(exc)

    def on_session_close(self) -> None:
        self._flush_deferred_generations()
        self._stop_user_speech()
        if self._active_gen is not None:
            active_gen = self._active_gen
            self._active_gen = None
            self._detach_bound_turn(active_gen)
            completed_reason = _completed_finish_reason(active_gen)
            if completed_reason is not None:
                self._populate_and_finish_generation(
                    active_gen, finish_reason=completed_reason, error=None
                )
            else:
                self._populate_and_finish_generation(
                    active_gen,
                    finish_reason="error",
                    error=Error(
                        message="Session closed before generation completed",
                        type=ErrorTypeValues.OTHER.value,
                    ),
                )

    def finalize_session(self, error: BaseException | None = None) -> None:
        if self._session_finalized:
            return
        self._session_finalized = True
        if error is not None:
            self.on_session_error(error)
        else:
            self.on_session_close()
        self._pending_context_updates.clear()
        effective_error = error if error is not None else self.terminal_error
        self._telemetry_handler.emit_realtime_session_ended(
            self.provider,
            request_model=self.request_model,
            realtime_session_id=self.realtime_session_id,
            server_address=self.server_address,
            server_port=self.server_port,
            error=effective_error,
            error_type_resolver=resolve_error_type,
            context=self.caller_context,
        )


async def _run_transactional_send(
    tracker: _LiveSessionTracker,
    commit_fn: Callable[[], None],
    call_fn: Callable[[], object],
    *,
    completes_turn: bool = False,
    completes_turn_if_nonempty: bool = False,
    has_input_payload: bool = False,
    omit_tool_response: bool = False,
    starts_new_turn: bool = False,
    is_tool_response: bool = False,
    tool_response_ids: tuple[str, ...] = (),
    tool_response_unidentified: int = 0,
    updates_system_instruction: bool = False,
    updates_tool_definitions: bool = False,
) -> object:
    existing_staged = _ACTIVE_SEND_CALLBACK.get()
    if existing_staged is not None:
        existing_staged.commit_fn = commit_fn
        existing_staged.completes_turn = completes_turn
        existing_staged.completes_turn_if_nonempty = completes_turn_if_nonempty
        existing_staged.has_input_payload = has_input_payload
        existing_staged.omit_tool_response = omit_tool_response
        existing_staged.starts_new_turn = starts_new_turn
        existing_staged.is_tool_response = is_tool_response
        existing_staged.tool_response_ids = tool_response_ids
        existing_staged.tool_response_unidentified = tool_response_unidentified
        existing_staged.updates_system_instruction = updates_system_instruction
        existing_staged.updates_tool_definitions = updates_tool_definitions
        res = call_fn()
        if inspect.isawaitable(res):
            return await res
        return res

    staged = _StagedSend(
        commit_fn=commit_fn,
        context=contextvars.copy_context(),
        completes_turn=completes_turn,
        completes_turn_if_nonempty=completes_turn_if_nonempty,
        has_input_payload=has_input_payload,
        omit_tool_response=omit_tool_response,
        starts_new_turn=starts_new_turn,
        is_tool_response=is_tool_response,
        tool_response_ids=tool_response_ids,
        tool_response_unidentified=tool_response_unidentified,
        updates_system_instruction=updates_system_instruction,
        updates_tool_definitions=updates_tool_definitions,
    )
    token = _ACTIVE_SEND_CALLBACK.set(staged)
    try:
        res = call_fn()
        if inspect.isawaitable(res):
            res = await res
    except BaseException as exc:
        tracker.abort_inflight_send(staged)
        if not isinstance(exc, (ValueError, TypeError)):
            tracker.on_send_error(exc)
        raise
    else:
        tracker.confirm_inflight_send(staged)
        return res
    finally:
        _ACTIVE_SEND_CALLBACK.reset(token)


def _wrap_session(session: AsyncSession, tracker: _LiveSessionTracker) -> None:
    """Wrap the methods of a live ``AsyncSession`` instance to record telemetry."""
    orig_send_realtime_input = session.send_realtime_input
    orig_send_client_content = session.send_client_content
    orig_send_tool_response = session.send_tool_response
    orig_send = getattr(session, "send", None)
    orig_send_context_update = getattr(session, "send_context_update", None)
    orig_close = getattr(session, "close", None)
    orig_receive = session._receive

    ws: object = getattr(session, "_ws", None)
    orig_ws_send = getattr(ws, "send", None) if ws is not None else None
    if ws is not None and callable(orig_ws_send):

        @functools.wraps(orig_ws_send)
        async def wrapped_ws_send(*args: object, **kwargs: object) -> object:
            staged = _ACTIVE_SEND_CALLBACK.get()
            if staged is not None:
                tracker.register_inflight_send(staged)
            try:
                res = orig_ws_send(*args, **kwargs)
                if inspect.isawaitable(res):
                    res = await res
            except BaseException as exc:
                if staged is not None:
                    tracker.abort_inflight_send(staged)
                tracker.on_send_error(exc)
                raise
            if staged is not None:
                tracker.confirm_inflight_send(staged)
            return res

        with contextlib.suppress(Exception):
            setattr(ws, "send", wrapped_ws_send)

    @functools.wraps(orig_send_realtime_input)
    async def wrapped_send_realtime_input(
        *args: object, **kwargs: object
    ) -> None:
        completes_turn = kwargs.get("text") is not None
        completes_turn_if_nonempty = bool(
            kwargs.get("activity_end") is not None
            or kwargs.get("audio_stream_end")
        )
        has_input_payload = any(
            kwargs.get(field_name) is not None
            for field_name in ("audio", "media", "video", "text")
        )
        starts_new_turn = kwargs.get("activity_start") is not None
        await _run_transactional_send(
            tracker,
            lambda: tracker.on_send_realtime_input(kwargs),
            lambda: orig_send_realtime_input(*args, **kwargs),  # pyright: ignore[reportArgumentType]
            completes_turn=completes_turn,
            completes_turn_if_nonempty=completes_turn_if_nonempty,
            has_input_payload=has_input_payload,
            starts_new_turn=starts_new_turn,
        )

    @functools.wraps(orig_send_client_content)
    async def wrapped_send_client_content(
        *args: object, **kwargs: object
    ) -> None:
        turns = cast(
            Content
            | ContentDict
            | list[Content | ContentDict]
            | ContentListUnion
            | ContentListUnionDict
            | None,
            get_argument("turns", orig_send_client_content, args, kwargs),
        )
        turn_complete = get_argument(
            "turn_complete", orig_send_client_content, args, kwargs
        )
        effective_turn_complete = (
            True if turn_complete is None else bool(turn_complete)
        )
        await _run_transactional_send(
            tracker,
            lambda: tracker.on_send_client_content(
                turns, turn_complete=effective_turn_complete
            ),
            lambda: orig_send_client_content(*args, **kwargs),  # pyright: ignore[reportArgumentType]
            completes_turn=effective_turn_complete,
            has_input_payload=turns is not None,
        )

    @functools.wraps(orig_send_tool_response)
    async def wrapped_send_tool_response(
        *args: object, **kwargs: object
    ) -> None:
        function_responses = cast(
            FunctionResponseOrDict | Sequence[FunctionResponseOrDict] | None,
            get_argument(
                "function_responses", orig_send_tool_response, args, kwargs
            ),
        )
        omit_tool_response = bool(
            tracker._active_gen is not None
            and tracker._active_gen.awaiting_tool_response
        )
        tool_ids, tool_unidentified = _extract_tool_response_keys(
            function_responses
        )
        await _run_transactional_send(
            tracker,
            lambda: tracker.on_send_tool_response(
                function_responses, omit_tool_response=omit_tool_response
            ),
            lambda: orig_send_tool_response(*args, **kwargs),  # pyright: ignore[reportArgumentType]
            completes_turn=not omit_tool_response,
            has_input_payload=not omit_tool_response,
            omit_tool_response=omit_tool_response,
            is_tool_response=True,
            tool_response_ids=tool_ids,
            tool_response_unidentified=tool_unidentified,
        )

    @functools.wraps(orig_receive)
    async def wrapped_receive(
        *args: object, **kwargs: object
    ) -> LiveServerMessage:
        active_gen = tracker._active_gen
        if active_gen is not None:
            with active_gen.invocation.activate():
                try:
                    message = await orig_receive(*args, **kwargs)
                except BaseException as exc:
                    tracker.on_receive_error(exc)
                    raise
        else:
            try:
                message = await orig_receive(*args, **kwargs)
            except BaseException as exc:
                tracker.on_receive_error(exc)
                raise
        tracker.on_server_message(message)
        return message

    setattr(session, "send_realtime_input", wrapped_send_realtime_input)
    setattr(session, "send_client_content", wrapped_send_client_content)
    setattr(session, "send_tool_response", wrapped_send_tool_response)
    setattr(session, "_receive", wrapped_receive)

    if callable(orig_send):

        @functools.wraps(orig_send)
        async def wrapped_send(*args: object, **kwargs: object) -> object:
            input_val = get_argument("input", orig_send, args, kwargs)
            end_of_turn = get_argument("end_of_turn", orig_send, args, kwargs)
            omit_tool_response = bool(
                tracker._active_gen is not None
                and tracker._active_gen.awaiting_tool_response
            )
            (
                completes_turn,
                omit_from_queue,
                is_tool_response,
                tool_ids,
                tool_unidentified,
            ) = _classify_legacy_send(
                input_val,
                bool(end_of_turn),
                omit_tool_response=omit_tool_response,
            )
            return await _run_transactional_send(
                tracker,
                lambda: tracker.on_send(
                    input_val,
                    end_of_turn=bool(end_of_turn),
                    omit_tool_response=omit_tool_response,
                ),
                lambda: orig_send(*args, **kwargs),
                completes_turn=completes_turn,
                has_input_payload=bool(input_val) and not omit_from_queue,
                omit_tool_response=omit_from_queue,
                is_tool_response=is_tool_response,
                tool_response_ids=tool_ids,
                tool_response_unidentified=tool_unidentified,
            )

        setattr(session, "send", wrapped_send)

    if callable(orig_send_context_update):

        @functools.wraps(orig_send_context_update)
        async def wrapped_send_context_update(
            *args: object, **kwargs: object
        ) -> object:
            sys_inst = cast(
                ContentUnion | ContentUnionDict | None,
                get_argument(
                    "system_instruction",
                    orig_send_context_update,
                    args,
                    kwargs,
                ),
            )
            tools = cast(
                ToolListUnion | ToolListUnionDict | None,
                get_argument("tools", orig_send_context_update, args, kwargs),
            )
            return await _run_transactional_send(
                tracker,
                lambda: tracker.on_send_context_update(
                    system_instruction=sys_inst, tools=tools
                ),
                lambda: orig_send_context_update(*args, **kwargs),
                updates_system_instruction=sys_inst is not None,
                updates_tool_definitions=tools is not None,
            )

        setattr(session, "send_context_update", wrapped_send_context_update)

    if callable(orig_close):

        @functools.wraps(orig_close)
        async def wrapped_close(*args: object, **kwargs: object) -> object:
            close_error: BaseException | None = None
            try:
                res = orig_close(*args, **kwargs)
                if inspect.isawaitable(res):
                    res = await res
                return res
            except BaseException as exc:
                close_error = exc
                raise
            finally:
                tracker.finalize_session(error=close_error)

        setattr(session, "close", wrapped_close)


@contextlib.asynccontextmanager
async def _connect_context_manager(
    live_instance: AsyncLive,
    original_connect: Callable[..., AbstractAsyncContextManager[AsyncSession]],
    telemetry_handler: TelemetryHandler,
    *args: object,
    **kwargs: object,
) -> AsyncIterator[AsyncSession]:
    bound_args = (live_instance, *args)
    raw_model = get_argument("model", original_connect, bound_args, kwargs)
    model = str(raw_model) if raw_model is not None else None
    config = cast(
        LiveConnectConfigOrDict | None,
        get_argument("config", original_connect, bound_args, kwargs),
    )
    request_model_name = _strip_model_prefix(model)
    provider, server_address, server_port = _extract_live_endpoint_info(
        live_instance, config=config
    )
    api_client: object = getattr(live_instance, "_api_client", None)
    is_vertex = bool(getattr(api_client, "vertexai", False))
    caller_context = get_current()

    session_started = False
    session_error: BaseException | None = None
    tracker: _LiveSessionTracker | None = None
    try:
        async with original_connect(live_instance, *args, **kwargs) as session:
            raw_session_id = getattr(session, "session_id", None)
            realtime_session_id = (
                str(raw_session_id) if raw_session_id else str(uuid.uuid4())
            )
            telemetry_handler.emit_realtime_session_started(
                provider,
                request_model=request_model_name,
                realtime_session_id=realtime_session_id,
                server_address=server_address,
                server_port=server_port,
                context=caller_context,
            )
            session_started = True
            tracker = _LiveSessionTracker(
                session=session,
                telemetry_handler=telemetry_handler,
                provider=provider,
                is_vertex=is_vertex,
                request_model=request_model_name,
                realtime_session_id=realtime_session_id,
                server_address=server_address,
                server_port=server_port,
                config=config,
                caller_context=caller_context,
            )
            _wrap_session(session, tracker)
            yield session
    except BaseException as exc:
        session_error = exc
        if tracker is not None:
            tracker.finalize_session(error=exc)
        raise
    finally:
        if session_started and tracker is not None:
            tracker.finalize_session(error=session_error)


class _LiveMethodsSnapshot:
    def __init__(self) -> None:
        self._original_connect = AsyncLive.connect

    def restore(self) -> None:
        setattr(AsyncLive, "connect", self._original_connect)


def instrument_live(
    telemetry_handler: TelemetryHandler,
) -> _LiveMethodsSnapshot:
    """Instrument ``AsyncLive.connect`` on ``google.genai.live``."""
    snapshot = _LiveMethodsSnapshot()
    original_connect = snapshot._original_connect

    @functools.wraps(original_connect)
    def instrumented_connect(
        self: AsyncLive, *args: object, **kwargs: object
    ) -> AbstractAsyncContextManager[AsyncSession]:
        return _connect_context_manager(
            self,
            original_connect,
            telemetry_handler,
            *args,
            **kwargs,
        )

    setattr(AsyncLive, "connect", instrumented_connect)
    return snapshot


def uninstrument_live(
    snapshot: _LiveMethodsSnapshot | None = None,
) -> None:
    """Restore ``AsyncLive.connect`` on ``google.genai.live``."""
    if snapshot is not None:
        snapshot.restore()
        return
    setattr(AsyncLive, "connect", _ORIGINAL_ASYNC_LIVE_CONNECT)
