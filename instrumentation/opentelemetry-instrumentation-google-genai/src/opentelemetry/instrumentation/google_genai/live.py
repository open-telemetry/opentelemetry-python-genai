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
    LiveConnectConfigOrDict,
    LiveServerContent,
    LiveServerMessage,
    Tool,
    ToolListUnion,
    ToolListUnionDict,
    ToolUnionDict,
    UsageMetadata,
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
    UriPart,
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
    if (
        not tools
        or not isinstance(tools, Sequence)
        or isinstance(tools, (str, bytes, bytearray))
    ):
        return []
    definitions: list[ToolDefinition] = []
    for item in tools:
        tool_item: object = item
        if tool_item is None:
            continue
        if isinstance(tool_item, Mapping):
            try:
                tool_item = Tool.model_validate(tool_item)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
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


def _modality_from_mime_type(mime_type: str | None) -> Modality:
    lower = (mime_type or "").lower()
    if "image" in lower:
        return Modality.IMAGE
    if "audio" in lower:
        return Modality.AUDIO
    if "video" in lower:
        return Modality.VIDEO
    if "text" in lower:
        return Modality.TEXT
    return Modality.DOCUMENT


def _extract_function_response_parts(item: object) -> list[MessagePart]:
    raw_parts = _get_config_field(item, "parts")
    if not isinstance(raw_parts, Sequence) or isinstance(
        raw_parts, (str, bytes, bytearray)
    ):
        return []
    converted: list[MessagePart] = []
    for raw_part in cast(Sequence[object], raw_parts):
        inline_data = _get_config_field(raw_part, "inline_data")
        if inline_data is not None:
            raw_mime = _get_config_field(inline_data, "mime_type")
            mime_type = str(raw_mime) if raw_mime else None
            raw_data = _get_config_field(inline_data, "data")
            data_bytes: bytes | None = None
            if isinstance(raw_data, (bytes, bytearray)):
                data_bytes = bytes(raw_data)
            elif isinstance(raw_data, str):
                data_bytes = decode_base64(raw_data) or raw_data.encode(
                    "utf-8"
                )
            if data_bytes is not None:
                converted.append(
                    BlobPart(
                        mime_type=mime_type,
                        modality=_modality_from_mime_type(mime_type),
                        content=data_bytes,
                    )
                )
            continue
        file_data = _get_config_field(raw_part, "file_data")
        if file_data is not None:
            file_uri = _get_config_field(file_data, "file_uri")
            if file_uri is not None and str(file_uri):
                raw_mime = _get_config_field(file_data, "mime_type")
                mime_type = str(raw_mime) if raw_mime else None
                converted.append(
                    UriPart(
                        mime_type=mime_type,
                        modality=_modality_from_mime_type(mime_type),
                        uri=str(file_uri),
                    )
                )
    return converted


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

    parts: list[MessagePart] = []
    for item in items:
        call_id = _get_config_field(item, "id")
        parts.append(
            ToolCallResponsePart(
                id=str(call_id) if call_id is not None else None,
                response=object_to_any_value(
                    _get_config_field(item, "response")
                ),
            )
        )
        parts.extend(_extract_function_response_parts(item))
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


def _is_silent_scheduling(scheduling: object) -> bool:
    if scheduling is None:
        return False
    val = str(getattr(scheduling, "value", scheduling)).upper()
    return val == "SILENT"


def _extract_tool_response_metadata(
    function_responses: FunctionResponseOrDict
    | Sequence[FunctionResponseOrDict]
    | None,
) -> tuple[tuple[str, ...], tuple[str, ...], int, bool, bool]:
    """Return ``(completed_ids, continuing_ids, completed_unidentified, has_non_silent, has_continuing_non_silent)``."""
    if function_responses is None:
        return (), (), 0, False, False
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
    completed_ids: list[str] = []
    continuing_ids: list[str] = []
    completed_unidentified = 0
    has_non_silent = False
    has_continuing_non_silent = False
    for item in items:
        will_continue = bool(_get_config_field(item, "will_continue"))
        is_silent = _is_silent_scheduling(
            _get_config_field(item, "scheduling")
        )
        if not is_silent:
            has_non_silent = True
            if will_continue:
                has_continuing_non_silent = True
        call_id = _get_config_field(item, "id")
        if will_continue:
            if call_id is not None and str(call_id):
                continuing_ids.append(str(call_id))
            continue
        if call_id is not None and str(call_id):
            completed_ids.append(str(call_id))
        else:
            completed_unidentified += 1
    return (
        tuple(completed_ids),
        tuple(continuing_ids),
        completed_unidentified,
        has_non_silent,
        has_continuing_non_silent,
    )


@dataclass(eq=False)
class _PendingToolCallBatch:
    """Tracks unanswered function calls from a Gemini 2.x ``tool_call`` batch."""

    seq: int = 0
    all_ids: set[str] = field(default_factory=set)
    remaining_ids: set[str] = field(default_factory=set)
    cancelled_ids: set[str] = field(default_factory=set)
    remaining_unidentified: int = 0
    turn: _OutboundTurn | None = None

    @property
    def is_complete(self) -> bool:
        return not self.remaining_ids and self.remaining_unidentified <= 0


@dataclass
class _SnapshotEffects:
    """Side-effect intents captured while snapshotting a confirmed send."""

    flush_before_audio: bool = False
    flush_after_audio: bool = False
    mark_new_user_utterance: bool = False
    start_user_speech: bool = False
    stop_user_speech: bool = False


@dataclass(eq=False)
class _StagedSend:
    """Represents an in-flight WebSocket send awaiting transport confirmation."""

    commit_fn: Callable[[], None]
    context: contextvars.Context
    completes_turn: bool = False
    completes_turn_if_nonempty: bool = False
    has_input_payload: bool = False
    omit_tool_response: bool = False
    is_tool_response: bool = False
    tool_response_ids: tuple[str, ...] = ()
    tool_response_continuing_ids: tuple[str, ...] = ()
    tool_response_unidentified: int = 0
    tool_response_has_non_silent: bool = False
    tool_response_has_continuing_non_silent: bool = False
    consumed_tool_entries: list[
        tuple[_PendingToolCallBatch, list[str], int]
    ] = field(default_factory=list)
    assigned_tool_batches: list[_PendingToolCallBatch] = field(
        default_factory=list
    )
    reassigned_tool_batches: list[
        tuple[_PendingToolCallBatch, _OutboundTurn]
    ] = field(default_factory=list)
    pending_merge_turns: list[_OutboundTurn] = field(default_factory=list)
    popped_tool_batch: bool = False
    updates_system_instruction: bool = False
    updates_tool_definitions: bool = False
    dependent_gens: list[_ActiveGeneration] = field(default_factory=list)
    snapshot_turn: _OutboundTurn | None = None
    snapshot_effects: _SnapshotEffects | None = None
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
    pending_audio_seq: int = 0
    input_entries: list[tuple[int, InputMessage]] = field(default_factory=list)
    input_transcript_deltas: list[str] = field(default_factory=list)
    accepting_input_transcription: bool = True
    capture_content: bool = False
    media_bytes: int = 0
    system_instruction: list[SystemInstructionPart] | None = None
    system_instruction_rev: int = 0
    tool_definitions: list[ToolDefinition] | None = None
    tool_definitions_rev: int = 0
    staged_sends: deque[_StagedSend] = field(default_factory=deque)
    sealed: bool = False
    retired: bool = False
    has_confirmed_turn_complete: bool = False
    has_confirmed_input_payload: bool = False
    has_confirmed_non_silent_tool_response: bool = False
    pending_merge_count: int = 0
    tool_batches: list[_PendingToolCallBatch] = field(default_factory=list)
    bound_gen: _ActiveGeneration | None = None

    def add_input_entry(self, seq: int, msg: InputMessage) -> None:
        if not self.input_entries or seq >= self.input_entries[-1][0]:
            self.input_entries.append((seq, msg))
            return
        idx = len(self.input_entries)
        while idx > 0 and self.input_entries[idx - 1][0] > seq:
            idx -= 1
        self.input_entries.insert(idx, (seq, msg))


def _has_incomplete_tool_batch(turn: _OutboundTurn) -> bool:
    return bool(turn.tool_batches) and any(
        not batch.is_complete for batch in turn.tool_batches
    )


def _classify_legacy_send(
    input_val: object,
    end_of_turn: bool,
    *,
    omit_tool_response: bool,
) -> tuple[
    bool, bool, bool, tuple[str, ...], tuple[str, ...], int, bool, bool
]:
    """Return ``(completes_turn, omit_from_queue, is_tool_response, tool_ids, continuing_ids, unidentified, has_non_silent, has_continuing_non_silent)``."""
    if not input_val:
        # AsyncSession._parse_client_message treats any falsy input as
        # client_content(turn_complete=True).
        return True, False, False, (), (), 0, False, False
    if isinstance(input_val, genai_types.LiveClientRealtimeInput) or (
        isinstance(input_val, Mapping)
        and ("media_chunks" in input_val or "mediaChunks" in input_val)
    ):
        return False, False, False, (), (), 0, False, False
    if isinstance(input_val, genai_types.Blob) or (
        isinstance(input_val, Mapping) and "data" in input_val
    ):
        return False, False, False, (), (), 0, False, False
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
        (
            ids,
            continuing_ids,
            unidentified,
            has_non_silent,
            has_continuing_non_silent,
        ) = _extract_tool_response_metadata(fn_resps)
        return (
            (not omit_tool_response and has_non_silent),
            omit_tool_response,
            True,
            ids,
            continuing_ids,
            unidentified,
            has_non_silent,
            has_continuing_non_silent,
        )
    if isinstance(input_val, genai_types.FunctionResponse) or (
        isinstance(input_val, Mapping)
        and "name" in input_val
        and "response" in input_val
    ):
        (
            ids,
            continuing_ids,
            unidentified,
            has_non_silent,
            has_continuing_non_silent,
        ) = _extract_tool_response_metadata(
            cast(FunctionResponseOrDict, input_val)
        )
        return (
            (not omit_tool_response and has_non_silent),
            omit_tool_response,
            True,
            ids,
            continuing_ids,
            unidentified,
            has_non_silent,
            has_continuing_non_silent,
        )
    if isinstance(input_val, genai_types.LiveClientContent) or (
        isinstance(input_val, Mapping)
        and ("turns" in input_val or "content" in input_val)
    ):
        tc = (
            input_val.turn_complete
            if isinstance(input_val, genai_types.LiveClientContent)
            else input_val.get("turn_complete")
        )
        return bool(tc), False, False, (), (), 0, False, False
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
            (
                ids,
                continuing_ids,
                unidentified,
                has_non_silent,
                has_continuing_non_silent,
            ) = _extract_tool_response_metadata(
                cast(Sequence[FunctionResponseOrDict], seq_val)
            )
            return (
                (not omit_tool_response and has_non_silent),
                omit_tool_response,
                True,
                ids,
                continuing_ids,
                unidentified,
                has_non_silent,
                has_continuing_non_silent,
            )
        if seq_val and any(
            isinstance(item, genai_types.Blob)
            or (isinstance(item, Mapping) and "data" in item)
            for item in seq_val
        ):
            return False, False, False, (), (), 0, False, False
    return bool(end_of_turn), False, False, (), (), 0, False, False


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
    input_entries: list[tuple[int, InputMessage]] = field(default_factory=list)
    input_media_bytes: int = 0
    input_audio_chunks: deque[bytes] = field(default_factory=deque)
    input_audio_bytes: int = 0
    input_audio_mime_type: str | None = None
    input_audio_seq: int = 0
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
    usage_metadata: UsageMetadata | None = None
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

    @property
    def input_messages(self) -> list[InputMessage]:
        return [msg for _, msg in self.input_entries]

    def _insert_input_entry(self, seq: int, msg: InputMessage) -> None:
        if not self.input_entries or seq >= self.input_entries[-1][0]:
            self.input_entries.append((seq, msg))
            return
        idx = len(self.input_entries)
        while idx > 0 and self.input_entries[idx - 1][0] > seq:
            idx -= 1
        self.input_entries.insert(idx, (seq, msg))

    def _enforce_input_media_budget(self) -> None:
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        excess = self.input_media_bytes + self.input_audio_bytes - max_bytes
        if excess <= 0:
            return

        # Evict oldest buffered media from input_entries first so the most
        # recent media preceding/during this generation is retained.
        msg_idx = 0
        while excess > 0 and msg_idx < len(self.input_entries):
            _, msg = self.input_entries[msg_idx]
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
                self.input_entries.pop(msg_idx)

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
            self.input_audio_seq = 0

    def append_input_audio(
        self, chunk: bytes, mime_type: str, *, seq: int = 0
    ) -> None:
        if not chunk:
            return
        if (
            self.input_audio_chunks
            and self.input_audio_mime_type
            and mime_type != self.input_audio_mime_type
        ):
            self.flush_input_audio()
        if (
            not self.input_audio_chunks
            and self.input_entries
            and seq > 0
            and seq < self.input_entries[-1][0]
        ):
            max_bytes = _MAX_BUFFERED_MEDIA_BYTES
            trimmed = chunk[-max_bytes:] if len(chunk) > max_bytes else chunk
            self._insert_input_entry(
                seq,
                InputMessage(
                    role="user",
                    parts=[
                        BlobPart(
                            mime_type=mime_type,
                            modality=Modality.AUDIO,
                            content=trimmed,
                        )
                    ],
                ),
            )
            self.input_media_bytes += len(trimmed)
            self._enforce_input_media_budget()
            return
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        if not self.input_audio_chunks:
            self.input_audio_seq = seq
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
        audio_seq = self.input_audio_seq
        self.input_audio_chunks.clear()
        self.input_audio_bytes = 0
        mime_type = self.input_audio_mime_type or "audio/pcm"
        self.input_audio_mime_type = None
        self.input_audio_seq = 0
        if len(combined) > _MAX_BUFFERED_MEDIA_BYTES:
            combined = combined[-_MAX_BUFFERED_MEDIA_BYTES:]
        self._insert_input_entry(
            audio_seq,
            InputMessage(
                role="user",
                parts=[
                    BlobPart(
                        mime_type=mime_type,
                        modality=Modality.AUDIO,
                        content=combined,
                    )
                ],
            ),
        )
        self.input_media_bytes += len(combined)
        self._enforce_input_media_budget()

    def append_input_entries(
        self, entries: Sequence[tuple[int, InputMessage]]
    ) -> None:
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        for seq, msg in entries:
            if (
                msg.role == "user"
                and len(msg.parts) == 1
                and isinstance(part := msg.parts[0], BlobPart)
                and part.modality == Modality.AUDIO
                and isinstance(part.content, bytes)
            ):
                self.append_input_audio(
                    part.content, part.mime_type or "audio/pcm", seq=seq
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
                self._insert_input_entry(seq, msg)
                if added_bytes:
                    self.input_media_bytes += added_bytes
                    self._enforce_input_media_budget()

    def append_input_transcription(self, input_transcript: str) -> None:
        user_msg: InputMessage | None = None
        for _, msg in reversed(self.input_entries):
            if msg.role == "user":
                user_msg = msg
                break
        if user_msg is not None:
            user_msg.parts.append(TranscriptionPart(content=input_transcript))
        else:
            seq = (self.input_entries[-1][0] + 1) if self.input_entries else 0
            self.input_entries.append(
                (
                    seq,
                    InputMessage(
                        role="user",
                        parts=[TranscriptionPart(content=input_transcript)],
                    ),
                )
            )

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
        self._next_tool_batch_seq: int = 0
        self._pending_context_updates: list[_StagedSend] = []
        self._write_turn: _OutboundTurn | None = None
        self._next_send_seq: int = 0
        self._current_send_seq: int = 0
        self._pending_input_transcript_deltas: list[str] = []
        self._deferred_generations: list[_ActiveGeneration] = []
        self._capture_content_override: bool | None = None
        self._snapshot_effects: _SnapshotEffects | None = None
        self._active_gen: _ActiveGeneration | None = None
        self._speech_invocation: UserSpeechInvocation | None = None
        self._awaiting_post_tool_call_turn_complete: bool = False
        self._awaiting_post_interrupted_turn_complete: bool = False
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
            messages.extend(msg for _, msg in turn.input_entries)
        return messages

    def _is_blocked_tool_turn(self, turn: _OutboundTurn) -> bool:
        return not turn.sealed and (
            _has_incomplete_tool_batch(turn) or turn.pending_merge_count > 0
        )

    def _current_open_turn(self) -> _OutboundTurn:
        if (
            not self._outbound_turns
            or self._outbound_turns[-1].sealed
            or self._is_blocked_tool_turn(self._outbound_turns[-1])
        ):
            turn = _OutboundTurn()
            self._outbound_turns.append(turn)
            return turn
        return self._outbound_turns[-1]

    def _reinsert_pending_tool_batch(
        self, batch: _PendingToolCallBatch
    ) -> None:
        if batch in self._pending_tool_call_batches:
            return
        inserted = False
        ordered: deque[_PendingToolCallBatch] = deque()
        for existing in self._pending_tool_call_batches:
            if not inserted and batch.seq < existing.seq:
                ordered.append(batch)
                inserted = True
            ordered.append(existing)
        if not inserted:
            ordered.append(batch)
        self._pending_tool_call_batches = ordered

    def _order_sealed_turn(self, turn: _OutboundTurn) -> None:
        if not turn.sealed or turn not in self._outbound_turns:
            return
        turns = list(self._outbound_turns)
        idx = turns.index(turn)
        target = idx
        while target > 0 and self._is_blocked_tool_turn(turns[target - 1]):
            target -= 1
        if target != idx:
            turns.pop(idx)
            turns.insert(target, turn)
            self._outbound_turns = deque(turns)

    def _reorder_sealed_turns(self) -> None:
        for candidate in list(self._outbound_turns):
            if candidate.sealed:
                self._order_sealed_turn(candidate)

    def _target_transcript_turn(self) -> _OutboundTurn | None:
        for turn in self._outbound_turns:
            if turn.accepting_input_transcription and not turn.tool_batches:
                return turn
        for turn in self._outbound_turns:
            if turn.accepting_input_transcription:
                return turn
        return None

    def _merge_outbound_turn(
        self, target: _OutboundTurn, source: _OutboundTurn
    ) -> None:
        if target is source:
            return
        if target.bound_gen is not None:
            target_gen = target.bound_gen
            self._consume_turn(
                source,
                target_gen,
                target_gen.invocation.should_capture_content,
            )
            self._apply_context_update_to_generation(target_gen, source)
            if source.staged_sends:
                for staged in source.staged_sends:
                    staged.turn = target
                target.staged_sends = deque(
                    sorted(
                        [*target.staged_sends, *source.staged_sends],
                        key=lambda s: s.seq,
                    )
                )
            target.has_confirmed_turn_complete = (
                target.has_confirmed_turn_complete
                or source.has_confirmed_turn_complete
            )
            target.has_confirmed_input_payload = (
                target.has_confirmed_input_payload
                or source.has_confirmed_input_payload
            )
            target.has_confirmed_non_silent_tool_response = (
                target.has_confirmed_non_silent_tool_response
                or source.has_confirmed_non_silent_tool_response
            )
            for batch in source.tool_batches:
                batch.turn = target
                if batch not in target.tool_batches:
                    target.tool_batches.append(batch)
            source.staged_sends.clear()
            source.tool_batches.clear()
            if source in self._outbound_turns:
                self._outbound_turns.remove(source)
            return
        if target.retired:
            self._apply_session_context_update(source)
            if source.staged_sends:
                for staged in source.staged_sends:
                    staged.turn = target
                target.staged_sends = deque(
                    sorted(
                        [*target.staged_sends, *source.staged_sends],
                        key=lambda s: s.seq,
                    )
                )
            source.input_entries.clear()
            source.pending_audio_chunks.clear()
            source.pending_audio_bytes = 0
            source.pending_audio_mime_type = None
            source.pending_audio_seq = 0
            source.media_bytes = 0
            source.input_transcript_deltas.clear()
            source.staged_sends.clear()
            source.tool_batches.clear()
            if source in self._outbound_turns:
                self._outbound_turns.remove(source)
            return
        has_interleaved_audio = bool(
            (target.pending_audio_chunks and source.input_entries)
            or (source.pending_audio_chunks and target.input_entries)
            or (
                target.pending_audio_chunks
                and source.pending_audio_chunks
                and target.pending_audio_mime_type
                and source.pending_audio_mime_type
                and source.pending_audio_mime_type
                != target.pending_audio_mime_type
            )
        )
        if has_interleaved_audio:
            if target.pending_audio_chunks:
                self._flush_pending_audio(turn=target, capture_content=True)
            if source.pending_audio_chunks:
                self._flush_pending_audio(turn=source, capture_content=True)
        if source.input_entries:
            target.input_entries = sorted(
                [*target.input_entries, *source.input_entries],
                key=lambda item: item[0],
            )
            target.media_bytes += source.media_bytes
        if source.pending_audio_chunks:
            if not target.pending_audio_chunks or (
                source.pending_audio_seq > 0
                and (
                    target.pending_audio_seq == 0
                    or source.pending_audio_seq < target.pending_audio_seq
                )
            ):
                if (
                    target.pending_audio_chunks
                    and source.pending_audio_seq < target.pending_audio_seq
                ):
                    target.pending_audio_chunks = deque(
                        [
                            *source.pending_audio_chunks,
                            *target.pending_audio_chunks,
                        ]
                    )
                else:
                    target.pending_audio_chunks.extend(
                        source.pending_audio_chunks
                    )
                target.pending_audio_seq = source.pending_audio_seq
            else:
                target.pending_audio_chunks.extend(source.pending_audio_chunks)
            target.pending_audio_bytes += source.pending_audio_bytes
            if not target.pending_audio_mime_type:
                target.pending_audio_mime_type = source.pending_audio_mime_type
        if source.input_transcript_deltas:
            target.input_transcript_deltas.extend(
                source.input_transcript_deltas
            )
        target.accepting_input_transcription = (
            target.accepting_input_transcription
            or source.accepting_input_transcription
        )
        target.capture_content = (
            target.capture_content or source.capture_content
        )
        if (
            source.system_instruction is not None
            and source.system_instruction_rev >= target.system_instruction_rev
        ):
            target.system_instruction = list(source.system_instruction)
            target.system_instruction_rev = source.system_instruction_rev
        if (
            source.tool_definitions is not None
            and source.tool_definitions_rev >= target.tool_definitions_rev
        ):
            target.tool_definitions = list(source.tool_definitions)
            target.tool_definitions_rev = source.tool_definitions_rev
        if source.staged_sends:
            for staged in source.staged_sends:
                staged.turn = target
            target.staged_sends = deque(
                sorted(
                    [*target.staged_sends, *source.staged_sends],
                    key=lambda s: s.seq,
                )
            )
        target.has_confirmed_turn_complete = (
            target.has_confirmed_turn_complete
            or source.has_confirmed_turn_complete
        )
        target.has_confirmed_input_payload = (
            target.has_confirmed_input_payload
            or source.has_confirmed_input_payload
        )
        target.has_confirmed_non_silent_tool_response = (
            target.has_confirmed_non_silent_tool_response
            or source.has_confirmed_non_silent_tool_response
        )
        for batch in source.tool_batches:
            batch.turn = target
            if batch not in target.tool_batches:
                target.tool_batches.append(batch)
        source.input_entries.clear()
        source.pending_audio_chunks.clear()
        source.pending_audio_bytes = 0
        source.pending_audio_mime_type = None
        source.pending_audio_seq = 0
        source.media_bytes = 0
        source.input_transcript_deltas.clear()
        source.staged_sends.clear()
        source.tool_batches.clear()
        if source in self._outbound_turns:
            self._outbound_turns.remove(source)
        self._enforce_pending_media_budget()

    def _target_turn(self) -> _OutboundTurn:
        if self._write_turn is not None:
            return self._write_turn
        return self._current_open_turn()

    def _ensure_config_content_extracted(self) -> None:
        if self._config_content_extracted:
            return
        self._config_content_extracted = True
        sys_inst = _get_live_config_field(self.config, "system_instruction")
        if sys_inst is not None and self._system_instruction_rev == 0:
            self.system_instruction = _extract_system_instruction(
                cast(ContentUnion | ContentUnionDict, sys_inst)
            )
        tools = _get_live_config_field(self.config, "tools")
        if tools is not None and self._tool_definitions_rev == 0:
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
        matched_batches: list[_PendingToolCallBatch] = []
        if staged.is_tool_response and self._pending_tool_call_batches:
            consumed_by_batch: dict[
                _PendingToolCallBatch, tuple[list[str], int]
            ] = {}

            def _record_matched(batch: _PendingToolCallBatch) -> None:
                if batch not in matched_batches:
                    matched_batches.append(batch)

            def _record_consumed(
                batch: _PendingToolCallBatch,
                call_id: str | None = None,
                unidentified: int = 0,
            ) -> None:
                _record_matched(batch)
                ids_list, unid_count = consumed_by_batch.get(batch, ([], 0))
                if call_id is not None:
                    ids_list.append(call_id)
                consumed_by_batch[batch] = (
                    ids_list,
                    unid_count + unidentified,
                )

            for call_id in staged.tool_response_ids:
                target_batch: _PendingToolCallBatch | None = None
                for candidate in self._pending_tool_call_batches:
                    if call_id in candidate.remaining_ids:
                        target_batch = candidate
                        break
                if target_batch is not None:
                    target_batch.remaining_ids.remove(call_id)
                    _record_consumed(target_batch, call_id=call_id)
                else:
                    for candidate in self._pending_tool_call_batches:
                        if call_id in candidate.all_ids:
                            _record_matched(candidate)
                            break

            for call_id in staged.tool_response_continuing_ids:
                for candidate in self._pending_tool_call_batches:
                    if (
                        call_id in candidate.remaining_ids
                        or call_id in candidate.all_ids
                    ):
                        _record_matched(candidate)
                        break

            unidentified_left = staged.tool_response_unidentified
            if unidentified_left > 0:
                fallback_batches: list[_PendingToolCallBatch] = list(
                    matched_batches
                )
                for candidate in self._pending_tool_call_batches:
                    if candidate not in fallback_batches:
                        fallback_batches.append(candidate)
                for candidate in fallback_batches:
                    while (
                        unidentified_left > 0
                        and candidate.remaining_unidentified > 0
                    ):
                        candidate.remaining_unidentified -= 1
                        _record_consumed(candidate, unidentified=1)
                        unidentified_left -= 1
                    while unidentified_left > 0 and candidate.remaining_ids:
                        popped_id = next(iter(candidate.remaining_ids))
                        candidate.remaining_ids.remove(popped_id)
                        _record_consumed(candidate, call_id=popped_id)
                        unidentified_left -= 1
                    if unidentified_left <= 0:
                        break

            staged.consumed_tool_entries = [
                (batch, ids_list, unid_count)
                for batch, (ids_list, unid_count) in consumed_by_batch.items()
            ]
            for batch in matched_batches:
                if (
                    batch.is_complete
                    and batch in self._pending_tool_call_batches
                ):
                    self._pending_tool_call_batches.remove(batch)
                    staged.popped_tool_batch = True

        matched_turn_set = {
            batch.turn
            for batch in matched_batches
            if batch.turn is not None
            and not batch.turn.sealed
            and batch.turn.bound_gen is None
            and batch.turn.pending_merge_count == 0
        }
        existing_turns = [
            candidate_turn
            for candidate_turn in self._outbound_turns
            if candidate_turn in matched_turn_set
        ]
        if existing_turns:
            turn = existing_turns[0]
            for other_turn in existing_turns[1:]:
                staged.pending_merge_turns.append(other_turn)
                other_turn.pending_merge_count += 1
        else:
            turn = self._current_open_turn()
        for batch in matched_batches:
            if batch.turn is None:
                batch.turn = turn
                staged.assigned_tool_batches.append(batch)
                if batch not in turn.tool_batches:
                    turn.tool_batches.append(batch)
            elif batch.turn is turn:
                if batch not in turn.tool_batches:
                    turn.tool_batches.append(batch)
            elif batch.turn not in staged.pending_merge_turns:
                staged.reassigned_tool_batches.append((batch, batch.turn))
                batch.turn = turn
                if batch not in turn.tool_batches:
                    turn.tool_batches.append(batch)
        if staged.is_tool_response:
            if matched_batches:
                if staged.tool_response_has_continuing_non_silent:
                    staged.completes_turn = True
                elif staged.popped_tool_batch:
                    candidate_turns = [turn, *staged.pending_merge_turns]
                    turn_has_non_silent = (
                        staged.tool_response_has_non_silent
                        or any(
                            candidate.has_confirmed_non_silent_tool_response
                            or any(
                                other.tool_response_has_non_silent
                                and other.status != "aborted"
                                for other in candidate.staged_sends
                            )
                            for candidate in candidate_turns
                        )
                    )
                    staged.completes_turn = turn_has_non_silent
                else:
                    staged.completes_turn = False
            else:
                staged.completes_turn = staged.tool_response_has_non_silent
        if staged.completes_turn_if_nonempty and not staged.completes_turn:
            has_prior_input = (
                turn.has_confirmed_input_payload
                or bool(turn.input_entries)
                or bool(turn.pending_audio_chunks)
                or bool(turn.input_transcript_deltas)
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
            self._order_sealed_turn(turn)

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

    def _release_pending_merge_turns(
        self, staged: _StagedSend
    ) -> list[_OutboundTurn]:
        released: list[_OutboundTurn] = []
        if not staged.pending_merge_turns:
            return released
        for other_turn in staged.pending_merge_turns:
            if other_turn.pending_merge_count > 0:
                other_turn.pending_merge_count -= 1
            released.append(other_turn)
        staged.pending_merge_turns.clear()
        return released

    def _snapshot_staged_send(self, staged: _StagedSend) -> None:
        if staged.snapshot_turn is not None:
            return
        staged_capture = staged.context.run(
            self._telemetry_handler.should_capture_realtime_content
        )
        scratch_turn = _OutboundTurn(capture_content=staged_capture)
        effects = _SnapshotEffects()
        saved_write_turn = self._write_turn
        saved_send_seq = self._current_send_seq
        saved_active_gen = self._active_gen
        saved_override = self._capture_content_override
        saved_snapshot_effects = self._snapshot_effects
        self._write_turn = scratch_turn
        self._current_send_seq = staged.seq
        self._active_gen = None
        self._capture_content_override = staged_capture
        self._snapshot_effects = effects
        try:
            staged.context.run(staged.commit_fn)
        finally:
            self._write_turn = saved_write_turn
            self._current_send_seq = saved_send_seq
            self._active_gen = saved_active_gen
            self._capture_content_override = saved_override
            self._snapshot_effects = saved_snapshot_effects
        staged.snapshot_turn = scratch_turn
        staged.snapshot_effects = effects

    def _apply_snapshot_to_turn(
        self, turn: _OutboundTurn, staged: _StagedSend
    ) -> None:
        snap = staged.snapshot_turn
        effects = staged.snapshot_effects
        if snap is None or effects is None:
            self._snapshot_staged_send(staged)
            snap = staged.snapshot_turn
            effects = staged.snapshot_effects
        if snap is None or effects is None:
            self._release_pending_context_update(staged)
            return
        detached_turn = turn not in self._outbound_turns
        if not detached_turn:
            if effects.start_user_speech:
                self._start_user_speech()
            elif effects.mark_new_user_utterance:
                self._mark_new_user_utterance()
        if snap.capture_content:
            turn.capture_content = True
        if effects.flush_before_audio:
            self._flush_pending_audio(turn=turn)
        if snap.input_entries:
            self._flush_pending_audio(turn=turn)
            for seq, msg in snap.input_entries:
                turn.add_input_entry(seq, msg)
            turn.media_bytes += snap.media_bytes
        if snap.pending_audio_chunks:
            mime_type = snap.pending_audio_mime_type or "audio/pcm"
            if (
                turn.pending_audio_chunks
                and turn.pending_audio_mime_type
                and mime_type != turn.pending_audio_mime_type
            ):
                self._flush_pending_audio(turn=turn)
            if not turn.pending_audio_chunks:
                turn.pending_audio_seq = snap.pending_audio_seq or staged.seq
                turn.pending_audio_mime_type = mime_type
            turn.pending_audio_chunks.extend(snap.pending_audio_chunks)
            turn.pending_audio_bytes += snap.pending_audio_bytes
        if effects.flush_after_audio:
            self._flush_pending_audio(turn=turn)
        if (
            snap.system_instruction is not None
            and snap.system_instruction_rev >= turn.system_instruction_rev
        ):
            turn.system_instruction = list(snap.system_instruction)
            turn.system_instruction_rev = snap.system_instruction_rev
        if (
            snap.tool_definitions is not None
            and snap.tool_definitions_rev >= turn.tool_definitions_rev
        ):
            turn.tool_definitions = list(snap.tool_definitions)
            turn.tool_definitions_rev = snap.tool_definitions_rev
        if not detached_turn and effects.stop_user_speech:
            self._stop_user_speech()
        if detached_turn:
            self._apply_session_context_update(turn)
        self._release_pending_context_update(staged, source=snap)
        if not detached_turn:
            self._enforce_pending_media_budget()

    def _commit_to_bound_generation(
        self, staged: _StagedSend, target_gen: _ActiveGeneration
    ) -> None:
        snap = staged.snapshot_turn
        effects = staged.snapshot_effects
        if snap is None or effects is None:
            self._snapshot_staged_send(staged)
            snap = staged.snapshot_turn
            effects = staged.snapshot_effects
        if snap is None:
            self._release_pending_context_update(staged)
            return
        if effects is not None:
            if effects.start_user_speech:
                saved_active_gen = self._active_gen
                self._active_gen = None
                try:
                    self._start_user_speech()
                finally:
                    self._active_gen = saved_active_gen
            if effects.stop_user_speech:
                self._stop_user_speech()
        if target_gen.invocation.should_capture_content:
            if snap.input_entries:
                target_gen.append_input_entries(snap.input_entries)
            if snap.pending_audio_chunks:
                mime_type = snap.pending_audio_mime_type or "audio/pcm"
                audio_seq = snap.pending_audio_seq or staged.seq
                for chunk in snap.pending_audio_chunks:
                    target_gen.append_input_audio(
                        chunk, mime_type, seq=audio_seq
                    )
            self._apply_context_update_to_generation(target_gen, snap)
        self._apply_session_context_update(snap)
        self._release_pending_context_update(staged, source=snap)

    def _drain_turn(self, turn: _OutboundTurn) -> None:
        while turn.staged_sends and turn.staged_sends[0].status != "pending":
            staged = turn.staged_sends.popleft()
            if staged.status != "confirmed" or staged.applied:
                continue
            staged.applied = True
            if staged.has_input_payload:
                turn.has_confirmed_input_payload = True
            if staged.tool_response_has_non_silent:
                turn.has_confirmed_non_silent_tool_response = True
            if staged.completes_turn and (
                not staged.consumed_tool_entries
                or staged.tool_response_has_continuing_non_silent
            ):
                turn.has_confirmed_turn_complete = True
            if turn.bound_gen is not None:
                self._commit_to_bound_generation(staged, turn.bound_gen)
            elif turn.retired:
                snap = staged.snapshot_turn
                if snap is not None and (
                    staged.updates_system_instruction
                    or staged.updates_tool_definitions
                ):
                    self._apply_session_context_update(snap)
                    self._release_pending_context_update(staged, source=snap)
                else:
                    self._release_pending_context_update(staged)
            else:
                self._apply_snapshot_to_turn(turn, staged)

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
            and turn.pending_merge_count == 0
            and not turn.has_confirmed_input_payload
            and not turn.input_entries
            and not turn.pending_audio_chunks
            and not turn.input_transcript_deltas
            and turn.system_instruction is None
            and turn.tool_definitions is None
            and turn in self._outbound_turns
        ):
            for batch in turn.tool_batches:
                if batch.turn is turn:
                    batch.turn = None
            turn.tool_batches.clear()
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
                self._release_pending_merge_turns(staged)
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
            self._release_pending_merge_turns(staged)
            self._release_pending_context_update(staged)
            return
        if staged.turn is None:
            self.register_inflight_send(staged)
        self._snapshot_staged_send(staged)
        staged.status = "confirmed"
        turn = staged.turn
        if turn is not None:
            if staged.pending_merge_turns:
                merge_sources = self._release_pending_merge_turns(staged)
                for other_turn in merge_sources:
                    self._merge_outbound_turn(target=turn, source=other_turn)
            self._drain_turn(turn)

    def abort_inflight_send(self, staged: _StagedSend) -> None:
        if staged.status != "pending":
            return
        staged.status = "aborted"
        self._release_pending_context_update(staged)
        released_merge_turns = self._release_pending_merge_turns(staged)
        if self._session_finalized:
            return
        reopened_batches: list[_PendingToolCallBatch] = []
        if staged.consumed_tool_entries:
            for (
                batch,
                consumed_ids,
                consumed_unidentified,
            ) in staged.consumed_tool_entries:
                restored_ids = [
                    cid
                    for cid in consumed_ids
                    if cid not in batch.cancelled_ids
                ]
                batch.remaining_ids.update(restored_ids)
                batch.remaining_unidentified += consumed_unidentified
                if not batch.is_complete:
                    reopened_batches.append(batch)
                    self._reinsert_pending_tool_batch(batch)
            staged.consumed_tool_entries.clear()
            staged.popped_tool_batch = False
        turn = staged.turn
        if turn is None:
            return
        if staged.assigned_tool_batches:
            assigned = list(staged.assigned_tool_batches)
            staged.assigned_tool_batches.clear()
            for batch in assigned:
                referenced_on_turn = any(
                    other is not staged
                    and other.status != "aborted"
                    and any(
                        entry_batch is batch
                        for entry_batch, _, _ in other.consumed_tool_entries
                    )
                    for other in turn.staged_sends
                )
                if not referenced_on_turn and batch.turn is turn:
                    batch.turn = None
                    if batch in turn.tool_batches:
                        turn.tool_batches.remove(batch)
        if staged.reassigned_tool_batches:
            reassigned = list(staged.reassigned_tool_batches)
            staged.reassigned_tool_batches.clear()
            for batch, previous_turn in reassigned:
                referenced_on_turn = any(
                    other is not staged
                    and other.status != "aborted"
                    and any(
                        entry_batch is batch
                        for entry_batch, _, _ in other.consumed_tool_entries
                    )
                    for other in turn.staged_sends
                )
                if not referenced_on_turn and batch.turn is turn:
                    batch.turn = previous_turn
                    if batch in turn.tool_batches:
                        turn.tool_batches.remove(batch)
        if reopened_batches:
            reopened_set = set(reopened_batches)
            for other in turn.staged_sends:
                if (
                    any(
                        entry_batch in reopened_set
                        for entry_batch, _, _ in other.consumed_tool_entries
                    )
                    and not any(
                        entry_batch.is_complete
                        for entry_batch, _, _ in other.consumed_tool_entries
                    )
                    and not other.tool_response_has_continuing_non_silent
                ):
                    other.completes_turn = False
        for other_turn in released_merge_turns:
            if (
                other_turn in self._outbound_turns
                and not other_turn.sealed
                and other_turn.pending_merge_count == 0
                and other_turn.tool_batches
                and any(batch.is_complete for batch in other_turn.tool_batches)
            ):
                other_has_non_silent = (
                    other_turn.has_confirmed_non_silent_tool_response
                    or any(
                        other.tool_response_has_non_silent
                        and other.status != "aborted"
                        for other in other_turn.staged_sends
                    )
                )
                if other_has_non_silent:
                    other_turn.sealed = True
        if (
            staged.completes_turn
            or reopened_batches
            or released_merge_turns
            or bool(turn.tool_batches)
        ):
            batch_has_active_non_silent_response = (
                turn.has_confirmed_non_silent_tool_response
                or any(
                    other.tool_response_has_non_silent
                    and other.status != "aborted"
                    for other in turn.staged_sends
                )
            )
            batch_completed = bool(
                turn.tool_batches
                and any(batch.is_complete for batch in turn.tool_batches)
                and batch_has_active_non_silent_response
            )
            turn.sealed = (
                turn.has_confirmed_turn_complete
                or batch_completed
                or any(
                    other.completes_turn and other.status != "aborted"
                    for other in turn.staged_sends
                )
            )
            self._reorder_sealed_turns()
        self._drain_turn(turn)

    def _mark_new_user_utterance(self) -> None:
        if self._snapshot_effects is not None:
            self._snapshot_effects.mark_new_user_utterance = True
            return
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
            while excess > 0 and msg_idx < len(turn.input_entries):
                _, msg = turn.input_entries[msg_idx]
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
                    turn.input_entries.pop(msg_idx)

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
                turn.pending_audio_seq = 0

            if (
                not turn.input_entries
                and not turn.pending_audio_chunks
                and not turn.input_transcript_deltas
                and not turn.staged_sends
                and not turn.sealed
                and turn.pending_merge_count == 0
                and not turn.has_confirmed_input_payload
                and turn.system_instruction is None
                and turn.tool_definitions is None
            ):
                for batch in turn.tool_batches:
                    if batch.turn is turn:
                        batch.turn = None
                turn.tool_batches.clear()
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
        if self._snapshot_effects is not None and self._write_turn is not None:
            if turn is None or turn is self._write_turn:
                if (
                    not self._write_turn.input_entries
                    and not self._write_turn.pending_audio_chunks
                ):
                    self._snapshot_effects.flush_before_audio = True
                    return
                if turn is None:
                    self._snapshot_effects.flush_after_audio = True
                    return
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
            (turn.capture_content or self._should_capture_content())
            if capture_content is None
            else capture_content
        )
        if should_capture:
            combined = b"".join(turn.pending_audio_chunks)
            mime_type = turn.pending_audio_mime_type or "audio/pcm"
            if len(combined) > _MAX_BUFFERED_MEDIA_BYTES:
                combined = combined[-_MAX_BUFFERED_MEDIA_BYTES:]
            turn.add_input_entry(
                turn.pending_audio_seq,
                InputMessage(
                    role="user",
                    parts=[
                        BlobPart(
                            mime_type=mime_type,
                            modality=Modality.AUDIO,
                            content=combined,
                        )
                    ],
                ),
            )
            turn.media_bytes += len(combined)
        turn.pending_audio_chunks.clear()
        turn.pending_audio_bytes = 0
        turn.pending_audio_mime_type = None
        turn.pending_audio_seq = 0
        self._enforce_pending_media_budget()

    def _start_user_speech(self) -> None:
        if self._snapshot_effects is not None:
            self._snapshot_effects.start_user_speech = True
            self._snapshot_effects.mark_new_user_utterance = True
            return
        if self._session_finalized:
            return
        self._mark_new_user_utterance()
        if self._speech_invocation is not None:
            return
        self._speech_invocation = self._telemetry_handler.user_speech(
            provider=self.provider,
            realtime_session_id=self.realtime_session_id,
            error_type_resolver=resolve_error_type,
            context=get_current(),
            _attach_to_context=False,
        )

    def _stop_user_speech(self, error: BaseException | None = None) -> None:
        if self._snapshot_effects is not None and error is None:
            self._snapshot_effects.stop_user_speech = True
            return
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
            if not turn.pending_audio_chunks:
                turn.pending_audio_seq = self._current_send_seq
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
        turn.add_input_entry(
            self._current_send_seq,
            InputMessage(
                role="user",
                parts=[
                    BlobPart(
                        mime_type=mime_type,
                        modality=modality,
                        content=data_bytes,
                    )
                ],
            ),
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
                self._target_turn().add_input_entry(
                    self._current_send_seq,
                    InputMessage(
                        role="user",
                        parts=[TextPart(content=str(text))],
                    ),
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
                    for msg in messages:
                        turn.add_input_entry(self._current_send_seq, msg)
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
            turn = self._target_turn()
            turn.add_input_entry(self._current_send_seq, tool_msg)
            self._track_input_message_media(turn, [tool_msg])

    def on_send_context_update(
        self,
        system_instruction: ContentUnion | ContentUnionDict | None = None,
        tools: ToolListUnion | ToolListUnionDict | None = None,
    ) -> None:
        if system_instruction is None and tools is None:
            return
        capture_content = self._should_capture_content()
        if capture_content:
            self._ensure_config_content_extracted()
        turn = self._target_turn()
        rev = self._current_send_seq
        if rev == 0:
            self._next_send_seq += 1
            rev = self._next_send_seq
        if system_instruction is not None:
            turn.system_instruction = (
                _extract_system_instruction(system_instruction)
                if capture_content
                else []
            )
            turn.system_instruction_rev = rev
        if tools is not None:
            turn.tool_definitions = (
                _extract_tool_definitions(tools) if capture_content else []
            )
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
            tc = (
                input_val.turn_complete
                if isinstance(input_val, genai_types.LiveClientContent)
                else input_val.get("turn_complete")
            )
            self.on_send_client_content(
                cast(ContentListUnion | ContentListUnionDict | None, turns),
                turn_complete=bool(tc),
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
        self._apply_session_context_update(turn)
        if capture_content:
            if turn.input_entries:
                active_gen.append_input_entries(turn.input_entries)
            if turn.pending_audio_chunks:
                mime_type = turn.pending_audio_mime_type or "audio/pcm"
                for chunk in turn.pending_audio_chunks:
                    active_gen.append_input_audio(
                        chunk, mime_type, seq=turn.pending_audio_seq
                    )
            if turn.input_transcript_deltas:
                active_gen.input_transcript_deltas.extend(
                    turn.input_transcript_deltas
                )
        turn.input_entries.clear()
        turn.pending_audio_chunks.clear()
        turn.pending_audio_bytes = 0
        turn.pending_audio_mime_type = None
        turn.pending_audio_seq = 0
        turn.media_bytes = 0
        turn.input_transcript_deltas.clear()

    def _claim_outbound_inputs(
        self, active_gen: _ActiveGeneration, capture_content: bool
    ) -> _OutboundTurn | None:
        bound_turn: _OutboundTurn | None = None
        claimed_any_input = False

        idx = 0
        while idx < len(self._outbound_turns):
            head = self._outbound_turns[idx]
            if self._is_blocked_tool_turn(head):
                idx += 1
                continue
            if head.staged_sends:
                if claimed_any_input:
                    # Do not bind a later turn's unconfirmed sends to a
                    # generation that already claimed prior input.
                    break
                turn = self._outbound_turns[idx]
                del self._outbound_turns[idx]
                self._consume_turn(turn, active_gen, capture_content)
                bound_turn = turn
                break
            turn = self._outbound_turns[idx]
            del self._outbound_turns[idx]
            if (
                turn.has_confirmed_input_payload
                or turn.input_entries
                or turn.pending_audio_chunks
                or turn.input_transcript_deltas
            ):
                claimed_any_input = True
            self._consume_turn(turn, active_gen, capture_content)
            if turn.sealed:
                break

        return bound_turn

    def _retire_no_output_turn(self) -> None:
        retired_any_input = False
        idx = 0
        while idx < len(self._outbound_turns):
            head = self._outbound_turns[idx]
            if self._is_blocked_tool_turn(head):
                idx += 1
                continue
            if head.staged_sends and retired_any_input:
                break
            turn = self._outbound_turns[idx]
            del self._outbound_turns[idx]
            turn.retired = True
            self._apply_session_context_update(turn)
            if (
                turn.input_entries
                or turn.pending_audio_chunks
                or turn.input_transcript_deltas
                or turn.has_confirmed_input_payload
                or any(
                    staged.has_input_payload and staged.status != "aborted"
                    for staged in turn.staged_sends
                )
            ):
                retired_any_input = True
            turn.input_entries.clear()
            turn.pending_audio_chunks.clear()
            turn.pending_audio_bytes = 0
            turn.pending_audio_mime_type = None
            turn.pending_audio_seq = 0
            turn.media_bytes = 0
            turn.input_transcript_deltas.clear()
            if turn.staged_sends or turn.sealed:
                break
        if not self._outbound_turns:
            self._pending_input_transcript_deltas.clear()

    def ensure_active_generation(self) -> _ActiveGeneration:
        if self._active_gen is None:
            self._awaiting_post_tool_call_turn_complete = False
            self._awaiting_post_interrupted_turn_complete = False
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
            active_gen.append_input_transcription(input_transcript)

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
            if usage.prompt_token_count is not None:
                invocation.input_tokens = usage.prompt_token_count

            thoughts_tokens = usage.thoughts_token_count
            if thoughts_tokens is not None:
                invocation.thinking_tokens = thoughts_tokens

            if usage.response_token_count is not None:
                # On Vertex Live, google-genai maps wire `candidatesTokenCount`
                # (which excludes reasoning tokens) to `response_token_count`.
                invocation.output_tokens = (
                    usage.response_token_count + (thoughts_tokens or 0)
                    if self.is_vertex
                    else usage.response_token_count
                )
            elif thoughts_tokens is not None:
                invocation.output_tokens = thoughts_tokens

            if usage.cached_content_token_count is not None:
                invocation.cache_read_input_tokens = (
                    usage.cached_content_token_count
                )

            if usage.prompt_tokens_details is not None:
                invocation.set_input_tokens(
                    (detail.modality, detail.token_count)
                    for detail in usage.prompt_tokens_details
                )

            if usage.response_tokens_details is not None:
                invocation.set_output_tokens(
                    (detail.modality, detail.token_count)
                    for detail in usage.response_tokens_details
                )

            if usage.cache_tokens_details is not None:
                invocation.set_cache_read_input_tokens(
                    (detail.modality, detail.token_count)
                    for detail in usage.cache_tokens_details
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
        server_content = message.server_content
        tool_call = message.tool_call
        tool_call_cancellation = message.tool_call_cancellation
        usage_metadata = message.usage_metadata

        if tool_call_cancellation is not None:
            cancelled_ids = tool_call_cancellation.ids or ()
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
                    for batch in turn.tool_batches:
                        _add_batch(batch)
                    for staged in turn.staged_sends:
                        for batch, _, _ in staged.consumed_tool_entries:
                            _add_batch(batch)

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
                        and batch_turn.pending_merge_count == 0
                    ):
                        batch_has_active_non_silent_response = (
                            batch_turn.has_confirmed_non_silent_tool_response
                            or any(
                                other.tool_response_has_non_silent
                                and other.status != "aborted"
                                for other in batch_turn.staged_sends
                            )
                        )
                        if batch_has_active_non_silent_response:
                            batch_turn.sealed = True
                            self._order_sealed_turn(batch_turn)

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
                if self._active_gen is None:
                    for turn in self._outbound_turns:
                        if turn.sealed and turn.input_transcript_deltas:
                            turn.accepting_input_transcription = False
                self._start_user_speech()
            elif "END" in vad_str or "EOS" in vad_str:
                if (
                    self._outbound_turns
                    and not self._outbound_turns[-1].sealed
                    and not self._is_blocked_tool_turn(
                        self._outbound_turns[-1]
                    )
                ):
                    open_turn = self._outbound_turns[-1]
                    self._flush_pending_audio(turn=open_turn)
                    if (
                        open_turn.has_confirmed_input_payload
                        or open_turn.input_entries
                        or open_turn.pending_audio_chunks
                        or open_turn.input_transcript_deltas
                        or any(
                            other.has_input_payload
                            and other.status != "aborted"
                            for other in open_turn.staged_sends
                        )
                    ):
                        open_turn.sealed = True
                        self._order_sealed_turn(open_turn)
                self._stop_user_speech()

        is_interrupted = bool(
            server_content is not None and server_content.interrupted
        )
        is_generation_complete = bool(
            server_content is not None and server_content.generation_complete
        )
        if is_interrupted:
            self._mark_new_user_utterance()

        # Accumulate input transcription deltas without starting the generation span.
        if server_content is not None:
            input_tx = server_content.input_transcription
            if input_tx is not None and input_tx.text:
                if (
                    self._active_gen is not None
                    and self._active_gen.accepting_input_transcription
                ):
                    if self._active_gen.invocation.should_capture_content:
                        self._active_gen.input_transcript_deltas.append(
                            str(input_tx.text)
                        )
                elif (
                    target_turn := self._target_transcript_turn()
                ) is not None:
                    if (
                        target_turn.capture_content
                        or self._telemetry_handler.should_capture_realtime_content()
                    ):
                        target_turn.input_transcript_deltas.append(
                            str(input_tx.text)
                        )
                elif self._telemetry_handler.should_capture_realtime_content():
                    self._pending_input_transcript_deltas.append(
                        str(input_tx.text)
                    )

        provider_reason = _turn_complete_reason(server_content)
        interaction_complete = _is_interaction_complete(server_content)
        has_model_output = bool(
            server_content is not None
            and (
                (
                    server_content.output_transcription is not None
                    and server_content.output_transcription.text
                )
                or (
                    server_content.model_turn is not None
                    and server_content.model_turn.parts
                )
            )
        )
        has_completion_signal = bool(
            server_content is not None
            and (
                is_interrupted
                or is_generation_complete
                or interaction_complete
                or provider_reason is not None
            )
        )
        has_tool_call = bool(
            tool_call is not None and tool_call.function_calls
        )

        if (
            self._active_gen is None
            and not has_model_output
            and not has_tool_call
        ):
            if (
                self._awaiting_post_tool_call_turn_complete
                and (interaction_complete or is_generation_complete)
                and not is_interrupted
            ):
                if interaction_complete:
                    self._awaiting_post_tool_call_turn_complete = False
                return
            if (
                self._awaiting_post_interrupted_turn_complete
                and interaction_complete
                and not is_interrupted
            ):
                self._awaiting_post_interrupted_turn_complete = False
                return
            if is_interrupted:
                self._awaiting_post_tool_call_turn_complete = False
                self._retire_no_output_turn()
                if not interaction_complete:
                    self._awaiting_post_interrupted_turn_complete = True
                return
            if interaction_complete and (
                provider_reason is not None or not is_generation_complete
            ):
                self._retire_no_output_turn()
            return

        active_gen = self.ensure_active_generation()
        capture_content = active_gen.invocation.should_capture_content

        if usage_metadata is not None:
            active_gen.usage_metadata = usage_metadata

        if server_content is not None:
            if provider_reason is not None:
                active_gen.turn_complete_reason = provider_reason

            if capture_content:
                output_tx = server_content.output_transcription
                if output_tx is not None and output_tx.text:
                    active_gen.output_transcript_deltas.append(
                        str(output_tx.text)
                    )

                model_turn = server_content.model_turn
                if model_turn is not None and model_turn.parts:
                    for part in model_turn.parts:
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

            if server_content.interrupted:
                active_gen.interrupted = True
            if server_content.generation_complete:
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
                    if fc.id is not None and str(fc.id):
                        batch_ids.add(str(fc.id))
                    else:
                        batch_unidentified += 1
                if batch_ids or batch_unidentified:
                    self._next_tool_batch_seq += 1
                    self._pending_tool_call_batches.append(
                        _PendingToolCallBatch(
                            seq=self._next_tool_batch_seq,
                            all_ids=set(batch_ids),
                            remaining_ids=batch_ids,
                            remaining_unidentified=batch_unidentified,
                        )
                    )
                if capture_content:
                    tool_parts: list[MessagePart] = [
                        ToolCallRequestPart(
                            id=fc.id,
                            name=fc.name or "",
                            arguments=dict(fc.args)
                            if fc.args is not None
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
                self._awaiting_post_tool_call_turn_complete = (
                    not interaction_complete
                )
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
    is_tool_response: bool = False,
    tool_response_ids: tuple[str, ...] = (),
    tool_response_continuing_ids: tuple[str, ...] = (),
    tool_response_unidentified: int = 0,
    tool_response_has_non_silent: bool = False,
    tool_response_has_continuing_non_silent: bool = False,
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
        existing_staged.is_tool_response = is_tool_response
        existing_staged.tool_response_ids = tool_response_ids
        existing_staged.tool_response_continuing_ids = (
            tool_response_continuing_ids
        )
        existing_staged.tool_response_unidentified = tool_response_unidentified
        existing_staged.tool_response_has_non_silent = (
            tool_response_has_non_silent
        )
        existing_staged.tool_response_has_continuing_non_silent = (
            tool_response_has_continuing_non_silent
        )
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
        is_tool_response=is_tool_response,
        tool_response_ids=tool_response_ids,
        tool_response_continuing_ids=tool_response_continuing_ids,
        tool_response_unidentified=tool_response_unidentified,
        tool_response_has_non_silent=tool_response_has_non_silent,
        tool_response_has_continuing_non_silent=tool_response_has_continuing_non_silent,
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
        await _run_transactional_send(
            tracker,
            lambda: tracker.on_send_realtime_input(kwargs),
            lambda: orig_send_realtime_input(*args, **kwargs),  # pyright: ignore[reportArgumentType]
            completes_turn=completes_turn,
            completes_turn_if_nonempty=completes_turn_if_nonempty,
            has_input_payload=has_input_payload,
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
        (
            tool_ids,
            tool_continuing_ids,
            tool_unidentified,
            tool_has_non_silent,
            tool_has_continuing_non_silent,
        ) = _extract_tool_response_metadata(function_responses)
        await _run_transactional_send(
            tracker,
            lambda: tracker.on_send_tool_response(
                function_responses, omit_tool_response=omit_tool_response
            ),
            lambda: orig_send_tool_response(*args, **kwargs),  # pyright: ignore[reportArgumentType]
            completes_turn=not omit_tool_response and tool_has_non_silent,
            has_input_payload=not omit_tool_response,
            omit_tool_response=omit_tool_response,
            is_tool_response=True,
            tool_response_ids=tool_ids,
            tool_response_continuing_ids=tool_continuing_ids,
            tool_response_unidentified=tool_unidentified,
            tool_response_has_non_silent=tool_has_non_silent,
            tool_response_has_continuing_non_silent=tool_has_continuing_non_silent,
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
                tool_continuing_ids,
                tool_unidentified,
                tool_has_non_silent,
                tool_has_continuing_non_silent,
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
                tool_response_continuing_ids=tool_continuing_ids,
                tool_response_unidentified=tool_unidentified,
                tool_response_has_non_silent=tool_has_non_silent,
                tool_response_has_continuing_non_silent=tool_has_continuing_non_silent,
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
