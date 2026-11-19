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


@dataclass(eq=False)
class _StagedSend:
    """Represents an in-flight WebSocket send awaiting transport confirmation."""

    commit_fn: Callable[[], None]
    context: contextvars.Context
    bound_gen: _ActiveGeneration | None = None
    settled: bool = False


def _merge_input_messages(
    target: list[InputMessage], committed: Sequence[InputMessage]
) -> None:
    """Merge newly confirmed input messages into ``target``, coalescing adjacent audio."""
    for msg in committed:
        if (
            target
            and target[-1].role == "user"
            and msg.role == "user"
            and len(target[-1].parts) == 1
            and len(msg.parts) == 1
            and isinstance(last_part := target[-1].parts[0], BlobPart)
            and isinstance(new_part := msg.parts[0], BlobPart)
            and last_part.modality == Modality.AUDIO
            and new_part.modality == Modality.AUDIO
            and last_part.mime_type == new_part.mime_type
            and isinstance(last_part.content, bytes)
            and isinstance(new_part.content, bytes)
        ):
            merged = last_part.content + new_part.content
            if len(merged) > _MAX_BUFFERED_MEDIA_BYTES:
                merged = merged[-_MAX_BUFFERED_MEDIA_BYTES:]
            last_part.content = merged
        else:
            target.append(msg)


@dataclass(eq=False)
class _ActiveGeneration:
    invocation: RealtimeInferenceInvocation
    input_messages: list[InputMessage] = field(default_factory=list)
    system_instruction: list[SystemInstructionPart] = field(
        default_factory=list
    )
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
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
    pending_sends: list[_StagedSend] = field(default_factory=list)
    deferred_finish_reason: str | None = None
    deferred_error: Error | BaseException | None = None

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

        self._pending_audio_chunks: deque[bytes] = deque()
        self._pending_audio_bytes: int = 0
        self._pending_audio_mime_type: str | None = None
        self._pending_input_messages: list[InputMessage] = []
        self._pending_media_bytes: int = 0
        self._pending_input_transcript_deltas: list[str] = []
        self._inflight_sends: list[_StagedSend] = []
        self._deferred_generations: list[_ActiveGeneration] = []
        self._capture_content_override: bool | None = None
        self._active_gen: _ActiveGeneration | None = None
        self._speech_invocation: UserSpeechInvocation | None = None
        self.terminal_error: BaseException | None = None

        self.system_instruction: list[SystemInstructionPart] = []
        self.tool_definitions: list[ToolDefinition] = []
        self._config_content_extracted: bool = False
        if self._should_capture_content():
            self._ensure_config_content_extracted()

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
        if staged.settled or staged in self._inflight_sends:
            return
        self._inflight_sends.append(staged)

    def _commit_to_bound_generation(
        self, staged: _StagedSend, target_gen: _ActiveGeneration
    ) -> None:
        # Execute the confirmed send callback against isolated pending buffers
        # so subsequent turns' pending state is unaffected, then attach any
        # resulting inputs/context updates to ``target_gen``.
        saved_chunks = self._pending_audio_chunks
        saved_audio_bytes = self._pending_audio_bytes
        saved_audio_mime = self._pending_audio_mime_type
        saved_messages = self._pending_input_messages
        saved_media_bytes = self._pending_media_bytes
        saved_sys = self.system_instruction
        saved_tools = self.tool_definitions
        saved_active_gen = self._active_gen
        saved_override = self._capture_content_override

        target_capture = target_gen.invocation.should_capture_content
        self._pending_audio_chunks = deque()
        self._pending_audio_bytes = 0
        self._pending_audio_mime_type = None
        self._pending_input_messages = []
        self._pending_media_bytes = 0
        self._active_gen = None
        self._capture_content_override = target_capture
        try:
            staged.context.run(staged.commit_fn)
            self._flush_pending_audio(capture_content=target_capture)
            committed_messages = self._pending_input_messages
        finally:
            self._pending_audio_chunks = saved_chunks
            self._pending_audio_bytes = saved_audio_bytes
            self._pending_audio_mime_type = saved_audio_mime
            self._pending_input_messages = saved_messages
            self._pending_media_bytes = saved_media_bytes
            self._active_gen = saved_active_gen
            self._capture_content_override = saved_override

        if committed_messages:
            _merge_input_messages(
                target_gen.input_messages, committed_messages
            )
        if target_capture:
            if self.system_instruction is not saved_sys:
                target_gen.system_instruction = list(self.system_instruction)
            if self.tool_definitions is not saved_tools:
                target_gen.tool_definitions = list(self.tool_definitions)

    def _detach_pending_sends(self, active_gen: _ActiveGeneration) -> None:
        for staged in active_gen.pending_sends:
            if staged.bound_gen is active_gen:
                staged.bound_gen = None
        active_gen.pending_sends.clear()

    def _maybe_finish_deferred_generation(
        self, active_gen: _ActiveGeneration
    ) -> None:
        if (
            active_gen.deferred_finish_reason is not None
            and not active_gen.pending_sends
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
        if active_gen.pending_sends and error is None:
            active_gen.invocation.mark_end()
            active_gen.deferred_finish_reason = finish_reason
            active_gen.deferred_error = None
            if active_gen not in self._deferred_generations:
                self._deferred_generations.append(active_gen)
            return
        self._detach_pending_sends(active_gen)
        self._populate_and_finish_generation(
            active_gen, finish_reason=finish_reason, error=error
        )

    def _flush_deferred_generations(self) -> None:
        while self._deferred_generations:
            gen = self._deferred_generations.pop(0)
            self._detach_pending_sends(gen)
            finish_reason = gen.deferred_finish_reason or "stop"
            error = gen.deferred_error
            gen.deferred_finish_reason = None
            gen.deferred_error = None
            self._populate_and_finish_generation(
                gen, finish_reason=finish_reason, error=error
            )

    def confirm_inflight_send(self, staged: _StagedSend) -> None:
        if staged.settled:
            return
        staged.settled = True
        if staged in self._inflight_sends:
            self._inflight_sends.remove(staged)
        target_gen = staged.bound_gen
        staged.bound_gen = None
        if target_gen is None:
            staged.context.run(staged.commit_fn)
            return
        if staged in target_gen.pending_sends:
            target_gen.pending_sends.remove(staged)
        self._commit_to_bound_generation(staged, target_gen)
        self._maybe_finish_deferred_generation(target_gen)

    def abort_inflight_send(self, staged: _StagedSend) -> None:
        if staged.settled:
            return
        staged.settled = True
        if staged in self._inflight_sends:
            self._inflight_sends.remove(staged)
        target_gen = staged.bound_gen
        staged.bound_gen = None
        if target_gen is not None:
            if staged in target_gen.pending_sends:
                target_gen.pending_sends.remove(staged)
            self._maybe_finish_deferred_generation(target_gen)

    def _mark_new_user_utterance(self) -> None:
        if self._active_gen is not None:
            self._active_gen.accepting_input_transcription = False

    def _enforce_pending_media_budget(self) -> None:
        max_bytes = _MAX_BUFFERED_MEDIA_BYTES
        excess = (
            self._pending_media_bytes + self._pending_audio_bytes - max_bytes
        )
        if excess <= 0:
            return

        # Evict oldest buffered media from pending input messages first so the
        # most recent audio/media preceding the next generation is preserved.
        idx = 0
        while excess > 0 and idx < len(self._pending_input_messages):
            msg = self._pending_input_messages[idx]
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
                        self._pending_media_bytes -= excess
                        excess = 0
                        new_parts.append(part)
                    else:
                        self._pending_media_bytes -= part_len
                        excess -= part_len
                else:
                    new_parts.append(part)
            if new_parts:
                msg.parts = new_parts
                idx += 1
            else:
                self._pending_input_messages.pop(idx)

        # If live streaming audio chunks alone still exceed the budget, evict
        # oldest chunks from the front of the deque.
        while excess > 0 and self._pending_audio_chunks:
            oldest = self._pending_audio_chunks[0]
            if len(oldest) <= excess:
                self._pending_audio_chunks.popleft()
                self._pending_audio_bytes -= len(oldest)
                excess -= len(oldest)
            else:
                self._pending_audio_chunks[0] = oldest[excess:]
                self._pending_audio_bytes -= excess
                excess = 0

        if not self._pending_audio_chunks:
            self._pending_audio_mime_type = None

    def _track_input_message_media(
        self, messages: Sequence[InputMessage]
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
            self._pending_media_bytes += added_bytes
            self._enforce_pending_media_budget()

    def _flush_pending_audio(
        self, *, capture_content: bool | None = None
    ) -> None:
        if not self._pending_audio_chunks:
            return
        should_capture = (
            self._should_capture_content()
            if capture_content is None
            else capture_content
        )
        if should_capture:
            combined = b"".join(self._pending_audio_chunks)
            mime_type = self._pending_audio_mime_type or "audio/pcm"
            if (
                self._pending_input_messages
                and self._pending_input_messages[-1].role == "user"
                and len(self._pending_input_messages[-1].parts) == 1
                and isinstance(
                    last_part := self._pending_input_messages[-1].parts[0],
                    BlobPart,
                )
                and last_part.modality == Modality.AUDIO
                and last_part.mime_type == mime_type
                and isinstance(last_part.content, bytes)
            ):
                merged = last_part.content + combined
                old_len = len(last_part.content)
                if len(merged) > _MAX_BUFFERED_MEDIA_BYTES:
                    merged = merged[-_MAX_BUFFERED_MEDIA_BYTES:]
                last_part.content = merged
                self._pending_media_bytes += len(merged) - old_len
            else:
                if len(combined) > _MAX_BUFFERED_MEDIA_BYTES:
                    combined = combined[-_MAX_BUFFERED_MEDIA_BYTES:]
                self._pending_input_messages.append(
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
                self._pending_media_bytes += len(combined)
        self._pending_audio_chunks.clear()
        self._pending_audio_bytes = 0
        self._pending_audio_mime_type = None
        self._enforce_pending_media_budget()

    def _start_user_speech(self) -> None:
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
        data_bytes, mime_type = extracted
        lower_mime = mime_type.lower()
        if not force_non_audio and "audio" in lower_mime:
            if not data_bytes:
                return
            if (
                self._pending_audio_chunks
                and self._pending_audio_mime_type
                and mime_type != self._pending_audio_mime_type
            ):
                self._flush_pending_audio()
            if len(data_bytes) >= _MAX_BUFFERED_MEDIA_BYTES:
                trimmed = data_bytes[-_MAX_BUFFERED_MEDIA_BYTES:]
                self._pending_audio_chunks.clear()
                self._pending_audio_chunks.append(trimmed)
                self._pending_audio_bytes = len(trimmed)
            else:
                self._pending_audio_chunks.append(data_bytes)
                self._pending_audio_bytes += len(data_bytes)
            if not self._pending_audio_mime_type:
                self._pending_audio_mime_type = mime_type
            self._enforce_pending_media_budget()
            return
        self._flush_pending_audio()
        if len(data_bytes) > _MAX_BUFFERED_MEDIA_BYTES:
            data_bytes = data_bytes[:_MAX_BUFFERED_MEDIA_BYTES]
        modality = (
            Modality.VIDEO
            if force_non_audio or "video" in lower_mime
            else Modality.IMAGE
            if "image" in lower_mime
            else Modality.DOCUMENT
        )
        self._pending_input_messages.append(
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
        self._pending_media_bytes += len(data_bytes)
        self._enforce_pending_media_budget()

    def on_send_realtime_input(self, kwargs: Mapping[str, object]) -> None:
        if kwargs.get("activity_start") is not None:
            self._start_user_speech()

        audio = kwargs.get("audio")
        media = kwargs.get("media")
        video = kwargs.get("video")
        text = kwargs.get("text")
        if (
            audio is not None
            or media is not None
            or video is not None
            or text is not None
        ):
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
                self._pending_input_messages.append(
                    InputMessage(
                        role="user",
                        parts=[TextPart(content=str(text))],
                    )
                )

        if kwargs.get("audio_stream_end"):
            self._flush_pending_audio()
        if kwargs.get("activity_end") is not None:
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
                    self._pending_input_messages.extend(messages)
                    self._track_input_message_media(messages)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

    def on_send_tool_response(
        self,
        function_responses: FunctionResponseOrDict
        | Sequence[FunctionResponseOrDict]
        | None,
    ) -> None:
        self._flush_pending_audio()
        if (
            self._active_gen is not None
            and self._active_gen.awaiting_tool_response
        ):
            return
        if not self._should_capture_content():
            return
        tool_msg = _build_tool_response_input_message(function_responses)
        if tool_msg is not None:
            self._pending_input_messages.append(tool_msg)

    def on_send_context_update(
        self,
        system_instruction: ContentUnion | ContentUnionDict | None = None,
        tools: ToolListUnion | ToolListUnionDict | None = None,
    ) -> None:
        if not self._should_capture_content():
            return
        self._ensure_config_content_extracted()
        if system_instruction is not None:
            self.system_instruction = _extract_system_instruction(
                system_instruction
            )
        if tools is not None:
            self.tool_definitions = _extract_tool_definitions(tools)

    def on_send(
        self,
        input_val: object = None,
        end_of_turn: bool = False,
    ) -> None:
        if input_val is None:
            self._flush_pending_audio()
            return

        if isinstance(input_val, genai_types.LiveClientRealtimeInput) or (
            isinstance(input_val, Mapping)
            and ("media_chunks" in input_val or "mediaChunks" in input_val)
        ):
            self._mark_new_user_utterance()
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
            self._mark_new_user_utterance()
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
            self.on_send_tool_response(fn_resps)
            return

        if isinstance(input_val, genai_types.FunctionResponse) or (
            isinstance(input_val, Mapping)
            and "name" in input_val
            and "response" in input_val
        ):
            self.on_send_tool_response(cast(FunctionResponseOrDict, input_val))
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
                    cast(Sequence[FunctionResponseOrDict], seq_val)
                )
                return
            if seq_val and any(
                isinstance(item, genai_types.Blob)
                or (isinstance(item, Mapping) and "data" in item)
                for item in seq_val
            ):
                self._mark_new_user_utterance()
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
            self._flush_pending_audio(capture_content=capture_content)
            if capture_content:
                self._ensure_config_content_extracted()
            invocation.output_type = self.output_type
            if self.max_tokens is not None:
                invocation.max_tokens = self.max_tokens
            if self.reasoning_level is not None:
                invocation.reasoning_level = self.reasoning_level

            input_msgs = list(self._pending_input_messages)
            self._pending_input_messages.clear()
            self._pending_media_bytes = 0
            input_transcript_deltas = list(
                self._pending_input_transcript_deltas
            )
            self._pending_input_transcript_deltas.clear()
            active_gen = _ActiveGeneration(
                invocation=invocation,
                input_messages=input_msgs,
                system_instruction=(
                    list(self.system_instruction) if capture_content else []
                ),
                tool_definitions=(
                    list(self.tool_definitions) if capture_content else []
                ),
                input_transcript_deltas=input_transcript_deltas,
            )
            if self._inflight_sends:
                pending = list(self._inflight_sends)
                self._inflight_sends.clear()
                for staged in pending:
                    staged.bound_gen = active_gen
                    active_gen.pending_sends.append(staged)
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
        usage_metadata = getattr(message, "usage_metadata", None)

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
                self._flush_pending_audio()
                self._stop_user_speech()

        capture_content = self._should_capture_content()

        # Accumulate input transcription deltas without starting the generation span.
        if server_content is not None and capture_content:
            input_tx = getattr(server_content, "input_transcription", None)
            if input_tx is not None and getattr(input_tx, "text", None):
                if (
                    self._active_gen is not None
                    and self._active_gen.accepting_input_transcription
                    and not self._pending_audio_chunks
                    and not self._pending_input_messages
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
            self._detach_pending_sends(active_gen)
            self._populate_and_finish_generation(
                active_gen, finish_reason="error", error=exc
            )

    def on_send_error(self, exc: BaseException) -> None:
        self._fail_active_operations(exc)

    def on_receive_error(self, exc: BaseException) -> None:
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
            self._detach_pending_sends(active_gen)
            finish_reason = active_gen.turn_complete_reason or (
                "interrupted"
                if active_gen.interrupted
                else "stop"
                if active_gen.generation_complete or active_gen.turn_complete
                else "error"
            )
            error = (
                Error(
                    message="Session closed before generation completed",
                    type=ErrorTypeValues.OTHER.value,
                )
                if finish_reason == "error"
                else None
            )
            self._populate_and_finish_generation(
                active_gen, finish_reason=finish_reason, error=error
            )


async def _run_transactional_send(
    tracker: _LiveSessionTracker,
    commit_fn: Callable[[], None],
    call_fn: Callable[[], object],
) -> object:
    existing_staged = _ACTIVE_SEND_CALLBACK.get()
    if existing_staged is not None:
        existing_staged.commit_fn = commit_fn
        res = call_fn()
        if inspect.isawaitable(res):
            return await res
        return res

    staged = _StagedSend(
        commit_fn=commit_fn,
        context=contextvars.copy_context(),
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
        await _run_transactional_send(
            tracker,
            lambda: tracker.on_send_realtime_input(kwargs),
            lambda: orig_send_realtime_input(*args, **kwargs),  # pyright: ignore[reportArgumentType]
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
        await _run_transactional_send(
            tracker,
            lambda: tracker.on_send_tool_response(function_responses),
            lambda: orig_send_tool_response(*args, **kwargs),  # pyright: ignore[reportArgumentType]
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
            return await _run_transactional_send(
                tracker,
                lambda: tracker.on_send(
                    input_val, end_of_turn=bool(end_of_turn)
                ),
                lambda: orig_send(*args, **kwargs),
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
            )

        setattr(session, "send_context_update", wrapped_send_context_update)


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
            tracker.on_session_error(exc)
        raise
    finally:
        if session_started and tracker is not None:
            if session_error is None:
                tracker.on_session_close()
            effective_error = (
                session_error
                if session_error is not None
                else tracker.terminal_error
            )
            telemetry_handler.emit_realtime_session_ended(
                provider,
                request_model=request_model_name,
                realtime_session_id=tracker.realtime_session_id,
                server_address=server_address,
                server_port=server_port,
                error=effective_error,
                error_type_resolver=resolve_error_type,
                context=caller_context,
            )


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
