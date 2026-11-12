# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import cast
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


def _strip_model_prefix(model: str | None) -> str | None:
    if not model:
        return None
    model_str = str(model)
    if model_str.startswith("models/"):
        return model_str[len("models/") :]
    return model_str


def _get_config_field(config: object, name: str) -> object:
    if config is None:
        return None
    if isinstance(config, Mapping):
        mapping = cast(Mapping[str, object], config)
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
                return "speech"
            if any(m.endswith("TEXT") for m in upper_modalities):
                return "text"
            if any(m.endswith("IMAGE") for m in upper_modalities):
                return "image"
    if is_vertex:
        # Vertex AI Live defaults responseModalities to ["AUDIO"] when omitted.
        return "speech"
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
        if isinstance(tool_item, Tool):
            definitions.extend(_to_tool_definition(tool_item))
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
    mime_type = (
        _get_config_field(blob_or_dict, "mime_type")
        or _get_config_field(blob_or_dict, "mimeType")
        or default_mime
    )
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


@dataclass
class _ActiveGeneration:
    invocation: RealtimeInferenceInvocation
    input_messages: list[InputMessage] = field(default_factory=list)
    completed_output_messages: list[OutputMessage] = field(
        default_factory=list
    )
    input_transcript_deltas: list[str] = field(default_factory=list)
    output_transcript_deltas: list[str] = field(default_factory=list)
    output_audio_chunks: list[bytes] = field(default_factory=list)
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
            str(getattr(thinking_level, "value", thinking_level)).lower()
            if thinking_level is not None
            else None
        )

        # Gemini Live 3.x models report the tool request and spoken answer
        # within a single generation; 2.x models report them as two sibling
        # generations.
        self.uses_single_generation_tool_exchange: bool = "gemini-3" in (
            (request_model or "").lower()
        )

        self.system_instruction: list[SystemInstructionPart] = []
        self.tool_definitions: list[ToolDefinition] = []
        if self._telemetry_handler.should_capture_content():
            sys_inst = _get_live_config_field(config, "system_instruction")
            if sys_inst is not None:
                self.system_instruction = _extract_system_instruction(
                    cast(ContentUnion | ContentUnionDict, sys_inst)
                )
            tools = _get_live_config_field(config, "tools")
            if tools is not None:
                self.tool_definitions = _extract_tool_definitions(
                    cast(ToolListUnion | ToolListUnionDict, tools)
                )

        self._pending_audio_chunks: list[bytes] = []
        self._pending_audio_mime_type: str | None = None
        self._pending_input_messages: list[InputMessage] = []
        self._pending_input_transcript_deltas: list[str] = []
        self._active_gen: _ActiveGeneration | None = None
        self._speech_invocation: UserSpeechInvocation | None = None
        self.terminal_error: BaseException | None = None

    def _mark_new_user_utterance(self) -> None:
        if self._active_gen is not None:
            self._active_gen.accepting_input_transcription = False

    def _flush_pending_audio(self) -> None:
        if not self._pending_audio_chunks:
            return
        if self._telemetry_handler.should_capture_content():
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
                last_part.content = last_part.content + combined
            else:
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
        self._pending_audio_chunks.clear()
        self._pending_audio_mime_type = None

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
            self._pending_audio_chunks.append(data_bytes)
            if not self._pending_audio_mime_type:
                self._pending_audio_mime_type = mime_type
            return
        self._flush_pending_audio()
        modality = (
            Modality.IMAGE
            if "image" in lower_mime
            else Modality.VIDEO
            if force_non_audio or "video" in lower_mime
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

        capture_content = self._telemetry_handler.should_capture_content()

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
                    default_mime="video/mp4",
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
        if (
            turns is not None
            and self._telemetry_handler.should_capture_content()
        ):
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
                    seq_turns = cast(Sequence[object], turns)
                    normalized_turns = [
                        Content.model_validate(item)
                        if isinstance(item, Mapping)
                        else cast(Content, item)
                        for item in seq_turns
                    ]
            if normalized_turns is not None:
                try:
                    messages = to_input_messages(contents=normalized_turns)
                    self._pending_input_messages.extend(messages)
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
        if not self._telemetry_handler.should_capture_content():
            return
        tool_msg = _build_tool_response_input_message(function_responses)
        if tool_msg is not None:
            self._pending_input_messages.append(tool_msg)

    def on_send_context_update(
        self,
        system_instruction: ContentUnion | ContentUnionDict | None = None,
        tools: ToolListUnion | ToolListUnionDict | None = None,
    ) -> None:
        if not self._telemetry_handler.should_capture_content():
            return
        if system_instruction is not None:
            self.system_instruction = _extract_system_instruction(
                system_instruction
            )
        if tools is not None:
            self.tool_definitions = _extract_tool_definitions(tools)

    def ensure_active_generation(self) -> _ActiveGeneration:
        if self._active_gen is None:
            self._flush_pending_audio()
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
            invocation.output_type = self.output_type
            if self.max_tokens is not None:
                invocation.max_tokens = self.max_tokens
            if self.reasoning_level is not None:
                invocation.reasoning_level = self.reasoning_level

            input_msgs = list(self._pending_input_messages)
            self._pending_input_messages.clear()
            input_transcript_deltas = list(
                self._pending_input_transcript_deltas
            )
            self._pending_input_transcript_deltas.clear()
            self._active_gen = _ActiveGeneration(
                invocation=invocation,
                input_messages=input_msgs,
                input_transcript_deltas=input_transcript_deltas,
            )
        return self._active_gen

    def _populate_and_finish_generation(
        self,
        active_gen: _ActiveGeneration,
        finish_reason: str,
        error: Error | BaseException | None = None,
    ) -> None:
        invocation = active_gen.invocation
        invocation.realtime_session_id = self.realtime_session_id

        # Attach any server-reported input audio transcription to the user message.
        if (
            active_gen.input_transcript_deltas
            and self._telemetry_handler.should_capture_content()
        ):
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

        if self._telemetry_handler.should_capture_content():
            active_gen.flush_current_turn_output_message(finish_reason)

        invocation.finish_reasons = [finish_reason]
        invocation.output_type = self.output_type

        if self._telemetry_handler.should_capture_content():
            invocation.input_messages = active_gen.input_messages
            invocation.output_messages = active_gen.completed_output_messages
            if self.system_instruction:
                invocation.system_instruction = self.system_instruction
            if self.tool_definitions:
                invocation.tool_definitions = self.tool_definitions

        usage = active_gen.usage_metadata
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            if prompt_tokens is not None:
                invocation.input_tokens = prompt_tokens

            response_tokens = getattr(usage, "response_token_count", None)
            if response_tokens is not None:
                invocation.output_tokens = response_tokens
            else:
                candidates_tokens = getattr(
                    usage, "candidates_token_count", None
                )
                thoughts_tokens = getattr(usage, "thoughts_token_count", None)
                if candidates_tokens is not None:
                    invocation.output_tokens = candidates_tokens + (
                        thoughts_tokens or 0
                    )

            cached_tokens = getattr(usage, "cached_content_token_count", None)
            if cached_tokens is not None:
                invocation.cache_read_input_tokens = cached_tokens

            thoughts_tokens = getattr(usage, "thoughts_token_count", None)
            if thoughts_tokens is not None:
                invocation.thinking_tokens = thoughts_tokens

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

        server_content = getattr(message, "server_content", None)
        tool_call = getattr(message, "tool_call", None)
        usage_metadata = getattr(message, "usage_metadata", None)
        capture_content = self._telemetry_handler.should_capture_content()

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
                or getattr(server_content, "turn_complete", False)
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
                            active_gen.output_audio_chunks.append(
                                bytes(raw_data)
                            )
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
            if getattr(server_content, "turn_complete", False):
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
                self._populate_and_finish_generation(
                    active_gen, finish_reason="tool_call"
                )
                return

        if _is_interaction_complete(server_content):
            finish_reason = active_gen.turn_complete_reason or (
                "interrupted" if active_gen.interrupted else "stop"
            )
            self._active_gen = None
            self._populate_and_finish_generation(
                active_gen, finish_reason=finish_reason
            )

    def on_receive_error(self, exc: BaseException) -> None:
        self.terminal_error = exc
        self._stop_user_speech(error=exc)
        if self._active_gen is not None:
            active_gen = self._active_gen
            self._active_gen = None
            self._populate_and_finish_generation(
                active_gen, finish_reason="error", error=exc
            )

    def on_session_error(self, exc: BaseException) -> None:
        self.on_receive_error(exc)

    def on_session_close(self) -> None:
        self._stop_user_speech()
        if self._active_gen is not None:
            active_gen = self._active_gen
            self._active_gen = None
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


def _wrap_session(session: AsyncSession, tracker: _LiveSessionTracker) -> None:
    """Wrap the methods of a live ``AsyncSession`` instance to record telemetry."""
    orig_send_realtime_input = session.send_realtime_input
    orig_send_client_content = session.send_client_content
    orig_send_tool_response = session.send_tool_response
    orig_send_context_update = getattr(session, "send_context_update", None)
    orig_receive = session._receive

    @functools.wraps(orig_send_realtime_input)
    async def wrapped_send_realtime_input(
        *args: object, **kwargs: object
    ) -> None:
        tracker.on_send_realtime_input(kwargs)
        await orig_send_realtime_input(*args, **kwargs)  # pyright: ignore[reportArgumentType]

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
        tracker.on_send_client_content(
            turns,
            turn_complete=True
            if turn_complete is None
            else bool(turn_complete),
        )
        await orig_send_client_content(*args, **kwargs)  # pyright: ignore[reportArgumentType]

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
        tracker.on_send_tool_response(function_responses)
        await orig_send_tool_response(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    @functools.wraps(orig_receive)
    async def wrapped_receive(
        *args: object, **kwargs: object
    ) -> LiveServerMessage:
        active_gen = tracker._active_gen
        try:
            if active_gen is not None:
                with active_gen.invocation.activate():
                    message = await orig_receive(*args, **kwargs)
            else:
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
            tracker.on_send_context_update(
                system_instruction=sys_inst, tools=tools
            )
            res = orig_send_context_update(*args, **kwargs)
            if hasattr(res, "__await__"):
                return await cast(
                    Callable[..., AsyncIterator[object]], lambda: res
                )()
            return res

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
