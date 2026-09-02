# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""wrapt wrapper factories for smolagents instrumentation.

Each factory takes the shared :class:`TelemetryHandler` and returns a wrapper
suitable for :func:`wrapt.wrap_function_wrapper`, applied to the in-process
model classes only (see ``_IN_PROCESS_MODEL_CLASSES``):

- :func:`model_generate` wraps ``generate`` -> ``chat`` span.
- :func:`model_generate_stream` wraps ``generate_stream`` -> ``chat`` span, held
  open until the stream is drained.

Original library exceptions are always re-raised unmodified; telemetry is
finalized via ``invocation.stop()`` / ``invocation.fail(exc)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Mapping
from inspect import signature
from typing import TYPE_CHECKING, Any, TypeAlias

from smolagents.models import REMOVE_PARAMETER

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import SyncStreamWrapper
from opentelemetry.util.genai.types import OutputMessage, Role, TextPart

from ._messages import (
    to_input_messages,
    to_output_message,
    to_tool_definitions,
)
from .provider import resolve_provider

if TYPE_CHECKING:
    from smolagents.models import (
        ChatMessage,
        ChatMessageStreamDelta,
        Model,
    )

_logger = logging.getLogger(__name__)

# ``Model`` is quoted because a type alias is evaluated at runtime, unlike an
# annotation.
_Wrapper: TypeAlias = Callable[
    [Callable[..., Any], "Model", tuple[Any, ...], dict[str, Any]], Any
]


def _bind_arguments(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bind call args to the wrapped callable's signature, applying defaults.

    smolagents passes the interesting arguments positionally
    (``model.generate(input_messages)``), so binding is what makes them
    readable by name. On a binding failure the keyword arguments are returned
    on their own, without the positional ones and without the defaults.
    """
    try:
        bound = signature(wrapped).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        _logger.debug(
            "Failed to bind arguments of %s; falling back to keyword arguments",
            getattr(wrapped, "__qualname__", wrapped),
            exc_info=True,
        )
        return dict(kwargs)


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _merged_request_kwargs(
    instance: Model, bound: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild the request keyword arguments smolagents will send.

    ``Model._prepare_completion_kwargs`` seeds ``stop`` from the
    ``stop_sequences`` argument and ``response_format`` from its own argument,
    applies the per-call ``**kwargs`` on top and the model-level ``self.kwargs``
    last, so a key set at two levels reaches the provider with the model-level
    value and a model-level ``REMOVE_PARAMETER`` drops it from the request
    entirely. Following that order here is what keeps a removed key off the span
    as well.
    """
    merged: dict[str, Any] = {}
    stop_sequences = bound.get("stop_sequences")
    if stop_sequences is not None and instance.supports_stop_parameter:
        merged["stop"] = stop_sequences
    response_format = bound.get("response_format")
    if response_format is not None:
        merged["response_format"] = response_format
    call_kwargs = bound.get("kwargs")
    if isinstance(call_kwargs, dict):
        merged.update(call_kwargs)
    for name, value in instance.kwargs.items():
        if value is REMOVE_PARAMETER:
            merged.pop(name, None)
        else:
            merged[name] = value
    return merged


def _stop_sequences(merged: dict[str, Any]) -> list[str] | None:
    """Return the stop sequences the request carries, if any."""
    stop = merged.get("stop", merged.get("stop_sequences"))
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [str(item) for item in stop]
    return None


# smolagents' ``response_format`` type -> ``gen_ai.output.type`` value, for the
# types whose provider spelling differs from the semconv one. The openai
# instrumentation maps the same two.
_OUTPUT_TYPE_MAP: dict[str, str] = {
    "json_object": GenAI.GenAiOutputTypeValues.JSON.value,
    "json_schema": GenAI.GenAiOutputTypeValues.JSON.value,
}

_OUTPUT_TYPE_VALUES = frozenset(
    value.value for value in GenAI.GenAiOutputTypeValues
)


def _output_type(merged: dict[str, Any]) -> str | None:
    """Map the request's ``response_format`` to ``gen_ai.output.type``.

    smolagents forwards ``response_format`` to the provider unchanged, so its
    ``type`` is whatever the provider accepts. Only the values the semconv
    defines are recorded; a provider-specific one is dropped rather than put on
    an enum attribute.
    """
    response_format = merged.get("response_format")
    if not isinstance(response_format, Mapping):
        return None
    format_type = response_format.get("type")
    if not isinstance(format_type, str):
        return None
    output_type = _OUTPUT_TYPE_MAP.get(format_type, format_type)
    if output_type not in _OUTPUT_TYPE_VALUES:
        _logger.debug(
            "No gen_ai.output.type value for response_format type %r",
            format_type,
        )
        return None
    return output_type


def _apply_request_parameters(
    invocation: InferenceInvocation, instance: Model, bound: dict[str, Any]
) -> None:
    """Copy the request parameters smolagents will send onto the span."""
    merged = _merged_request_kwargs(instance, bound)

    invocation.temperature = _coerce_float(merged.get("temperature"))
    invocation.top_p = _coerce_float(merged.get("top_p"))
    invocation.top_k = _coerce_float(merged.get("top_k"))
    invocation.frequency_penalty = _coerce_float(
        merged.get("frequency_penalty")
    )
    invocation.presence_penalty = _coerce_float(merged.get("presence_penalty"))
    # TransformersModel takes the generation limit as max_new_tokens (which its
    # constructor defaults to 4096) and treats max_tokens as an alias for it.
    invocation.max_tokens = _coerce_int(
        merged.get("max_tokens", merged.get("max_new_tokens"))
    )
    invocation.seed = _coerce_int(merged.get("seed"))
    invocation.stop_sequences = _stop_sequences(merged)
    invocation.output_type = _output_type(merged)


def _apply_token_usage(
    invocation: InferenceInvocation, output_message: ChatMessage
) -> None:
    # ChatMessage.token_usage is the only source: the per-model
    # last_input_token_count / last_output_token_count counters were removed
    # before the oldest supported smolagents. The in-process runtimes count the
    # prompt and generated tokens themselves and report them here.
    token_usage = output_message.token_usage
    if token_usage is None:
        return
    invocation.input_tokens = token_usage.input_tokens
    invocation.output_tokens = token_usage.output_tokens


def _record_request(
    handler: TelemetryHandler,
    invocation: InferenceInvocation,
    wrapped: Callable[..., Any],
    instance: Model,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Record the request on the invocation. Extraction errors are dropped.

    The messages, tools and keyword arguments come from the caller, so the
    conversion can get a shape it does not handle. The span is already
    started and the model has not been called yet, so an error raised here
    would both break the call and leave the span unfinished.
    """
    try:
        bound = _bind_arguments(wrapped, args, kwargs)
        _apply_request_parameters(invocation, instance, bound)
        invocation.tool_definitions = to_tool_definitions(
            bound.get("tools_to_call_from")
        )
        if handler.should_capture_content():
            invocation.input_messages = to_input_messages(
                bound.get("messages")
            )
    except Exception:  # pylint: disable=broad-except
        _logger.debug("Failed to record the request", exc_info=True)


def _record_response(
    handler: TelemetryHandler,
    invocation: InferenceInvocation,
    output_message: ChatMessage,
) -> None:
    """Record the response on the invocation. Extraction errors are dropped.

    The model call has already succeeded at this point, so an error raised
    here would turn a completed call into a failed one.
    """
    try:
        _apply_token_usage(invocation, output_message)
        if handler.should_capture_content():
            invocation.output_messages = [to_output_message(output_message)]
    except Exception:  # pylint: disable=broad-except
        _logger.debug("Failed to record the response", exc_info=True)


def _start_inference(
    handler: TelemetryHandler,
    wrapped: Callable[..., Any],
    instance: Model,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> InferenceInvocation:
    """Start the ``chat`` span and record the request.

    ``generate`` and ``generate_stream`` take the same parameters. An in-process
    runtime listens on no socket, so the span carries no ``server.address`` or
    ``server.port``.
    """
    provider = resolve_provider(instance)
    invocation = handler.inference(
        provider,
        request_model=instance.model_id,
    )
    _record_request(handler, invocation, wrapped, instance, args, kwargs)
    return invocation


def model_generate(handler: TelemetryHandler) -> _Wrapper:
    """Wrap a defining ``Model.generate`` to emit a ``chat`` span.

    An in-process runtime returns the generated text, the token counts it made
    itself, and no response envelope, so the span carries no
    ``gen_ai.response.id``, ``gen_ai.response.model`` or
    ``gen_ai.response.finish_reasons``.
    """

    def wrapper(
        wrapped: Callable[..., ChatMessage],
        instance: Model,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ChatMessage:
        invocation = _start_inference(handler, wrapped, instance, args, kwargs)
        with invocation:
            output_message = wrapped(*args, **kwargs)
            _record_response(handler, invocation, output_message)
            return output_message

    return wrapper


class _ModelStreamWrapper(SyncStreamWrapper["ChatMessageStreamDelta"]):
    """Keep the ``chat`` span open until the delta stream is drained.

    Passing the invocation to ``super().__init__()`` turns on
    ``gen_ai.request.stream`` and the per-chunk timing metrics.

    ``TransformersModel`` is the only in-process runtime with a
    ``generate_stream``. Its deltas carry the generated text and per-delta token
    counts, and never any tool calls.
    """

    def __init__(
        self,
        stream: Generator[ChatMessageStreamDelta, None, None],
        invocation: InferenceInvocation,
        handler: TelemetryHandler,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_inference = invocation
        self._self_capture_content = handler.should_capture_content()
        self._self_content: list[str] = []
        self._self_input_tokens = 0
        self._self_output_tokens = 0
        self._self_saw_token_usage = False

    def _process_chunk(self, chunk: ChatMessageStreamDelta) -> None:
        content = chunk.content
        if content and self._self_capture_content:
            self._self_content.append(content)
        token_usage = chunk.token_usage
        if token_usage is not None:
            # Summed like agglomerate_stream_deltas, so the span agrees with the
            # totals the agent's monitor reports.
            self._self_saw_token_usage = True
            self._self_input_tokens += token_usage.input_tokens
            self._self_output_tokens += token_usage.output_tokens

    def _output_message(self) -> OutputMessage | None:
        content = "".join(self._self_content)
        if not content:
            # Closed before it was drained, so there is no response to report.
            return None
        # Deltas carry no finish reason, and defaulting to "stop" would hide a
        # generation cut short by a token limit.
        return OutputMessage(
            role=Role.ASSISTANT.value,
            parts=[TextPart(content=content)],
            finish_reason="",
        )

    def _finalize(self, error: BaseException | None = None) -> None:
        invocation = self._self_inference
        if self._self_saw_token_usage:
            invocation.input_tokens = self._self_input_tokens
            invocation.output_tokens = self._self_output_tokens
        if self._self_capture_content:
            output = self._output_message()
            if output is not None:
                invocation.output_messages = [output]
        if error is not None:
            invocation.fail(error)
        else:
            invocation.stop()

    def _on_stream_end(self) -> None:
        self._finalize()

    def _on_stream_error(self, error: BaseException) -> None:
        # Records what was streamed before the failure.
        self._finalize(error)


def model_generate_stream(handler: TelemetryHandler) -> _Wrapper:
    """Wrap a defining ``Model.generate_stream`` to emit a ``chat`` span.

    ``stream_outputs=True`` routes an agent's model calls here. The span stays
    open until the caller drains the deltas.
    """

    def wrapper(
        wrapped: Callable[..., Generator[ChatMessageStreamDelta, None, None]],
        instance: Model,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _ModelStreamWrapper:
        invocation = _start_inference(handler, wrapped, instance, args, kwargs)
        try:
            stream = wrapped(*args, **kwargs)
            return _ModelStreamWrapper(stream, invocation, handler)
        except Exception as error:
            invocation.fail(error)
            raise

    return wrapper
