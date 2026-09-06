# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""wrapt wrapper factories for smolagents instrumentation.

Each factory takes the shared :class:`TelemetryHandler` and returns a wrapper
suitable for :func:`wrapt.wrap_function_wrapper`:

- :func:`model_generate` wraps ``generate`` -> ``chat`` span, applied to the
  in-process model classes only (see ``_IN_PROCESS_MODEL_CLASSES``).
- :func:`model_generate_stream` wraps ``generate_stream`` -> ``chat`` span, held
  open until the stream is drained.
- :func:`agent_run` wraps ``MultiStepAgent.run`` -> ``invoke_agent`` span.

Original library exceptions are re-raised unmodified. Telemetry is finalized
through ``invocation.stop()`` or ``invocation.fail(exc)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack
from inspect import Signature, signature
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from smolagents import AgentMaxStepsError
from smolagents.agents import RunResult
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import REMOVE_PARAMETER

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    GenAIInvocation,
    InferenceInvocation,
)
from opentelemetry.util.genai.stream import SyncStreamWrapper
from opentelemetry.util.genai.types import OutputMessage, Role, TextPart

from ._messages import (
    final_answer_parts,
    task_to_input_messages,
    to_input_messages,
    to_output_message,
    to_tool_definitions,
)
from .provider import resolve_provider

if TYPE_CHECKING:
    from smolagents.agents import MultiStepAgent
    from smolagents.memory import PlanningStep
    from smolagents.models import (
        ChatMessage,
        ChatMessageStreamDelta,
        Model,
    )

_logger = logging.getLogger(__name__)

_InstanceT = TypeVar("_InstanceT")

_Wrapper: TypeAlias = Callable[
    [Callable[..., Any], _InstanceT, tuple[Any, ...], dict[str, Any]], Any
]

# Quote the alias because ``PlanningStep`` and ``ChatMessageStreamDelta`` are
# imported only for type checking.
_RunStreamChunk: TypeAlias = (
    "ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta"
)


def _finish(
    invocation: GenAIInvocation, error: BaseException | None = None
) -> None:
    """End the invocation. An interrupt ends the span without an error."""
    if isinstance(error, Exception):
        invocation.fail(error)
    else:
        invocation.stop()


# Keyed on the underlying function: wrapt hands the wrapper a freshly bound
# method per call, so keying on that would retain every model and agent.
_signatures: dict[object, Signature] = {}


def _bind_arguments(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bind call args to the wrapped callable's signature, applying defaults.

    smolagents passes the interesting arguments positionally
    (``model.generate(input_messages)``, ``agent.run(task)``), so binding is
    what makes them readable by name. On a binding failure the keyword
    arguments are returned on their own, without the positional ones and
    without the defaults.
    """
    try:
        function: object | None = getattr(wrapped, "__func__", None)
        call_signature = (
            _signatures.get(function) if function is not None else None
        )
        if call_signature is None:
            call_signature = signature(wrapped)
            if function is not None:
                _signatures[function] = call_signature
        bound = call_signature.bind(*args, **kwargs)
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
    bound = _bind_arguments(wrapped, args, kwargs)
    _apply_request_parameters(invocation, instance, bound)
    invocation.tool_definitions = to_tool_definitions(
        bound.get("tools_to_call_from")
    )
    if handler.should_capture_content():
        invocation.input_messages = to_input_messages(bound.get("messages"))


def _record_response(
    handler: TelemetryHandler,
    invocation: InferenceInvocation,
    output_message: ChatMessage,
) -> None:
    _apply_token_usage(invocation, output_message)
    if handler.should_capture_content():
        invocation.output_messages = [to_output_message(output_message)]


def _start_inference(
    handler: TelemetryHandler, instance: Model
) -> InferenceInvocation:
    """In-process runtimes have no server address or port."""
    return handler.inference(
        resolve_provider(instance),
        request_model=instance.model_id,
    )


def model_generate(handler: TelemetryHandler) -> _Wrapper[Model]:
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
        invocation = _start_inference(handler, instance)
        with invocation:
            _record_request(
                handler, invocation, wrapped, instance, args, kwargs
            )
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

    def __del__(self) -> None:
        try:
            self._finalize_success()
        except BaseException:  # pylint: disable=broad-except
            pass

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
        _finish(invocation, error)

    def _on_stream_end(self) -> None:
        self._finalize()

    def _on_stream_error(self, error: BaseException) -> None:
        # Records what was streamed before the failure.
        self._finalize(error)


def model_generate_stream(handler: TelemetryHandler) -> _Wrapper[Model]:
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
        invocation = _start_inference(handler, instance)
        with ExitStack() as finishing:
            finishing.enter_context(invocation)
            _record_request(
                handler, invocation, wrapped, instance, args, kwargs
            )
            stream = _ModelStreamWrapper(
                wrapped(*args, **kwargs), invocation, handler
            )
            # Transfer invocation finalization to the stream.
            finishing.pop_all()
            return stream

    return wrapper


def _record_run_answer(
    invocation: AgentInvocation,
    agent: MultiStepAgent,
    output: object,
    *,
    capture_content: bool,
) -> None:
    # Reaching ``max_steps`` returns normally with ``AgentMaxStepsError`` on the
    # final ``ActionStep``.
    steps = agent.memory.steps
    last_step = steps[-1] if steps else None
    error = last_step.error if isinstance(last_step, ActionStep) else None
    if isinstance(error, AgentMaxStepsError):
        finish_reason = "max_steps"
    elif error is not None:
        finish_reason = "error"
    else:
        finish_reason = "stop"
    invocation.finish_reasons = [finish_reason]
    if capture_content:
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=final_answer_parts(output),
                finish_reason=finish_reason,
            )
        ]


class _AgentRunStreamWrapper(SyncStreamWrapper[_RunStreamChunk]):
    def __init__(
        self,
        stream: Generator[_RunStreamChunk, None, None],
        invocation: AgentInvocation,
        agent: MultiStepAgent,
        *,
        capture_content: bool,
    ) -> None:
        # Model-stream metrics do not apply to agent steps, so do not pass the
        # invocation to the base class.
        super().__init__(stream)
        self._self_agent_invocation = invocation
        self._self_agent = agent
        self._self_capture_content = capture_content
        self._self_final_output: object = None
        self._self_saw_final = False
        self._self_finished = False

    def __del__(self) -> None:
        try:
            self._finalize_success()
        except BaseException:  # pylint: disable=broad-except
            pass

    def _finish_once(self, error: BaseException | None = None) -> None:
        if self._self_finished:
            return
        self._self_finished = True
        _finish(self._self_agent_invocation, error)

    # The base wrapper finalizes on StopIteration and on an Exception, but not
    # on an interrupt, which would leave the span open.
    def __next__(self) -> _RunStreamChunk:
        try:
            return super().__next__()
        except StopIteration:
            raise
        except BaseException as error:
            self._finish_once(error)
            raise

    def close(self) -> None:
        try:
            super().close()
        except BaseException as error:
            self._finish_once(error)
            raise

    def _process_chunk(self, chunk: _RunStreamChunk) -> None:
        if isinstance(chunk, FinalAnswerStep):
            self._self_final_output = chunk.output
            self._self_saw_final = True
            self._finalize_success()

    def _on_stream_end(self) -> None:
        try:
            if self._self_saw_final:
                _record_run_answer(
                    self._self_agent_invocation,
                    self._self_agent,
                    self._self_final_output,
                    capture_content=self._self_capture_content,
                )
        finally:
            self._finish_once()

    def _on_stream_error(self, error: BaseException) -> None:
        self._finish_once(error)


def _record_agent(
    invocation: AgentInvocation,
    agent: MultiStepAgent,
    bound: dict[str, Any],
    *,
    capture_content: bool,
) -> None:
    invocation.agent_description = agent.description
    # Managed agents are exposed to the model as tools.
    invocation.tool_definitions = to_tool_definitions(
        [*agent.tools.values(), *agent.managed_agents.values()]
    )
    if capture_content:
        invocation.input_messages = task_to_input_messages(
            bound.get("task"), bound.get("images")
        )


def _record_agent_run(
    invocation: AgentInvocation,
    agent: MultiStepAgent,
    bound: dict[str, Any],
    result: object,
    *,
    capture_content: bool,
) -> _AgentRunStreamWrapper | None:
    """Record a finished run or wrap a streamed run until it is drained."""
    if capture_content and bound.get("additional_args"):
        # ``run()`` appends ``additional_args`` to ``agent.task``.
        invocation.input_messages = task_to_input_messages(
            agent.task or bound.get("task"), bound.get("images")
        )

    if bound.get("stream") and isinstance(result, Generator):
        return _AgentRunStreamWrapper(
            result,
            invocation,
            agent,
            capture_content=capture_content,
        )

    # ``return_full_result`` can be set on the agent, so ``run(task)`` can
    # return ``RunResult``.
    output = result.output if isinstance(result, RunResult) else result
    _record_run_answer(
        invocation, agent, output, capture_content=capture_content
    )
    return None


def agent_run(handler: TelemetryHandler) -> _Wrapper[MultiStepAgent]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: MultiStepAgent,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        agent = instance
        bound = _bind_arguments(wrapped, args, kwargs)
        capture_content = handler.should_capture_content()
        # Only managed agents require names. Use the class name for unnamed agents.
        # Models may omit ``model_id``.
        invocation = handler.invoke_local_agent(
            agent_name=agent.name or type(agent).__name__,
            # ``gen_ai.request.model`` applies because an agent uses one model.
            request_model=getattr(agent.model, "model_id", None),
        )

        with ExitStack() as finishing:
            finishing.enter_context(invocation)
            _record_agent(
                invocation, agent, bound, capture_content=capture_content
            )
            result = wrapped(*args, **kwargs)
            stream = _record_agent_run(
                invocation,
                agent,
                bound,
                result,
                capture_content=capture_content,
            )
            if stream is None:
                return result
            # Transfer invocation finalization to the stream.
            finishing.pop_all()
            return stream

    return wrapper
