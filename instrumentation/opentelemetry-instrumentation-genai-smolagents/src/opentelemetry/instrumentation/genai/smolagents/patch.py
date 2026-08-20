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
- :func:`tool_call` wraps ``Tool.__call__`` -> ``execute_tool`` span.
- :func:`agent_tool_calls` wraps ``ToolCallingAgent.process_tool_calls``. No
  span; it publishes the step's tool call ids for the ``execute_tool`` spans.

Original library exceptions are always re-raised unmodified; telemetry is
finalized via ``invocation.stop()`` / ``invocation.fail(exc)``. The one
exception is documented on :func:`_tolerant_run_generator`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Mapping
from contextvars import ContextVar, Token
from inspect import (
    GEN_SUSPENDED,
    Parameter,
    getgeneratorstate,
    signature,
)
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from smolagents import AgentMaxStepsError
from smolagents.agents import RunResult
from smolagents.memory import ActionStep, FinalAnswerStep, ToolCall
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
from opentelemetry.util.genai.types import OutputMessage, Text

from ._messages import (
    final_answer_parts,
    task_to_input_message,
    to_content_value,
    to_input_messages,
    to_output_message,
    to_tool_definitions,
)
from .provider import resolve_provider

if TYPE_CHECKING:
    from smolagents.agents import MultiStepAgent, ToolCallingAgent, ToolOutput
    from smolagents.memory import PlanningStep
    from smolagents.models import (
        ChatMessage,
        ChatMessageStreamDelta,
        Model,
    )
    from smolagents.tools import Tool

_logger = logging.getLogger(__name__)

# The type of the instance the wrapper is applied to: a model, an agent, or a
# tool.
_InstanceT = TypeVar("_InstanceT")

_Wrapper: TypeAlias = Callable[
    [Callable[..., Any], _InstanceT, tuple[Any, ...], dict[str, Any]], Any
]

# What ``MultiStepAgent._run_stream`` yields, as smolagents declares it
# (``agents.py``). A ``ChatMessageStreamDelta`` appears only when the agent runs
# its model with ``stream_outputs=True``.
_RunStreamChunk: TypeAlias = (
    "ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta"
)


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
            role="assistant", parts=[Text(content=content)], finish_reason=""
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
        invocation = _start_inference(handler, wrapped, instance, args, kwargs)
        try:
            stream = wrapped(*args, **kwargs)
            return _ModelStreamWrapper(stream, invocation, handler)
        except Exception as error:
            invocation.fail(error)
            raise

    return wrapper


def _positional_parameter_names(instance: Tool) -> list[str]:
    """Names ``Tool.forward`` gives its positional parameters, in order.

    ``Tool.__call__`` forwards positional arguments straight to ``forward``, so
    the signature is what names them. The declared ``inputs`` mapping is not:
    ``Tool.validate_arguments`` only checks that the two agree as *sets*. A tool
    whose hand-written ``inputs`` is ordered differently would record every
    argument under the wrong name. ``@tool`` builds ``SimpleTool`` with an
    unbound ``forward``, hence the ``self`` filter.

    ``skip_forward_signature_validation`` opts out of that check, so ``forward``
    says nothing about the declared inputs and those are used instead. A
    ``PipelineTool`` is why this matters: its ``forward(inputs)`` names the
    *encoded* inputs, not the caller's argument. Only the classes that opt out
    define the attribute, which is how smolagents itself reads it.
    """
    if getattr(instance, "skip_forward_signature_validation", False) is True:
        return list(instance.inputs)
    try:
        parameters = list(signature(instance.forward).parameters.values())
    except (TypeError, ValueError):
        _logger.debug(
            "Failed to read the signature of %s.forward",
            type(instance).__name__,
            exc_info=True,
        )
        parameters = []
    names = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name != "self"
    ]
    return names or list(instance.inputs)


def _tool_arguments(
    instance: Tool, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Map a ``Tool.__call__`` invocation onto ``{input name: value}``."""
    call_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key != "sanitize_inputs_outputs"
    }
    if not args:
        return call_kwargs
    names = _positional_parameter_names(instance)
    # A lone dict whose keys are all declared inputs is forwarded as kwargs by
    # Tool.__call__ itself, so record it the same way.
    if (
        not call_kwargs
        and len(args) == 1
        and isinstance(args[0], dict)
        and all(key in names for key in args[0])
    ):
        return args[0]
    # zip truncates: positional arguments beyond the declared inputs are dropped
    # rather than recorded under a wrong name.
    return {**dict(zip(names, args)), **call_kwargs}


# The tool calls a ToolCallingAgent step is about to run, published by
# agent_tool_calls for the execute_tool spans of that step. It is a ContextVar
# because smolagents runs parallel tool calls in worker threads, and each thread
# copies the caller's context.
_PENDING_TOOL_CALLS: ContextVar[tuple[ToolCall, ...]] = ContextVar(
    "otel_smolagents_pending_tool_calls", default=()
)


def _claim_tool_call_id(tool_name: str) -> str | None:
    """The id of the pending tool call for ``tool_name``, if there is just one.

    ``Tool.__call__`` gets only the argument values, so the id comes from the
    ``ToolCall`` objects the step yielded. The match is by name, because
    ``execute_tool_call`` rewrites state-variable arguments first. Two calls to
    one tool in a step have no unambiguous match, so the id is left off.
    """
    matches = [
        pending
        for pending in _PENDING_TOOL_CALLS.get()
        if pending.name == tool_name
    ]
    if len(matches) != 1:
        return None
    return matches[0].id or None


def _reset_pending_tool_calls(token: Token[tuple[ToolCall, ...]]) -> None:
    try:
        _PENDING_TOOL_CALLS.reset(token)
    except ValueError:
        # Token from another context: a generator closed on a different thread.
        # Telemetry must not raise over it.
        _logger.debug("Failed to reset the pending tool calls", exc_info=True)


def agent_tool_calls(
    wrapped: Callable[..., Generator[ToolCall | ToolOutput, None, None]],
    instance: ToolCallingAgent,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Generator[ToolCall | ToolOutput, None, None]:
    """Publish a ``ToolCallingAgent`` step's tool calls. Emits no span.

    ``process_tool_calls`` yields every ``ToolCall`` before it runs any of them,
    so the published tuple is complete by the time a tool executes. Parallel
    calls run in worker threads that copy the context, so they see it too.
    """
    pending: list[ToolCall] = []
    token = _PENDING_TOOL_CALLS.set(())
    try:
        for item in wrapped(*args, **kwargs):
            if isinstance(item, ToolCall):
                pending.append(item)
                _PENDING_TOOL_CALLS.set(tuple(pending))
            yield item
    finally:
        _reset_pending_tool_calls(token)


def tool_call(handler: TelemetryHandler) -> _Wrapper[Tool]:
    """Wrap ``Tool.__call__`` to emit an ``execute_tool`` span."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Tool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Tool.validate_arguments() runs after every Tool.__init__ and rejects
        # a tool without a str name and description.
        invocation = handler.tool(
            name=instance.name,
            tool_call_id=_claim_tool_call_id(instance.name),
            tool_type="function",
            tool_description=instance.description,
        )
        # util-genai puts tool arguments and results on span attributes and
        # nowhere else, so the gate here is the span one rather than the
        # handler's should_capture_content(), which an agent run uses.
        with invocation:
            if invocation.should_capture_content_on_span:
                invocation.arguments = _tool_arguments(instance, args, kwargs)
            result = wrapped(*args, **kwargs)
            if invocation.should_capture_content_on_span:
                invocation.tool_result = to_content_value(result)
            return result

    return wrapper


def _run_finish_reason(agent: MultiStepAgent) -> str:
    """Why the run stopped: ``"stop"`` for an answer, ``"length"`` for a budget.

    A run that uses up ``max_steps`` does not raise. smolagents asks the model
    for a closing answer, appends an ``ActionStep`` carrying
    ``AgentMaxStepsError`` and returns normally, so without this check a run that
    gave up is indistinguishable from one that answered.
    ``MultiStepAgent.run`` detects it the same way to set its ``RunResult`` state
    to ``"max_steps_error"`` (``agents.py``).
    """
    steps = agent.memory.steps
    # Only an ActionStep carries an error; the other step types have no such
    # attribute at all.
    last_step = steps[-1] if steps else None
    if isinstance(last_step, ActionStep) and isinstance(
        last_step.error, AgentMaxStepsError
    ):
        return "length"
    return "stop"


def _run_output(result: object) -> object:
    # ``run()`` returns a RunResult when ``return_full_result`` is true, which
    # defaults to the agent-level ``self.return_full_result``, so a plain
    # ``run(task)`` can return one too.
    return result.output if isinstance(result, RunResult) else result


def _assistant_message(output: object, finish_reason: str) -> OutputMessage:
    return OutputMessage(
        role="assistant",
        parts=final_answer_parts(output),
        finish_reason=finish_reason,
    )


def _tolerant_run_generator(
    generator: Generator[_RunStreamChunk, None, None],
) -> Generator[_RunStreamChunk, None, None]:
    """Delegate to the run generator, tolerating its refusal to shut down.

    ``MultiStepAgent._run_stream`` yields from inside a ``finally`` block
    (``agents.py``), so closing it mid-run makes CPython raise ``RuntimeError:
    generator ignored GeneratorExit`` and leaves the generator suspended. That is
    the generator declining to stop, not a failed run: the steps that did run
    succeeded. Returning instead keeps ``with agent.run(..., stream=True)`` from
    reporting an error the caller never wrote, and leaves ``close()``,
    ``__exit__`` and finalization to :class:`SyncStreamWrapper`.

    A generator that fails mid-iteration, or whose cleanup code raises, ends up
    *closed*; that is how a real failure is told apart, and it propagates and
    fails the span.

    This is a generator function rather than an adapter class so that the object
    :class:`SyncStreamWrapper` proxies is a real generator. A proxy forwards
    ``__class__``, so ``isinstance(stream, Generator)`` and
    ``inspect.isgenerator(stream)`` keep answering what they answer for an
    uninstrumented run.
    """
    try:
        yield from generator
    except RuntimeError:
        if getgeneratorstate(generator) != GEN_SUSPENDED:
            raise
        _logger.debug(
            "The smolagents run generator ignored GeneratorExit on close",
            exc_info=True,
        )


class _AgentRunStreamWrapper(SyncStreamWrapper[_RunStreamChunk]):
    """Keep the ``invoke_agent`` span open until the run generator is drained."""

    def __init__(
        self,
        stream: Generator[_RunStreamChunk, None, None],
        invocation: AgentInvocation,
        handler: TelemetryHandler,
        agent: MultiStepAgent,
    ) -> None:
        # The invocation is not passed to super().__init__(): that turns on the
        # base class's per-chunk timing, which records
        # gen_ai.client.operation.time_to_first_chunk and
        # time_per_output_chunk. Both metrics describe the response stream of a
        # generation call and require gen_ai.provider.name. A chunk of an agent
        # run is a step object, and the agent span carries no provider.
        super().__init__(_tolerant_run_generator(stream))
        self._self_agent_invocation = invocation
        self._self_handler = handler
        self._self_agent = agent
        self._self_final_output: object = None
        self._self_saw_final = False

    def _process_chunk(self, chunk: _RunStreamChunk) -> None:
        if isinstance(chunk, FinalAnswerStep):
            self._self_final_output = chunk.output
            self._self_saw_final = True

    def _on_stream_end(self) -> None:
        invocation = self._self_agent_invocation
        if self._self_saw_final:
            finish_reason = _run_finish_reason(self._self_agent)
            invocation.finish_reasons = [finish_reason]
            if self._self_handler.should_capture_content():
                invocation.output_messages = [
                    _assistant_message(self._self_final_output, finish_reason)
                ]
        invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_agent_invocation.fail(error)


def _finalize_failed(
    invocation: GenAIInvocation, error: BaseException
) -> None:
    """End an invocation whose call raised, the way ``with invocation`` does.

    ``GenAIInvocation.__exit__`` records an ``Exception`` as a failure and ends
    the span for anything else, so interrupting a long agent run ends its span
    instead of leaving it open.
    """
    if isinstance(error, Exception):
        invocation.fail(error)
    else:
        invocation.stop()


def _start_agent_run(
    handler: TelemetryHandler, agent: MultiStepAgent
) -> AgentInvocation:
    """Start the ``invoke_agent`` span and record what the agent was built with."""
    # An agent only has a name when it was given one, which is required of
    # managed agents and optional everywhere else.
    agent_name = agent.name or type(agent).__name__
    invocation = handler.invoke_local_agent(
        agent_name=agent_name,
        # An agent runs exactly one model, which is the condition semconv
        # puts on gen_ai.request.model for an agent span.
        request_model=agent.model.model_id,
    )
    invocation.agent_description = agent.description
    # A managed agent is callable by the model just like a tool is:
    # ToolCallingAgent passes it in tools_to_call_from, and a CodeAgent
    # calls it from generated code.
    invocation.tool_definitions = to_tool_definitions(
        [*agent.tools.values(), *agent.managed_agents.values()]
    )
    # gen_ai.client.operation.duration requires gen_ai.provider.name, and
    # invoke_local_agent() exposes no provider argument. Resolve it from the
    # agent's model and set it on the metric attributes, the way the
    # openai-agents package does for tool spans. The local agent span
    # deliberately carries no provider, so this stays off the span.
    invocation.metric_attributes[GenAI.GEN_AI_PROVIDER_NAME] = (
        resolve_provider(agent.model)
    )
    return invocation


def _record_agent_run(
    handler: TelemetryHandler,
    invocation: AgentInvocation,
    agent: MultiStepAgent,
    bound: dict[str, Any],
    result: Any,
    *,
    capture_content: bool,
) -> _AgentRunStreamWrapper | None:
    """Record what ``run`` returned, or wrap a stream the caller has to drain.

    A stream has no answer yet, so it gets the wrapper that keeps the invocation
    open until the caller drains it. Anything else is a finished run: its
    outcome is recorded here and the caller ends the invocation.
    """
    if capture_content and bound.get("additional_args"):
        # run() appends the additional arguments to the task, so the effective
        # task is only readable after the call.
        invocation.input_messages = [
            task_to_input_message(
                agent.task or bound.get("task"), bound.get("images")
            )
        ]

    if bound.get("stream") and isinstance(result, Generator):
        return _AgentRunStreamWrapper(result, invocation, handler, agent)

    finish_reason = _run_finish_reason(agent)
    invocation.finish_reasons = [finish_reason]
    if capture_content:
        invocation.output_messages = [
            _assistant_message(_run_output(result), finish_reason)
        ]
    return None


def agent_run(handler: TelemetryHandler) -> _Wrapper[MultiStepAgent]:
    """Wrap ``MultiStepAgent.run`` to emit an ``invoke_agent`` span."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: MultiStepAgent,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        agent = instance
        # Bound before the span starts: a binding failure then costs no span.
        bound = _bind_arguments(wrapped, args, kwargs)
        invocation = _start_agent_run(handler, agent)
        capture_content = handler.should_capture_content()

        # A managed agent runs inside the manager's step; its tools must not
        # claim the manager's call ids.
        token = _PENDING_TOOL_CALLS.set(())
        try:
            if capture_content:
                invocation.input_messages = [
                    task_to_input_message(
                        bound.get("task"), bound.get("images")
                    )
                ]
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            _finalize_failed(invocation, error)
            raise
        finally:
            _reset_pending_tool_calls(token)

        stream: _AgentRunStreamWrapper | None = None
        try:
            stream = _record_agent_run(
                handler,
                invocation,
                agent,
                bound,
                result,
                capture_content=capture_content,
            )
        except Exception:  # pylint: disable=broad-except
            # The run itself succeeded, so a failure to record it is not the
            # caller's to handle, and the caller still gets what run() returned.
            _logger.debug("Failed to record the agent run", exc_info=True)
        finally:
            if stream is None:
                invocation.stop()
        return stream if stream is not None else result

    return wrapper
