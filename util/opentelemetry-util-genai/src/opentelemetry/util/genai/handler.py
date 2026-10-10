# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Telemetry handler for GenAI invocations.

This module exposes the `TelemetryHandler` class, which manages the lifecycle of
GenAI (Generative AI) invocations and emits telemetry data (spans and related attributes).

Classes:
    - TelemetryHandler: Manages GenAI invocation lifecycles and emits telemetry.

Functions:

Usage:
    handler = TelemetryHandler(
        instrumentation_scope_name=__name__,
        instrumentation_scope_version=__version__,
    )

    # Factory method: construct and start in one call, then stop or fail.
    invocation = handler.inference("my-provider", request_model="my-model")
    invocation.input_messages = [...]
    invocation.temperature = 0.7
    try:
        # ... call the underlying library ...
        invocation.output_messages = [...]
        invocation.stop()
    except Exception as exc:
        invocation.fail(exc)
        raise

    # Or use the context manager form — exception handling is automatic.
    with handler.inference("my-provider", request_model="my-model") as invocation:
        invocation.input_messages = [...]
        # ... call the underlying library ...
        invocation.output_messages = [...]
"""

from __future__ import annotations

from opentelemetry._logs import (
    LoggerProvider,
    get_logger,
)
from opentelemetry.context import Context, get_value
from opentelemetry.metrics import Meter, MeterProvider, get_meter
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import (
    TracerProvider,
    get_tracer,
)
from opentelemetry.util.genai._embedding_invocation import (
    SuppressedEmbeddingInvocation,
)
from opentelemetry.util.genai._fetch_response_invocation import (
    SuppressedFetchResponseInvocation,
)
from opentelemetry.util.genai._inference_invocation import (
    SuppressedInferenceInvocation,
)
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._retrieval_invocation import (
    SuppressedRetrievalInvocation,
)
from opentelemetry.util.genai._tool_invocation import (
    SuppressedToolInvocation,
)
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
    _SafeCompletionHook,
)
from opentelemetry.util.genai.invocation import (
    CLIENT_INFERENCE_CONTEXT_KEY,
    EMBEDDING_CONTEXT_KEY,
    FETCH_RESPONSE_CONTEXT_KEY,
    RETRIEVAL_CONTEXT_KEY,
    TOOL_CONTEXT_KEY,
    EmbeddingData,
    EmbeddingInvocation,
    FetchResponseData,
    FetchResponseInvocation,
    InferenceData,
    InferenceInvocation,
    LocalAgentInvocation,
    RemoteAgentInvocation,
    RetrievalData,
    RetrievalInvocation,
    ToolData,
    ToolInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    ErrorTypeResolver,
)
from opentelemetry.util.genai.utils import get_content_capturing_mode
from opentelemetry.util.genai.version import __version__


class TelemetryHandler:
    """
    High-level handler managing GenAI invocation lifecycles and emitting
    them as spans, metrics, and events.
    """

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        logger_provider: LoggerProvider | None = None,
        completion_hook: CompletionHook | None = None,
        instrumentation_scope_name: str | None = None,
        instrumentation_scope_version: str | None = None,
    ):
        """Creates a new telemetry handler.

        Args:
            instrumentation_scope_name: the name of the instrumentation library
                emitting the telemetry, reported as the instrumentation scope
                name. Instrumentations must pass their dotted package path so
                telemetry is attributable to them; it defaults to this module for
                backwards compatibility.
            instrumentation_scope_version: the version of the instrumentation
                library, reported as the instrumentation scope version.
        """
        schema_url = Schemas.V1_37_0.value
        # The scope name and version must describe the same library, so a
        # version passed without a name is dropped rather than pinned onto the
        # util's own scope name.
        if instrumentation_scope_name is None:
            instrumentation_scope_name = __name__
            instrumentation_scope_version = __version__
        version = instrumentation_scope_version or ""
        self._tracer = get_tracer(
            instrumentation_scope_name,
            version,
            tracer_provider,
            schema_url=schema_url,
        )
        meter: Meter = get_meter(
            instrumentation_scope_name,
            version,
            meter_provider=meter_provider,
            schema_url=schema_url,
        )
        self._instruments = _Instruments(meter)
        self._logger = get_logger(
            instrumentation_scope_name,
            version,
            logger_provider,
            schema_url=schema_url,
        )
        self._content_capturing_mode = get_content_capturing_mode()
        if completion_hook is None or isinstance(
            completion_hook, (_NoOpCompletionHook, _SafeCompletionHook)
        ):
            self._completion_hook: CompletionHook = (
                completion_hook or _NoOpCompletionHook()
            )
        else:
            self._completion_hook = _SafeCompletionHook(completion_hook)
        self._capture_content = (
            self._content_capturing_mode
            in (
                ContentCapturingMode.SPAN_ONLY,
                ContentCapturingMode.EVENT_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )
        ) or not isinstance(self._completion_hook, _NoOpCompletionHook)

    def should_capture_content(self) -> bool:
        """Returns True when message content should be captured by the instrumentation library.

        Message content includes the following attributes:
            - input_messages
            - output_messages
            - system_instructions
            - For tool invocations: tool args and results
            - For tool definitions: tool args and description

        The util library will decide when and where the message content will be
        added to the telemetry data.

        Content should be captured when the content capturing mode requires it, or
        when a real completion hook is configured (not a no-op).
        """
        return self._capture_content

    def retrieval(
        self,
        *,
        data_source_id: str | None = None,
        provider: str | None = None,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> RetrievalInvocation:
        """Returns a Retrieval invocation. Starts span when called.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        Only set data attributes on the invocation object, do not modify the span or context.
        """
        invocation_cls: type[RetrievalInvocation] = (
            SuppressedRetrievalInvocation
            if isinstance(
                get_value(RETRIEVAL_CONTEXT_KEY, context=context),
                RetrievalData,
            )
            else RetrievalInvocation
        )
        return invocation_cls(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            data_source_id=data_source_id,
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
        )

    def inference(
        self,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        operation_name: str | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> InferenceInvocation:
        """Returns an Inference invocation. Starts span when called.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        ``context`` parents the span and becomes the base of the invocation's
        own context. ``conversation_id`` overrides the one inherited from it.

        Only set data attributes on the invocation object, do not modify the span or context.
        """
        invocation_cls: type[InferenceInvocation] = (
            SuppressedInferenceInvocation
            if isinstance(
                get_value(CLIENT_INFERENCE_CONTEXT_KEY, context=context),
                InferenceData,
            )
            else InferenceInvocation
        )
        return invocation_cls(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            operation_name=operation_name,
            error_type_resolver=error_type_resolver,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
        )

    def embedding(
        self,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> EmbeddingInvocation:
        """Returns an Embedding invocation. Starts span when called.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        Only set data attributes on the invocation object, do not modify the span or context.
        """
        invocation_cls: type[EmbeddingInvocation] = (
            SuppressedEmbeddingInvocation
            if isinstance(
                get_value(EMBEDDING_CONTEXT_KEY, context=context),
                EmbeddingData,
            )
            else EmbeddingInvocation
        )
        return invocation_cls(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
        )

    def fetch_response(
        self,
        provider: str,
        *,
        response_id: str,
        request_stream: bool | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> FetchResponseInvocation:
        """Returns a Fetch Response invocation. Starts span when called.

        Describes fetching a previously generated model response by its
        identifier. No inference is performed and no tokens are consumed, so
        the fetched response's token counts must not be recorded here.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        Only set data attributes on the invocation object, do not modify the span or context.
        """
        invocation_cls: type[FetchResponseInvocation] = (
            SuppressedFetchResponseInvocation
            if isinstance(
                get_value(FETCH_RESPONSE_CONTEXT_KEY, context=context),
                FetchResponseData,
            )
            else FetchResponseInvocation
        )
        return invocation_cls(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            provider=provider,
            response_id=response_id,
            request_stream=request_stream,
            server_address=server_address,
            server_port=server_port,
            error_type_resolver=error_type_resolver,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
        )

    def tool(
        self,
        name: str,
        *,
        tool_type: str | None = None,
        agent_name: str | None = None,
        tool_call_id: str | None = None,
        tool_description: str | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> ToolInvocation:
        """Returns a Tool invocation. Starts span when called.

        .. deprecated:: 1.2b0
            Passing ``tool_call_id`` or ``tool_description`` as keyword
            arguments is deprecated. Set ``invocation.tool_call_id`` and
            ``invocation.tool_description`` on the returned invocation instead.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        Only set data attributes on the invocation object, do not modify the span or context.
        Recommended to set ``invocation.arguments`` and ``invocation.tool_result`` on the
        invocation object but only if `invocation.should_capture_content` is True.
        """
        ctx_data = get_value(TOOL_CONTEXT_KEY, context=context)
        is_suppressed = isinstance(ctx_data, ToolData) and (
            ctx_data.tool_name is None or ctx_data.tool_name == name
        )
        invocation_cls: type[ToolInvocation] = (
            SuppressedToolInvocation if is_suppressed else ToolInvocation
        )
        return invocation_cls(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            name,
            tool_type=tool_type,
            agent_name=agent_name,
            tool_call_id=tool_call_id,
            tool_description=tool_description,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
        )

    def invoke_local_agent(
        self,
        *,
        request_model: str | None = None,
        agent_name: str | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> LocalAgentInvocation:
        """Returns an agent invocation (INTERNAL span kind). Starts span when called.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        Use for agents running within the same process (e.g. LangChain, CrewAI).

        ``context`` parents the span and becomes the base of the invocation's
        own context. ``conversation_id`` overrides the one inherited from it.

        Only set data attributes on the invocation object, do not modify the span or context.
        """
        return LocalAgentInvocation(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            request_model=request_model,
            agent_name=agent_name,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
        )

    def invoke_remote_agent(
        self,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        agent_name: str | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> RemoteAgentInvocation:
        """Returns an agent invocation (CLIENT span kind). Starts span when called.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        Use for agents invoked over a remote service (e.g. OpenAI Assistants, AWS Bedrock).

        ``context`` parents the span and becomes the base of the invocation's
        own context. ``conversation_id`` overrides the one inherited from it.

        Only set data attributes on the invocation object, do not modify the span or context.
        """
        return RemoteAgentInvocation(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            provider=provider,
            request_model=request_model,
            agent_name=agent_name,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
        )

    def workflow(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> WorkflowInvocation:
        """Returns a Workflow invocation. Starts a span when called.

        Args:
            context: An optional OpenTelemetry Context to parent the span.

        Returned object can be used as a ContextManager which automatically calls `stop` or `fail`
        to finalize the span upon exiting. If not used as a ContextManager, the caller is
        responsible for calling `stop` or `fail` to finalize the span.

        ``context`` parents the span and becomes the base of the invocation's
        own context. ``conversation_id`` overrides the one inherited from it.

        Only set data attributes on the invocation object, do not modify the span or context.
        """
        return WorkflowInvocation(
            self._tracer,
            self._instruments,
            self._logger,
            self._completion_hook,
            name,
            content_capturing_mode=self._content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
        )


def get_telemetry_handler(
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
    logger_provider: LoggerProvider | None = None,
    completion_hook: CompletionHook | None = None,
) -> TelemetryHandler:
    """
    Returns a singleton TelemetryHandler instance.

    .. deprecated::1.2b0
        Construct a :class:`TelemetryHandler` directly instead.
    """
    handler: TelemetryHandler | None = getattr(
        get_telemetry_handler, "_default_handler", None
    )
    if handler is None:
        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=completion_hook,
        )
        setattr(get_telemetry_handler, "_default_handler", handler)
    return handler
