# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)

from opentelemetry.instrumentation.genai.langchain.agent_context import (
    claim_agent,
)
from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.instrumentation.genai.langchain.operation_mapping import (
    OperationName,
    classify_chain_run,
    resolve_agent_name,
)
from opentelemetry.instrumentation.genai.langchain.utils import (
    _legacy_function_call_request,
    _message_name,
    _normalize_role,
    extract_token_details,
    extract_usage_tokens,
    is_stream_end_marker,
    make_input_message,
    make_last_output_message,
    modality_tokens,
    normalize_provider,
    prepare_tool_definitions,
    resolve_response_model_and_id,
    response_fields_from_generation,
    to_input_messages,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
    LocalAgentInvocation,
    RetrievalInvocation,
    ToolInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    Modality,
    ModalityTokens,
    OutputMessage,
    RetrievalDocument,
    Role,
    TextPart,
    ToolCallRequestPart,
)

SUPPORTED_RAPI_RESPONSE_HEADERS = ("x-ms-served-model",)

# Conversation identifier in precedence order.
CONVERSATION_ID_METADATA_KEYS = (
    "thread_id",
    "session_id",
    "conversation_id",
)


def _conversation_id(metadata: dict[str, Any] | None) -> str | None:
    """Return the conversation id from a run's own metadata."""
    if not metadata:
        return None
    for key in CONVERSATION_ID_METADATA_KEYS:
        conversation_id = metadata.get(key)
        if conversation_id:
            return str(conversation_id)
    return None


def _usage_metadata_candidates(
    chat_generation: Any,
    llm_output: Any,
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    message = getattr(chat_generation, "message", None)
    usage_metadata = _usage_mapping(getattr(message, "usage_metadata", None))
    if usage_metadata is not None:
        candidates.append(usage_metadata)

    generation_info = _usage_mapping(
        getattr(chat_generation, "generation_info", None)
    )
    if generation_info is not None:
        usage_metadata = _usage_mapping(generation_info.get("usage_metadata"))
        if usage_metadata is not None:
            candidates.append(usage_metadata)

    llm_output_mapping = _usage_mapping(llm_output)
    if llm_output_mapping is not None:
        for key in ("token_usage", "usage"):
            usage_metadata = _usage_mapping(llm_output_mapping.get(key))
            if usage_metadata is not None:
                candidates.append(usage_metadata)

    return candidates


def _usage_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return None


def _contains_supported_modality_tokens(
    entries: ModalityTokens | None,
) -> bool:
    supported_modalities = {
        Modality.TEXT.value,
        Modality.IMAGE.value,
        Modality.AUDIO.value,
    }
    return entries is not None and any(
        str(getattr(modality, "value", modality)).lower()
        in supported_modalities
        and isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and token_count >= 0
        for modality, token_count in entries
    )


def _extract_document_score(doc: Any) -> float | int | None:
    """Extract relevance score polymorphically from a Document or Mapping.

    Retrieval scores are grounded in standard LangChain retrievers:
    - Direct knowledge base and vector retrievers (e.g. AmazonKnowledgeBasesRetriever,
      TavilySearchAPIRetriever) attach confidence/similarity scores to
      ``doc.metadata["score"]``.
    - Contextual compression retrievers wrapping rerankers (e.g. CohereRerank via
      ContextualCompressionRetriever) populate ``doc.metadata["relevance_score"]``.
    - Custom or duck-typed documents may provide a top-level ``score`` (or
      ``relevance_score``) attribute or key.

    Non-finite floats (NaN, +/-Inf) and boolean values are filtered out to ensure
    valid RFC 8259 JSON serialization in gen_ai.retrieval.documents.
    """
    score: Any = None
    if isinstance(doc, Mapping):
        doc_map = cast(Mapping[str, Any], doc)
        score = doc_map.get("score")
        if score is None:
            score = doc_map.get("relevance_score")
        if score is None:
            metadata = doc_map.get("metadata")
            if isinstance(metadata, Mapping):
                meta_map = cast(Mapping[str, Any], metadata)
                score = meta_map.get("score")
                if score is None:
                    score = meta_map.get("relevance_score")
            elif metadata is not None:
                score = getattr(metadata, "score", None)
                if score is None:
                    score = getattr(metadata, "relevance_score", None)
    else:
        score = getattr(doc, "score", None)
        if score is None:
            score = getattr(doc, "relevance_score", None)
        if score is None:
            metadata = getattr(doc, "metadata", None)
            if isinstance(metadata, Mapping):
                meta_map = cast(Mapping[str, Any], metadata)
                score = meta_map.get("score")
                if score is None:
                    score = meta_map.get("relevance_score")
            elif metadata is not None:
                score = getattr(metadata, "score", None)
                if score is None:
                    score = getattr(metadata, "relevance_score", None)

    if (
        score is not None
        and not isinstance(score, bool)
        and isinstance(score, (int, float))
    ):
        if isinstance(score, float) and not math.isfinite(score):
            return None
        return score

    return None


def _document_to_retrieval_document(doc: object) -> RetrievalDocument:
    """Extract only the standard document ID and relevance score."""
    if isinstance(doc, Mapping):
        doc_map = cast(Mapping[str, object], doc)
        doc_id = doc_map.get("id")
    else:
        doc_id = getattr(doc, "id", None)

    return RetrievalDocument(
        id=doc_id if isinstance(doc_id, str) else None,
        score=_extract_document_score(doc),
    )


class OpenTelemetryLangChainCallbackHandler(BaseCallbackHandler):
    """
    A callback handler for LangChain that uses OpenTelemetry to create spans for LLM calls and chains, tools etc,. in future.
    """

    def __init__(
        self,
        telemetry_handler: TelemetryHandler,
        *,
        _attach_to_context: bool = True,
        invocation_manager: _InvocationManager | None = None,
    ) -> None:
        super().__init__()
        self._telemetry_handler = telemetry_handler
        self._attach_to_context = _attach_to_context
        self._invocation_manager = (
            invocation_manager
            if invocation_manager is not None
            else _InvocationManager()
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        parent_agent_name, ancestor_agent_names = self._find_agent_context(
            parent_run_id
        )
        # A claimed announcement is proof this run is a create_agent root, which
        # the callback metadata alone cannot establish for a nested agent.
        agent_announcement = claim_agent()
        declared_agent_name = (
            agent_announcement.name if agent_announcement else None
        )
        operation = classify_chain_run(
            serialized,
            metadata,
            kwargs,
            parent_run_id,
            declared_agent_name,
            agent_announcement is not None,
            ancestor_agent_names,
        )
        parent_context = self._invocation_manager.get_parent_context(
            parent_run_id
        )
        conversation_id = _conversation_id(metadata)
        capture_content = self._telemetry_handler.should_capture_content()
        if operation == OperationName.INVOKE_WORKFLOW:
            workflow_name = kwargs.get("name") or serialized.get("name")
            workflow_name_override = (
                metadata.get("workflow_name") if metadata else None
            )
            workflow = self._telemetry_handler.workflow(
                name=workflow_name_override or workflow_name,
                context=parent_context,
                _attach_to_context=self._attach_to_context,
            )
            workflow.conversation_id = conversation_id
            if capture_content:
                workflow.input_messages = make_input_message(inputs)
            self._invocation_manager.add_invocation_state(
                run_id, parent_run_id, workflow
            )
        elif operation == OperationName.INVOKE_AGENT:
            # agent name passed by the user
            suggested_agent_name = resolve_agent_name(
                serialized,
                metadata,
                kwargs,
                declared_agent_name,
                ancestor_agent_names,
                agent_announcement is not None,
            )
            # find if there is an agent already
            if suggested_agent_name:
                suggested_agent_name_lower = suggested_agent_name.lower()
                parent_agent_name_lower = (
                    parent_agent_name.lower() if parent_agent_name else None
                )
                # An announced create_agent root always opens its own layer. For
                # non-announced runs, suppress a repeated metadata name matching the
                # enclosing agent - that repetition is inherited config, not a new agent.
                if (
                    agent_announcement is not None
                    or suggested_agent_name_lower != parent_agent_name_lower
                ):
                    agent = self._telemetry_handler.invoke_local_agent(
                        agent_name=suggested_agent_name,
                        context=parent_context,
                        _attach_to_context=self._attach_to_context,
                    )
                    agent.conversation_id = conversation_id
                    if capture_content:
                        agent.input_messages = make_input_message(inputs)

                    if metadata:
                        agent.agent_description = metadata.get(
                            "agent_description"
                        )

                    self._invocation_manager.add_invocation_state(
                        run_id,
                        parent_run_id,
                        agent,
                        agent_name=suggested_agent_name,
                    )
                else:
                    # We create invoke_agent span for the initial chain for agent. All follow-up chains invoked for agent invocation will not create agent span.
                    self._invocation_manager.add_invocation_state(
                        run_id, parent_run_id, None
                    )
            elif agent_announcement is not None:
                agent = self._telemetry_handler.invoke_local_agent(
                    agent_name=None,
                    context=parent_context,
                    _attach_to_context=self._attach_to_context,
                )
                agent.input_messages = make_input_message(inputs)
                self._invocation_manager.add_invocation_state(
                    run_id, parent_run_id, agent
                )
            else:
                # No agent name could be resolved; still register the run_id so that
                # parent-child traversal through _find_agent_context is not broken for
                # any children of this node.
                self._invocation_manager.add_invocation_state(
                    run_id, parent_run_id, None
                )
        else:
            # For unclassified chains, we still want to track them in the invocation manager to maintain the parent-child relationships, even though we won't create spans for them.
            self._invocation_manager.add_invocation_state(
                run_id, parent_run_id, None
            )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if invocation is None or not isinstance(
            invocation, (WorkflowInvocation, LocalAgentInvocation)
        ):
            # If the invocation does not exist, we cannot set attributes or end it
            self._invocation_manager.delete_invocation_state(run_id)
            return

        if self._telemetry_handler.should_capture_content():
            invocation.output_messages = make_last_output_message(outputs)

        invocation.stop()
        self._invocation_manager.delete_invocation_state(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if invocation is None or not isinstance(
            invocation, (WorkflowInvocation, LocalAgentInvocation)
        ):
            # If the invocation does not exist, we cannot set attributes or end it
            self._invocation_manager.delete_invocation_state(run_id)
            return

        invocation.fail(error)
        self._invocation_manager.delete_invocation_state(run_id=run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if "invocation_params" in kwargs:
            params = (
                kwargs["invocation_params"].get("params")
                or kwargs["invocation_params"]
            )
        else:
            params = kwargs

        # Resolve request_model from common provider-specific keys.
        request_model: str | None = None
        for model_tag in (
            "model_name",  # ChatOpenAI / ChatAnthropic
            "model_id",  # ChatBedrock
            "model",  # ChatGoogleGenerativeAI / ChatVertexAI / ChatGroq / ChatMistralAI / ChatCohere / ChatOllama / ChatDeepSeek / ChatXAI
        ):
            if (model := (params or {}).get(model_tag)) is not None:
                request_model = str(model)
                break
            if (model := (metadata or {}).get(model_tag)) is not None:
                request_model = str(model)
                break

        if request_model is None and metadata:
            if model := metadata.get("ls_model_name"):
                request_model = str(model)

        # Skip telemetry for unsupported request models
        if request_model is None:
            return

        request_model = request_model.removeprefix("models/")

        # Initialize variables with default values to avoid "possibly unbound" errors
        request_choice_count = None
        top_k = None
        top_p = None
        frequency_penalty = None
        presence_penalty = None
        stop_sequences = None
        seed = None
        temperature = None
        max_tokens = None

        if params is not None:
            request_choice_count = params.get("n")
            top_k = params.get("top_k")
            top_p = params.get("top_p")
            frequency_penalty = params.get("frequency_penalty")
            presence_penalty = params.get("presence_penalty")
            stop_sequences = params.get("stop")
            if stop_sequences is None:
                stop_sequences = params.get("stop_sequences")
            if stop_sequences is None:
                serialized_kwargs: dict[str, Any] = (
                    serialized.get("kwargs") or {}
                )
                stop_sequences = serialized_kwargs.get("stop_sequences")
            seed = params.get("seed")
            temperature = params.get("temperature")
            max_tokens = (
                params.get("max_completion_tokens")
                if params.get("max_completion_tokens") is not None
                else params.get("max_tokens")
            )

        provider = normalize_provider(metadata) or "unknown"
        if metadata is not None:
            # Override with ChatBedrock values if present
            if "ls_temperature" in metadata:
                temperature = metadata.get("ls_temperature")
            if "ls_max_tokens" in metadata:
                max_tokens = metadata.get("ls_max_tokens")

        # ``messages`` from on_chat_model_start is ``list[list[BaseMessage]]``
        # (one inner list per generation request). Flatten and let
        # :func:`to_input_messages` produce spec-conformant ``InputMessage`` s
        # with proper roles, tool-call requests, tool results, and reasoning.
        flattened: list[BaseMessage] = [msg for sub in messages for msg in sub]
        input_messages: list[InputMessage] = []
        if self._telemetry_handler.should_capture_content():
            input_messages = to_input_messages(flattened)

        parent_context = self._invocation_manager.get_parent_context(
            parent_run_id
        )
        llm_invocation = self._telemetry_handler.inference(
            provider,
            request_model=request_model,
            context=parent_context,
            _attach_to_context=self._attach_to_context,
        )
        llm_invocation.conversation_id = _conversation_id(metadata)
        llm_invocation.input_messages = input_messages
        llm_invocation.top_p = top_p
        llm_invocation.top_k = top_k
        llm_invocation.request_choice_count = request_choice_count
        llm_invocation.frequency_penalty = frequency_penalty
        llm_invocation.presence_penalty = presence_penalty
        llm_invocation.stop_sequences = stop_sequences
        llm_invocation.seed = seed
        llm_invocation.temperature = temperature
        llm_invocation.max_tokens = max_tokens
        if params is not None:
            tools = params.get("tools") or params.get("functions")
            if tools:
                tool_definitions = prepare_tool_definitions(tools)
                llm_invocation.tool_definitions = tool_definitions
        self._invocation_manager.add_invocation_state(
            run_id=run_id,
            parent_run_id=parent_run_id,
            invocation=llm_invocation,
        )

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if is_stream_end_marker(token, chunk):
            return

        llm_invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if not isinstance(llm_invocation, InferenceInvocation):
            return

        llm_invocation.record_stream_chunk()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        llm_invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if llm_invocation is None or not isinstance(
            llm_invocation,
            InferenceInvocation,
        ):
            # If the invocation does not exist, we cannot set attributes or end it
            return

        output_messages: list[OutputMessage] = []
        finish_reasons: list[str] = []
        served_model: str | None = None
        generation_model: str | None = None
        generation_response_id: str | None = None
        llm_output = getattr(response, "llm_output", None)
        for generation in getattr(response, "generations", []):
            for chat_generation in generation:
                message = chat_generation.message
                if message is None:
                    continue

                if generation_model is None or generation_response_id is None:
                    gen_model, gen_response_id = (
                        response_fields_from_generation(chat_generation)
                    )
                    generation_model = generation_model or gen_model
                    generation_response_id = (
                        generation_response_id or gen_response_id
                    )

                # Resolve finish_reason from generation_info or response
                # metadata. Modern langchain-aws (>= 0.2) emits ``stop_reason``
                # (snake_case); older versions used ``stopReason``.
                finish_reason: str | None = None
                generation_info = getattr(
                    chat_generation, "generation_info", None
                )
                if generation_info is not None:
                    finish_reason = generation_info.get("finish_reason")

                if chat_generation.message:
                    # Responses API (RAPI) may include the served model in the
                    # response headers, which accurately returns the served
                    # model name for the request.
                    if (
                        served_model is None
                        and chat_generation.message.response_metadata
                    ):
                        headers = (
                            chat_generation.message.response_metadata.get(
                                "headers"
                            )
                        )
                        if isinstance(headers, Mapping):
                            headers_map = cast(Mapping[Any, Any], headers)
                            for name, value in headers_map.items():
                                if (
                                    isinstance(name, str)
                                    and name.lower()
                                    in SUPPORTED_RAPI_RESPONSE_HEADERS
                                    and isinstance(value, str)
                                    and value.strip()
                                ):
                                    served_model = str(value)
                                    break

                    # Get finish reason if not found in generation_info above
                    if (
                        not finish_reason
                        and chat_generation.message.response_metadata
                    ):
                        finish_reason = (
                            chat_generation.message.response_metadata.get(
                                "stopReason"
                            )
                            or chat_generation.message.response_metadata.get(
                                "stop_reason"
                            )
                        )
                    finish_reason = finish_reason or "error"

                    name_str = _message_name(chat_generation.message)

                    if finish_reason in ("tool_calls", "tool_use"):
                        tool_calls: list[ToolCallRequestPart] = []
                        for tool_call in chat_generation.message.tool_calls:
                            tool_call_request = ToolCallRequestPart(
                                name=tool_call["name"],
                                id=tool_call["id"],
                                arguments=tool_call["args"],
                            )
                            tool_calls.append(tool_call_request)
                        output_message = OutputMessage(
                            role=_normalize_role(chat_generation.message)
                            or Role.ASSISTANT.value,
                            parts=cast(list[MessagePart], tool_calls),
                            finish_reason=finish_reason,
                            name=name_str,
                        )
                    elif (
                        legacy_call := _legacy_function_call_request(
                            chat_generation.message
                        )
                    ) is not None:
                        # Pre-tools OpenAI ``function_call`` present in
                        # ``additional_kwargs`` — surface it as a tool-call
                        # request part like the modern ``tool_calls`` path.
                        output_message = OutputMessage(
                            role=_normalize_role(chat_generation.message)
                            or Role.ASSISTANT.value,
                            parts=cast(list[MessagePart], [legacy_call]),
                            finish_reason=finish_reason,
                            name=name_str,
                        )
                    else:
                        parts = [
                            TextPart(
                                content=chat_generation.message.content,
                                type="text",
                            )
                        ]
                        role = (
                            _normalize_role(chat_generation.message)
                            or Role.ASSISTANT.value
                        )
                        output_message = OutputMessage(
                            role=role,
                            parts=cast(list[MessagePart], parts),
                            finish_reason=finish_reason,
                            name=name_str,
                        )
                    output_messages.append(output_message)
                    finish_reasons.append(finish_reason)

                    # Get token usage if available
                    has_input_tokens = False
                    has_output_tokens = False
                    input_modality_tokens = None
                    output_modality_tokens = None
                    for usage_metadata in _usage_metadata_candidates(
                        chat_generation,
                        llm_output,
                    ):
                        input_tokens, output_tokens = extract_usage_tokens(
                            usage_metadata
                        )
                        if input_tokens is not None and not has_input_tokens:
                            llm_invocation.input_tokens = input_tokens
                            has_input_tokens = True
                        if output_tokens is not None and not has_output_tokens:
                            llm_invocation.output_tokens = output_tokens
                            has_output_tokens = True

                        # Cache, reasoning, and modality token break-downs
                        token_details = extract_token_details(usage_metadata)
                        if (
                            cache_write := token_details.get(
                                "cache_write_input_tokens"
                            )
                        ) is not None:
                            llm_invocation.cache_write_input_tokens = (
                                cache_write
                            )
                        if (
                            cache_read := token_details.get(
                                "cache_read_input_tokens"
                            )
                        ) is not None:
                            llm_invocation.cache_read_input_tokens = cache_read
                        if (
                            reasoning_tokens := token_details.get(
                                "reasoning_tokens"
                            )
                        ) is not None:
                            llm_invocation.thinking_tokens = reasoning_tokens

                        candidate_input_modalities = modality_tokens(
                            usage_metadata, "input_token_details"
                        )
                        if (
                            input_modality_tokens is None
                            and _contains_supported_modality_tokens(
                                candidate_input_modalities
                            )
                        ):
                            input_modality_tokens = candidate_input_modalities

                        candidate_output_modalities = modality_tokens(
                            usage_metadata, "output_token_details"
                        )
                        if (
                            output_modality_tokens is None
                            and _contains_supported_modality_tokens(
                                candidate_output_modalities
                            )
                        ):
                            output_modality_tokens = (
                                candidate_output_modalities
                            )

                    llm_invocation.set_input_tokens(input_modality_tokens)
                    llm_invocation.set_output_tokens(output_modality_tokens)

        llm_invocation.output_messages = output_messages
        if finish_reasons:
            llm_invocation.finish_reasons = finish_reasons

        response_model, response_id = resolve_response_model_and_id(
            llm_output=llm_output,
            served_model=served_model,
            generation_model=generation_model,
            generation_response_id=generation_response_id,
        )
        if response_model is not None:
            llm_invocation.response_model_name = response_model
        if response_id is not None:
            llm_invocation.response_id = response_id

        llm_invocation.stop()
        self._invocation_manager.delete_invocation_state(run_id=run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        llm_invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if llm_invocation is None or not isinstance(
            llm_invocation,
            InferenceInvocation,
        ):
            # If the invocation does not exist, we cannot set attributes or end it
            return

        llm_invocation.fail(error)
        self._invocation_manager.delete_invocation_state(run_id=run_id)

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = "unknown"
        description = None
        if serialized is not None:
            name = serialized.get("name") or "unknown"
            description = serialized.get("description")

        arguments: Any
        if inputs is not None:
            arguments = inputs
        else:
            try:
                arguments = json.loads(input_str)
            except (json.JSONDecodeError, ValueError):
                arguments = input_str
        agent_name, _ = self._find_agent_context(parent_run_id)
        parent_context = self._invocation_manager.get_parent_context(
            parent_run_id
        )

        tool_invocation = self._telemetry_handler.tool(
            name=name,
            tool_type="function",
            agent_name=agent_name,
            context=parent_context,
            _attach_to_context=self._attach_to_context,
        )
        tool_invocation.tool_description = description
        tool_invocation.arguments = arguments
        tool_call_id = kwargs.get("tool_call_id")
        if tool_call_id:
            tool_invocation.tool_call_id = tool_call_id
        self._invocation_manager.add_invocation_state(
            run_id, parent_run_id, tool_invocation
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_kwargs: Any,
    ) -> None:
        tool_invocation = self._invocation_manager.get_invocation(run_id)
        if not isinstance(tool_invocation, ToolInvocation):
            return
        end_tool_call_id = getattr(output, "tool_call_id", None)
        if end_tool_call_id and not tool_invocation.tool_call_id:
            tool_invocation.tool_call_id = end_tool_call_id
        tool_invocation.tool_result = getattr(output, "content", None)
        tool_invocation.stop()
        self._invocation_manager.delete_invocation_state(run_id=run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **_: Any,
    ) -> None:
        tool_invocation = self._invocation_manager.get_invocation(run_id)
        if not isinstance(tool_invocation, ToolInvocation):
            return
        tool_invocation.fail(error)
        self._invocation_manager.delete_invocation_state(run_id=run_id)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        meta = metadata or {}
        provider = meta.get("ls_vector_store_provider") or None
        request_model = meta.get("ls_embedding_model") or None
        parent_context = self._invocation_manager.get_parent_context(
            parent_run_id
        )
        retrieval = self._telemetry_handler.retrieval(
            provider=provider,
            request_model=request_model,
            context=parent_context,
            _attach_to_context=self._attach_to_context,
        )
        retrieval.query_text = query
        self._invocation_manager.add_invocation_state(
            run_id, parent_run_id, retrieval
        )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if invocation is None or not isinstance(
            invocation, RetrievalInvocation
        ):
            self._invocation_manager.delete_invocation_state(run_id)
            return

        if self._telemetry_handler.should_capture_content():
            invocation.documents = [
                _document_to_retrieval_document(doc) for doc in documents
            ]
        invocation.stop()
        self._invocation_manager.delete_invocation_state(run_id)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if invocation is None or not isinstance(
            invocation, RetrievalInvocation
        ):
            self._invocation_manager.delete_invocation_state(run_id)
            return

        invocation.fail(error)
        self._invocation_manager.delete_invocation_state(run_id=run_id)

    def _find_agent_context(
        self, run_id: UUID | None
    ) -> tuple[str | None, set[str]]:
        current = run_id
        visited: set[UUID] = set()
        nearest_agent_name: str | None = None
        found_nearest_agent = False
        ancestor_agent_names: set[str] = set()
        while current is not None and current not in visited:
            visited.add(current)
            entity = self._invocation_manager.get_invocation(current)
            if isinstance(entity, LocalAgentInvocation):
                agent_name = self._invocation_manager.get_agent_name(current)
                if not found_nearest_agent:
                    nearest_agent_name = agent_name
                    found_nearest_agent = True
                if agent_name:
                    ancestor_agent_names.add(agent_name.lower())
            current = self._invocation_manager.get_parent_run_id(current)
        return nearest_agent_name, ancestor_agent_names
