# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.instrumentation.genai.langchain.operation_mapping import (
    OperationName,
    classify_chain_run,
    resolve_agent_name,
)
from opentelemetry.instrumentation.genai.langchain.utils import (
    make_input_message,
    make_last_output_message,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    InferenceInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
)


class OpenTelemetryLangChainCallbackHandler(BaseCallbackHandler):
    """
    A callback handler for LangChain that uses OpenTelemetry to create spans for LLM calls and chains, tools etc,. in future.
    """

    def __init__(self, telemetry_handler: TelemetryHandler) -> None:
        super().__init__()
        self._telemetry_handler = telemetry_handler
        self._invocation_manager = _InvocationManager()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        operation = classify_chain_run(
            serialized, metadata, kwargs, parent_run_id
        )

        if operation == OperationName.INVOKE_WORKFLOW:
            workflow_name = kwargs.get("name") or serialized.get("name")
            workflow_name_override = (
                metadata.get("workflow_name") if metadata else None
            )
            workflow = self._telemetry_handler.workflow(
                name=workflow_name_override or workflow_name
            )
            workflow.input_messages = make_input_message(inputs)
            self._invocation_manager.add_invocation_state(
                run_id, parent_run_id, workflow
            )
        elif operation == OperationName.INVOKE_AGENT:
            # agent name passed by the user
            suggested_agent_name = resolve_agent_name(
                serialized, metadata, kwargs
            )
            # find if there is an agent already
            agent_invocation = self._find_nearest_agent(parent_run_id)
            agent_invocation_name = (
                agent_invocation.agent_name if agent_invocation else None
            )
            if suggested_agent_name:
                suggested_agent_name_lower = suggested_agent_name.lower()
                agent_invocation_name_lower = (
                    agent_invocation_name.lower()
                    if agent_invocation_name
                    else None
                )
                if suggested_agent_name_lower != agent_invocation_name_lower:
                    agent = self._telemetry_handler.invoke_local_agent(
                        agent_name=suggested_agent_name,
                    )
                    agent.input_messages = make_input_message(inputs)

                    if metadata:
                        agent.agent_id = metadata.get("agent_id")
                        agent.agent_description = metadata.get(
                            "agent_description"
                        )

                        for key in (
                            "thread_id",
                            "session_id",
                            "conversation_id",
                        ):
                            conv_id = metadata.get(key)
                            if conv_id:
                                agent.conversation_id = conv_id
                                break

                    self._invocation_manager.add_invocation_state(
                        run_id, parent_run_id, agent
                    )
                else:
                    # We create invoke_agent span for the initial chain for agent. All follow-up chains invoked for agent invocation will not create agent span.
                    self._invocation_manager.add_invocation_state(
                        run_id, parent_run_id, None
                    )
            else:
                # No agent name could be resolved; still register the run_id so that
                # parent-child traversal (e.g. _find_nearest_agent) is not broken for
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
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if invocation is None or not isinstance(
            invocation, (WorkflowInvocation, AgentInvocation)
        ):
            # If the invocation does not exist, we cannot set attributes or end it
            self._invocation_manager.delete_invocation_state(run_id)
            return

        invocation.output_messages = make_last_output_message(outputs)

        invocation.stop()

        if not invocation.span.is_recording():
            self._invocation_manager.delete_invocation_state(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        invocation = self._invocation_manager.get_invocation(run_id=run_id)
        if invocation is None or not isinstance(
            invocation, (WorkflowInvocation, AgentInvocation)
        ):
            # If the invocation does not exist, we cannot set attributes or end it
            self._invocation_manager.delete_invocation_state(run_id)
            return

        invocation.fail(error)
        if not invocation.span.is_recording():
            self._invocation_manager.delete_invocation_state(run_id=run_id)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Other providers/LLMs may be supported in the future and telemetry for them is skipped for now.
        if serialized.get("name") not in ("ChatOpenAI", "ChatBedrock"):
            return

        if "invocation_params" in kwargs:
            params = (
                kwargs["invocation_params"].get("params")
                or kwargs["invocation_params"]
            )
        else:
            params = kwargs

        request_model = "unknown"
        for model_tag in (
            "model_name",  # ChatOpenAI
            "model_id",  # ChatBedrock
        ):
            if (model := (params or {}).get(model_tag)) is not None:
                request_model = model
                break
            elif (model := (metadata or {}).get(model_tag)) is not None:
                request_model = model
                break

        # Skip telemetry for unsupported request models
        if request_model == "unknown":
            return

        # Initialize variables with default values to avoid "possibly unbound" errors
        top_p = None
        frequency_penalty = None
        presence_penalty = None
        stop_sequences = None
        seed = None
        temperature = None
        max_tokens = None

        if params is not None:
            top_p = params.get("top_p")
            frequency_penalty = params.get("frequency_penalty")
            presence_penalty = params.get("presence_penalty")
            stop_sequences = params.get("stop")
            seed = params.get("seed")
            temperature = params.get("temperature")
            max_tokens = params.get("max_completion_tokens")

        provider = "unknown"
        if metadata is not None:
            provider = metadata.get("ls_provider", "unknown")

            # Override with ChatBedrock values if present
            if "ls_temperature" in metadata:
                temperature = metadata.get("ls_temperature")
            if "ls_max_tokens" in metadata:
                max_tokens = metadata.get("ls_max_tokens")

        input_messages: list[InputMessage] = []
        for sub_messages in messages:
            for message in sub_messages:
                # Cast to Any to avoid type checking issues with LangChain's complex content type
                raw_content: Any = message.content
                role = message.type
                parts: list[Text] = []

                if isinstance(raw_content, str):
                    parts = [Text(content=raw_content, type="text")]
                elif isinstance(raw_content, list):
                    for item in raw_content:  # type: ignore[misc]
                        if isinstance(item, str):
                            parts.append(Text(content=item, type="text"))
                        elif isinstance(item, dict):
                            # Safely extract text content from dict
                            text_value = item.get("text")  # type: ignore[misc]
                            if isinstance(text_value, str) and text_value:
                                parts.append(
                                    Text(content=text_value, type="text")
                                )

                input_messages.append(
                    InputMessage(
                        parts=cast(list[MessagePart], parts), role=role
                    )
                )

        llm_invocation = self._telemetry_handler.inference(
            provider,
            request_model=request_model,
        )
        llm_invocation.input_messages = input_messages
        llm_invocation.top_p = top_p
        llm_invocation.frequency_penalty = frequency_penalty
        llm_invocation.presence_penalty = presence_penalty
        llm_invocation.stop_sequences = stop_sequences
        llm_invocation.seed = seed
        llm_invocation.temperature = temperature
        llm_invocation.max_tokens = max_tokens
        self._invocation_manager.add_invocation_state(
            run_id=run_id,
            parent_run_id=parent_run_id,
            invocation=llm_invocation,
        )

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
        for generation in getattr(response, "generations", []):
            for chat_generation in generation:
                # Get finish reason
                finish_reason = "unknown"  # Default value
                generation_info = getattr(
                    chat_generation, "generation_info", None
                )
                if generation_info is not None:
                    finish_reason = generation_info.get(
                        "finish_reason", "unknown"
                    )

                if chat_generation.message:
                    # Get finish reason if generation_info is None above
                    if (
                        generation_info is None
                        and chat_generation.message.response_metadata
                    ):
                        finish_reason = (
                            chat_generation.message.response_metadata.get(
                                "stopReason", "unknown"
                            )
                        )

                    # Get message content
                    parts = [
                        Text(
                            content=chat_generation.message.content,
                            type="text",
                        )
                    ]
                    role = chat_generation.message.type
                    output_message = OutputMessage(
                        role=role,
                        parts=cast(list[MessagePart], parts),
                        finish_reason=finish_reason,
                    )
                    output_messages.append(output_message)

                    # Get token usage if available
                    if chat_generation.message.usage_metadata:
                        input_tokens = (
                            chat_generation.message.usage_metadata.get(
                                "input_tokens", 0
                            )
                        )
                        llm_invocation.input_tokens = input_tokens

                        output_tokens = (
                            chat_generation.message.usage_metadata.get(
                                "output_tokens", 0
                            )
                        )
                        llm_invocation.output_tokens = output_tokens

        llm_invocation.output_messages = output_messages

        llm_output = getattr(response, "llm_output", None)
        if llm_output is not None:
            response_model = llm_output.get("model_name") or llm_output.get(
                "model"
            )
            if response_model is not None:
                llm_invocation.response_model_name = str(response_model)

            response_id = llm_output.get("id")
            if response_id is not None:
                llm_invocation.response_id = str(response_id)

        llm_invocation.stop()
        if not llm_invocation.span.is_recording():
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
        if not llm_invocation.span.is_recording():
            self._invocation_manager.delete_invocation_state(run_id=run_id)

    def _find_nearest_agent(
        self, run_id: Optional[UUID]
    ) -> Optional[AgentInvocation]:
        current = run_id
        visited: set[UUID] = set()
        while current is not None and current not in visited:
            visited.add(current)
            entity = self._invocation_manager.get_invocation(current)
            if isinstance(entity, AgentInvocation):
                return entity
            current = self._invocation_manager.get_parent_run_id(current)
        return None
