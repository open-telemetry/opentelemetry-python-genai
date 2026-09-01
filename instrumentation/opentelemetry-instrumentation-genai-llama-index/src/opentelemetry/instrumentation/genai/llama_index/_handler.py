# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from mimetypes import guess_type
from typing import Any, cast

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.multi_agent_workflow import AgentWorkflow
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    AgentSetup,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput
from pydantic import PrivateAttr

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    GenAIInvocation,
    ToolInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    ToolDefinition,
    UriPart,
)

_ToolExecutionAttributes = tuple[str, str | None]
_AGENT_TOOL_ATTRIBUTES: ContextVar[
    dict[str, _ToolExecutionAttributes] | None
] = ContextVar("llama_index_agent_tool_attributes", default=None)


def _method_name(span_id: str) -> str:
    """Extract the method used to route a LlamaIndex dispatcher span.

    Dispatcher IDs have the form ``Class.method-UUID``; the UUID makes the
    full ID unsuitable for matching agent and tool operations.
    """
    return span_id.partition("-")[0].rsplit(".", 1)[-1]


def _chat_message_parts(message: ChatMessage) -> list[MessagePart]:
    """Convert LlamaIndex content blocks into semconv message parts.

    Keeping the conversion in one place preserves structured tool calls and
    multimodal content for both input and output messages.
    """
    parts: list[MessagePart] = []
    for block in message.blocks:
        if isinstance(block, TextBlock) and block.text:
            parts.append(TextPart(content=block.text))
        elif isinstance(block, ToolCallBlock):
            parts.append(
                ToolCallRequestPart(
                    arguments=block.tool_kwargs,
                    name=block.tool_name,
                    id=block.tool_call_id,
                )
            )
        elif isinstance(block, ThinkingBlock) and block.content:
            parts.append(ReasoningPart(content=block.content))
        elif isinstance(block, ImageBlock):
            part = _media_part(
                data=block.image,
                path=block.path,
                url=block.url,
                mime_type=block.image_mimetype,
                modality="image",
            )
            if part is not None:
                parts.append(part)
        elif isinstance(block, AudioBlock):
            part = _media_part(
                data=block.audio,
                path=block.path,
                url=block.url,
                mime_type=_audio_mime_type(block.format),
                modality="audio",
            )
            if part is not None:
                parts.append(part)
        elif isinstance(block, DocumentBlock):
            part = _media_part(
                data=block.data,
                path=block.path,
                url=block.url,
                mime_type=block.document_mimetype,
                modality="document",
            )
            if part is not None:
                parts.append(part)
    return parts


def _media_part(
    *,
    data: object,
    path: object,
    url: object,
    mime_type: str | None,
    modality: str,
) -> MessagePart | None:
    """Represent inline or referenced media as a semconv message part.

    LlamaIndex can carry media as normalized base64 bytes, a URL, or a local
    path, while the GenAI model distinguishes embedded blobs from URIs.
    """
    if isinstance(data, bytes):
        # LlamaIndex normalizes inline media to base64 bytes during validation.
        try:
            return BlobPart(
                content=b64decode(data, validate=True),
                mime_type=mime_type,
                modality=modality,
            )
        except (BinasciiError, ValueError):
            pass
    reference = url or path
    if reference is not None:
        return UriPart(
            uri=str(reference),
            mime_type=mime_type,
            modality=modality,
        )
    return None


def _audio_mime_type(format_: str | None) -> str | None:
    """Normalize LlamaIndex's audio format into a MIME type when possible."""
    if not format_ or "/" in format_:
        return format_
    return guess_type(f"file.{format_}")[0] or f"audio/{format_}"


def _input_message(message: ChatMessage) -> InputMessage:
    """Map a LlamaIndex chat message to a semconv input message."""
    return InputMessage(
        role=message.role.value,
        parts=_chat_message_parts(message),
    )


def _output_message(message: ChatMessage) -> OutputMessage:
    """Map an assistant message and its tool-call state to semconv output."""
    return OutputMessage(
        role=message.role.value,
        parts=_chat_message_parts(message),
        finish_reason=(
            "tool_calls"
            if any(
                isinstance(block, ToolCallBlock) for block in message.blocks
            )
            else "stop"
        ),
    )


def _agent_input(bound_args: inspect.BoundArguments) -> list[InputMessage]:
    """Recover agent input messages from LlamaIndex's workflow start event.

    ``BaseWorkflowAgent.run`` normalizes the current message and chat history
    into ``start_event``, so reading ordinary call arguments would miss them.
    """
    start_event = bound_args.arguments.get("start_event")
    if start_event is None:
        return []

    history_value: object = start_event.get("chat_history", None)
    history: Sequence[object] = (
        cast(Sequence[object], history_value)
        if isinstance(history_value, Sequence)
        else cast(Sequence[object], ())
    )
    messages = [
        _input_message(message)
        for message in history or []
        if isinstance(message, ChatMessage)
    ]
    user_message = start_event.get("user_msg", None)
    if isinstance(user_message, ChatMessage):
        messages.append(_input_message(user_message))
    elif isinstance(user_message, str) and user_message:
        messages.append(
            InputMessage(role="user", parts=[TextPart(content=user_message)])
        )
    return messages


def _agent_step_input(
    event: AgentSetup, system_prompt: str | None
) -> list[InputMessage]:
    """Recover the member agent input from an AgentWorkflow step.

    AgentWorkflow prepends the member's system prompt to ``AgentSetup.input``;
    it is captured separately as the agent's system instruction.
    """
    messages = list(event.input)
    if (
        system_prompt
        and messages
        and messages[0].role.value == "system"
        and messages[0].content == system_prompt
    ):
        messages.pop(0)
    return [_input_message(message) for message in messages]


def _request_model(agent: BaseWorkflowAgent) -> str | None:
    """Best-effort extraction of the model name across LLM integrations."""
    try:
        model_name = agent.llm.metadata.model_name
    except Exception:  # LLM integrations can compute metadata dynamically.
        model_name = getattr(agent.llm, "model", None)
    return model_name if isinstance(model_name, str) and model_name else None


def _tool_attributes(
    candidate: object,
) -> tuple[str, str, str | None] | None:
    """Return the semconv name, type, and description for a tool.

    Tool metadata is user-extensible and may raise or be incomplete; skipping
    invalid metadata keeps telemetry from breaking the agent invocation.
    """
    if not isinstance(candidate, BaseTool):
        return None
    try:
        metadata = candidate.metadata
        name = metadata.name
        if not name:
            return None
        description = metadata.description or None
    except Exception:
        return None
    tool_type = (
        "function"
        if isinstance(candidate, FunctionTool)
        else type(candidate).__name__
    )
    return name, tool_type, description


def _tool_definition(candidate: object) -> ToolDefinition | None:
    """Convert a usable LlamaIndex tool into its semconv definition."""
    attributes = _tool_attributes(candidate)
    if attributes is None:
        return None
    name, tool_type, description = attributes
    if not isinstance(candidate, FunctionTool):
        return GenericToolDefinition(name=name, type=tool_type)
    try:
        parameters = cast(
            dict[str, Any],
            cast(Any, candidate.metadata).get_parameters_dict(),
        )
    except Exception:
        return None
    return FunctionToolDefinition(
        name=name,
        description=description,
        type="function",
        parameters=parameters,
    )


def _agent_tool_attributes(
    agent: object, tool_name: str
) -> tuple[str | None, str | None]:
    """Find execution attributes for a named tool exposed by an agent."""
    try:
        tools = cast(Sequence[object], cast(Any, agent).tools or ())
    except Exception:
        tools = ()
    for candidate in tools:
        attributes = _tool_attributes(candidate)
        if attributes is not None and attributes[0] == tool_name:
            _, tool_type, description = attributes
            return tool_type, description
    contextual_attributes = _AGENT_TOOL_ATTRIBUTES.get()
    if contextual_attributes is None:
        return None, None
    return contextual_attributes.get(tool_name, (None, None))


def _agent_tool_attribute_map(
    agent: BaseWorkflowAgent,
) -> dict[str, _ToolExecutionAttributes]:
    """Keep agent tool metadata available to callbacks without an instance."""
    attributes_by_name: dict[str, _ToolExecutionAttributes] = {}
    for candidate in cast(Sequence[object], cast(Any, agent).tools or ()):
        attributes = _tool_attributes(candidate)
        if attributes is not None:
            name, tool_type, description = attributes
            attributes_by_name[name] = tool_type, description
    return attributes_by_name


def _tool_definitions(agent: BaseWorkflowAgent) -> list[ToolDefinition] | None:
    """Collect valid agent tool metadata for every content-capture mode."""
    definitions = [
        definition
        for candidate in cast(Sequence[object], cast(Any, agent).tools or ())
        if (definition := _tool_definition(candidate)) is not None
    ]
    return definitions or None


def _set_agent_output(invocation: AgentInvocation, result: Any) -> None:
    """Copy the final chat response out of LlamaIndex's workflow result."""
    output = getattr(result, "result", None)
    response = getattr(output, "response", None)
    if isinstance(response, ChatMessage):
        invocation.output_messages = [_output_message(response)]


def _set_agent_step_output(invocation: AgentInvocation, result: Any) -> None:
    """Copy a member agent's response out of an AgentWorkflow step."""
    if isinstance(result, AgentOutput):
        invocation.output_messages = [_output_message(result.response)]


def _set_workflow_output(invocation: WorkflowInvocation, result: Any) -> None:
    """Copy the final response out of an AgentWorkflow stop event."""
    output = getattr(result, "result", None)
    response = getattr(output, "response", None)
    if isinstance(response, ChatMessage):
        invocation.output_messages = [_output_message(response)]


def _tool_arguments(
    tool: FunctionTool, bound_args: inspect.BoundArguments
) -> dict[str, Any]:
    """Bind tool arguments to user-facing parameter names.

    LlamaIndex exposes positional values under ``args`` and may inject a
    workflow context parameter, neither of which should appear in telemetry.
    """
    positional = bound_args.arguments.get("args")
    args = (
        tuple(cast(Sequence[Any], positional))
        if isinstance(positional, Sequence)
        else ()
    )
    keyword = bound_args.arguments.get("kwargs")
    kwargs: dict[str, Any] = {}
    if isinstance(keyword, Mapping):
        kwargs.update(cast(Mapping[str, Any], keyword))
    try:
        arguments = dict(
            inspect.signature(tool.real_fn)
            .bind_partial(*args, **kwargs)
            .arguments
        )
    except (TypeError, ValueError):
        arguments = {"args": list(args), **kwargs}
    if tool.ctx_param_name:
        arguments.pop(tool.ctx_param_name, None)
    return arguments


class _LlamaIndexInvocation(BaseSpan):
    """Pair a LlamaIndex span ID with the GenAI invocation it controls."""

    _invocation: GenAIInvocation = PrivateAttr()
    _tool_attributes_token: (
        Token[dict[str, _ToolExecutionAttributes] | None] | None
    ) = PrivateAttr()
    _workflow_agents: dict[str, BaseWorkflowAgent] = PrivateAttr()

    def __init__(
        self,
        *,
        id_: str,
        parent_id: str | None,
        invocation: GenAIInvocation,
        tool_attributes_token: Token[
            dict[str, _ToolExecutionAttributes] | None
        ]
        | None = None,
        workflow_agents: Mapping[str, BaseWorkflowAgent] | None = None,
    ) -> None:
        """Create the adapter used by LlamaIndex's span-handler lifecycle."""
        super().__init__(id_=id_, parent_id=parent_id)
        self._invocation = invocation
        self._tool_attributes_token = tool_attributes_token
        self._workflow_agents = dict(workflow_agents or {})

    def workflow_agent(self, name: str) -> BaseWorkflowAgent | None:
        """Return a member agent owned by this workflow invocation."""
        return self._workflow_agents.get(name)

    def workflow_tool_attributes(
        self, name: str
    ) -> tuple[str | None, str | None]:
        """Find a configured tool exposed by a workflow member agent."""
        for agent in self._workflow_agents.values():
            tool_type, description = _agent_tool_attributes(agent, name)
            if tool_type is not None:
                return tool_type, description
        return None, None

    def reset_tool_attributes(self) -> None:
        """Restore task-local tool metadata after an agent run finishes."""
        if self._tool_attributes_token is not None:
            try:
                _AGENT_TOOL_ATTRIBUTES.reset(self._tool_attributes_token)
            except ValueError:
                pass
            self._tool_attributes_token = None


class LlamaIndexSpanHandler(BaseSpanHandler[_LlamaIndexInvocation]):
    """Map LlamaIndex-owned agent and tool operations to GenAI spans."""

    _handler: TelemetryHandler = PrivateAttr()

    def __init__(self, handler: TelemetryHandler) -> None:
        """Initialize the bridge to ``opentelemetry-util-genai``."""
        super().__init__()
        self._handler = handler

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexInvocation | None:
        """Start GenAI invocations for LlamaIndex-owned agents and tools.

        Provider inference is deliberately ignored so its own instrumentation
        can emit inference telemetry, and nested tool callbacks are deduplicated.
        """
        method_name = _method_name(id_)
        invocation: GenAIInvocation
        tool_attributes_token: (
            Token[dict[str, _ToolExecutionAttributes] | None] | None
        ) = None
        workflow_agents: Mapping[str, BaseWorkflowAgent] | None = None

        if isinstance(instance, AgentWorkflow) and method_name == "run":
            capture_content = self._handler.should_capture_content()
            input_messages = (
                _agent_input(bound_args) if capture_content else []
            )
            workflow_agents = instance.agents
            workflow_invocation = self._handler.workflow(
                name=type(instance).__name__
            )
            workflow_invocation.input_messages = input_messages
            invocation = workflow_invocation
        elif isinstance(instance, BaseWorkflowAgent) and method_name == "run":
            capture_content = self._handler.should_capture_content()
            agent_name = instance.name or type(instance).__name__
            request_model = _request_model(instance)
            agent_description = instance.description
            input_messages = (
                _agent_input(bound_args) if capture_content else []
            )
            tool_definitions = _tool_definitions(instance)
            system_prompt = instance.system_prompt
            system_instruction: list[MessagePart] = (
                [TextPart(content=system_prompt)]
                if capture_content and system_prompt
                else []
            )
            agent_invocation = self._handler.invoke_local_agent(
                request_model=request_model,
                agent_name=agent_name,
            )
            agent_invocation.agent_description = agent_description
            agent_invocation.input_messages = input_messages
            agent_invocation.tool_definitions = tool_definitions
            agent_invocation.system_instruction = system_instruction
            invocation = agent_invocation
            tool_attributes_token = _AGENT_TOOL_ATTRIBUTES.set(
                _agent_tool_attribute_map(instance)
            )
        elif method_name == "run_agent_step" and isinstance(
            (agent_setup := bound_args.arguments.get("ev")), AgentSetup
        ):
            parent = self.open_spans.get(parent_span_id or "")
            agent = (
                parent.workflow_agent(agent_setup.current_agent_name)
                if parent is not None
                else None
            )
            if agent is None:
                return None
            capture_content = self._handler.should_capture_content()
            agent_name = agent.name or type(agent).__name__
            request_model = _request_model(agent)
            agent_description = agent.description
            input_messages = (
                _agent_step_input(agent_setup, agent.system_prompt)
                if capture_content
                else []
            )
            tool_definitions = _tool_definitions(agent)
            system_instruction: list[MessagePart] = (
                [TextPart(content=agent.system_prompt)]
                if capture_content and agent.system_prompt
                else []
            )
            agent_invocation = self._handler.invoke_local_agent(
                request_model=request_model,
                agent_name=agent_name,
            )
            agent_invocation.agent_description = agent_description
            agent_invocation.input_messages = input_messages
            agent_invocation.tool_definitions = tool_definitions
            agent_invocation.system_instruction = system_instruction
            invocation = agent_invocation
        elif method_name == "call_tool" and isinstance(
            (tool_call := bound_args.arguments.get("ev")), ToolCall
        ):
            parent = self.open_spans.get(parent_span_id or "")
            tool_type, tool_description = (
                parent.workflow_tool_attributes(tool_call.tool_name)
                if parent is not None
                else (None, None)
            )
            if tool_type is None:
                tool_type, tool_description = _agent_tool_attributes(
                    instance or bound_args.arguments.get("self"),
                    tool_call.tool_name,
                )
            tool_invocation = self._handler.tool(
                tool_call.tool_name,
                tool_call_id=tool_call.tool_id,
                tool_type=tool_type,
                tool_description=tool_description,
            )
            if tool_invocation.should_capture_content_on_span:
                tool_invocation.arguments = cast(
                    dict[str, Any], cast(Any, tool_call).tool_kwargs
                )
            invocation = tool_invocation
        elif isinstance(instance, FunctionTool) and method_name in {
            "call",
            "acall",
        }:
            parent = self.open_spans.get(parent_span_id or "")
            # LlamaIndex reports an agent tool execution through both call_tool
            # and the nested FunctionTool.call/acall; the parent records it.
            if parent is not None and isinstance(
                parent._invocation, ToolInvocation
            ):
                return None
            metadata = instance.metadata
            tool_invocation = self._handler.tool(
                metadata.get_name(),
                tool_type="function",
                tool_description=metadata.description or None,
            )
            if tool_invocation.should_capture_content_on_span:
                tool_invocation.arguments = _tool_arguments(
                    instance, bound_args
                )
            invocation = tool_invocation
        else:
            return None

        return _LlamaIndexInvocation(
            id_=id_,
            parent_id=parent_span_id,
            invocation=invocation,
            tool_attributes_token=tool_attributes_token,
            workflow_agents=workflow_agents,
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexInvocation | None:
        """Finalize successful dispatcher spans with agent or tool results.

        LlamaIndex can return a failed ``ToolOutput`` instead of raising, so
        tool-result inspection is required to assign the correct span status.
        """
        span = self.open_spans.get(id_)
        if span is None:
            return None
        if isinstance(span._invocation, WorkflowInvocation):
            if self._handler.should_capture_content():
                _set_workflow_output(span._invocation, result)
        elif isinstance(span._invocation, AgentInvocation):
            span.reset_tool_attributes()
            if self._handler.should_capture_content():
                if isinstance(result, AgentOutput):
                    _set_agent_step_output(span._invocation, result)
                else:
                    _set_agent_output(span._invocation, result)
        elif isinstance(span._invocation, ToolInvocation):
            tool_output: ToolOutput | None = None
            if isinstance(result, ToolCallResult):
                tool_output = result.tool_output
            elif isinstance(result, ToolOutput):
                tool_output = result
            if tool_output is not None:
                if span._invocation.should_capture_content_on_span:
                    span._invocation.tool_result = tool_output.raw_output
                if tool_output.is_error:
                    # LlamaIndex reports failures such as unknown tools without an
                    # exception, so provide one to record error telemetry:
                    # https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/agent/workflow/base_agent.py
                    error = (
                        tool_output.exception
                        if isinstance(tool_output.exception, BaseException)
                        else RuntimeError(tool_output.content)
                    )
                    span._invocation.fail(error)
                    return span
        span._invocation.stop()
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexInvocation | None:
        """Finalize a dropped dispatcher span with its original exception."""
        span = self.open_spans.get(id_)
        if span is None:
            return None
        span.reset_tool_attributes()
        if err is None:
            span._invocation.stop()
        else:
            span._invocation.fail(err)
        return span
