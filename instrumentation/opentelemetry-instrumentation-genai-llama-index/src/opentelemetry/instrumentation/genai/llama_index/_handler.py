# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import inspect
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextvars import ContextVar, Token
from mimetypes import guess_type
from typing import Any, cast
from weakref import WeakKeyDictionary

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.multi_agent_workflow import AgentWorkflow
from llama_index.core.agent.workflow.workflow_events import (
    AgentOutput,
    AgentSetup,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.base.base_retriever import BaseRetriever
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
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import BaseTool, FunctionTool, ToolOutput
from pydantic import PrivateAttr

from opentelemetry.context import Context, attach, detach
from opentelemetry.trace import set_span_in_context
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    GenAIInvocation,
    LocalAgentInvocation,
    RetrievalInvocation,
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
    Role,
    SystemInstructionPart,
    TextPart,
    ToolCallRequestPart,
    ToolDefinition,
    UriPart,
)

_ToolExecutionAttributes = tuple[str, str | None]
_AGENT_TOOL_ATTRIBUTES: ContextVar[
    dict[str, _ToolExecutionAttributes] | None
] = ContextVar("llama_index_agent_tool_attributes", default=None)
_ACTIVE_WORKFLOW_TOOL: ContextVar[tuple[str, ToolInvocation] | None] = (
    ContextVar("llama_index_active_workflow_tool", default=None)
)


_MEMBER_AGENT_CONTEXTS: MutableMapping[
    LocalAgentInvocation, contextvars.Context
] = WeakKeyDictionary()


def _start_member_agent(
    start: Callable[[], LocalAgentInvocation],
) -> LocalAgentInvocation:
    """Open a member-agent invocation inside a context this module owns.

    The invocation stays open across workflow steps, and each step runs in its
    own asyncio task. ``TelemetryHandler`` attaches the span to whatever context
    is current when the invocation starts, and that attachment can only be undone
    from the same context -- so the context is kept and reused to finish in.
    """
    context = contextvars.copy_context()
    invocation = context.run(start)
    _MEMBER_AGENT_CONTEXTS[invocation] = context
    return invocation


def _finish_member_agent(
    invocation: LocalAgentInvocation, error: BaseException | None = None
) -> None:
    """Finish a member-agent invocation in the context that started it."""

    def finish() -> None:
        if error is None:
            invocation.stop()
        else:
            invocation.fail(error)

    context = _MEMBER_AGENT_CONTEXTS.pop(invocation, None)
    if context is None:
        finish()
        return
    try:
        context.run(finish)
    except RuntimeError:
        # Already entered further up the stack; finishing is idempotent.
        finish()


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
            InputMessage(
                role=Role.USER.value, parts=[TextPart(content=user_message)]
            )
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


def _retrieval_query(bound_args: inspect.BoundArguments) -> str | None:
    """Extract text from either accepted LlamaIndex retrieval query form."""
    query = bound_args.arguments.get("str_or_query_bundle")
    if isinstance(query, str):
        return query
    if isinstance(query, QueryBundle):
        return query.query_str
    return None


def _retrieval_top_k(retriever: BaseRetriever) -> int | None:
    """Read the common top-k setting without requiring a retriever subtype."""
    try:
        top_k = getattr(retriever, "similarity_top_k", None)
    except BaseException:
        return None
    if isinstance(top_k, int) and not isinstance(top_k, bool):
        return top_k
    return None


def _retrieval_documents(
    result: object,
) -> list[dict[str, Any]] | None:
    """Convert retrieved LlamaIndex nodes to semconv document objects."""
    if not isinstance(result, Sequence):
        return None
    candidates = cast(Sequence[object], result)
    documents: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, NodeWithScore):
            continue
        try:
            document: dict[str, Any] = {
                "id": candidate.node_id,
                "content": candidate.node.get_content(),
            }
            if candidate.score is not None:
                document["score"] = candidate.score
            documents.append(document)
        except BaseException:
            continue
    # Preserve [] for a genuine empty result, but omit the attribute when a
    # non-empty result could not be converted into semantic-convention docs.
    if documents:
        return documents
    return [] if len(candidates) == 0 else None


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


def _workflow_tool_definitions(
    workflow: AgentWorkflow, agent: BaseWorkflowAgent
) -> list[ToolDefinition] | None:
    """Capture static tools plus AgentWorkflow's generated handoff tool.

    ``get_tools`` may perform retrieval, so observe only the workflow's
    deterministic handoff resolution here rather than triggering another
    retrieval pass solely for telemetry.
    """
    tools_value: object = getattr(agent, "tools", None)
    candidates: list[object] = (
        list(cast(Sequence[object], tools_value))
        if isinstance(tools_value, Sequence)
        else []
    )
    get_handoff_tool = getattr(workflow, "_get_handoff_tool", None)
    if callable(get_handoff_tool):
        try:
            handoff_tool = get_handoff_tool(agent)
        except Exception:
            handoff_tool = None
        if handoff_tool is not None:
            candidates.append(handoff_tool)
    definitions = [
        definition
        for candidate in candidates
        if (definition := _tool_definition(candidate)) is not None
    ]
    return definitions or None


def _set_agent_output(invocation: LocalAgentInvocation, result: Any) -> None:
    """Copy the final chat response out of LlamaIndex's workflow result."""
    output = getattr(result, "result", None)
    response = getattr(output, "response", None)
    if isinstance(response, ChatMessage):
        invocation.output_messages = [_output_message(response)]


def _set_agent_step_output(
    invocation: LocalAgentInvocation, result: Any
) -> None:
    """Copy a member agent's response out of an AgentWorkflow step."""
    if isinstance(result, AgentOutput):
        invocation.output_messages = [_output_message(result.response)]


def _set_return_direct_agent_output(
    invocation: LocalAgentInvocation, tool_output: ToolOutput
) -> None:
    """Record the response synthesized by a return-direct tool execution."""
    invocation.output_messages = [
        _output_message(
            ChatMessage(role="assistant", content=tool_output.content)
        )
    ]


def _agent_step_is_complete(result: Any) -> bool:
    """Return whether a workflow agent step produced a final response.

    ``response.blocks`` does not reliably contain the tool selections that
    drive the next workflow step; ``AgentOutput`` exposes those selections and
    retry messages explicitly.
    """
    if not isinstance(result, AgentOutput):
        return False
    if result.retry_messages:
        return False
    return not result.tool_calls


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
    _workflow_agents_by_run_id: dict[str, BaseWorkflowAgent] = PrivateAttr()
    _workflow_invocations_by_run_id: dict[str, LocalAgentInvocation] = (
        PrivateAttr()
    )
    _workflow_invocations_by_key: dict[
        tuple[str, str], LocalAgentInvocation
    ] = PrivateAttr()
    _workflow_agent_invocation: LocalAgentInvocation | None = PrivateAttr()
    _workflow_handoff: bool = PrivateAttr()
    _workflow_agent_context_token: Token[Context] | None = PrivateAttr()
    _tool_parent_context_token: Token[Context] | None = PrivateAttr()
    _workflow_tool_token: Token[tuple[str, ToolInvocation] | None] | None = (
        PrivateAttr()
    )
    _workflow_run_id: str | None = PrivateAttr()
    _workflow_tool_counts: dict[str, int] = PrivateAttr()
    _workflow_return_direct_runs: set[str] = PrivateAttr()
    _workflow_tool_errors: dict[str, BaseException] = PrivateAttr()
    _workflow_pending_handoffs: dict[str, LocalAgentInvocation] = PrivateAttr()

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
        workflow_tool_token: Token[tuple[str, ToolInvocation] | None]
        | None = None,
        workflow_agents: Mapping[str, BaseWorkflowAgent] | None = None,
        workflow_run_id: str | None = None,
        workflow_agent: BaseWorkflowAgent | None = None,
        workflow_agent_invocation: LocalAgentInvocation | None = None,
        workflow_handoff: bool = False,
        tool_parent_context_token: Token[Context] | None = None,
    ) -> None:
        """Create the adapter used by LlamaIndex's span-handler lifecycle."""
        super().__init__(id_=id_, parent_id=parent_id)
        self._invocation = invocation
        self._tool_attributes_token = tool_attributes_token
        self._workflow_tool_token = workflow_tool_token
        self._workflow_run_id = workflow_run_id
        self._workflow_agents = dict(workflow_agents or {})
        self._workflow_agents_by_run_id = {}
        self._workflow_invocations_by_run_id = {}
        self._workflow_invocations_by_key = {}
        self._workflow_agent_invocation = workflow_agent_invocation
        self._workflow_handoff = workflow_handoff
        self._workflow_agent_context_token = None
        self._tool_parent_context_token = tool_parent_context_token
        self._workflow_tool_counts = {}
        self._workflow_return_direct_runs = set()
        self._workflow_tool_errors = {}
        self._workflow_pending_handoffs = {}
        if workflow_run_id is not None and workflow_agent is not None:
            self.register_workflow_agent(workflow_run_id, workflow_agent)
        if (
            workflow_run_id is not None
            and workflow_agent_invocation is not None
        ):
            self._workflow_invocations_by_run_id[workflow_run_id] = (
                workflow_agent_invocation
            )

    def workflow_agent(self, name: str) -> BaseWorkflowAgent | None:
        """Return a member agent owned by this workflow invocation."""
        return self._workflow_agents.get(name)

    def register_workflow_agent(
        self, run_id: str, agent: BaseWorkflowAgent
    ) -> None:
        """Associate a workflow run with its currently executing agent."""
        self._workflow_agents_by_run_id[run_id] = agent

    def workflow_agent_for_run_id(
        self, run_id: str | None
    ) -> BaseWorkflowAgent | None:
        """Return the agent executing the current step for a workflow run."""
        if run_id is None:
            return None
        return self._workflow_agents_by_run_id.get(run_id)

    def workflow_invocation_for_run_id(
        self, run_id: str | None, agent_name: str | None = None
    ) -> LocalAgentInvocation | None:
        """Return the reusable member-agent invocation for a workflow run."""
        if self._workflow_agent_invocation is not None:
            return self._workflow_agent_invocation
        if run_id is None:
            return None
        if agent_name is not None:
            return self._workflow_invocations_by_key.get((run_id, agent_name))
        return self._workflow_invocations_by_run_id.get(run_id)

    def register_workflow_invocation(
        self, run_id: str, agent_name: str, invocation: LocalAgentInvocation
    ) -> None:
        """Keep one member-agent invocation open across workflow turns."""
        self._workflow_invocations_by_run_id[run_id] = invocation
        self._workflow_invocations_by_key[(run_id, agent_name)] = invocation

    def remove_workflow_invocation(
        self, invocation: LocalAgentInvocation
    ) -> None:
        """Forget a completed member invocation so a later turn can restart it."""
        for key, value in list(self._workflow_invocations_by_key.items()):
            if value is invocation:
                del self._workflow_invocations_by_key[key]
                run_id = key[0]
                if (
                    self._workflow_invocations_by_run_id.get(run_id)
                    is invocation
                ):
                    del self._workflow_invocations_by_run_id[run_id]

    def reset_tool_attributes(self) -> None:
        """Restore task-local tool metadata after an agent run finishes."""
        if self._tool_attributes_token is not None:
            try:
                _AGENT_TOOL_ATTRIBUTES.reset(self._tool_attributes_token)
            except ValueError:
                pass
            self._tool_attributes_token = None

    def reset_workflow_tool(self) -> None:
        """Stop exposing a workflow tool while its nested SDK call unwinds."""
        if self._workflow_tool_token is not None:
            try:
                _ACTIVE_WORKFLOW_TOOL.reset(self._workflow_tool_token)
            except ValueError:
                pass
            self._workflow_tool_token = None

    def reset_tool_parent_context(self) -> None:
        """Detach the agent context after the tool span has finished."""
        if self._tool_parent_context_token is not None:
            try:
                detach(self._tool_parent_context_token)
            except ValueError:
                pass
            self._tool_parent_context_token = None

    def expect_workflow_tools(self, run_id: str, count: int) -> None:
        """Record how many tool calls the agent's current turn requested.

        AgentWorkflow dispatches one ``ToolCall`` event per selection and runs
        them as separate steps, so counting the ``call_tool`` spans that have
        already opened would miss the ones still queued.
        """
        if count:
            self._workflow_tool_counts[run_id] = count
            self._workflow_return_direct_runs.discard(run_id)
            self._workflow_tool_errors.pop(run_id, None)
        else:
            self._workflow_tool_counts.pop(run_id, None)
            self._workflow_return_direct_runs.discard(run_id)
            self._workflow_tool_errors.pop(run_id, None)

    def set_return_direct_agent_output(
        self,
        run_id: str,
        invocation: LocalAgentInvocation,
        tool_output: ToolOutput,
    ) -> None:
        """Keep the first successful return-direct result to arrive."""
        if run_id in self._workflow_return_direct_runs:
            return
        self._workflow_return_direct_runs.add(run_id)
        _set_return_direct_agent_output(invocation, tool_output)

    def record_workflow_tool_error(
        self, run_id: str, error: BaseException
    ) -> None:
        """Preserve the first failure across a concurrent tool turn."""
        self._workflow_tool_errors.setdefault(run_id, error)

    def take_workflow_tool_error(self, run_id: str) -> BaseException | None:
        """Return and clear the failure recorded for a completed tool turn."""
        return self._workflow_tool_errors.pop(run_id, None)

    def release_workflow_tool(self, run_id: str | None) -> bool:
        """Release one completed tool and report whether the turn is drained."""
        if run_id is None:
            return False
        remaining = self._workflow_tool_counts.get(run_id, 0) - 1
        if remaining > 0:
            self._workflow_tool_counts[run_id] = remaining
            return False
        self._workflow_tool_counts.pop(run_id, None)
        self._workflow_return_direct_runs.discard(run_id)
        return True

    def set_pending_handoff(
        self, run_id: str, invocation: LocalAgentInvocation
    ) -> None:
        """Hold a handing-off agent open until its whole turn has drained."""
        self._workflow_pending_handoffs[run_id] = invocation

    def take_pending_handoff(self, run_id: str) -> LocalAgentInvocation | None:
        """Claim the handing-off agent owed a close, if there is one."""
        return self._workflow_pending_handoffs.pop(run_id, None)

    def activate_workflow_agent(self) -> None:
        """Make a resumed member-agent span current for this workflow step."""
        if self._workflow_agent_context_token is None:
            self._workflow_agent_context_token = attach(
                set_span_in_context(self._invocation.span)
            )

    def reset_workflow_agent(self) -> None:
        """Detach the temporary context used by a resumed agent step."""
        if self._workflow_agent_context_token is not None:
            detach(self._workflow_agent_context_token)
            self._workflow_agent_context_token = None

    def finalize_workflow_agents(
        self,
        error: BaseException | None = None,
        result: Any | None = None,
    ) -> None:
        """Finish member-agent spans left open when the workflow terminates."""
        if error is None and isinstance(
            (output := getattr(result, "result", None)), AgentOutput
        ):
            # ``early_stopping_method="generate"`` creates the final response
            # in ``parse_agent_output`` rather than another agent step.
            agent_name = output.current_agent_name
            for (
                _,
                name,
            ), invocation in self._workflow_invocations_by_key.items():
                if name == agent_name:
                    _set_agent_step_output(invocation, output)
                    break
        invocations: list[LocalAgentInvocation] = []
        for candidate in self._workflow_invocations_by_key.values():
            if all(candidate is not existing for existing in invocations):
                invocations.append(candidate)
        for agent_invocation in invocations:
            _finish_member_agent(agent_invocation, error)
        self._workflow_invocations_by_key.clear()
        self._workflow_invocations_by_run_id.clear()


class LlamaIndexSpanHandler(BaseSpanHandler[_LlamaIndexInvocation]):
    """Map LlamaIndex-owned agent, tool, and retrieval operations to spans."""

    _handler: TelemetryHandler = PrivateAttr()

    def __init__(self, handler: TelemetryHandler) -> None:
        """Initialize the bridge to ``opentelemetry-util-genai``."""
        super().__init__()
        self._handler = handler

    def _is_open_tool(self, invocation: ToolInvocation) -> bool:
        """Check that a task-local tool still belongs to this handler.

        ``BaseSpanHandler`` mutates ``open_spans`` under its lock from worker
        threads, so iterating it unguarded can raise ``RuntimeError``.
        """
        with self.lock:
            adapters = list(self.open_spans.values())
        return any(adapter._invocation is invocation for adapter in adapters)

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _LlamaIndexInvocation | None:
        """Start GenAI invocations for agents, tools, and retrievers.

        Provider inference is deliberately ignored so its own instrumentation
        can emit inference telemetry, and nested tool callbacks are deduplicated.
        """
        method_name = _method_name(id_)
        invocation: GenAIInvocation
        tool_attributes_token: (
            Token[dict[str, _ToolExecutionAttributes] | None] | None
        ) = None
        workflow_agents: Mapping[str, BaseWorkflowAgent] | None = None
        workflow_run_id: str | None = None
        workflow_agent: BaseWorkflowAgent | None = None
        workflow_agent_invocation: LocalAgentInvocation | None = None
        workflow_handoff = False
        member_agent_step = False
        workflow_tool_token: (
            Token[tuple[str, ToolInvocation] | None] | None
        ) = None
        tool_parent_context_token: Token[Context] | None = None

        if isinstance(instance, AgentWorkflow) and method_name == "run":
            capture_content = self._handler.should_capture_content()
            input_messages = (
                _agent_input(bound_args) if capture_content else []
            )
            workflow_agents = instance.agents
            workflow_name = getattr(instance, "workflow_name", None)
            default_workflow_name = (
                f"{type(instance).__module__}.{type(instance).__qualname__}"
            )
            if (
                not isinstance(workflow_name, str)
                or not workflow_name
                or workflow_name == default_workflow_name
            ):
                workflow_name = type(instance).__name__
            workflow_invocation = self._handler.workflow(name=workflow_name)
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
            agent_system_instruction: list[SystemInstructionPart] = (
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
            agent_invocation.system_instruction = agent_system_instruction
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
            tool_definitions = _workflow_tool_definitions(
                cast(AgentWorkflow, instance), agent
            )
            system_instruction: list[SystemInstructionPart] = (
                [TextPart(content=agent.system_prompt)]
                if capture_content and agent.system_prompt
                else []
            )
            workflow_run_id = (
                tags.get("llamaindex.run_id") if tags is not None else None
            )
            workflow_agent_invocation = (
                parent.workflow_invocation_for_run_id(
                    workflow_run_id, agent_name
                )
                if parent is not None
                else None
            )
            member_agent_step = True
            if workflow_agent_invocation is None:
                workflow_agent_invocation = _start_member_agent(
                    lambda: self._handler.invoke_local_agent(
                        request_model=request_model,
                        agent_name=agent_name,
                    )
                )
                workflow_agent_invocation.agent_description = agent_description
                workflow_agent_invocation.input_messages = input_messages
                workflow_agent_invocation.tool_definitions = tool_definitions
                workflow_agent_invocation.system_instruction = (
                    system_instruction
                )
                if parent is not None and workflow_run_id is not None:
                    parent.register_workflow_invocation(
                        workflow_run_id, agent_name, workflow_agent_invocation
                    )
            agent_invocation = workflow_agent_invocation
            invocation = agent_invocation
            if parent is not None:
                if workflow_run_id is not None:
                    parent.register_workflow_agent(workflow_run_id, agent)
            workflow_agent = agent
        elif isinstance(instance, BaseRetriever) and method_name in {
            "retrieve",
            "aretrieve",
        }:
            retrieval_invocation = self._handler.retrieval()
            retrieval_invocation.top_k = _retrieval_top_k(instance)
            if retrieval_invocation.should_capture_content:
                retrieval_invocation.query_text = _retrieval_query(bound_args)
            invocation = retrieval_invocation
        elif method_name == "call_tool" and isinstance(
            (tool_call := bound_args.arguments.get("ev")), ToolCall
        ):
            parent = self.open_spans.get(parent_span_id or "")
            active_agent = (
                parent.workflow_agent_for_run_id(
                    tags.get("llamaindex.run_id") if tags is not None else None
                )
                if parent is not None
                else None
            )
            active_invocation = (
                parent.workflow_invocation_for_run_id(
                    tags.get("llamaindex.run_id") if tags is not None else None
                )
                if parent is not None
                else None
            )
            if active_invocation is None and parent is not None:
                if isinstance(parent._invocation, LocalAgentInvocation):
                    active_invocation = parent._invocation
            workflow_run_id = (
                tags.get("llamaindex.run_id") if tags is not None else None
            )
            tool_type, tool_description = (
                _agent_tool_attributes(active_agent, tool_call.tool_name)
                if active_agent is not None
                else (None, None)
            )
            if tool_type is None:
                tool_type, tool_description = _agent_tool_attributes(
                    instance or bound_args.arguments.get("self"),
                    tool_call.tool_name,
                )
            if tool_type is None and tool_call.tool_name == "handoff":
                # AgentWorkflow's built-in handoff is emitted as a ToolCall,
                # although its generated tool metadata is not available here.
                tool_type = "function"
            # The member-agent span stays open across workflow steps, each of
            # which runs in its own asyncio task. Pass its context explicitly
            # so the tool span nests under the agent that requested the call.
            agent_context_token = (
                attach(set_span_in_context(active_invocation.span))
                if active_invocation is not None
                else None
            )
            try:
                tool_invocation = self._handler.tool(
                    tool_call.tool_name,
                    tool_type=tool_type,
                    agent_name=getattr(active_invocation, "_agent_name", None),
                )
            except BaseException:
                if agent_context_token is not None:
                    detach(agent_context_token)
                raise
            tool_parent_context_token = agent_context_token
            workflow_tool_token = _ACTIVE_WORKFLOW_TOOL.set(
                (tool_call.tool_name, tool_invocation)
            )
            tool_invocation.tool_call_id = tool_call.tool_id
            tool_invocation.tool_description = tool_description
            if tool_invocation.should_capture_content:
                tool_invocation.arguments = cast(
                    dict[str, Any], cast(Any, tool_call).tool_kwargs
                )
            invocation = tool_invocation
            if parent is not None and isinstance(
                parent._invocation, WorkflowInvocation
            ):
                workflow_agent_invocation = active_invocation
                workflow_handoff = tool_call.tool_name == "handoff"
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
                # The workflow callback identifies the tool by name only; the
                # nested FunctionTool call is the authoritative executing tool.
                parent._invocation.tool_description = (
                    instance.metadata.description or None
                )
                return None
            active_workflow_tool = _ACTIVE_WORKFLOW_TOOL.get()
            if active_workflow_tool is not None:
                active_tool_name, active_tool = active_workflow_tool
                if active_tool_name == instance.metadata.get_name() and (
                    self._is_open_tool(active_tool)
                ):
                    active_tool.tool_description = (
                        instance.metadata.description or None
                    )
                    return None
            metadata = instance.metadata
            tool_invocation = self._handler.tool(
                metadata.get_name(),
                tool_type="function",
            )
            tool_invocation.tool_description = metadata.description or None
            if tool_invocation.should_capture_content:
                tool_invocation.arguments = _tool_arguments(
                    instance, bound_args
                )
            invocation = tool_invocation
        else:
            return None

        adapter = _LlamaIndexInvocation(
            id_=id_,
            parent_id=parent_span_id,
            invocation=invocation,
            tool_attributes_token=tool_attributes_token,
            workflow_tool_token=workflow_tool_token,
            tool_parent_context_token=tool_parent_context_token,
            workflow_agents=workflow_agents,
            workflow_run_id=workflow_run_id,
            workflow_agent=workflow_agent,
            workflow_agent_invocation=workflow_agent_invocation,
            workflow_handoff=workflow_handoff,
        )
        if method_name == "run_agent_step" and member_agent_step:
            adapter.activate_workflow_agent()
        return adapter

    def _expect_workflow_tools(
        self, span: _LlamaIndexInvocation, result: Any
    ) -> None:
        """Record the tool calls a member agent's turn just requested."""
        run_id = span._workflow_run_id
        if run_id is None or not isinstance(result, AgentOutput):
            return
        parent = self.open_spans.get(span.parent_id or "")
        if parent is not None:
            parent.expect_workflow_tools(run_id, len(result.tool_calls))

    def _release_workflow_invocation(
        self,
        span: _LlamaIndexInvocation,
        invocation: LocalAgentInvocation,
    ) -> None:
        """Drop a finished member invocation so a later turn opens a new span."""
        parent = self.open_spans.get(span.parent_id or "")
        if parent is not None:
            parent.remove_workflow_invocation(invocation)

    def _finish_workflow_tool(
        self,
        span: _LlamaIndexInvocation,
        handoff_succeeded: bool = True,
        error: BaseException | None = None,
    ) -> None:
        """Release one tool of a member agent's turn and close the agent last.

        AgentWorkflow reports a handoff as a tool call made by the agent that is
        stepping down, and that turn can request other tools alongside it. The
        agent's span has to outlive every one of them, so it is closed only once
        the turn's last tool call ends.
        """
        run_id = span._workflow_run_id
        if run_id is None:
            return
        parent = self.open_spans.get(span.parent_id or "")
        if parent is None:
            return
        if error is not None:
            parent.record_workflow_tool_error(run_id, error)
        invocation = span._workflow_agent_invocation
        if (
            handoff_succeeded
            and span._workflow_handoff
            and invocation is not None
        ):
            parent.set_pending_handoff(run_id, invocation)
        if not parent.release_workflow_tool(run_id):
            return
        pending = parent.take_pending_handoff(run_id)
        if pending is not None:
            _finish_member_agent(
                pending, parent.take_workflow_tool_error(run_id)
            )
            parent.remove_workflow_invocation(pending)
        else:
            parent.take_workflow_tool_error(run_id)

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
            capture_content = self._handler.should_capture_content()
            if capture_content:
                _set_workflow_output(span._invocation, result)
            span.finalize_workflow_agents(
                result=result if capture_content else None
            )
        elif isinstance(span._invocation, LocalAgentInvocation):
            span.reset_tool_attributes()
            span.reset_workflow_tool()
            if self._handler.should_capture_content():
                if isinstance(result, AgentOutput):
                    _set_agent_step_output(span._invocation, result)
                else:
                    _set_agent_output(span._invocation, result)
            if span._workflow_agent_invocation is not None:
                span.reset_workflow_agent()
                if not _agent_step_is_complete(result):
                    self._expect_workflow_tools(span, result)
                return span
        elif isinstance(span._invocation, RetrievalInvocation):
            if span._invocation.should_capture_content:
                span._invocation.documents = _retrieval_documents(result)
        elif isinstance(span._invocation, ToolInvocation):
            span.reset_workflow_tool()
            tool_output: ToolOutput | None = None
            if isinstance(result, ToolCallResult):
                tool_output = result.tool_output
            elif isinstance(result, ToolOutput):
                tool_output = result
            if tool_output is not None:
                if (
                    isinstance(result, ToolCallResult)
                    and result.return_direct
                    and not tool_output.is_error
                    and span._workflow_agent_invocation is not None
                    and self._handler.should_capture_content()
                ):
                    parent = self.open_spans.get(span.parent_id or "")
                    if (
                        parent is not None
                        and span._workflow_run_id is not None
                    ):
                        parent.set_return_direct_agent_output(
                            span._workflow_run_id,
                            span._workflow_agent_invocation,
                            tool_output,
                        )
                if span._invocation.should_capture_content:
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
                    span.reset_tool_parent_context()
                    self._finish_workflow_tool(
                        span,
                        handoff_succeeded=False,
                        error=error,
                    )
                    return span
        span._invocation.stop()
        span.reset_tool_parent_context()
        if isinstance(span._invocation, ToolInvocation):
            self._finish_workflow_tool(span)
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
        span.reset_workflow_tool()
        if isinstance(span._invocation, WorkflowInvocation):
            span.finalize_workflow_agents(err)
            if err is None:
                span._invocation.stop()
            else:
                span._invocation.fail(err)
        elif isinstance(span._invocation, LocalAgentInvocation):
            _finish_member_agent(span._invocation, err)
        elif err is None:
            span._invocation.stop()
        else:
            span._invocation.fail(err)
        span.reset_tool_parent_context()
        if isinstance(span._invocation, LocalAgentInvocation):
            span.reset_workflow_agent()
            self._release_workflow_invocation(span, span._invocation)
        elif isinstance(span._invocation, ToolInvocation):
            self._finish_workflow_tool(
                span,
                handoff_succeeded=err is None,
                error=err,
            )
        return span
