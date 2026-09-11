# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Agno instrumentation."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Sequence,
)
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agno.agent import RunOutput
    from agno.knowledge.document.base import Document
    from agno.knowledge.knowledge import Knowledge
    from agno.run.workflow import WorkflowRunOutput
    from agno.team import TeamRunOutput
    from agno.tools.function import FunctionCall, FunctionExecutionResult

    AgnoRunOutput = RunOutput | TeamRunOutput | WorkflowRunOutput

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.instrumentation.genai.agno.stream import (
    AgnoAgentStreamWrapper,
    AgnoToolStreamWrapper,
    AgnoWorkflowStreamWrapper,
    AsyncAgnoAgentStreamWrapper,
    AsyncAgnoToolStreamWrapper,
    AsyncAgnoWorkflowStreamWrapper,
)
from opentelemetry.instrumentation.genai.agno.utils import (
    _get_property_value,
    format_content,
    format_retrieval_document,
    prepare_tool_definitions,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ErrorTypeValues,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    RetrievalInvocation,
    ToolInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    OutputMessage,
    Role,
    TextPart,
)

logger = logging.getLogger(__name__)

_AGNO_MODULE = "agno.agent"
_AGENT_CLASS = "Agent"
_AGNO_TEAM_MODULE = "agno.team"
_TEAM_CLASS = "Team"
_AGNO_TOOLS_MODULE = "agno.tools.function"
_FUNCTION_CALL_CLASS = "FunctionCall"
_AGNO_WORKFLOW_MODULE = "agno.workflow.workflow"
_WORKFLOW_CLASS = "Workflow"
_AGNO_KNOWLEDGE_MODULE = "agno.knowledge.knowledge"
_KNOWLEDGE_CLASS = "Knowledge"


# wrapt has no unregister API for post-import hooks; monotonic generations
# invalidate deferred hooks registered during prior instrumentation cycles.
_instrumentation_generation: int = 0
_is_instrumented: bool = False


def _safe_wrap_function(
    target_module: str,
    target_name: str,
    wrapper: Callable[..., Any],
    generation: int,
) -> None:
    """Safely wrap a method if it exists, deferring if module is not yet imported."""

    def _apply(mod: Any) -> None:
        if not _is_instrumented or _instrumentation_generation != generation:
            return
        try:
            parts = target_name.split(".")
            curr = mod
            for part in parts:
                curr = getattr(curr, part)
        except AttributeError:
            # Target class or method may not exist across all supported Agno versions.
            return
        if hasattr(curr, "__wrapped__"):
            return
        wrap_function_wrapper(mod, target_name, wrapper)

    if target_module in sys.modules:
        _apply(sys.modules[target_module])
        return

    # Defer wrapping to avoid eagerly importing submodules with heavy or optional dependencies.
    register_post_import_hook(_apply, target_module)


def patch_agent(handler: TelemetryHandler) -> None:
    """Apply patches to Agno class methods."""
    global _instrumentation_generation, _is_instrumented
    _instrumentation_generation += 1
    _is_instrumented = True
    current_generation = _instrumentation_generation

    _safe_wrap_function(
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.run",
        _agent_run(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.arun",
        _agent_arun(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_TEAM_MODULE,
        f"{_TEAM_CLASS}.run",
        _agent_run(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_TEAM_MODULE,
        f"{_TEAM_CLASS}.arun",
        _agent_arun(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_TOOLS_MODULE,
        f"{_FUNCTION_CALL_CLASS}.execute",
        _tool_call_execute(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_TOOLS_MODULE,
        f"{_FUNCTION_CALL_CLASS}.aexecute",
        _tool_call_aexecute(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_WORKFLOW_MODULE,
        f"{_WORKFLOW_CLASS}.run",
        _workflow_run(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_WORKFLOW_MODULE,
        f"{_WORKFLOW_CLASS}.arun",
        _workflow_arun(handler),
        current_generation,
    )
    # Knowledge.retrieve and aretrieve delegate to search and asearch, so wrapping
    # search/asearch avoids duplicate spans.
    _safe_wrap_function(
        _AGNO_KNOWLEDGE_MODULE,
        f"{_KNOWLEDGE_CLASS}.search",
        _knowledge_search(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_KNOWLEDGE_MODULE,
        f"{_KNOWLEDGE_CLASS}.asearch",
        _knowledge_asearch(handler),
        current_generation,
    )


def unpatch_agent() -> None:
    """Remove patches from Agno class methods."""
    global _instrumentation_generation, _is_instrumented
    _instrumentation_generation += 1
    _is_instrumented = False
    if _AGNO_MODULE in sys.modules:
        try:
            import agno.agent

            unwrap(agno.agent.Agent, "run")
            unwrap(agno.agent.Agent, "arun")
        except (ImportError, AttributeError):
            pass
    if _AGNO_TEAM_MODULE in sys.modules:
        try:
            import agno.team

            unwrap(agno.team.Team, "run")
            unwrap(agno.team.Team, "arun")
        except (ImportError, AttributeError):
            pass
    if _AGNO_TOOLS_MODULE in sys.modules:
        try:
            import agno.tools.function

            unwrap(agno.tools.function.FunctionCall, "execute")
            unwrap(agno.tools.function.FunctionCall, "aexecute")
        except (ImportError, AttributeError):
            pass
    if _AGNO_WORKFLOW_MODULE in sys.modules:
        try:
            import agno.workflow.workflow

            unwrap(agno.workflow.workflow.Workflow, "run")
            unwrap(agno.workflow.workflow.Workflow, "arun")
        except (ImportError, AttributeError):
            pass
    if _AGNO_KNOWLEDGE_MODULE in sys.modules:
        try:
            import agno.knowledge.knowledge

            unwrap(agno.knowledge.knowledge.Knowledge, "search")
            unwrap(agno.knowledge.knowledge.Knowledge, "asearch")
        except (ImportError, AttributeError):
            pass


def _extract_input_content(input_val: Any) -> str:
    if input_val is None:
        return ""
    content = _get_property_value(input_val, "content")
    if content is not None:
        return format_content(content)
    return format_content(input_val)


def _extract_output_content(result: Any) -> str:
    if result is None:
        return ""
    content = _get_property_value(result, "content")
    if content is not None:
        return format_content(content)
    val = _get_property_value(result, "result")
    if val is not None:
        return format_content(val)
    return format_content(result)


def _extract_arguments_str(args_val: Any) -> str:
    return format_content(args_val)


def _set_tool_invocation_input(
    invocation: ToolInvocation,
    instance: FunctionCall,
    capture_content: bool,
) -> None:
    if capture_content:
        arguments = instance.arguments
        if arguments is not None:
            invocation.arguments = _extract_arguments_str(arguments)


def _fail_tool_invocation(
    invocation: ToolInvocation,
    result: FunctionExecutionResult,
) -> None:
    error = result.error
    invocation.fail(
        Error(
            type=ErrorTypeValues.OTHER.value,
            message=str(error) if error else None,
        )
    )


def _set_tool_invocation_output(
    invocation: ToolInvocation,
    result: FunctionExecutionResult,
    capture_content: bool,
) -> None:
    if result.status == "failure":
        return
    if capture_content:
        invocation.tool_result = _extract_output_content(result)


def _set_invocation_input(
    invocation: AgentInvocation | WorkflowInvocation,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> None:
    if capture_content and (args or "input" in kwargs):
        input_val = args[0] if args else kwargs.get("input")
        if input_val is not None:
            content_str = _extract_input_content(input_val)
            invocation.input_messages = [
                InputMessage(
                    role=Role.USER.value, parts=[TextPart(content=content_str)]
                )
            ]


def _extract_finish_reason(result: object) -> str:
    if "error" in str(getattr(result, "status", "")).lower():
        return "error"
    return "stop"


def _set_invocation_output(
    invocation: AgentInvocation | WorkflowInvocation,
    result: object | None,
    capture_content: bool,
) -> None:
    if capture_content and result is not None:
        output_str = _extract_output_content(result)
        invocation.output_messages = [
            OutputMessage(
                role=Role.ASSISTANT.value,
                parts=[TextPart(content=output_str)],
                finish_reason=_extract_finish_reason(result),
            )
        ]
    session_id = getattr(result, "session_id", None)
    if session_id:
        invocation.conversation_id = str(session_id)
    if isinstance(invocation, AgentInvocation):
        model = getattr(result, "model", None)
        if model:
            invocation.attributes.setdefault(
                GenAI.GEN_AI_REQUEST_MODEL, str(model)
            )


def _start_agent_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> AgentInvocation:
    agent_name = getattr(instance, "name", None)
    model_obj = getattr(instance, "model", None)
    request_model = None
    if model_obj is not None:
        request_model = getattr(model_obj, "id", None) or (
            model_obj if isinstance(model_obj, str) else None
        )

    invocation = handler.invoke_local_agent(
        agent_name=str(agent_name) if agent_name else None,
        request_model=str(request_model) if request_model else None,
    )
    if request_model:
        invocation.attributes[GenAI.GEN_AI_REQUEST_MODEL] = str(request_model)
    description = getattr(instance, "description", None)
    if description:
        invocation.agent_description = str(description)

    _set_invocation_input(invocation, instance, args, kwargs, capture_content)
    invocation.tool_definitions = prepare_tool_definitions(
        getattr(instance, "tools", None)
    )
    return invocation


def _start_tool_invocation(
    handler: TelemetryHandler,
    instance: FunctionCall,
    capture_content: bool,
) -> ToolInvocation:
    function_obj = instance.function
    tool_name = getattr(function_obj, "name", None) or "tool"
    tool_desc = getattr(function_obj, "description", None)
    tool_call_id = instance.call_id

    invocation = handler.tool(
        name=str(tool_name),
        tool_type="function",
    )
    if tool_call_id:
        invocation.tool_call_id = str(tool_call_id)
    if tool_desc:
        invocation.tool_description = str(tool_desc)
    _set_tool_invocation_input(invocation, instance, capture_content)
    return invocation


def _agent_run(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_agent_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(error)
            raise

        if isinstance(result, Iterator):
            return AgnoAgentStreamWrapper(result, invocation, capture_content)

        _set_invocation_output(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _agent_arun(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            invocation = _start_agent_invocation(
                handler, instance, args, kwargs, capture_content
            )
            invocation.fail(error)
            raise

        if isinstance(result, AsyncIterator):
            invocation = _start_agent_invocation(
                handler, instance, args, kwargs, capture_content
            )
            return AsyncAgnoAgentStreamWrapper(
                result, invocation, capture_content
            )

        if isinstance(result, Awaitable):

            @functools.wraps(wrapped)
            async def _await_result() -> object:
                invocation = _start_agent_invocation(
                    handler, instance, args, kwargs, capture_content
                )
                try:
                    awaitable = cast(Awaitable[object], result)
                    response: object = await awaitable
                    if isinstance(response, AsyncIterator):
                        return AsyncAgnoAgentStreamWrapper(
                            response, invocation, capture_content
                        )
                    _set_invocation_output(
                        invocation, response, capture_content
                    )
                    invocation.stop()
                    return response
                except Exception as error:
                    invocation.fail(error)
                    raise

            return _await_result()

        invocation = _start_agent_invocation(
            handler, instance, args, kwargs, capture_content
        )
        _set_invocation_output(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _handle_tool_result(
    invocation: ToolInvocation,
    instance: FunctionCall,
    result: FunctionExecutionResult,
    capture_content: bool,
) -> FunctionExecutionResult:
    if result.status == "failure":
        _fail_tool_invocation(invocation, result)
        return result

    tool_res = result.result
    wrapped_stream: Any = None
    if isinstance(tool_res, AsyncIterator):
        wrapped_stream = AsyncAgnoToolStreamWrapper(
            cast(Any, tool_res), invocation, capture_content
        )
    elif isinstance(tool_res, Iterator):
        wrapped_stream = AgnoToolStreamWrapper(
            cast(Any, tool_res), invocation, capture_content
        )

    if wrapped_stream is not None:
        result.result = wrapped_stream
        instance.result = wrapped_stream
        return result

    _set_tool_invocation_output(invocation, result, capture_content)
    invocation.stop()
    return result


def _tool_call_execute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., FunctionExecutionResult],
        instance: FunctionCall,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> FunctionExecutionResult:
        invocation = _start_tool_invocation(handler, instance, capture_content)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(error)
            raise
        return _handle_tool_result(
            invocation, instance, result, capture_content
        )

    return traced_method


def _tool_call_aexecute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Awaitable[FunctionExecutionResult]],
        instance: FunctionCall,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> FunctionExecutionResult:
        invocation = _start_tool_invocation(handler, instance, capture_content)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(error)
            raise
        return _handle_tool_result(
            invocation, instance, result, capture_content
        )

    return cast(Callable[..., Any], traced_method)


def _start_workflow_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> WorkflowInvocation:
    workflow_name = getattr(instance, "name", None)
    invocation = handler.workflow(name=workflow_name)
    _set_invocation_input(invocation, instance, args, kwargs, capture_content)
    return invocation


def _workflow_run(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_workflow_invocation(
            handler, instance, args, kwargs, capture_content
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(error)
            raise

        if isinstance(result, Iterator):
            return AgnoWorkflowStreamWrapper(
                result, invocation, capture_content
            )

        _set_invocation_output(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _workflow_arun(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            invocation = _start_workflow_invocation(
                handler, instance, args, kwargs, capture_content
            )
            invocation.fail(error)
            raise

        if isinstance(result, AsyncIterator):
            invocation = _start_workflow_invocation(
                handler, instance, args, kwargs, capture_content
            )
            return AsyncAgnoWorkflowStreamWrapper(
                result, invocation, capture_content
            )

        if isinstance(result, Awaitable):

            @functools.wraps(wrapped)
            async def _await_result() -> object:
                invocation = _start_workflow_invocation(
                    handler, instance, args, kwargs, capture_content
                )
                try:
                    awaitable = cast(Awaitable[object], result)
                    response: object = await awaitable
                    if isinstance(response, AsyncIterator):
                        return AsyncAgnoWorkflowStreamWrapper(
                            response, invocation, capture_content
                        )
                    _set_invocation_output(
                        invocation, response, capture_content
                    )
                    invocation.stop()
                    return response
                except Exception as error:
                    invocation.fail(error)
                    raise

            return _await_result()

        invocation = _start_workflow_invocation(
            handler, instance, args, kwargs, capture_content
        )
        _set_invocation_output(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _start_retrieval_invocation(
    handler: TelemetryHandler,
    instance: Knowledge,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> RetrievalInvocation:
    vector_db = instance.vector_db
    data_source_id = (
        instance.name
        or (
            getattr(vector_db, "name", None) if vector_db is not None else None
        )
        or (
            getattr(vector_db, "collection", None)
            if vector_db is not None
            else None
        )
        or (
            getattr(vector_db, "table_name", None)
            if vector_db is not None
            else None
        )
    )
    provider = None
    if vector_db is not None:
        provider = (
            getattr(vector_db, "provider", None)
            or vector_db.__class__.__name__.lower()
        )

    embedder = (
        getattr(vector_db, "embedder", None) if vector_db is not None else None
    )
    request_model = None
    if embedder is not None:
        request_model = getattr(embedder, "id", None) or getattr(
            embedder, "model", None
        )

    invocation = handler.retrieval(
        data_source_id=str(data_source_id)
        if data_source_id is not None
        else None,
        provider=str(provider).lower() if provider is not None else None,
        request_model=str(request_model)
        if request_model is not None
        else None,
    )

    query = args[0] if args else kwargs.get("query")
    if query is not None:
        invocation.query_text = str(query)

    max_results = None
    if len(args) > 1 and args[1] is not None:
        max_results = args[1]
    elif kwargs.get("max_results") is not None:
        max_results = kwargs.get("max_results")
    else:
        max_results = instance.max_results

    if max_results is not None:
        try:
            invocation.top_k = int(max_results)
        except (ValueError, TypeError):
            pass

    return invocation


def _knowledge_search(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Sequence[Document] | None],
        instance: Knowledge,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_retrieval_invocation(
            handler, instance, args, kwargs
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(error)
            raise

        if capture_content and result is not None:
            invocation.documents = [
                format_retrieval_document(doc) for doc in result
            ]
        invocation.stop()
        return result

    return traced_method


def _knowledge_asearch(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Awaitable[Sequence[Document] | None]],
        instance: Knowledge,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_retrieval_invocation(
            handler, instance, args, kwargs
        )
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as error:
            invocation.fail(error)
            raise

        if capture_content and result is not None:
            invocation.documents = [
                format_retrieval_document(doc) for doc in result
            ]
        invocation.stop()
        return result

    return cast(Callable[..., Any], traced_method)
