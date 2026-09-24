# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Agno instrumentation."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import functools
import json
import logging
import sys
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agno.agent import Agent, RunOutput
    from agno.knowledge.document.base import Document
    from agno.knowledge.knowledge import Knowledge
    from agno.run.workflow import WorkflowRunOutput
    from agno.team import Team, TeamRunOutput
    from agno.tools.function import FunctionCall, FunctionExecutionResult
    from agno.workflow import Workflow

    AgnoRunOutput = RunOutput | TeamRunOutput | WorkflowRunOutput

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.instrumentation.genai.agno.stream import (
    AgnoAgentStreamWrapper,
    AgnoWorkflowStreamWrapper,
    AsyncAgnoAgentStreamWrapper,
    AsyncAgnoWorkflowStreamWrapper,
)
from opentelemetry.instrumentation.genai.agno.utils import (
    _get_property_value,
    format_content,
    format_retrieval_document,
    prepare_tool_definitions,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ErrorTypeValues,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    LocalAgentInvocation,
    RetrievalInvocation,
    ToolInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncToolStreamWrapper,
    SyncToolStreamWrapper,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    OutputMessage,
    Role,
    TextPart,
    ToolCallResponsePart,
)
from opentelemetry.util.genai.utils import get_argument

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
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.continue_run",
        _agent_run(handler, is_continue=True),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.acontinue_run",
        _agent_arun(handler, is_continue=True),
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
        _AGNO_TEAM_MODULE,
        f"{_TEAM_CLASS}.continue_run",
        _agent_run(handler, is_continue=True),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_TEAM_MODULE,
        f"{_TEAM_CLASS}.acontinue_run",
        _agent_arun(handler, is_continue=True),
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
    _safe_wrap_function(
        _AGNO_WORKFLOW_MODULE,
        f"{_WORKFLOW_CLASS}.continue_run",
        _workflow_run(handler, is_continue=True),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_WORKFLOW_MODULE,
        f"{_WORKFLOW_CLASS}.acontinue_run",
        _workflow_arun(handler, is_continue=True),
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

    def _safe_unwrap(target: Any, attr: str) -> None:
        try:
            unwrap(target, attr)
        except (AttributeError, ValueError):
            pass

    if _AGNO_MODULE in sys.modules:
        try:
            import agno.agent

            for attr in ("run", "arun", "continue_run", "acontinue_run"):
                _safe_unwrap(agno.agent.Agent, attr)
        except ImportError:
            pass
    if _AGNO_TEAM_MODULE in sys.modules:
        try:
            import agno.team

            for attr in ("run", "arun", "continue_run", "acontinue_run"):
                _safe_unwrap(agno.team.Team, attr)
        except ImportError:
            pass
    if _AGNO_TOOLS_MODULE in sys.modules:
        try:
            import agno.tools.function

            for attr in ("execute", "aexecute"):
                _safe_unwrap(agno.tools.function.FunctionCall, attr)
        except ImportError:
            pass
    if _AGNO_WORKFLOW_MODULE in sys.modules:
        try:
            import agno.workflow.workflow

            for attr in ("run", "arun", "continue_run", "acontinue_run"):
                _safe_unwrap(agno.workflow.workflow.Workflow, attr)
        except ImportError:
            pass
    if _AGNO_KNOWLEDGE_MODULE in sys.modules:
        try:
            import agno.knowledge.knowledge

            for attr in ("search", "asearch"):
                _safe_unwrap(agno.knowledge.knowledge.Knowledge, attr)
        except ImportError:
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
) -> None:
    if invocation.should_capture_content:
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
) -> None:
    if result.status == "failure":
        return
    if invocation.should_capture_content:
        invocation.tool_result = _extract_output_content(result)


def _set_invocation_input(
    invocation: LocalAgentInvocation | WorkflowInvocation,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
    wrapped: Callable[..., Any],
) -> None:
    if capture_content:
        input_val = get_argument("input", wrapped, args, kwargs)
        if input_val is not None:
            content_str = _extract_input_content(input_val)
            invocation.input_messages = [
                InputMessage(
                    role=Role.USER.value, parts=[TextPart(content=content_str)]
                )
            ]


def _extract_continue_input(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    for param in (
        "input",
        "additional_instructions",
        "additionalInstructions",
    ):
        val = get_argument(param, wrapped, args, kwargs)
        if val is not None:
            return val
    return None


def _extract_continue_tool_results(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[tuple[str, Any]]:
    """Extract tool call results passed to continue_run.

    In Agno, tool results can be passed as:
    - ``tools``: JSON string (e.g. from the continue-run REST API) or list of
      tool execution dicts / ToolExecution objects
    - ``updated_tools``: list of ToolExecution objects or dicts
    - ``requirements``: list of RunRequirement objects containing tool_execution
    """
    raw_tools: Any = (
        get_argument("tools", wrapped, args, kwargs)
        or get_argument("updated_tools", wrapped, args, kwargs)
        or get_argument("requirements", wrapped, args, kwargs)
    )
    if not raw_tools:
        return []

    items: list[Any]
    if isinstance(raw_tools, str):
        try:
            parsed: Any = json.loads(raw_tools)
            items = (
                cast(list[Any], parsed)
                if isinstance(parsed, list)
                else [parsed]
            )
        except Exception:
            return []
    elif isinstance(raw_tools, Iterable):
        items = list(cast(Iterable[Any], raw_tools))
    else:
        items = [raw_tools]

    tool_results: list[tuple[str, Any]] = []
    for item_raw in items:
        item: Any = item_raw
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                pass

        if (tool_exec := getattr(item, "tool_execution", None)) is not None:
            item = tool_exec

        if isinstance(item, dict):
            item_dict = cast(dict[str, Any], item)
            call_id = item_dict.get("tool_call_id") or item_dict.get("id")
            if not call_id:
                continue
            resp: Any = item_dict.get("result")
            if resp is None:
                resp = (
                    {"confirmed": item_dict["confirmed"]}
                    if "confirmed" in item_dict
                    else item_dict
                )
            tool_results.append((str(call_id), resp))
        else:
            call_id = getattr(item, "tool_call_id", None)
            if not call_id:
                continue
            resp = getattr(item, "result", None)
            if resp is None:
                confirmed = getattr(item, "confirmed", None)
                if confirmed is not None:
                    resp = {"confirmed": confirmed}
                else:
                    to_dict = getattr(item, "to_dict", None)
                    resp = to_dict() if callable(to_dict) else str(item)
            tool_results.append((str(call_id), resp))

    return tool_results


def _extract_continue_session_id(
    instance: Agent | Team | Workflow,
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    session_id = get_argument("session_id", wrapped, args, kwargs)
    if session_id:
        return str(session_id)
    run_response = get_argument("run_response", wrapped, args, kwargs)
    if run_response is not None:
        sid = getattr(run_response, "session_id", None)
        if sid:
            return str(sid)
    sid = getattr(instance, "session_id", None)
    if sid:
        return str(sid)
    return None


def _set_continue_invocation_input(
    invocation: LocalAgentInvocation | WorkflowInvocation,
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> None:
    if not capture_content:
        return
    messages: list[InputMessage] = []

    tool_results = _extract_continue_tool_results(wrapped, args, kwargs)
    for call_id, resp in tool_results:
        messages.append(
            InputMessage(
                role=Role.TOOL.value,
                parts=[
                    # format_content stringifies responses (dumping structured objects to JSON).
                    # When gen_ai.input.messages is serialized to a JSON string attribute on the span,
                    # structured responses become inner JSON strings that consumers can json.loads().
                    # If the response is already a string, it remains a normal string and is not double-encoded.
                    ToolCallResponsePart(
                        id=call_id,
                        response=format_content(resp),
                    )
                ],
            )
        )

    input_val = _extract_continue_input(wrapped, args, kwargs)
    if input_val is not None:
        content_str = _extract_input_content(input_val)
        if content_str:
            messages.append(
                InputMessage(
                    role=Role.USER.value,
                    parts=[TextPart(content=content_str)],
                )
            )

    invocation.input_messages = messages


def _extract_finish_reason(result: object) -> str:
    if "error" in str(getattr(result, "status", "")).lower():
        return "error"
    return "stop"


def _set_invocation_output(
    invocation: LocalAgentInvocation | WorkflowInvocation,
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


def _start_agent_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
    wrapped: Callable[..., Any],
    *,
    is_continue: bool = False,
) -> LocalAgentInvocation:
    agent_name = getattr(instance, "name", None)
    model_obj = get_argument("model", wrapped, args, kwargs) or getattr(
        instance, "model", None
    )
    request_model = None
    if model_obj is not None:
        request_model = (
            getattr(model_obj, "id", None)
            or getattr(model_obj, "name", None)
            or (model_obj if isinstance(model_obj, str) else None)
        )

    invocation = handler.invoke_local_agent(
        agent_name=str(agent_name) if agent_name else None,
        request_model=str(request_model) if request_model else None,
    )
    description = getattr(instance, "description", None)
    if description:
        invocation.agent_description = str(description)

    if is_continue:
        _set_continue_invocation_input(
            invocation, wrapped, args, kwargs, capture_content
        )
        invocation.conversation_id = _extract_continue_session_id(
            instance, wrapped, args, kwargs
        )
    else:
        _set_invocation_input(
            invocation, instance, args, kwargs, capture_content, wrapped
        )

    tool_defs = prepare_tool_definitions(getattr(instance, "tools", None))
    if not tool_defs and not is_continue:
        tools_arg: Any = get_argument("tools", wrapped, args, kwargs)
        if tools_arg is not None:
            tool_defs = prepare_tool_definitions(tools_arg)
    invocation.tool_definitions = tool_defs
    return invocation


def _start_tool_invocation(
    handler: TelemetryHandler,
    instance: FunctionCall,
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
    _set_tool_invocation_input(invocation, instance)
    return invocation


def _agent_run(
    handler: TelemetryHandler,
    *,
    is_continue: bool = False,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_agent_invocation(
            handler,
            instance,
            args,
            kwargs,
            capture_content,
            wrapped=wrapped,
            is_continue=is_continue,
        )
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
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
    *,
    is_continue: bool = False,
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
        except BaseException as error:
            invocation = _start_agent_invocation(
                handler,
                instance,
                args,
                kwargs,
                capture_content,
                wrapped=wrapped,
                is_continue=is_continue,
            )
            invocation.fail(error)
            raise

        if isinstance(result, AsyncIterator):
            invocation = _start_agent_invocation(
                handler,
                instance,
                args,
                kwargs,
                capture_content,
                wrapped=wrapped,
                is_continue=is_continue,
            )
            return AsyncAgnoAgentStreamWrapper(
                result, invocation, capture_content
            )

        if isinstance(result, Awaitable):

            @functools.wraps(wrapped)
            async def _await_result() -> object:
                invocation = _start_agent_invocation(
                    handler,
                    instance,
                    args,
                    kwargs,
                    capture_content,
                    wrapped=wrapped,
                    is_continue=is_continue,
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
                except BaseException as error:
                    invocation.fail(error)
                    raise

            return _await_result()

        invocation = _start_agent_invocation(
            handler,
            instance,
            args,
            kwargs,
            capture_content,
            wrapped=wrapped,
            is_continue=is_continue,
        )
        _set_invocation_output(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _handle_tool_result(
    invocation: ToolInvocation,
    instance: FunctionCall,
    result: FunctionExecutionResult,
) -> FunctionExecutionResult:
    if result.status == "failure":
        _fail_tool_invocation(invocation, result)
        return result

    tool_res = result.result
    wrapped_stream: Any = None
    if isinstance(tool_res, AsyncIterator):
        wrapped_stream = AsyncToolStreamWrapper(
            cast(Any, tool_res), invocation
        )
    elif isinstance(tool_res, Iterator):
        wrapped_stream = SyncToolStreamWrapper(cast(Any, tool_res), invocation)

    if wrapped_stream is not None:
        result.result = wrapped_stream
        instance.result = wrapped_stream
        return result

    _set_tool_invocation_output(invocation, result)
    invocation.stop()
    return result


def _tool_call_execute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., FunctionExecutionResult],
        instance: FunctionCall,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> FunctionExecutionResult:
        invocation = _start_tool_invocation(handler, instance)
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise
        return _handle_tool_result(invocation, instance, result)

    return traced_method


def _tool_call_aexecute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[FunctionExecutionResult]],
        instance: FunctionCall,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> FunctionExecutionResult:
        invocation = _start_tool_invocation(handler, instance)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise
        return _handle_tool_result(invocation, instance, result)

    return cast(Callable[..., Any], traced_method)


def _start_workflow_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
    wrapped: Callable[..., Any],
    *,
    is_continue: bool = False,
) -> WorkflowInvocation:
    workflow_name = getattr(instance, "name", None)
    invocation = handler.workflow(name=workflow_name)
    if is_continue:
        _set_continue_invocation_input(
            invocation, wrapped, args, kwargs, capture_content
        )
        invocation.conversation_id = _extract_continue_session_id(
            instance, wrapped, args, kwargs
        )
    else:
        _set_invocation_input(
            invocation, instance, args, kwargs, capture_content, wrapped
        )
    return invocation


def _workflow_run(
    handler: TelemetryHandler,
    *,
    is_continue: bool = False,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_workflow_invocation(
            handler,
            instance,
            args,
            kwargs,
            capture_content,
            wrapped=wrapped,
            is_continue=is_continue,
        )
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
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
    *,
    is_continue: bool = False,
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
        except BaseException as error:
            invocation = _start_workflow_invocation(
                handler,
                instance,
                args,
                kwargs,
                capture_content,
                wrapped=wrapped,
                is_continue=is_continue,
            )
            invocation.fail(error)
            raise

        if isinstance(result, AsyncIterator):
            invocation = _start_workflow_invocation(
                handler,
                instance,
                args,
                kwargs,
                capture_content,
                wrapped=wrapped,
                is_continue=is_continue,
            )
            return AsyncAgnoWorkflowStreamWrapper(
                result, invocation, capture_content
            )

        if isinstance(result, Awaitable):

            @functools.wraps(wrapped)
            async def _await_result() -> object:
                invocation = _start_workflow_invocation(
                    handler,
                    instance,
                    args,
                    kwargs,
                    capture_content,
                    wrapped=wrapped,
                    is_continue=is_continue,
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
                except BaseException as error:
                    invocation.fail(error)
                    raise

            return _await_result()

        invocation = _start_workflow_invocation(
            handler,
            instance,
            args,
            kwargs,
            capture_content,
            wrapped=wrapped,
            is_continue=is_continue,
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
    wrapped: Callable[..., Any],
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

    if invocation.should_capture_content:
        query = get_argument("query", wrapped, args, kwargs)
        if query is not None:
            invocation.query_text = str(query)

    max_results = get_argument("max_results", wrapped, args, kwargs)
    if max_results is None:
        max_results = instance.max_results

    if isinstance(max_results, (int, float, str)):
        try:
            invocation.top_k = int(max_results)
        except (ValueError, TypeError):
            pass

    return invocation


def _knowledge_search(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Sequence[Document] | None],
        instance: Knowledge,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_retrieval_invocation(
            handler, instance, args, kwargs, wrapped=wrapped
        )
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise

        if invocation.should_capture_content and result is not None:
            invocation.documents = [
                format_retrieval_document(doc) for doc in result
            ]
        invocation.stop()
        return result

    return traced_method


def _knowledge_asearch(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Sequence[Document] | None]],
        instance: Knowledge,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation = _start_retrieval_invocation(
            handler, instance, args, kwargs, wrapped=wrapped
        )
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise

        if invocation.should_capture_content and result is not None:
            invocation.documents = [
                format_retrieval_document(doc) for doc in result
            ]
        invocation.stop()
        return result

    return cast(Callable[..., Any], traced_method)
