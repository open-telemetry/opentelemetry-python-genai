# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Agno instrumentation."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import contextvars
import functools
import logging
import sys
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from agno.knowledge.document.base import Document
    from agno.knowledge.embedder.base import Embedder
    from agno.knowledge.knowledge import Knowledge
    from agno.models.response import ToolExecution
    from agno.tools.function import FunctionCall, FunctionExecutionResult

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.instrumentation.genai.agno.stream import (
    AgnoAgentStreamWrapper,
    AgnoWorkflowStreamWrapper,
    AsyncAgnoAgentStreamWrapper,
    AsyncAgnoWorkflowStreamWrapper,
)
from opentelemetry.instrumentation.genai.agno.utils import (
    _get_property_value,
    extract_session_id,
    format_content,
    format_retrieval_document,
    prepare_tool_definitions,
    resolve_embedder_provider,
    safe_int,
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
    SystemInstructionPart,
    TextPart,
    ToolCallResponsePart,
)
from opentelemetry.util.genai.utils import (
    bind_arguments,
    get_argument,
    object_to_any_value,
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

_ACTIVE_WORKFLOW_RUN_IDS: set[str] = set()
_SUPPRESS_EMBEDDING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_SUPPRESS_EMBEDDING", default=False
)
_patched_embedder_classes: set[type[Any]] = set()
_KNOWN_EMBEDDERS: tuple[tuple[str, str], ...] = (
    ("agno.knowledge.embedder.base", "Embedder"),
    ("agno.knowledge.embedder.openai", "OpenAIEmbedder"),
    ("agno.knowledge.embedder.azure_openai", "AzureOpenAIEmbedder"),
    ("agno.knowledge.embedder.aws_bedrock", "AwsBedrockEmbedder"),
    ("agno.knowledge.embedder.google", "GoogleEmbedder"),
    ("agno.knowledge.embedder.google", "GeminiEmbedder"),
    ("agno.knowledge.embedder.ollama", "OllamaEmbedder"),
    ("agno.knowledge.embedder.mistral", "MistralEmbedder"),
    ("agno.knowledge.embedder.cohere", "CohereEmbedder"),
    ("agno.knowledge.embedder.fireworks", "FireworksEmbedder"),
    ("agno.knowledge.embedder.jina", "JinaEmbedder"),
    ("agno.knowledge.embedder.together", "TogetherEmbedder"),
    ("agno.knowledge.embedder.voyageai", "VoyageAIEmbedder"),
    ("agno.knowledge.embedder.fastembed", "FastEmbedEmbedder"),
    (
        "agno.knowledge.embedder.sentence_transformer",
        "SentenceTransformerEmbedder",
    ),
    ("agno.knowledge.embedder.huggingface", "HuggingfaceCustomEmbedder"),
    ("agno.knowledge.embedder.langdb", "LangDBEmbedder"),
    ("agno.knowledge.embedder.nebius", "NebiusEmbedder"),
    ("agno.knowledge.embedder.openai_like", "OpenAILikeEmbedder"),
    ("agno.knowledge.embedder.vllm", "VLLMEmbedder"),
)
_EMBEDDER_METHODS: frozenset[str] = frozenset(
    {
        "get_embedding",
        "get_embedding_and_usage",
        "async_get_embedding",
        "async_get_embedding_and_usage",
    }
)


def _release_run_id(run_id: str | None) -> None:
    if run_id:
        _ACTIVE_WORKFLOW_RUN_IDS.discard(run_id)


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
    _safe_wrap_function(
        _AGNO_WORKFLOW_MODULE,
        f"{_WORKFLOW_CLASS}._aexecute",
        _workflow_aexecute(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_WORKFLOW_MODULE,
        f"{_WORKFLOW_CLASS}._aexecute_stream",
        _workflow_aexecute_stream(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_WORKFLOW_MODULE,
        f"{_WORKFLOW_CLASS}._aexecute_workflow_agent",
        _workflow_aexecute_workflow_agent(handler),
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

    embedder_wrappers = _embedder_method_wrappers(handler)
    for mod_name, cls_name in _KNOWN_EMBEDDERS:
        for method_name, wrapper_fn in embedder_wrappers:
            _safe_wrap_function(
                mod_name,
                f"{cls_name}.{method_name}",
                wrapper_fn,
                current_generation,
            )

    try:
        from agno.knowledge.embedder.base import Embedder

        # Wrap all existing custom embedder classes
        def _wrap_subclasses(base_cls: type[Any]) -> None:
            for sub in base_cls.__subclasses__():
                _wrap_embedder_class(sub, handler)
                _wrap_subclasses(sub)

        _wrap_subclasses(Embedder)

        # Wrap new embedder classes created during runtime
        def _traced_init_subclass(cls: type[Embedder], **kwargs: Any) -> None:
            super(Embedder, cls).__init_subclass__(**kwargs)
            if _is_instrumented:
                _wrap_embedder_class(cls, handler)

        setattr(
            Embedder, "__init_subclass__", classmethod(_traced_init_subclass)
        )
    except Exception:
        pass


def unpatch_agent() -> None:
    """Remove patches from Agno class methods."""
    global _instrumentation_generation, _is_instrumented
    _instrumentation_generation += 1
    _is_instrumented = False
    _ACTIVE_WORKFLOW_RUN_IDS.clear()

    def _safe_unwrap(target: Any, attr: str) -> None:
        try:
            unwrap(target, attr)
        except (AttributeError, ValueError):
            pass

    try:
        from agno.knowledge.embedder.base import Embedder

        if "__init_subclass__" in Embedder.__dict__:
            delattr(Embedder, "__init_subclass__")
    except Exception:
        pass

    for cls in list(_patched_embedder_classes):
        for attr in _EMBEDDER_METHODS:
            _safe_unwrap(cls, attr)
    _patched_embedder_classes.clear()
    for mod_name, cls_name in _KNOWN_EMBEDDERS:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                for attr in _EMBEDDER_METHODS:
                    _safe_unwrap(cls, attr)

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

            for attr in (
                "run",
                "arun",
                "continue_run",
                "acontinue_run",
                "_aexecute",
                "_aexecute_stream",
                "_aexecute_workflow_agent",
            ):
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


_extract_embedder_provider = resolve_embedder_provider


def _extract_embedder_model(embedder: Embedder) -> str | None:
    model = (
        getattr(embedder, "id", None)
        or getattr(embedder, "model", None)
        or getattr(embedder, "name", None)
    )
    return str(model) if model is not None else None


def _extract_embedder_input_tokens(usage: dict[str, Any]) -> int | None:
    for key in ("prompt_tokens", "input_tokens", "total_tokens"):
        if (tok := safe_int(usage.get(key))) is not None:
            return tok
    return None


def _embedder_get_embedding(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Sequence[float]],
        instance: Embedder,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Sequence[float]:
        if _SUPPRESS_EMBEDDING.get():
            return wrapped(*args, **kwargs)

        token = _SUPPRESS_EMBEDDING.set(True)
        provider = _extract_embedder_provider(instance)
        request_model = _extract_embedder_model(instance)
        invocation = handler.embedding(
            provider=provider,
            request_model=request_model,
        )
        if request_model:
            invocation.response_model_name = request_model
        if encoding_format := getattr(instance, "encoding_format", None):
            invocation.encoding_formats = [str(encoding_format)]

        try:
            result = wrapped(*args, **kwargs)
            if result:
                invocation.dimension_count = len(result)
            else:
                invocation.dimension_count = safe_int(instance.dimensions)
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise
        finally:
            _SUPPRESS_EMBEDDING.reset(token)

    return traced_method


def _embedder_get_embedding_and_usage(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., tuple[Sequence[float], dict[str, Any] | None]],
        instance: Embedder,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Sequence[float], dict[str, Any] | None]:
        if _SUPPRESS_EMBEDDING.get():
            return wrapped(*args, **kwargs)

        token = _SUPPRESS_EMBEDDING.set(True)
        provider = _extract_embedder_provider(instance)
        request_model = _extract_embedder_model(instance)
        invocation = handler.embedding(
            provider=provider,
            request_model=request_model,
        )
        if request_model:
            invocation.response_model_name = request_model
        if encoding_format := getattr(instance, "encoding_format", None):
            invocation.encoding_formats = [str(encoding_format)]

        try:
            result = wrapped(*args, **kwargs)
            embedding, usage = result
            if embedding:
                invocation.dimension_count = len(embedding)
            else:
                invocation.dimension_count = safe_int(instance.dimensions)

            if isinstance(usage, dict):
                invocation.input_tokens = _extract_embedder_input_tokens(usage)
                if model := usage.get("model"):
                    invocation.response_model_name = str(model)

            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise
        finally:
            _SUPPRESS_EMBEDDING.reset(token)

    return traced_method


def _embedder_async_get_embedding(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Sequence[float]]],
        instance: Embedder,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Sequence[float]:
        if _SUPPRESS_EMBEDDING.get():
            return await wrapped(*args, **kwargs)

        token = _SUPPRESS_EMBEDDING.set(True)
        provider = _extract_embedder_provider(instance)
        request_model = _extract_embedder_model(instance)
        invocation = handler.embedding(
            provider=provider,
            request_model=request_model,
        )
        if request_model:
            invocation.response_model_name = request_model
        if encoding_format := getattr(instance, "encoding_format", None):
            invocation.encoding_formats = [str(encoding_format)]

        try:
            result = await wrapped(*args, **kwargs)
            if result:
                invocation.dimension_count = len(result)
            else:
                invocation.dimension_count = safe_int(instance.dimensions)
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise
        finally:
            _SUPPRESS_EMBEDDING.reset(token)

    return cast(Callable[..., Any], traced_method)


def _embedder_async_get_embedding_and_usage(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[
            ..., Awaitable[tuple[Sequence[float], dict[str, Any] | None]]
        ],
        instance: Embedder,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Sequence[float], dict[str, Any] | None]:
        if _SUPPRESS_EMBEDDING.get():
            return await wrapped(*args, **kwargs)

        token = _SUPPRESS_EMBEDDING.set(True)
        provider = _extract_embedder_provider(instance)
        request_model = _extract_embedder_model(instance)
        invocation = handler.embedding(
            provider=provider,
            request_model=request_model,
        )
        if request_model:
            invocation.response_model_name = request_model
        if encoding_format := getattr(instance, "encoding_format", None):
            invocation.encoding_formats = [str(encoding_format)]

        try:
            result = await wrapped(*args, **kwargs)
            embedding, usage = result
            if embedding:
                invocation.dimension_count = len(embedding)
            else:
                invocation.dimension_count = safe_int(instance.dimensions)

            if isinstance(usage, dict):
                invocation.input_tokens = _extract_embedder_input_tokens(usage)
                if model := usage.get("model"):
                    invocation.response_model_name = str(model)

            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise
        finally:
            _SUPPRESS_EMBEDDING.reset(token)

    return cast(Callable[..., Any], traced_method)


def _embedder_method_wrappers(
    handler: TelemetryHandler,
) -> tuple[tuple[str, Callable[..., Any]], ...]:
    return (
        ("get_embedding", _embedder_get_embedding(handler)),
        (
            "get_embedding_and_usage",
            _embedder_get_embedding_and_usage(handler),
        ),
        ("async_get_embedding", _embedder_async_get_embedding(handler)),
        (
            "async_get_embedding_and_usage",
            _embedder_async_get_embedding_and_usage(handler),
        ),
    )


def _wrap_embedder_class(
    cls: type[Any],
    handler: TelemetryHandler,
) -> None:
    for method_name, wrapper_fn in _embedder_method_wrappers(handler):
        if hasattr(cls, method_name):
            target = getattr(cls, method_name)
            if not hasattr(target, "__wrapped__"):
                try:
                    wrap_function_wrapper(
                        cast(Any, cls), method_name, wrapper_fn
                    )
                    _patched_embedder_classes.add(cls)
                except Exception:
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
    return format_content(_extract_output_value(result))


def _extract_output_value(result: Any) -> Any:
    content = _get_property_value(result, "content")
    if content is not None:
        return content
    val = _get_property_value(result, "result")
    if val is not None:
        return val
    return result


def _set_tool_invocation_input(
    invocation: ToolInvocation,
    instance: FunctionCall,
) -> None:
    if invocation.should_capture_content:
        invocation.arguments = object_to_any_value(instance.arguments)


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
        invocation.tool_result = object_to_any_value(
            _extract_output_value(result)
        )


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


def _to_tool_executions(raw_tools: object) -> list[ToolExecution]:
    """Normalize raw tools/requirements into a list of Agno ToolExecution instances."""
    if not raw_tools:
        return []

    from agno.models.response import ToolExecution

    items: Iterable[object]
    if isinstance(raw_tools, Iterable) and not isinstance(
        raw_tools, (str, bytes, Mapping)
    ):
        items = cast(Iterable[object], raw_tools)
    else:
        items = cast(Iterable[object], [raw_tools])

    tool_executions: list[ToolExecution] = []
    for item in items:
        if (tool_exec := getattr(item, "tool_execution", None)) is not None:
            item = tool_exec

        if isinstance(item, Mapping):
            try:
                from agno.run.requirement import RunRequirement

                item = RunRequirement.from_dict(
                    cast(dict[str, Any], item)
                ).tool_execution
            except Exception:
                pass

        # Agno populates tool_call_id on paused tools; without an ID the result
        # cannot be correlated to a prior tool call.
        if isinstance(item, ToolExecution) and item.tool_call_id:
            tool_executions.append(item)

    return tool_executions


_RESUME_FLAGS = (
    "external_execution_required",
    "requires_confirmation",
    "requires_user_input",
    "is_paused",
)


def _has_result(item: object) -> bool:
    return (
        _get_property_value(item, "result") is not None
        or _get_property_value(item, "confirmed") is not None
        or _get_property_value(item, "user_input") is not None
    )


def _extract_resumed_tools(run_response: object) -> list[object]:
    """Extract resumed tool executions or requirements from run_response."""
    if isinstance(run_response, str) or run_response is None:
        return []

    tools: object = _get_property_value(run_response, "tools")
    if isinstance(tools, Iterable) and not isinstance(
        tools, (str, bytes, Mapping)
    ):
        tool_items: list[object] = [
            t
            for t in cast(Iterable[object], tools)
            if _has_result(t)
            and any(_get_property_value(t, flag) for flag in _RESUME_FLAGS)
        ]
        if tool_items:
            return tool_items

    requirements: object = _get_property_value(run_response, "requirements")
    if requirements is not None:
        if isinstance(requirements, Iterable) and not isinstance(
            requirements, (str, bytes, Mapping)
        ):
            result: list[object] = []
            for req in cast(Iterable[object], requirements):
                target: object = getattr(req, "tool_execution", req)
                is_resolved = getattr(req, "is_resolved", None)
                if _has_result(target) or (
                    callable(is_resolved) and bool(is_resolved())
                ):
                    result.append(req)
            if result:
                return result

    return []


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

    if isinstance(invocation, LocalAgentInvocation):
        # In Agno agents, tool results can be passed as:
        # - updated_tools: list of ToolExecution objects or dicts
        # - requirements: list of RunRequirement objects containing tool_execution
        # - run_response: RunOutput containing resumed tool executions or requirements
        updated_tools: object = get_argument(
            "updated_tools", wrapped, args, kwargs
        )
        requirements_arg: object = get_argument(
            "requirements", wrapped, args, kwargs
        )
        raw_tools: object = updated_tools or requirements_arg
        # Explicit tool arguments take precedence as incoming updates; when omitted,
        # callers resume by modifying tool executions on run_response in-place.
        if not raw_tools:
            run_response = get_argument("run_response", wrapped, args, kwargs)
            if run_response is not None:
                raw_tools = _extract_resumed_tools(run_response)
        for tool_exec in _to_tool_executions(raw_tools):
            resp: Any = tool_exec.result
            # In Agno HITL confirmation, the user approves/denies the tool call.
            # When confirmed=True without a result, Agno executes the tool during
            # this continued run (emitting an execute_tool child span).
            if resp is None and tool_exec.confirmed is not None:
                resp = {"confirmed": tool_exec.confirmed}
                if getattr(tool_exec, "confirmation_note", None):
                    resp["confirmation_note"] = tool_exec.confirmation_note
            elif resp is None:
                input_schema = getattr(tool_exec, "user_input_schema", None)
                feedback_schema = getattr(
                    tool_exec, "user_feedback_schema", None
                )
                if input_schema:
                    resp = {
                        field.name: field.value
                        for field in input_schema
                        if getattr(field, "value", None) is not None
                    }
                elif feedback_schema:
                    resp = {
                        q.question: q.selected_options
                        for q in feedback_schema
                        if getattr(q, "selected_options", None) is not None
                    }

            messages.append(
                InputMessage(
                    role=Role.TOOL.value,
                    parts=[
                        ToolCallResponsePart(
                            id=tool_exec.tool_call_id,
                            response=resp,
                        )
                    ],
                )
            )

    input_val = (
        get_argument("input", wrapped, args, kwargs)
        if isinstance(invocation, LocalAgentInvocation)
        else None
    )
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

    if isinstance(invocation, LocalAgentInvocation):
        instructions_val = get_argument(
            "additional_instructions", wrapped, args, kwargs
        )
        if instructions_val is not None:
            instr_str = _extract_input_content(instructions_val)
            if instr_str:
                instructions: list[SystemInstructionPart] = [
                    TextPart(content=instr_str)
                ]
                invocation.system_instruction = instructions


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
    conversation_id = extract_session_id(
        instance, args, kwargs, wrapped=wrapped, is_continue=is_continue
    )
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
        conversation_id=conversation_id,
    )
    description = getattr(instance, "description", None)
    if description:
        invocation.agent_description = str(description)

    if is_continue:
        _set_continue_invocation_input(
            invocation, wrapped, args, kwargs, capture_content
        )
    else:
        _set_invocation_input(
            invocation, instance, args, kwargs, capture_content, wrapped
        )

    tool_defs = prepare_tool_definitions(getattr(instance, "tools", None))
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
            if isinstance(result, Iterator):
                return AgnoAgentStreamWrapper(
                    result,
                    invocation,
                    capture_content,
                )

            _set_invocation_output(invocation, result, capture_content)
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise

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
                result,
                invocation,
                capture_content,
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
                            response,
                            invocation,
                            capture_content,
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

    return traced_method


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
    conversation_id = extract_session_id(
        instance, args, kwargs, wrapped=wrapped, is_continue=is_continue
    )
    invocation = handler.workflow(
        name=workflow_name, conversation_id=conversation_id
    )
    if is_continue:
        _set_continue_invocation_input(
            invocation, wrapped, args, kwargs, capture_content
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
            if isinstance(result, Iterator):
                return AgnoWorkflowStreamWrapper(
                    result,
                    invocation,
                    capture_content,
                )

            _set_invocation_output(invocation, result, capture_content)
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise

    return traced_method


def _is_background_requested(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    return bool(get_argument("background", wrapped, args, kwargs))


def _extract_workflow_run_id(
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    try:
        bound = bind_arguments(wrapped, args, kwargs)
    except Exception:
        bound = kwargs
    if run_id := bound.get("run_id"):
        return str(run_id)
    if run_context := bound.get("run_context"):
        if run_id := getattr(run_context, "run_id", None):
            return str(run_id)
    if run_resp := bound.get("workflow_run_response"):
        if run_id := getattr(run_resp, "run_id", None):
            return str(run_id)
    if run_resp := bound.get("run_response"):
        if run_id := getattr(run_resp, "run_id", None):
            return str(run_id)
    return None


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
        # If background execution is requested, skip wrapping here.
        # Agno returns a pending placeholder immediately and runs execution
        # in a background task via _aexecute / _aexecute_stream.
        if not is_continue and _is_background_requested(wrapped, args, kwargs):
            return wrapped(*args, **kwargs)

        run_id = _extract_workflow_run_id(wrapped, args, kwargs)
        if not run_id:
            # Matches what agno does internally when no run id is set.
            run_id = str(uuid4())
            # Do not inject run_id on continue runs; Agno requires the caller
            # to supply run_response or run_id and fails validation otherwise.
            if not is_continue:
                kwargs["run_id"] = run_id

        _ACTIVE_WORKFLOW_RUN_IDS.add(run_id)

        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            _release_run_id(run_id)
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
                result,
                invocation,
                capture_content,
                on_close=functools.partial(_release_run_id, run_id),
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
                discard_on_exit = True
                try:
                    awaitable = cast(Awaitable[object], result)
                    response: object = await awaitable
                    if isinstance(response, AsyncIterator):
                        discard_on_exit = False
                        return AsyncAgnoWorkflowStreamWrapper(
                            response,
                            invocation,
                            capture_content,
                            on_close=functools.partial(
                                _release_run_id, run_id
                            ),
                        )
                    _set_invocation_output(
                        invocation, response, capture_content
                    )
                    invocation.stop()
                    return response
                except BaseException as error:
                    invocation.fail(error)
                    raise
                finally:
                    if discard_on_exit:
                        _release_run_id(run_id)

            return _await_result()

        try:
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
        finally:
            _release_run_id(run_id)

    return traced_method


def _start_background_workflow_invocation(
    handler: TelemetryHandler,
    instance: Any,
    wrapped: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> WorkflowInvocation:
    bound_args = bind_arguments(wrapped, args, kwargs)
    run_resp = bound_args.get("workflow_run_response")
    session_obj = bound_args.get("session")
    session_id_val = (
        bound_args.get("session_id")
        or getattr(run_resp, "session_id", None)
        or getattr(session_obj, "session_id", None)
    )
    session_id = (
        str(session_id_val)
        if session_id_val is not None
        else extract_session_id(instance, args, kwargs, wrapped=wrapped)
    )
    exec_input = bound_args.get("execution_input")
    input_val = getattr(exec_input, "input", None) or getattr(
        run_resp, "input", None
    )

    workflow_name = getattr(instance, "name", None)
    invocation = handler.workflow(
        name=workflow_name,
        conversation_id=session_id,
    )

    if capture_content and input_val is not None:
        content_str = _extract_input_content(input_val)
        if content_str:
            invocation.input_messages = [
                InputMessage(
                    role=Role.USER.value,
                    parts=[TextPart(content=content_str)],
                )
            ]

    return invocation


def _workflow_aexecute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        run_id = _extract_workflow_run_id(wrapped, args, kwargs)
        if run_id and run_id in _ACTIVE_WORKFLOW_RUN_IDS:
            return await wrapped(*args, **kwargs)

        if run_id:
            _ACTIVE_WORKFLOW_RUN_IDS.add(run_id)
        invocation = _start_background_workflow_invocation(
            handler, instance, wrapped, args, kwargs, capture_content
        )
        try:
            result = await wrapped(*args, **kwargs)
            _set_invocation_output(invocation, result, capture_content)
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise
        finally:
            _release_run_id(run_id)

    return traced_method


def _workflow_aexecute_stream(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        run_id = _extract_workflow_run_id(wrapped, args, kwargs)
        if run_id and run_id in _ACTIVE_WORKFLOW_RUN_IDS:
            return wrapped(*args, **kwargs)

        if run_id:
            _ACTIVE_WORKFLOW_RUN_IDS.add(run_id)
        invocation = _start_background_workflow_invocation(
            handler, instance, wrapped, args, kwargs, capture_content
        )
        try:
            result = wrapped(*args, **kwargs)
            if isinstance(result, AsyncIterator):
                return AsyncAgnoWorkflowStreamWrapper(
                    result,
                    invocation,
                    capture_content,
                    on_close=functools.partial(_release_run_id, run_id),
                )
            _release_run_id(run_id)
            _set_invocation_output(invocation, result, capture_content)
            invocation.stop()
            return result
        except BaseException as error:
            _release_run_id(run_id)
            invocation.fail(error)
            raise

    return traced_method


def _workflow_aexecute_workflow_agent(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        run_id = _extract_workflow_run_id(wrapped, args, kwargs)
        if run_id and run_id in _ACTIVE_WORKFLOW_RUN_IDS:
            return wrapped(*args, **kwargs)

        bound_args = bind_arguments(wrapped, args, kwargs)
        user_input = bound_args.get("user_input")
        run_context = bound_args.get("run_context")
        session_id = (
            str(sid)
            if (sid := getattr(run_context, "session_id", None)) is not None
            else extract_session_id(instance, args, kwargs, wrapped=wrapped)
        )

        workflow_name = getattr(instance, "name", None)
        invocation = handler.workflow(
            name=workflow_name,
            conversation_id=session_id,
        )

        if capture_content and user_input is not None:
            content_str = _extract_input_content(user_input)
            if content_str:
                invocation.input_messages = [
                    InputMessage(
                        role=Role.USER.value,
                        parts=[TextPart(content=content_str)],
                    )
                ]

        if run_id:
            _ACTIVE_WORKFLOW_RUN_IDS.add(run_id)

        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            _release_run_id(run_id)
            invocation.fail(error)
            raise

        if isinstance(result, AsyncIterator):
            return AsyncAgnoWorkflowStreamWrapper(
                result,
                invocation,
                capture_content,
                on_close=functools.partial(_release_run_id, run_id),
            )

        if isinstance(result, Awaitable):

            @functools.wraps(wrapped)
            async def _await_result() -> object:
                discard_on_exit = True
                try:
                    awaitable = cast(Awaitable[object], result)
                    response: object = await awaitable
                    if isinstance(response, AsyncIterator):
                        discard_on_exit = False
                        return AsyncAgnoWorkflowStreamWrapper(
                            response,
                            invocation,
                            capture_content,
                            on_close=functools.partial(
                                _release_run_id, run_id
                            ),
                        )
                    _set_invocation_output(
                        invocation, response, capture_content
                    )
                    invocation.stop()
                    return response
                except BaseException as error:
                    invocation.fail(error)
                    raise
                finally:
                    if discard_on_exit:
                        _release_run_id(run_id)

            return _await_result()

        try:
            _set_invocation_output(invocation, result, capture_content)
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise
        finally:
            _release_run_id(run_id)

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

    return traced_method
