# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Agno instrumentation."""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import contextvars
import functools
import json
import logging
import sys
import urllib.parse
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
    from agno.knowledge.embedder.base import Embedder
    from agno.knowledge.knowledge import Knowledge
    from agno.models.base import MessageData, Model
    from agno.models.message import Message
    from agno.models.response import ModelResponse
    from agno.run.workflow import WorkflowRunOutput
    from agno.team import Team, TeamRunOutput
    from agno.tools.function import FunctionCall, FunctionExecutionResult
    from agno.workflow import Workflow

    AgnoRunOutput = RunOutput | TeamRunOutput | WorkflowRunOutput

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.instrumentation.genai.agno.stream import (
    AgnoAgentStreamWrapper,
    AgnoModelStreamWrapper,
    AgnoWorkflowStreamWrapper,
    AsyncAgnoAgentStreamWrapper,
    AsyncAgnoModelStreamWrapper,
    AsyncAgnoWorkflowStreamWrapper,
)
from opentelemetry.instrumentation.genai.agno.utils import (
    _get_property_value,
    extract_model_finish_reasons,
    extract_session_id,
    extract_user_id,
    format_content,
    format_model_input_messages,
    format_model_output_message,
    format_retrieval_document,
    has_model_output_content,
    prepare_tool_definitions,
    resolve_embedder_provider,
    resolve_model_provider,
    safe_float,
    safe_int,
    set_invocation_user_id,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ErrorTypeValues,
)
from opentelemetry.semconv._incubating.attributes.user_attributes import (
    USER_ID,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
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
from opentelemetry.util.genai.utils import bind_arguments, get_argument

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
_AGNO_MODELS_MODULE = "agno.models.base"
_MODEL_CLASS = "Model"

_ACTIVE_FOREGROUND_WORKFLOWS: contextvars.ContextVar[frozenset[int]] = (
    contextvars.ContextVar("_ACTIVE_FOREGROUND_WORKFLOWS", default=frozenset())
)
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
    _safe_wrap_function(
        _AGNO_MODELS_MODULE,
        f"{_MODEL_CLASS}._process_model_response",
        _model_process_response(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_MODELS_MODULE,
        f"{_MODEL_CLASS}._aprocess_model_response",
        _model_aprocess_response(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_MODELS_MODULE,
        f"{_MODEL_CLASS}.process_response_stream",
        _model_process_response_stream(handler),
        current_generation,
    )
    _safe_wrap_function(
        _AGNO_MODELS_MODULE,
        f"{_MODEL_CLASS}.aprocess_response_stream",
        _model_aprocess_response_stream(handler),
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
    if _AGNO_MODELS_MODULE in sys.modules:
        try:
            import agno.models.base

            for attr in (
                "_process_model_response",
                "_aprocess_model_response",
                "process_response_stream",
                "aprocess_response_stream",
            ):
                _safe_unwrap(agno.models.base.Model, attr)
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
        set_invocation_user_id(
            invocation, instance, args, kwargs, wrapped=wrapped
        )
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
        set_invocation_user_id(
            invocation, instance, args, kwargs, wrapped=wrapped
        )
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
        set_invocation_user_id(
            invocation, instance, args, kwargs, wrapped=wrapped
        )
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
        set_invocation_user_id(
            invocation, instance, args, kwargs, wrapped=wrapped
        )
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
        session_id = extract_session_id(
            instance, args, kwargs, wrapped=wrapped
        )
        if session_id:
            invocation.conversation_id = str(session_id)

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
    set_invocation_user_id(invocation, instance)
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
        user_id = extract_user_id(instance, args, kwargs, wrapped=wrapped)
        if user_id is not None:
            invocation.attributes[USER_ID] = user_id
        try:
            result = wrapped(*args, **kwargs)
            if isinstance(result, Iterator):
                return AgnoAgentStreamWrapper(
                    result,
                    invocation,
                    capture_content,
                )

            _set_invocation_output(invocation, result, capture_content)
            set_invocation_user_id(
                invocation,
                instance,
                args,
                kwargs,
                run_response=result,
                wrapped=wrapped,
            )
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
            set_invocation_user_id(
                invocation, instance, args, kwargs, wrapped=wrapped
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
            user_id = extract_user_id(instance, args, kwargs, wrapped=wrapped)
            if user_id is not None:
                invocation.attributes[USER_ID] = user_id
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
                user_id = extract_user_id(
                    instance, args, kwargs, wrapped=wrapped
                )
                if user_id is not None:
                    invocation.attributes[USER_ID] = user_id
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
                    set_invocation_user_id(
                        invocation,
                        instance,
                        args,
                        kwargs,
                        run_response=response,
                        wrapped=wrapped,
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
        set_invocation_user_id(
            invocation, instance, args, kwargs, wrapped=wrapped
        )
        _set_invocation_output(invocation, result, capture_content)
        set_invocation_user_id(
            invocation,
            instance,
            args,
            kwargs,
            run_response=result,
            wrapped=wrapped,
        )
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
        session_id = extract_session_id(
            instance, args, kwargs, wrapped=wrapped
        )
        if session_id:
            invocation.conversation_id = str(session_id)
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
        user_id = extract_user_id(instance, args, kwargs, wrapped=wrapped)
        if user_id is not None:
            invocation.attributes[USER_ID] = user_id
        try:
            result = wrapped(*args, **kwargs)
            if isinstance(result, Iterator):
                return AgnoWorkflowStreamWrapper(
                    result,
                    invocation,
                    capture_content,
                )

            _set_invocation_output(invocation, result, capture_content)
            set_invocation_user_id(
                invocation,
                instance,
                args,
                kwargs,
                run_response=result,
                wrapped=wrapped,
            )
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
        if _is_background_requested(wrapped, args, kwargs):
            return wrapped(*args, **kwargs)

        instance_id = id(instance)
        token = _ACTIVE_FOREGROUND_WORKFLOWS.set(
            _ACTIVE_FOREGROUND_WORKFLOWS.get() | {instance_id}
        )
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            _ACTIVE_FOREGROUND_WORKFLOWS.reset(token)
            invocation = _start_workflow_invocation(
                handler,
                instance,
                args,
                kwargs,
                capture_content,
                wrapped=wrapped,
                is_continue=is_continue,
            )
            set_invocation_user_id(
                invocation, instance, args, kwargs, wrapped=wrapped
            )
            invocation.fail(error)
            raise

        if isinstance(result, AsyncIterator):
            _ACTIVE_FOREGROUND_WORKFLOWS.reset(token)
            invocation = _start_workflow_invocation(
                handler,
                instance,
                args,
                kwargs,
                capture_content,
                wrapped=wrapped,
                is_continue=is_continue,
            )
            user_id = extract_user_id(instance, args, kwargs, wrapped=wrapped)
            if user_id is not None:
                invocation.attributes[USER_ID] = user_id
            return AsyncAgnoWorkflowStreamWrapper(
                result,
                invocation,
                capture_content,
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
                user_id = extract_user_id(
                    instance, args, kwargs, wrapped=wrapped
                )
                if user_id is not None:
                    invocation.attributes[USER_ID] = user_id
                sub_token = _ACTIVE_FOREGROUND_WORKFLOWS.set(
                    _ACTIVE_FOREGROUND_WORKFLOWS.get() | {instance_id}
                )
                try:
                    awaitable = cast(Awaitable[object], result)
                    response: object = await awaitable
                    if isinstance(response, AsyncIterator):
                        return AsyncAgnoWorkflowStreamWrapper(
                            response,
                            invocation,
                            capture_content,
                        )
                    _set_invocation_output(
                        invocation, response, capture_content
                    )
                    set_invocation_user_id(
                        invocation,
                        instance,
                        args,
                        kwargs,
                        run_response=response,
                        wrapped=wrapped,
                    )
                    invocation.stop()
                    return response
                except BaseException as error:
                    invocation.fail(error)
                    raise
                finally:
                    _ACTIVE_FOREGROUND_WORKFLOWS.reset(sub_token)

            _ACTIVE_FOREGROUND_WORKFLOWS.reset(token)
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
            set_invocation_user_id(
                invocation, instance, args, kwargs, wrapped=wrapped
            )
            _set_invocation_output(invocation, result, capture_content)
            set_invocation_user_id(
                invocation,
                instance,
                args,
                kwargs,
                run_response=result,
                wrapped=wrapped,
            )
            invocation.stop()
            return result
        finally:
            _ACTIVE_FOREGROUND_WORKFLOWS.reset(token)

    return traced_method


def _workflow_aexecute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Awaitable[WorkflowRunOutput]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> WorkflowRunOutput:
        if id(instance) in _ACTIVE_FOREGROUND_WORKFLOWS.get():
            return await wrapped(*args, **kwargs)

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
        user_id_val = (
            bound_args.get("user_id")
            or getattr(run_resp, "user_id", None)
            or getattr(session_obj, "user_id", None)
        )
        user_id = (
            str(user_id_val)
            if user_id_val is not None
            else extract_user_id(instance, args, kwargs, wrapped=wrapped)
        )
        exec_input = bound_args.get("execution_input")
        input_val = getattr(exec_input, "input", None) or getattr(
            run_resp, "input", None
        )

        workflow_name = getattr(instance, "name", None)
        invocation = handler.workflow(name=workflow_name)
        if session_id is not None:
            invocation.conversation_id = str(session_id)
        if user_id is not None:
            invocation.attributes[USER_ID] = str(user_id)

        if capture_content and input_val is not None:
            content_str = _extract_input_content(input_val)
            if content_str:
                invocation.input_messages = [
                    InputMessage(
                        role=Role.USER.value,
                        parts=[TextPart(content=content_str)],
                    )
                ]

        try:
            result = await wrapped(*args, **kwargs)
            _set_invocation_output(invocation, result, capture_content)
            set_invocation_user_id(
                invocation,
                instance,
                args,
                kwargs,
                run_response=result,
                wrapped=wrapped,
            )
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise

    return cast(Callable[..., Any], traced_method)


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
        if id(instance) in _ACTIVE_FOREGROUND_WORKFLOWS.get():
            return wrapped(*args, **kwargs)

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
        user_id_val = (
            bound_args.get("user_id")
            or getattr(run_resp, "user_id", None)
            or getattr(session_obj, "user_id", None)
        )
        user_id = (
            str(user_id_val)
            if user_id_val is not None
            else extract_user_id(instance, args, kwargs, wrapped=wrapped)
        )
        exec_input = bound_args.get("execution_input")
        input_val = getattr(exec_input, "input", None) or getattr(
            run_resp, "input", None
        )

        workflow_name = getattr(instance, "name", None)
        invocation = handler.workflow(name=workflow_name)
        if session_id is not None:
            invocation.conversation_id = str(session_id)
        if user_id is not None:
            invocation.attributes[USER_ID] = str(user_id)

        if capture_content and input_val is not None:
            content_str = _extract_input_content(input_val)
            if content_str:
                invocation.input_messages = [
                    InputMessage(
                        role=Role.USER.value,
                        parts=[TextPart(content=content_str)],
                    )
                ]

        try:
            result = wrapped(*args, **kwargs)
            if isinstance(result, AsyncIterator):
                return AsyncAgnoWorkflowStreamWrapper(
                    result,
                    invocation,
                    capture_content,
                )
            return result
        except BaseException as error:
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
        if id(instance) in _ACTIVE_FOREGROUND_WORKFLOWS.get():
            return wrapped(*args, **kwargs)

        bound_args = bind_arguments(wrapped, args, kwargs)
        user_input = bound_args.get("user_input")
        run_context = bound_args.get("run_context")
        session_id = (
            str(sid)
            if (sid := getattr(run_context, "session_id", None)) is not None
            else extract_session_id(instance, args, kwargs, wrapped=wrapped)
        )
        user_id = (
            str(uid)
            if (uid := getattr(run_context, "user_id", None)) is not None
            else extract_user_id(instance, args, kwargs, wrapped=wrapped)
        )

        workflow_name = getattr(instance, "name", None)
        invocation = handler.workflow(name=workflow_name)
        if session_id is not None:
            invocation.conversation_id = str(session_id)
        if user_id is not None:
            invocation.attributes[USER_ID] = str(user_id)

        if capture_content and user_input is not None:
            content_str = _extract_input_content(user_input)
            if content_str:
                invocation.input_messages = [
                    InputMessage(
                        role=Role.USER.value,
                        parts=[TextPart(content=content_str)],
                    )
                ]

        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise

        if isinstance(result, AsyncIterator):
            return AsyncAgnoWorkflowStreamWrapper(
                result,
                invocation,
                capture_content,
            )

        if isinstance(result, Awaitable):

            @functools.wraps(wrapped)
            async def _await_result() -> object:
                try:
                    awaitable = cast(Awaitable[object], result)
                    response: object = await awaitable
                    if isinstance(response, AsyncIterator):
                        return AsyncAgnoWorkflowStreamWrapper(
                            response,
                            invocation,
                            capture_content,
                        )
                    _set_invocation_output(
                        invocation, response, capture_content
                    )
                    set_invocation_user_id(
                        invocation,
                        instance,
                        args,
                        kwargs,
                        run_response=response,
                        wrapped=wrapped,
                    )
                    invocation.stop()
                    return response
                except BaseException as error:
                    invocation.fail(error)
                    raise

            return _await_result()

        try:
            set_invocation_user_id(
                invocation, instance, args, kwargs, wrapped=wrapped
            )
            _set_invocation_output(invocation, result, capture_content)
            set_invocation_user_id(
                invocation,
                instance,
                args,
                kwargs,
                run_response=result,
                wrapped=wrapped,
            )
            invocation.stop()
            return result
        except BaseException as error:
            invocation.fail(error)
            raise

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
    set_invocation_user_id(invocation, instance, args, kwargs, wrapped=wrapped)

    if invocation.should_capture_content:
        query = get_argument("query", wrapped, args, kwargs)
        if query is not None:
            invocation.query_text = str(query)

    max_results = get_argument("max_results", wrapped, args, kwargs)
    if max_results is None:
        max_results = instance.max_results

    invocation.top_k = safe_int(max_results)
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


def _start_model_inference(
    handler: TelemetryHandler,
    instance: Model,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    wrapped: Callable[..., Any],
) -> tuple[
    InferenceInvocation,
    Message | None,
    ModelResponse | MessageData | None,
]:
    """Start an InferenceInvocation for an Agno model response call."""
    messages = cast(
        "Sequence[Message] | None",
        get_argument("messages", wrapped, args, kwargs),
    )
    assistant_message = cast(
        "Message | None",
        get_argument("assistant_message", wrapped, args, kwargs),
    )
    response_obj = cast(
        "ModelResponse | MessageData | None",
        get_argument("model_response", wrapped, args, kwargs)
        or get_argument("stream_data", wrapped, args, kwargs),
    )
    tools = cast(
        "Iterable[Any] | str | None",
        get_argument("tools", wrapped, args, kwargs),
    )
    run_response = get_argument("run_response", wrapped, args, kwargs)

    provider = resolve_model_provider(instance)
    request_model = (
        getattr(instance, "id", None)
        or getattr(instance, "model", None)
        or getattr(instance, "name", None)
    )
    server_address: str | None = None
    server_port: int | None = None
    base_url = getattr(instance, "base_url", None)
    if base_url:
        try:
            parsed = urllib.parse.urlparse(str(base_url))
            server_address = parsed.hostname
            server_port = safe_int(parsed.port)
        except Exception:
            pass

    session_id = extract_session_id(
        instance=instance,
        args=args,
        kwargs=kwargs,
        run_response=run_response,
        wrapped=wrapped,
    )
    invocation = handler.inference(
        provider,
        request_model=str(request_model)
        if request_model is not None
        else None,
        server_address=server_address,
        server_port=server_port,
        conversation_id=str(session_id) if session_id is not None else None,
    )

    invocation.temperature = safe_float(getattr(instance, "temperature", None))
    invocation.top_p = safe_float(getattr(instance, "top_p", None))
    invocation.top_k = safe_int(getattr(instance, "top_k", None))
    invocation.max_tokens = safe_int(
        getattr(instance, "max_tokens", None)
        or getattr(instance, "max_completion_tokens", None)
    )
    invocation.frequency_penalty = safe_float(
        getattr(instance, "frequency_penalty", None)
    )
    invocation.presence_penalty = safe_float(
        getattr(instance, "presence_penalty", None)
    )

    stop = getattr(instance, "stop_sequences", None) or getattr(
        instance, "stop", None
    )
    if isinstance(stop, str):
        invocation.stop_sequences = [stop]
    elif isinstance(stop, (list, tuple)):
        stop_seqs = cast(Sequence[object], stop)
        invocation.stop_sequences = [str(s) for s in stop_seqs]

    invocation.seed = safe_int(getattr(instance, "seed", None))

    if tools:
        invocation.tool_definitions = prepare_tool_definitions(tools)

    if invocation.should_capture_content and messages:
        invocation.input_messages = format_model_input_messages(messages)

    set_invocation_user_id(
        invocation,
        instance=instance,
        args=args,
        kwargs=kwargs,
        run_response=run_response,
        wrapped=wrapped,
    )

    return invocation, assistant_message, response_obj


def _populate_model_response_telemetry(
    invocation: InferenceInvocation,
    assistant_message: Message | None,
    model_response: ModelResponse | None,
) -> None:
    """Populate final response telemetry on an InferenceInvocation."""
    provider_data = None
    if assistant_message is not None:
        provider_data = getattr(assistant_message, "provider_data", None)
    if not provider_data and model_response is not None:
        provider_data = getattr(model_response, "provider_data", None)

    if isinstance(provider_data, dict):
        provider_dict = cast(dict[str, Any], provider_data)
        if "id" in provider_dict and not invocation.response_id:
            invocation.response_id = str(cast(object, provider_dict["id"]))
        if "model" in provider_dict and not invocation.response_model_name:
            invocation.response_model_name = str(
                cast(object, provider_dict["model"])
            )

    source_metrics = None
    if assistant_message is not None:
        source_metrics = getattr(assistant_message, "metrics", None)
    if source_metrics is None and model_response is not None:
        source_metrics = getattr(
            model_response, "response_usage", model_response
        )

    invocation.input_tokens = safe_int(
        getattr(source_metrics, "input_tokens", None)
    )
    invocation.output_tokens = safe_int(
        getattr(source_metrics, "output_tokens", None)
    )
    invocation.cache_read_input_tokens = safe_int(
        getattr(source_metrics, "cache_read_tokens", None)
    )
    invocation.cache_write_input_tokens = safe_int(
        getattr(source_metrics, "cache_write_tokens", None)
    )
    invocation.thinking_tokens = safe_int(
        getattr(source_metrics, "reasoning_tokens", None)
    )

    finish_reasons = extract_model_finish_reasons(
        assistant_message, model_response
    )
    invocation.finish_reasons = finish_reasons

    if invocation.should_capture_content:
        if assistant_message is not None and has_model_output_content(
            assistant_message
        ):
            invocation.output_messages = [
                format_model_output_message(
                    assistant_message,
                    finish_reason=finish_reasons[0]
                    if finish_reasons
                    else "stop",
                )
            ]
        elif model_response is not None and has_model_output_content(
            model_response
        ):
            invocation.output_messages = [
                format_model_output_message(
                    model_response,
                    finish_reason=finish_reasons[0]
                    if finish_reasons
                    else "stop",
                )
            ]


def _model_process_response(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Model,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation, assistant_message, model_response = _start_model_inference(
            handler, instance, args, kwargs, wrapped=wrapped
        )
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise

        _populate_model_response_telemetry(
            invocation,
            assistant_message=assistant_message,
            model_response=cast("ModelResponse | None", model_response),
        )
        invocation.stop()
        return result

    return traced_method


def _model_aprocess_response(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Model,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation, assistant_message, model_response = _start_model_inference(
            handler, instance, args, kwargs, wrapped=wrapped
        )
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise

        _populate_model_response_telemetry(
            invocation,
            assistant_message=assistant_message,
            model_response=cast("ModelResponse | None", model_response),
        )
        invocation.stop()
        return result

    return cast(Callable[..., Any], traced_method)


def _model_process_response_stream(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Model,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation, assistant_message, stream_data = _start_model_inference(
            handler, instance, args, kwargs, wrapped=wrapped
        )
        try:
            stream = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise

        return AgnoModelStreamWrapper(
            stream,
            invocation=invocation,
            assistant_message=assistant_message,
            stream_data=cast("MessageData | None", stream_data),
        )

    return traced_method


def _model_aprocess_response_stream(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    def traced_method(
        wrapped: Callable[..., Any],
        instance: Model,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        invocation, assistant_message, stream_data = _start_model_inference(
            handler, instance, args, kwargs, wrapped=wrapped
        )
        try:
            stream = wrapped(*args, **kwargs)
        except BaseException as error:
            invocation.fail(error)
            raise

        return AsyncAgnoModelStreamWrapper(
            stream,
            invocation=invocation,
            assistant_message=assistant_message,
            stream_data=cast("MessageData | None", stream_data),
        )

    return traced_method
