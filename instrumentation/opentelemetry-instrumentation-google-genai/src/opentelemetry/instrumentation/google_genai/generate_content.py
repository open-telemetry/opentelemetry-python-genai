# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import json
import os
from typing import Any, AsyncIterator, Awaitable, Iterator, Optional, Union

from google.genai.models import AsyncModels, Models
from google.genai.models import t as transformers
from google.genai.types import (
    ContentListUnion,
    ContentListUnionDict,
    ContentUnion,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    Tool,
    ToolListUnionDict,
    ToolUnionDict,
)

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import get_content_capturing_mode
from opentelemetry.util.types import AttributeValue

from .allowlist_util import AllowList
from .custom_semconv import GCP_GENAI_OPERATION_CONFIG
from .dict_util import flatten_dict
from .message import (
    to_input_messages,
    to_output_messages,
    to_system_instructions,
)
from .tool_call_wrapper import wrapped_tool

_is_mcp_imported = False
McpClientSession = McpTool = None
try:
    from mcp import ClientSession as McpClientSession
    from mcp import Tool as McpTool

    _is_mcp_imported = True
except ImportError:
    pass

GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY = context_api.create_key(
    "generate_content_extra_attributes_context_key"
)


class _MethodsSnapshot:
    def __init__(self):
        self._original_generate_content = Models.generate_content
        self._original_generate_content_stream = Models.generate_content_stream
        self._original_async_generate_content = AsyncModels.generate_content
        self._original_async_generate_content_stream = (
            AsyncModels.generate_content_stream
        )

    @property
    def generate_content(self):
        return self._original_generate_content

    @property
    def generate_content_stream(self):
        return self._original_generate_content_stream

    @property
    def async_generate_content(self):
        return self._original_async_generate_content

    @property
    def async_generate_content_stream(self):
        return self._original_async_generate_content_stream

    def restore(self):
        Models.generate_content = self._original_generate_content
        Models.generate_content_stream = self._original_generate_content_stream
        AsyncModels.generate_content = self._original_async_generate_content
        AsyncModels.generate_content_stream = (
            self._original_async_generate_content_stream
        )


def _guess_genai_system_from_env():
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in [
        "true",
        "1",
    ]:
        return gen_ai_attributes.GenAiSystemValues.VERTEX_AI.name.lower()
    return gen_ai_attributes.GenAiSystemValues.GEMINI.name.lower()


def _get_is_vertexai(models_object: Union[Models, AsyncModels]):
    # Since commit 8e561de04965bb8766db87ad8eea7c57c1040442 of "googleapis/python-genai",
    # it is possible to obtain the information using a documented property.
    if hasattr(models_object, "vertexai"):
        vertexai_attr = getattr(models_object, "vertexai")
        if vertexai_attr is not None:
            return vertexai_attr
    # For earlier revisions, it is necessary to deeply inspect the internals.
    if hasattr(models_object, "_api_client"):
        client = getattr(models_object, "_api_client")
        if not client:
            return None
        if hasattr(client, "vertexai"):
            return getattr(client, "vertexai")
    return None


def _determine_genai_system(models_object: Union[Models, AsyncModels]):
    vertexai_attr = _get_is_vertexai(models_object)
    if vertexai_attr is None:
        return _guess_genai_system_from_env()
    if vertexai_attr:
        return gen_ai_attributes.GenAiSystemValues.VERTEX_AI.name.lower()
    return gen_ai_attributes.GenAiSystemValues.GEMINI.name.lower()


def _to_dict(value: object):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except TypeError:
            return {"ModelName": str(value)}

    return json.loads(json.dumps(value))


def _model_dump_to_tool_definition(tool: Any) -> ToolDefinition:
    model_dump = tool.model_dump(exclude_none=True)

    name = (
        model_dump.get("name")
        or getattr(tool, "name", None)
        or type(tool).__name__
    )
    description = model_dump.get("description") or getattr(
        tool, "description", None
    )
    parameters = model_dump.get("parameters") or model_dump.get("inputSchema")
    return FunctionToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
    )


def _clean_parameters(params: Any) -> Any:
    """Converts parameter objects into plain dicts."""
    if params is None:
        return None
    if isinstance(params, dict):
        return params
    if hasattr(params, "to_dict"):
        return params.to_dict()
    if hasattr(params, "model_dump"):
        return params.model_dump(exclude_none=True)

    try:
        # Check if it's already a standard JSON type.
        json.dumps(params)
        return params

    except (TypeError, ValueError):
        return {
            "type": "object",
            "properties": {
                "serialization_error": {
                    "type": "string",
                    "description": f"Failed to serialize parameters: {type(params).__name__}",
                }
            },
        }


def _tool_to_tool_definition(tool: Tool) -> list[ToolDefinition]:
    definitions = []
    if tool.function_declarations:
        for fd in tool.function_declarations:
            definitions.append(
                FunctionToolDefinition(
                    name=getattr(fd, "name", type(fd).__name__),
                    description=getattr(fd, "description", None),
                    parameters=_clean_parameters(
                        getattr(fd, "parameters", None)
                    ),
                )
            )

    # Generic types
    if hasattr(tool, "model_dump"):
        exclude_fields = {"function_declarations"}
        fields = {
            k: v
            for k, v in tool.model_dump().items()
            if v is not None and k not in exclude_fields
        }

        for tool_type, _ in fields.items():
            definitions.append(
                GenericToolDefinition(
                    type=tool_type,
                    name=tool_type,
                )
            )

    return definitions


def _callable_tool_to_tool_definition(tool: Any) -> ToolDefinition:
    doc = getattr(tool, "__doc__", "") or ""
    return FunctionToolDefinition(
        name=getattr(tool, "__name__", type(tool).__name__),
        description=doc.strip(),
        parameters=None,
    )


def _mcp_tool_to_tool_definition(tool: McpTool) -> ToolDefinition:
    if hasattr(tool, "model_dump"):
        return _model_dump_to_tool_definition(tool)

    return FunctionToolDefinition(
        name=getattr(tool, "name", type(tool).__name__),
        description=getattr(tool, "description", None),
        parameters=getattr(tool, "input_schema", None),
    )


def _to_tool_definition_common(tool: ToolUnionDict) -> list[ToolDefinition]:
    if isinstance(tool, Tool):
        return _tool_to_tool_definition(tool)

    if callable(tool):
        return [_callable_tool_to_tool_definition(tool)]

    if _is_mcp_imported and isinstance(tool, McpTool):
        return [_mcp_tool_to_tool_definition(tool)]

    return [
        GenericToolDefinition(
            name="UnserializableTool",
            type=type(tool).__name__,
        )
    ]


def _to_tool_definition(tool: ToolUnionDict) -> list[ToolDefinition]:
    if _is_mcp_imported and isinstance(tool, McpClientSession):
        return []

    return _to_tool_definition_common(tool)


async def _to_tool_definition_async(
    tool: ToolUnionDict,
) -> list[ToolDefinition]:
    if _is_mcp_imported and isinstance(tool, McpClientSession):
        result = await tool.list_tools()
        return [_model_dump_to_tool_definition(t) for t in result.tools]

    return _to_tool_definition_common(tool)


def _create_request_attributes(
    config: Optional[GenerateContentConfigOrDict],
    allow_list: AllowList,
) -> dict[str, AttributeValue]:
    if not config:
        return {}
    config = _to_dict(config)
    attributes = flatten_dict(
        config,
        # A custom prefix is used, because the names/structure of the
        # configuration is likely to be specific to Google Gen AI SDK.
        key_prefix=GCP_GENAI_OPERATION_CONFIG,
        exclude_keys=[
            # System instruction can be overly long for a span attribute.
            # Additionally, it is recorded as an event (log), instead.
            "gcp.gen_ai.operation.config.system_instruction",
        ],
        # Although a custom prefix is used by default, some of the attributes
        # are captured in common, standard, Semantic Conventions. For the
        # well-known properties whose values align with Semantic Conventions,
        # we ensure that the key name matches the standard SemConv name.
        rename_keys={
            # TODO: add more entries here as more semantic conventions are
            # generalized to cover more of the available config options.
            "gcp.gen_ai.operation.config.temperature": gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE,
            "gcp.gen_ai.operation.config.top_k": gen_ai_attributes.GEN_AI_REQUEST_TOP_K,
            "gcp.gen_ai.operation.config.top_p": gen_ai_attributes.GEN_AI_REQUEST_TOP_P,
            "gcp.gen_ai.operation.config.candidate_count": gen_ai_attributes.GEN_AI_REQUEST_CHOICE_COUNT,
            "gcp.gen_ai.operation.config.max_output_tokens": gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS,
            "gcp.gen_ai.operation.config.stop_sequences": gen_ai_attributes.GEN_AI_REQUEST_STOP_SEQUENCES,
            "gcp.gen_ai.operation.config.frequency_penalty": gen_ai_attributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
            "gcp.gen_ai.operation.config.presence_penalty": gen_ai_attributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
            "gcp.gen_ai.operation.config.seed": gen_ai_attributes.GEN_AI_REQUEST_SEED,
        },
    )
    response_mime_type = config.get("response_mime_type")
    if response_mime_type:
        if response_mime_type == "text/plain":
            attributes[gen_ai_attributes.GEN_AI_OUTPUT_TYPE] = "text"
        elif response_mime_type == "application/json":
            attributes[gen_ai_attributes.GEN_AI_OUTPUT_TYPE] = "json"
        else:
            attributes[gen_ai_attributes.GEN_AI_OUTPUT_TYPE] = (
                response_mime_type
            )
    for key in list(attributes.keys()):
        if key.startswith(
            GCP_GENAI_OPERATION_CONFIG
        ) and not allow_list.allowed(key):
            del attributes[key]
    return attributes


def _get_response_property(response: GenerateContentResponse, path: str):
    path_segments = path.split(".")
    current_context = response
    for path_segment in path_segments:
        if current_context is None:
            return None
        if isinstance(current_context, dict):
            current_context = current_context.get(path_segment)
        else:
            current_context = getattr(current_context, path_segment)
    return current_context


def _coerce_config_to_object(
    config: GenerateContentConfigOrDict,
) -> GenerateContentConfig:
    if isinstance(config, GenerateContentConfig):
        return config
    # Input must be a dictionary; convert by invoking the constructor.
    return GenerateContentConfig(**config)


def _wrapped_config_with_tools(
    telemetry_handler: TelemetryHandler,
    config: GenerateContentConfig,
):
    if not config.tools:
        return config
    result = copy.copy(config)
    result.tools = [
        wrapped_tool(tool, telemetry_handler) for tool in config.tools
    ]
    return result


def _config_to_system_instruction(
    config: Union[GenerateContentConfigOrDict, None],
) -> Union[ContentUnion, None]:
    if not config:
        return None

    if isinstance(config, dict):
        return GenerateContentConfig.model_validate(config).system_instruction
    return config.system_instruction


def _config_to_tools(
    config: Union[GenerateContentConfigOrDict, None],
) -> Union[ToolListUnionDict, None]:
    if not config:
        return None

    if isinstance(config, dict):
        return GenerateContentConfig.model_validate(config).tools
    return config.tools


def _get_extra_generate_content_attributes() -> dict[str, AttributeValue]:
    attrs = context_api.get_value(
        GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
    )
    return dict(attrs or {})


def _maybe_update_token_counts_and_finish_reasons(
    response: GenerateContentResponse,
    finish_reasons: list[str],
    invocation: InferenceInvocation,
):
    for candidate in response.candidates or []:
        if candidate.finish_reason:
            finish_reasons.append(candidate.finish_reason.value.lower())
    invocation.finish_reasons = finish_reasons
    input_tokens = _get_response_property(
        response, "usage_metadata.prompt_token_count"
    )
    output_tokens = _get_response_property(
        response, "usage_metadata.candidates_token_count"
    )
    cached_tokens = _get_response_property(
        response, "usage_metadata.cached_content_token_count"
    )
    thinking_tokens = _get_response_property(
        response, "usage_metadata.thoughts_token_count"
    )
    if cached_tokens is not None and isinstance(cached_tokens, int):
        invocation.cache_read_input_tokens = cached_tokens
    if input_tokens is not None and isinstance(input_tokens, int):
        invocation.input_tokens = input_tokens
    if output_tokens is not None and isinstance(output_tokens, int):
        invocation.output_tokens = output_tokens
    if thinking_tokens is not None and isinstance(thinking_tokens, int):
        # The util library will add this total to output tokens.
        invocation.thinking_tokens = thinking_tokens


def _maybe_get_tool_definitions(config) -> list[ToolDefinition]:
    if tools := _config_to_tools(config):
        return [de for tool in tools for de in _to_tool_definition(tool) if de]
    return []


async def _maybe_get_tool_definitions_async(config) -> list[ToolDefinition]:
    tool_definitions = []
    if tools := _config_to_tools(config):
        for tool in tools:
            definitions = await _to_tool_definition_async(tool)
            for de in definitions:
                if de:
                    tool_definitions.append(de)

    return tool_definitions


def _create_instrumented_generate_content(
    snapshot: _MethodsSnapshot,
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
    content_recording_enabled: bool,
):
    wrapped_func = snapshot.generate_content

    @functools.wraps(wrapped_func)
    def instrumented_generate_content(
        self: Models,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> GenerateContentResponse:
        wrapped_config = (
            _wrapped_config_with_tools(
                telemetry_handler,
                _coerce_config_to_object(config),
            )
            if config
            else None
        )
        finish_reasons = []
        extra_attributes = (
            _get_extra_generate_content_attributes()
            | _create_request_attributes(
                config,
                generate_content_config_key_allowlist,
            )
        )
        with telemetry_handler.inference(
            provider=_determine_genai_system(self),
            request_model=model,
            operation_name="generate_content",
        ) as invocation:
            invocation.attributes.update(extra_attributes)
            invocation.tool_definitions = _maybe_get_tool_definitions(config)

            if content_recording_enabled:
                invocation.input_messages = to_input_messages(
                    contents=transformers.t_contents(contents)
                )
                if system_content := _config_to_system_instruction(config):
                    invocation.system_instruction = to_system_instructions(
                        content=transformers.t_contents(system_content)[0]
                    )
            candidates = []
            try:
                response = wrapped_func(
                    self,
                    model=model,
                    contents=contents,
                    config=wrapped_config,
                    **kwargs,
                )
                _maybe_update_token_counts_and_finish_reasons(
                    response, finish_reasons, invocation
                )
                if response.candidates:
                    candidates.extend(response.candidates)
                return response
            finally:
                if content_recording_enabled and candidates:
                    invocation.output_messages = to_output_messages(
                        candidates=candidates
                    )

    return instrumented_generate_content


def _create_instrumented_generate_content_stream(
    snapshot: _MethodsSnapshot,
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
    content_recording_enabled: bool,
):
    wrapped_func = snapshot.generate_content_stream

    @functools.wraps(wrapped_func)
    def instrumented_generate_content_stream(
        self: Models,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> Iterator[GenerateContentResponse]:
        wrapped_config = (
            _wrapped_config_with_tools(
                telemetry_handler,
                _coerce_config_to_object(config),
            )
            if config
            else None
        )
        finish_reasons = []
        extra_attributes = (
            _get_extra_generate_content_attributes()
            | _create_request_attributes(
                config,
                generate_content_config_key_allowlist,
            )
        )
        with telemetry_handler.inference(
            provider=_determine_genai_system(self),
            request_model=model,
            operation_name="generate_content",
        ) as invocation:
            invocation.attributes.update(extra_attributes)
            invocation.tool_definitions = _maybe_get_tool_definitions(config)

            if content_recording_enabled:
                invocation.input_messages = to_input_messages(
                    contents=transformers.t_contents(contents)
                )
                if system_content := _config_to_system_instruction(config):
                    invocation.system_instruction = to_system_instructions(
                        content=transformers.t_contents(system_content)[0]
                    )
            candidates = []
            try:
                for resp in wrapped_func(
                    self,
                    model=model,
                    contents=contents,
                    config=wrapped_config,
                    **kwargs,
                ):
                    _maybe_update_token_counts_and_finish_reasons(
                        resp, finish_reasons, invocation
                    )
                    if resp.candidates:
                        candidates += resp.candidates
                    yield resp
            finally:
                if content_recording_enabled and candidates:
                    invocation.output_messages = to_output_messages(
                        candidates=candidates
                    )

    return instrumented_generate_content_stream


def _create_instrumented_async_generate_content(
    snapshot: _MethodsSnapshot,
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
    content_recording_enabled: bool,
):
    wrapped_func = snapshot.async_generate_content

    @functools.wraps(wrapped_func)
    async def instrumented_generate_content(
        self: AsyncModels,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> GenerateContentResponse:
        wrapped_config = (
            _wrapped_config_with_tools(
                telemetry_handler,
                _coerce_config_to_object(config),
            )
            if config
            else None
        )
        finish_reasons = []
        extra_attributes = (
            _get_extra_generate_content_attributes()
            | _create_request_attributes(
                config,
                generate_content_config_key_allowlist,
            )
        )
        with telemetry_handler.inference(
            provider=_determine_genai_system(self),
            request_model=model,
            operation_name="generate_content",
        ) as invocation:
            invocation.attributes.update(extra_attributes)
            invocation.tool_definitions = (
                await _maybe_get_tool_definitions_async(config)
            )

            if content_recording_enabled:
                invocation.input_messages = to_input_messages(
                    contents=transformers.t_contents(contents)
                )
                if system_content := _config_to_system_instruction(config):
                    invocation.system_instruction = to_system_instructions(
                        content=transformers.t_contents(system_content)[0]
                    )
            candidates = []
            try:
                response = await wrapped_func(
                    self,
                    model=model,
                    contents=contents,
                    config=wrapped_config,
                    **kwargs,
                )
                _maybe_update_token_counts_and_finish_reasons(
                    response, finish_reasons, invocation
                )
                if response.candidates:
                    candidates += response.candidates
                return response
            finally:
                if content_recording_enabled and candidates:
                    invocation.output_messages = to_output_messages(
                        candidates=candidates
                    )

    return instrumented_generate_content


# Disabling type checking because this is not yet implemented and tested fully.
def _create_instrumented_async_generate_content_stream(  # type: ignore
    snapshot: _MethodsSnapshot,
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
    content_recording_enabled: bool,
):
    wrapped_func = snapshot.async_generate_content_stream

    @functools.wraps(wrapped_func)
    async def instrumented_generate_content_stream(
        self: AsyncModels,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        **kwargs: Any,
    ) -> Awaitable[AsyncIterator[GenerateContentResponse]]:  # type: ignore
        wrapped_config = (
            _wrapped_config_with_tools(
                telemetry_handler,
                _coerce_config_to_object(config),
            )
            if config
            else None
        )
        finish_reasons = []
        extra_attributes = (
            _get_extra_generate_content_attributes()
            | _create_request_attributes(
                config,
                generate_content_config_key_allowlist,
            )
        )
        invocation = telemetry_handler.inference(
            provider=_determine_genai_system(self),
            request_model=model,
            operation_name="generate_content",
        )
        invocation.attributes.update(extra_attributes)
        invocation.tool_definitions = await _maybe_get_tool_definitions_async(
            config
        )

        if content_recording_enabled:
            invocation.input_messages = to_input_messages(
                contents=transformers.t_contents(contents)
            )
            if system_content := _config_to_system_instruction(config):
                invocation.system_instruction = to_system_instructions(
                    content=transformers.t_contents(system_content)[0]
                )

        async def _response_async_generator_wrapper():
            candidates = []
            try:
                async for resp in await wrapped_func(
                    self,
                    model=model,
                    contents=contents,
                    config=wrapped_config,
                    **kwargs,
                ):
                    _maybe_update_token_counts_and_finish_reasons(
                        resp, finish_reasons, invocation
                    )
                    if resp.candidates:
                        candidates += resp.candidates
                    yield resp
                if content_recording_enabled and candidates:
                    invocation.output_messages = to_output_messages(
                        candidates=candidates
                    )
                invocation.stop()
            except Exception as exc:
                if content_recording_enabled and candidates:
                    invocation.output_messages = to_output_messages(
                        candidates=candidates
                    )
                invocation.fail(exc)
                raise

        return _response_async_generator_wrapper()

    return instrumented_generate_content_stream


def uninstrument_generate_content(snapshot: object):
    assert isinstance(snapshot, _MethodsSnapshot)
    snapshot.restore()


def instrument_generate_content(
    telemetry_handler: TelemetryHandler,
    generate_content_config_key_allowlist: AllowList,
) -> object:
    os.environ["OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT"] = "true"
    snapshot = _MethodsSnapshot()
    content_recording_enabled = get_content_capturing_mode()
    Models.generate_content = _create_instrumented_generate_content(
        snapshot,
        telemetry_handler,
        generate_content_config_key_allowlist,
        content_recording_enabled,
    )
    Models.generate_content_stream = (
        _create_instrumented_generate_content_stream(
            snapshot,
            telemetry_handler,
            generate_content_config_key_allowlist,
            content_recording_enabled,
        )
    )
    AsyncModels.generate_content = _create_instrumented_async_generate_content(
        snapshot,
        telemetry_handler,
        generate_content_config_key_allowlist,
        content_recording_enabled,
    )
    AsyncModels.generate_content_stream = (
        _create_instrumented_async_generate_content_stream(
            snapshot,
            telemetry_handler,
            generate_content_config_key_allowlist,
            content_recording_enabled,
        )
    )
    return snapshot
