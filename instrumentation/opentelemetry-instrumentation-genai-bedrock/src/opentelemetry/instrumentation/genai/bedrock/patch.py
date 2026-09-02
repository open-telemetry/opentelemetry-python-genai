# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from botocore.client import BaseClient
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiProviderNameValues,
)
from opentelemetry.util.genai.handler import TelemetryHandler

from .extractors import (
    extract_converse_request,
    extract_converse_response,
    extract_server_address_and_port,
)
from .stream import BedrockConverseStreamWrapper

_logger = logging.getLogger(__name__)

BEDROCK_RUNTIME = "bedrock-runtime"


def _handle_converse(
    wrapped: Callable[..., Any],
    instance: BaseClient,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    api_params: dict[str, Any],
    handler: TelemetryHandler,
    *,
    is_stream: bool = False,
) -> Any:
    endpoint_url = getattr(
        getattr(instance, "meta", None), "endpoint_url", None
    )
    server_address, server_port = extract_server_address_and_port(endpoint_url)
    raw_model_id = api_params.get("modelId")
    model_id = str(raw_model_id) if raw_model_id else None
    invocation = handler.inference(
        provider=GenAiProviderNameValues.AWS_BEDROCK.value,
        request_model=model_id,
        server_address=server_address,
        server_port=server_port,
        operation_name=GenAiOperationNameValues.CHAT.value,
    )
    capture_content = handler.should_capture_content()
    extract_converse_request(
        api_params, invocation, capture_content=capture_content
    )
    try:
        response: Any = wrapped(*args, **kwargs)
    except Exception as exc:
        invocation.fail(exc)
        raise

    if is_stream:
        # Any 200 response is guaranteed to have this..
        if "stream" in response:
            response["stream"] = BedrockConverseStreamWrapper(
                response["stream"],
                invocation=invocation,
                capture_content=capture_content,
            )
            return response
    else:
        extract_converse_response(
            response,
            invocation,
            capture_content=capture_content,
        )

    invocation.stop()
    return response


def _make_api_call_wrapper(handler: TelemetryHandler) -> Callable[..., Any]:
    def _wrapper(
        wrapped: Callable[..., Any],
        instance: BaseClient,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        service_name = getattr(
            getattr(instance, "_service_model", None), "service_name", None
        )
        if service_name != BEDROCK_RUNTIME:
            return wrapped(*args, **kwargs)

        operation_name = args[0] if args else kwargs.get("operation_name")
        raw_params = args[1] if len(args) > 1 else kwargs.get("api_params")
        api_params: dict[str, Any] = (
            cast(dict[str, Any], raw_params)
            if isinstance(raw_params, dict)
            else {}
        )

        if operation_name in ("Converse", "ConverseStream"):
            return _handle_converse(
                wrapped,
                instance,
                args,
                kwargs,
                api_params,
                handler,
                is_stream=(operation_name == "ConverseStream"),
            )

        return wrapped(*args, **kwargs)

    return _wrapper


def patch_bedrock(handler: TelemetryHandler) -> None:
    """Patch botocore BaseClient to instrument Bedrock runtime operations."""
    wrap_function_wrapper(
        "botocore.client",
        "BaseClient._make_api_call",
        _make_api_call_wrapper(handler),
    )


def unpatch_bedrock() -> None:
    """Unpatch botocore BaseClient._make_api_call."""
    import botocore.client

    try:
        unwrap(botocore.client.BaseClient, "_make_api_call")
    except Exception:
        _logger.debug(
            "Failed to unwrap BaseClient._make_api_call", exc_info=True
        )
