# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for AWS Bedrock instrumentation."""

from __future__ import annotations

import logging
from typing import Any, Callable
from urllib.parse import urlparse

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation

from .extractors import (
    get_input_messages,
    get_output_messages,
    get_request_attributes,
    get_response_attributes,
    get_system_instruction,
)
from .wrappers import ConverseStreamWrapper

_logger = logging.getLogger(__name__)

_SUPPORTED_OPERATIONS = frozenset({"Converse", "ConverseStream"})
_BEDROCK_RUNTIME_SERVICE = "bedrock-runtime"
_PROVIDER = "aws.bedrock"


def make_api_call_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    """Wrap ``BaseClient._make_api_call`` to trace Bedrock Runtime calls."""
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # _make_api_call signature: (operation_name, api_params)
        if len(args) < 2:
            return wrapped(*args, **kwargs)

        operation_name: str = args[0]
        api_params: dict[str, Any] = args[1]

        # Only intercept Bedrock Runtime Converse/ConverseStream
        service_id = _get_service_id(instance)
        if service_id != _BEDROCK_RUNTIME_SERVICE:
            return wrapped(*args, **kwargs)

        if operation_name not in _SUPPORTED_OPERATIONS:
            return wrapped(*args, **kwargs)

        invocation = _create_invocation(
            handler, instance, api_params, capture_content
        )

        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise

        if operation_name == "ConverseStream":
            return ConverseStreamWrapper(result, invocation, capture_content)

        # Non-streaming Converse response
        _set_response_attributes(invocation, result, capture_content)
        invocation.stop()
        return result

    return traced_method


def _get_service_id(client: Any) -> str:
    """Extract the service identifier from a botocore client instance."""
    try:
        service_model = client._service_model
        return service_model.endpoint_prefix
    except AttributeError:
        return ""


def _get_server_address(client: Any) -> str | None:
    """Extract the server address (hostname) from the client's endpoint URL."""
    try:
        endpoint_url = client._endpoint.host
        parsed = urlparse(endpoint_url)
        return parsed.hostname
    except (AttributeError, ValueError):
        return None


def _get_server_port(client: Any) -> int | None:
    """Extract the server port from the client's endpoint URL if non-standard."""
    try:
        endpoint_url = client._endpoint.host
        parsed = urlparse(endpoint_url)
        port = parsed.port
        if port and port != 443 and port > 0:
            return port
    except (AttributeError, ValueError):
        pass
    return None


def _create_invocation(
    handler: TelemetryHandler,
    client: Any,
    api_params: dict[str, Any],
    capture_content: bool,
) -> InferenceInvocation:
    """Create and configure an InferenceInvocation from Converse parameters."""
    model_id = api_params.get("modelId") or ""
    server_address = _get_server_address(client)
    server_port = _get_server_port(client)

    invocation = handler.start_inference(
        provider=_PROVIDER,
        request_model=model_id,
        server_address=server_address,
        server_port=server_port,
    )

    invocation.attributes = get_request_attributes(api_params)

    if capture_content:
        invocation.input_messages = get_input_messages(api_params)
        invocation.system_instruction = get_system_instruction(api_params)

    return invocation


def _set_response_attributes(
    invocation: InferenceInvocation,
    result: dict[str, Any],
    capture_content: bool,
) -> None:
    """Extract response attributes from a Converse result and apply them."""
    response_attrs = get_response_attributes(result)

    invocation.response_model_name = response_attrs.get("response_model")
    invocation.response_id = response_attrs.get("response_id")

    finish_reasons = response_attrs.get("finish_reasons")
    if finish_reasons:
        invocation.finish_reasons = finish_reasons

    input_tokens = response_attrs.get("input_tokens")
    if input_tokens is not None:
        invocation.input_tokens = input_tokens

    output_tokens = response_attrs.get("output_tokens")
    if output_tokens is not None:
        invocation.output_tokens = output_tokens

    if capture_content:
        invocation.output_messages = get_output_messages(result)
