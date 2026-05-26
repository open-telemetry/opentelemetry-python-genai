# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry AWS Bedrock Instrumentation
==========================================

Instrumentation for the AWS Bedrock Runtime service via ``botocore``.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
    import boto3

    # Enable instrumentation
    BedrockInstrumentor().instrument()

    # Use Bedrock Runtime client normally
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
    )

Configuration
-------------

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

API
---
"""

from __future__ import annotations

from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments
from .patch import make_api_call_wrapper


class BedrockInstrumentor(BaseInstrumentor):
    """An instrumentor for AWS Bedrock Runtime via botocore.

    This instrumentor automatically traces Bedrock Converse and ConverseStream
    API calls and optionally captures message content as events.
    """

    def __init__(self) -> None:
        super().__init__()

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Bedrock instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
        """
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        wrap_function_wrapper(
            "botocore.client",
            "BaseClient._make_api_call",
            make_api_call_wrapper(handler),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Bedrock instrumentation.

        This removes all patches applied during instrumentation.
        """
        import botocore.client  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        unwrap(
            botocore.client.BaseClient,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            "_make_api_call",
        )
