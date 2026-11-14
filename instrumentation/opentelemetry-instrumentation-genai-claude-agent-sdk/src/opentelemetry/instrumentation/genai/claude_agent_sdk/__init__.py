# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Claude Agent SDK Instrumentation
===============================================

Instrumentation for the `Claude Agent SDK
<https://github.com/anthropics/claude-agent-sdk-python>`_.

Each ``query()`` stream becomes one ``invoke_agent`` span, and each tool call
observed in that stream an ``execute_tool`` span. This is agent-level tracing:
LLM requests the CLI makes internally are not reported here.

Usage
-----

.. code-block:: python

    import claude_agent_sdk
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (
        ClaudeAgentSDKInstrumentor,
    )
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        AgentDefinition,
        AssistantMessage,
        TextBlock,
    )

    # Enable instrumentation. Do this before taking a reference to query():
    # the function is patched as a module attribute, so ``from claude_agent_sdk
    # import query`` evaluated first would keep the uninstrumented function.
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK normally
    import anyio


    async def main():
        options = ClaudeAgentOptions(
            agents={
                "assistant": AgentDefinition(
                    description="A helpful assistant",
                    prompt="You are a helpful assistant.",
                    tools=["Read"],
                    model="sonnet",
                ),
            },
        )

        async for message in claude_agent_sdk.query(
            prompt="Hello!", options=options
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


    anyio.run(main)

Configuration
-------------

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

API
---
"""

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.genai.claude_agent_sdk.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.claude_agent_sdk.version import (
    __version__,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """An instrumentor for the Claude Agent SDK.

    This instrumentor will automatically trace Anthropic API calls and
    optionally capture message content as events.
    """

    def __init__(self) -> None:
        super().__init__()
        self._handler: TelemetryHandler | None = None

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Claude Agent SDK instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
        """

        # Get providers from kwargs
        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")
        meter_provider = kwargs.get("meter_provider")

        self._handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
            instrumentation_scope_name=__package__,
            instrumentation_scope_version=__version__,
        )

        from opentelemetry.instrumentation.genai.claude_agent_sdk import patch

        patch.instrument(self._handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Claude Agent SDK instrumentation.

        This removes all patches applied during instrumentation.
        """
        from opentelemetry.instrumentation.genai.claude_agent_sdk import patch

        patch.uninstrument()
