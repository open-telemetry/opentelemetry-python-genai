OpenTelemetry Claude Agent SDK Instrumentation
==============================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-claude-agent-sdk.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-claude-agent-sdk/

This library allows tracing the runs started by the
`Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_.

Each ``query()`` stream is reported as one ``invoke_agent`` span, and every tool
call observed in that stream as an ``execute_tool`` span. This is agent-level
tracing: LLM requests the CLI makes internally are not reported here. Only
``claude_agent_sdk.query`` is instrumented today; ``ClaudeSDKClient`` is not.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-claude-agent-sdk

If you don't have a Claude Agent SDK application yet, try our `examples <examples>`_
which only need a valid Anthropic API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Claude Agent SDK instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    import claude_agent_sdk
    from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition, AssistantMessage, TextBlock
    from opentelemetry.instrumentation.genai.claude_agent_sdk import ClaudeAgentSDKInstrumentor

    # Instrument Claude Agent SDK first: query() is patched as a module
    # attribute, so a ``from claude_agent_sdk import query`` taken before this
    # line would keep calling the uninstrumented function.
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK as normal
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
            prompt="Hello, Claude!",
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

    anyio.run(main)


Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Claude Agent SDK (Python) <https://github.com/anthropics/claude-agent-sdk-python>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
