OpenTelemetry AgentScope Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-agentscope.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-agentscope/

This library allows tracing LLM, embedding, agent, and tool calls made by the
`AgentScope <https://github.com/agentscope-ai/agentscope>`_ framework. It
supports ``agentscope >= 1.0.0, < 3.0.0`` and emits telemetry through the shared
``opentelemetry-util-genai`` utilities following the GenAI semantic conventions.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-agentscope

Usage
-----

.. code-block:: python

    import asyncio

    from agentscope.model import DashScopeChatModel

    from opentelemetry.instrumentation.genai.agentscope import (
        AgentScopeInstrumentor,
    )

    AgentScopeInstrumentor().instrument()

    model = DashScopeChatModel(model_name="qwen-max")
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    asyncio.run(model(messages))


Content capture
---------------

This instrumentation follows the shared ``opentelemetry-util-genai`` content
capture controls. Set ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to
select whether prompt/response content is attached to spans and/or events; see
the ``opentelemetry-util-genai`` documentation for the supported capture modes.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_
