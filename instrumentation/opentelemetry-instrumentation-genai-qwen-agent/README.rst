OpenTelemetry Qwen-Agent Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-qwen-agent.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-qwen-agent/

This library allows tracing agent runs, LLM requests, and tool executions
made through the `Qwen-Agent framework <https://pypi.org/project/qwen-agent/>`_.

It produces spans following the `GenAI semantic conventions
<https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_:

- ``invoke_agent`` for ``Agent.run()``
- ``chat`` for ``BaseChatModel.chat()``
- ``execute_tool`` for ``Agent._call_tool()``

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-qwen-agent

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.genai.qwen_agent import (
        QwenAgentInstrumentor,
    )
    from qwen_agent.agents import Assistant

    QwenAgentInstrumentor().instrument()

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="my-assistant",
    )
    for responses in bot.run([{"role": "user", "content": "Hello!"}]):
        pass

Message content capture is disabled by default. Set
``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` and
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`` to record
prompts, completions, and tool arguments/results on spans.

References
----------

* `OpenTelemetry Qwen-Agent Instrumentation <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/qwen_agent/qwen_agent.html>`_
* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_
