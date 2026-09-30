# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Langchain instrumentation supporting `ChatOpenAI` and `ChatBedrock`, it can be enabled by
using ``LangChainInstrumentor``. Other providers/LLMs may be supported in the future and telemetry for them is skipped for now.

Usage
-----
.. code:: python
    from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    LangChainInstrumentor().instrument()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)
    messages = [
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="What is the capital of France?"),
    ]
    result = llm.invoke(messages)
    LangChainInstrumentor().uninstrument()

API
---
"""

from collections.abc import Callable, Collection
from typing import Any

from langchain_core.callbacks import BaseCallbackManager
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.langchain._execution_context import (
    _ExecutionContext,
)
from opentelemetry.instrumentation.genai.langchain.agent_context import (
    wrap_astream,
    wrap_stream,
)
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.instrumentation.genai.langchain.package import _instruments
from opentelemetry.instrumentation.genai.langchain.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler


class LangChainInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for LangChain.
    This adds a custom callback handler to the LangChain callback manager
    to capture LLM telemetry.
    """

    _execution_context: _ExecutionContext | None = None

    def __init__(
        self,
    ):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any):
        """
        Enable Langchain instrumentation.
        """
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        telemetry_handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
            instrumentation_scope_name=__package__,
            instrumentation_scope_version=__version__,
        )
        invocation_manager = _InvocationManager()
        handler = OpenTelemetryLangChainCallbackHandler(
            telemetry_handler=telemetry_handler,
            invocation_manager=invocation_manager,
        )

        wrap_function_wrapper(
            "langchain_core.callbacks",
            "BaseCallbackManager.__init__",
            _BaseCallbackManagerInitWrapper(handler),
        )
        self._execution_context = _ExecutionContext(
            invocation_manager,
            handler.on_tool_error,
            handler.on_retriever_error,
        )
        self._execution_context.instrument()
        self._instrument_agent_entry_points()

    @staticmethod
    def _instrument_agent_entry_points() -> None:
        """Recover the create_agent provenance the callback metadata does not carry."""
        for method, wrapper in (
            ("Pregel.stream", wrap_stream),
            ("Pregel.astream", wrap_astream),
        ):
            try:
                wrap_function_wrapper("langgraph.pregel", method, wrapper)
            except (ImportError, AttributeError):
                # Without langgraph there are no agent graphs to announce, and
                # classification stays metadata-based.
                return

    def _uninstrument(self, **kwargs: Any):
        """
        Cleanup instrumentation (unwrap).
        """
        unwrap("langchain_core.callbacks.base.BaseCallbackManager", "__init__")
        try:
            import langgraph.pregel

            for method in ("stream", "astream"):
                unwrap(langgraph.pregel.Pregel, method)
        except (ImportError, AttributeError):
            pass
        if self._execution_context is not None:
            self._execution_context.uninstrument()
            self._execution_context = None


class _BaseCallbackManagerInitWrapper:
    """
    Wrap the BaseCallbackManager __init__ to insert custom callback handler in the manager's handlers list.
    """

    def __init__(
        self,
        handler: OpenTelemetryLangChainCallbackHandler,
    ):
        self._handler = handler

    def __call__(
        self,
        wrapped: Callable[..., None],
        instance: BaseCallbackManager,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        wrapped(*args, **kwargs)
        if not any(
            isinstance(handler, OpenTelemetryLangChainCallbackHandler)
            for handler in instance.inheritable_handlers
        ):
            instance.add_handler(self._handler, inherit=True)
