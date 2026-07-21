# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
AgentScope instrumentation supporting ``agentscope >= 1.0.0, < 3.0.0``.

Usage
-----
.. code:: python

    import asyncio
    from opentelemetry.instrumentation.genai.agentscope import (
        AgentScopeInstrumentor,
    )
    from agentscope.model import DashScopeChatModel

    AgentScopeInstrumentor().instrument()

    model = DashScopeChatModel(model_name="qwen-max")

    messages = [{"role": "user", "content": "Hello, how are you?"}]

    async def call_model():
        response = await model(messages)
        if hasattr(response, "__aiter__"):
            result = []
            async for chunk in response:
                result.append(chunk)
            return result[-1] if result else response
        return response

    result = asyncio.run(call_model())

    AgentScopeInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.agentscope.package import (
    get_installed_instrumentation_dependencies,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler

logger = logging.getLogger(__name__)

_MODEL_MODULE = "agentscope.model"
_AGENT_MODULE = "agentscope.agent"
_EMBEDDING_MODULE = "agentscope.embedding"
_TOOL_MODULE = "agentscope.tool"

__all__ = ["AgentScopeInstrumentor"]


class AgentScopeInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for the AgentScope framework."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: TelemetryHandler | None = None
        self._agentscope_major: int | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return get_installed_instrumentation_dependencies()

    def _setup_tracing_patch(self, wrapped, instance, args, kwargs) -> None:  # noqa: ARG002
        """Replace setup_tracing with a no-op so OTel handles tracing."""

    def _check_tracing_enabled_patch(  # noqa: ARG002
        self, wrapped, instance, args, kwargs
    ) -> bool:
        """Return False to disable tracing in the native AgentScope library."""
        return False

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        self._agentscope_major = _get_agentscope_major()
        if self._agentscope_major >= 2:
            self._instrument_v2()
        else:
            self._instrument_v1()

    def _instrument_v1(self) -> None:
        from ._wrapper import (  # noqa: PLC0415
            AgentScopeAgentWrapper,
            AgentScopeChatModelWrapper,
            AgentScopeEmbeddingModelWrapper,
        )
        from .patch import wrap_tool_call  # noqa: PLC0415

        try:
            wrap_function_wrapper(
                _MODEL_MODULE,
                "ChatModelBase.__init__",
                AgentScopeChatModelWrapper(handler=self._handler),
            )
        except Exception as e:
            logger.warning("Failed to instrument ChatModelBase: %s", e)

        try:
            wrap_function_wrapper(
                _AGENT_MODULE,
                "AgentBase.__init__",
                AgentScopeAgentWrapper(handler=self._handler),
            )
        except Exception as e:
            logger.warning("Failed to instrument AgentBase: %s", e)

        try:
            wrap_function_wrapper(
                _EMBEDDING_MODULE,
                "EmbeddingModelBase.__init__",
                AgentScopeEmbeddingModelWrapper(handler=self._handler),
            )
        except Exception as e:
            logger.warning("Failed to instrument EmbeddingModelBase: %s", e)

        try:

            def wrap_tool_with_handler(wrapped, instance, args, kwargs):
                return wrap_tool_call(
                    wrapped, instance, args, kwargs, handler=self._handler
                )

            wrap_function_wrapper(
                _TOOL_MODULE,
                "Toolkit.call_tool_function",
                wrap_tool_with_handler,
            )
        except Exception as e:
            logger.warning("Failed to instrument Toolkit: %s", e)

        # Disable AgentScope's native tracing to avoid duplicate spans.
        try:
            wrap_function_wrapper(
                "agentscope.tracing",
                "setup_tracing",
                self._setup_tracing_patch,
            )
        except Exception as e:
            logger.warning("Failed to patch setup_tracing: %s", e)

        try:
            wrap_function_wrapper(
                "agentscope.tracing._trace",
                "_check_tracing_enabled",
                self._check_tracing_enabled_patch,
            )
        except Exception as e:
            logger.warning("Failed to patch _check_tracing_enabled: %s", e)

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        if self._agentscope_major is None:
            self._agentscope_major = _get_agentscope_major()
        if self._agentscope_major >= 2:
            self._uninstrument_v2()
        else:
            self._uninstrument_v1()
        self._handler = None
        self._agentscope_major = None

    def _uninstrument_v1(self) -> None:
        try:
            from ._wrapper import (  # noqa: PLC0415
                AgentScopeAgentWrapper,
                AgentScopeChatModelWrapper,
                AgentScopeEmbeddingModelWrapper,
            )
        except Exception as e:
            logger.warning("Failed to import AgentScope wrappers: %s", e)
            AgentScopeAgentWrapper = None
            AgentScopeChatModelWrapper = None
            AgentScopeEmbeddingModelWrapper = None

        for wrapper_cls in (
            AgentScopeChatModelWrapper,
            AgentScopeAgentWrapper,
            AgentScopeEmbeddingModelWrapper,
        ):
            if wrapper_cls is not None:
                try:
                    wrapper_cls.restore_original_methods()
                except Exception as e:
                    logger.warning(
                        "Failed to restore %s methods: %s",
                        wrapper_cls.__name__,
                        e,
                    )

        _unwrap_safely(_MODEL_MODULE, "ChatModelBase", "__init__")
        _unwrap_safely(_AGENT_MODULE, "AgentBase", "__init__")
        _unwrap_safely(_EMBEDDING_MODULE, "EmbeddingModelBase", "__init__")
        _unwrap_safely(_TOOL_MODULE, "Toolkit", "call_tool_function")

        try:
            import agentscope.tracing  # noqa: PLC0415

            unwrap(agentscope.tracing, "setup_tracing")
        except Exception as e:
            logger.warning("Failed to uninstrument setup_tracing: %s", e)

        try:
            import agentscope.tracing._trace as agentscope_tracing_trace  # noqa: PLC0415

            unwrap(agentscope_tracing_trace, "_check_tracing_enabled")
        except Exception as e:
            logger.warning(
                "Failed to uninstrument _check_tracing_enabled: %s", e
            )

    def _instrument_v2(self) -> None:
        from ._v2_middleware import (  # noqa: PLC0415
            AgentScopeV2Middleware,
            append_agentscope_middleware,
        )

        try:

            def wrap_agent_init(wrapped, instance, args, kwargs):
                args, kwargs = append_agentscope_middleware(
                    args,
                    kwargs,
                    # Resolve the handler lazily so reinstrumentation uses the
                    # current handler instead of the one captured at init time.
                    AgentScopeV2Middleware(handler=lambda: self._handler),
                )
                return wrapped(*args, **kwargs)

            wrap_function_wrapper(
                _AGENT_MODULE,
                "Agent.__init__",
                wrap_agent_init,
            )
        except Exception as e:
            logger.warning("Failed to instrument AgentScope v2 Agent: %s", e)

    def _uninstrument_v2(self) -> None:
        _unwrap_safely(_AGENT_MODULE, "Agent", "__init__")


def _unwrap_safely(module_name: str, class_name: str, method: str) -> None:
    try:
        module = __import__(module_name, fromlist=[class_name])
        target = getattr(module, class_name)
        unwrap(target, method)
    except Exception as e:
        logger.warning(
            "Failed to uninstrument %s.%s.%s: %s",
            module_name,
            class_name,
            method,
            e,
        )


def _get_agentscope_major() -> int:
    try:
        installed_version = metadata_version("agentscope")
    except PackageNotFoundError:
        return 1

    major_text = installed_version.split(".", 1)[0]
    try:
        return int(major_text)
    except ValueError:
        return 1
