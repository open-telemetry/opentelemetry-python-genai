# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import Context as PythonContext
from functools import wraps
from importlib import import_module
from inspect import iscoroutinefunction, signature
from typing import Any
from uuid import UUID

from wrapt import wrap_function_wrapper

from opentelemetry.context import attach, detach
from opentelemetry.instrumentation.genai.langchain._run_context import (
    _astart_run,
    _start_run,
    _wrap_call,
    _wrap_run,
    _wrap_stream,
)
from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.instrumentation.utils import unwrap

__all__ = ["_ExecutionContext"]

_METHODS = (
    (
        "langchain_core.language_models.chat_models",
        "BaseChatModel",
        "_generate_with_cache",
    ),
    (
        "langchain_core.language_models.chat_models",
        "BaseChatModel",
        "_agenerate_with_cache",
    ),
)


class _ExecutionContext:
    def __init__(
        self,
        invocations: _InvocationManager,
        on_tool_error: Callable[..., None],
        on_retriever_error: Callable[..., None],
    ) -> None:
        self._invocations = invocations
        self._on_tool_error = on_tool_error
        self._on_retriever_error = on_retriever_error
        self._patched: list[tuple[Any, str]] = []

    @contextmanager
    def _activate(self, run_id: UUID | None) -> Iterator[None]:
        context = self._invocations.get_parent_context(run_id)
        if context is None:
            yield
            return
        token = attach(context)
        try:
            yield
        finally:
            detach(token)

    def _wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        parameters = signature(original)

        def run_id_for(
            instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> UUID | None:
            run_manager = parameters.bind(
                instance, *args, **kwargs
            ).arguments.get("run_manager")
            return run_manager.run_id if run_manager is not None else None

        @wraps(original)
        def sync(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            with self._activate(run_id_for(instance, args, kwargs)):
                return wrapped(*args, **kwargs)

        @wraps(original)
        async def asynchronous(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            with self._activate(run_id_for(instance, args, kwargs)):
                return await wrapped(*args, **kwargs)

        return asynchronous if iscoroutinefunction(original) else sync

    @contextmanager
    def _graph_context(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Iterator[PythonContext]:
        config = args[0] if args else kwargs["config"]
        callbacks = config.get("callbacks")
        parent = self._invocations.get_parent_context(
            getattr(callbacks, "parent_run_id", None)
        )
        with wrapped(*args, **kwargs) as context:
            # LangGraph executes nodes in this copied context. Restore the token
            # in that same context, never in a callback or the consuming task.
            token = context.run(attach, parent) if parent is not None else None
            try:
                yield context
            finally:
                if token is not None:
                    context.run(detach, token)

    def instrument(self) -> None:
        for module_name, class_name, method in _METHODS:
            cls = getattr(import_module(module_name), class_name)
            wrap_function_wrapper(
                cls, method, self._wrap(getattr(cls, method))
            )
            self._patched.append((cls, method))

        for module_name, class_name, methods, on_error in (
            (
                "langchain_core.tools.base",
                "BaseTool",
                ("run", "arun"),
                self._on_tool_error,
            ),
            (
                "langchain_core.retrievers",
                "BaseRetriever",
                ("invoke", "ainvoke"),
                self._on_retriever_error,
            ),
        ):
            cls = getattr(import_module(module_name), class_name)
            for method, asynchronous in zip(methods, (False, True)):
                wrap_function_wrapper(
                    cls,
                    method,
                    _wrap_run(
                        getattr(cls, method),
                        self._invocations,
                        on_error,
                        asynchronous,
                    ),
                )
                self._patched.append((cls, method))

        # These helpers start the chain run themselves, so the run id is chosen
        # up front for _started to match.
        cls = getattr(
            import_module("langchain_core.runnables.base"), "Runnable"
        )
        for method, asynchronous in (
            ("_call_with_config", False),
            ("_acall_with_config", True),
        ):
            wrap_function_wrapper(
                cls,
                method,
                _wrap_call(
                    getattr(cls, method), self._invocations, asynchronous
                ),
            )
            self._patched.append((cls, method))

        for class_name, wrapper in (
            ("CallbackManager", _start_run),
            ("AsyncCallbackManager", _astart_run),
        ):
            cls = getattr(
                import_module("langchain_core.callbacks.manager"), class_name
            )
            for method in (
                "on_chat_model_start",
                "on_chain_start",
                "on_tool_start",
                "on_retriever_start",
            ):
                wrap_function_wrapper(cls, method, wrapper)
                self._patched.append((cls, method))

        for module_name, class_name, methods in (
            (
                "langchain_core.language_models.chat_models",
                "BaseChatModel",
                ("stream", "astream"),
            ),
            (
                "langchain_core.runnables.base",
                "Runnable",
                (
                    "_transform_stream_with_config",
                    "_atransform_stream_with_config",
                ),
            ),
            # RunnableLambda.astream does not forward aclose to its inner
            # transform iterator, so its own boundary needs finalization too.
            (
                "langchain_core.runnables.base",
                "RunnableLambda",
                ("stream", "astream"),
            ),
            ("langgraph.pregel", "Pregel", ("stream", "astream")),
        ):
            try:
                cls = getattr(import_module(module_name), class_name)
            except ImportError:
                continue
            for method, asynchronous in zip(methods, (False, True)):
                wrap_function_wrapper(
                    cls,
                    method,
                    _wrap_stream(
                        getattr(cls, method), self._invocations, asynchronous
                    ),
                )
                self._patched.append((cls, method))

        try:
            module = import_module("langgraph._internal._runnable")
        except ImportError:
            return
        if hasattr(module, "set_config_context"):
            wrap_function_wrapper(
                module, "set_config_context", self._graph_context
            )
            self._patched.append((module, "set_config_context"))

    def uninstrument(self) -> None:
        for owner, name in reversed(self._patched):
            unwrap(owner, name)
        self._patched.clear()
