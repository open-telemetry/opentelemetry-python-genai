# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import Context as PythonContext
from functools import partial, wraps
from importlib import import_module
from importlib.util import find_spec
from inspect import iscoroutinefunction, signature
from typing import Any
from uuid import UUID

from wrapt import wrap_function_wrapper

from opentelemetry.context import attach, detach, get_current
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

_logger = logging.getLogger(__name__)

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
    def _config_context(
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
            # LangChain runs each child (a sequence step, a parallel branch, a
            # graph node) in this copied context. Restore the token in that
            # same context, never in a callback or the consuming task. A child
            # whose own boundary already attached the parent needs no second
            # token.
            token = (
                context.run(attach, parent)
                if parent is not None
                and context.run(get_current) is not parent
                else None
            )
            try:
                yield context
            finally:
                if token is not None:
                    context.run(detach, token)

    def _patch(
        self,
        module_name: str,
        class_name: str | None,
        method: str,
        wrapper_for: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> None:
        try:
            owner: Any = import_module(module_name)
            if class_name is not None:
                owner = getattr(owner, class_name)
            original = getattr(owner, method)
        except (ImportError, AttributeError):
            target = ".".join(filter(None, (module_name, class_name, method)))
            library = module_name.partition(".")[0]
            if find_spec(library) is None:
                _logger.debug(
                    "Skipping execution boundary %s: %s is not installed",
                    target,
                    library,
                )
            else:
                # Installed but changed: spans under it will not correlate.
                _logger.warning(
                    "Skipping execution boundary %s: not found in the "
                    "installed %s, so context is not propagated across it",
                    target,
                    library,
                )
            return
        wrap_function_wrapper(owner, method, wrapper_for(original))
        self._patched.append((owner, method))

    def instrument(self) -> None:
        for module_name, class_name, method in _METHODS:
            self._patch(module_name, class_name, method, self._wrap)

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
            for method, asynchronous in zip(methods, (False, True)):
                self._patch(
                    module_name,
                    class_name,
                    method,
                    partial(
                        _wrap_run,
                        invocations=self._invocations,
                        on_error=on_error,
                        asynchronous=asynchronous,
                    ),
                )

        # These helpers start the chain run themselves, so the run id is chosen
        # up front for _started to match.
        for method, asynchronous in (
            ("_call_with_config", False),
            ("_acall_with_config", True),
        ):
            self._patch(
                "langchain_core.runnables.base",
                "Runnable",
                method,
                partial(
                    _wrap_call,
                    invocations=self._invocations,
                    asynchronous=asynchronous,
                ),
            )

        for class_name, wrapper in (
            ("CallbackManager", _start_run),
            ("AsyncCallbackManager", _astart_run),
        ):
            for method in (
                "on_chat_model_start",
                "on_chain_start",
                "on_tool_start",
                "on_retriever_start",
            ):
                self._patch(
                    "langchain_core.callbacks.manager",
                    class_name,
                    method,
                    lambda _original, wrapper=wrapper: wrapper,
                )

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
            for method, asynchronous in zip(methods, (False, True)):
                self._patch(
                    module_name,
                    class_name,
                    method,
                    partial(
                        _wrap_stream,
                        invocations=self._invocations,
                        asynchronous=asynchronous,
                    ),
                )

        # Composite runnables drive steps that may start no run of their own
        # (a plain ``invoke`` override), so the child context is attached where
        # LangChain and LangGraph enter the step's copied context.
        for module_name in (
            "langchain_core.runnables.base",
            "langchain_core.runnables.fallbacks",
            "langgraph._internal._runnable",
        ):
            self._patch(
                module_name,
                None,
                "set_config_context",
                lambda _original: self._config_context,
            )

    def uninstrument(self) -> None:
        for owner, name in reversed(self._patched):
            unwrap(owner, name)
        self._patched.clear()
