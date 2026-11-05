# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import Context as PythonContext
from functools import partial, wraps
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from inspect import iscoroutinefunction, signature
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from wrapt import wrap_function_wrapper

from opentelemetry.context import attach, detach, get_current
from opentelemetry.instrumentation.genai.langchain._run_context import (
    _astart_run,
    _RunScope,
    _start_run,
    _wrap_call,
    _wrap_run,
    _wrap_stream,
)
from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _INFERENCE_CALLS_KEY,
    _inference_calls,
    _InferenceCallScope,
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
        refresh_handlers: Callable[[BaseCallbackManager], None],
    ) -> None:
        self._invocations = invocations
        self._on_tool_error = on_tool_error
        self._on_retriever_error = on_retriever_error
        self._refresh_handlers = refresh_handlers
        self._patched: list[tuple[Any, str]] = []

    @contextmanager
    def _activate(
        self,
        run_id: UUID | None,
        configured: _InferenceCallScope | None = None,
    ) -> Iterator[None]:
        context = self._invocations.get_parent_context(run_id)
        token = (
            attach(context)
            if context is not None and get_current() is not context
            else None
        )
        scope = self._invocations.get_inference_scope(run_id, configured)
        scope_token = (
            _inference_calls.set(scope) if scope is not None else None
        )
        try:
            yield
        finally:
            if scope_token is not None:
                _inference_calls.reset(scope_token)
            if token is not None:
                detach(token)

    def _wrap_input_config(
        self, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        parameters = signature(original)

        def context_for(
            instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> AbstractContextManager[None]:
            # One input runs in this frame; several run in the executor or
            # as tasks, each in a copied context that this frame never sees.
            bound = parameters.bind(instance, *args, **kwargs).arguments
            if "inputs" in bound and len(bound["inputs"]) != 1:
                return self._activate(None)
            config = bound.get("config")
            if isinstance(config, Sequence):
                config = cast(Any, config[0]) if config else None
            typed_config = cast(RunnableConfig, config or {})
            callbacks = typed_config.get("callbacks")
            configured = typed_config.get("configurable", {}).get(
                _INFERENCE_CALLS_KEY
            )
            return self._activate(
                getattr(callbacks, "parent_run_id", None),
                configured
                if isinstance(configured, _InferenceCallScope)
                else None,
            )

        @wraps(original)
        def sync(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            with context_for(instance, args, kwargs):
                return wrapped(*args, **kwargs)

        @wraps(original)
        async def asynchronous(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            with context_for(instance, args, kwargs):
                return await wrapped(*args, **kwargs)

        return asynchronous if iscoroutinefunction(original) else sync

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
            run_id = run_id_for(instance, args, kwargs)
            with self._activate(run_id):
                try:
                    return await wrapped(*args, **kwargs)
                except BaseException as error:
                    # agenerate reports an Exception from its gather results
                    # and reaches on_llm_error; a cancellation or interrupt
                    # leaves the gather before that, with no handler around
                    # the await.
                    if not isinstance(error, Exception):
                        _RunScope(run_id, self._invocations).finish(error)
                    raise

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
        configured = config.get("configurable", {}).get(_INFERENCE_CALLS_KEY)
        scope = self._invocations.get_inference_scope(
            getattr(callbacks, "parent_run_id", None),
            configured
            if isinstance(configured, _InferenceCallScope)
            else None,
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
            scope_token = (
                context.run(_inference_calls.set, scope)
                if scope is not None
                else None
            )
            try:
                yield context
            finally:
                if scope_token is not None:
                    context.run(_inference_calls.reset, scope_token)
                if token is not None:
                    context.run(detach, token)

    def _wrap_attempt_config(
        self, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        @wraps(original)
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            config: RunnableConfig = wrapped(*args, **kwargs)
            run_id = getattr(config.get("callbacks"), "parent_run_id", None)
            configured = config.get("configurable", {}).get(
                _INFERENCE_CALLS_KEY
            )
            if isinstance(run_id, UUID):
                scope = self._invocations.start_inference_attempt(
                    run_id,
                    configured
                    if isinstance(configured, _InferenceCallScope)
                    else None,
                )
                if scope is not None:
                    # The reserved prefix prevents promotion into callback metadata.
                    config["configurable"] = {
                        **config.get("configurable", {}),
                        _INFERENCE_CALLS_KEY: scope,
                    }
            return config

        return wrapper

    def _wrap_model_attempts(
        self, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        parameters = signature(original)

        @wraps(original)
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            bound = parameters.bind(instance, *args, **kwargs)
            handler = cast(Callable[..., Any], bound.arguments["handler"])
            config = ensure_config()
            configured = config.get("configurable", {}).get(
                _INFERENCE_CALLS_KEY
            )
            scope = _InferenceCallScope(
                uuid4(),
                self._invocations.get_inference_scope(
                    getattr(config.get("callbacks"), "parent_run_id", None),
                    configured
                    if isinstance(configured, _InferenceCallScope)
                    else None,
                ),
            )

            @wraps(handler)
            def sync(*args: Any, **kwargs: Any) -> Any:
                with _InferenceCallScope(
                    scope.run_id, scope.parent, keys=scope.keys
                ).activate():
                    return handler(*args, **kwargs)

            @wraps(handler)
            async def asynchronous(*args: Any, **kwargs: Any) -> Any:
                with _InferenceCallScope(
                    scope.run_id, scope.parent, keys=scope.keys
                ).activate():
                    return await handler(*args, **kwargs)

            bound.arguments["handler"] = (
                asynchronous if iscoroutinefunction(original) else sync
            )
            return wrapped(*bound.args[1:], **bound.kwargs)

        return wrapper

    def _patch(
        self,
        module_name: str,
        class_name: str | None,
        method: str,
        wrapper_for: Callable[[Callable[..., Any]], Callable[..., Any]],
        *,
        fallback_modules: tuple[str, ...] = (),
    ) -> None:
        for name in (module_name, *fallback_modules):
            try:
                owner: Any = import_module(name)
                if class_name is not None:
                    owner = getattr(owner, class_name)
                original = getattr(owner, method)
            except (ImportError, AttributeError):
                continue
            wrap_function_wrapper(owner, method, wrapper_for(original))
            self._patched.append((owner, method))
            return
        target = ".".join(filter(None, (module_name, class_name, method)))
        library = module_name.partition(".")[0]
        try:
            # The namespace alone proves nothing: langgraph-checkpoint fills
            # it without the runtime. Look for the distribution instead.
            distribution(library)
        except PackageNotFoundError:
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

    def instrument(self) -> None:
        for module_name in (
            "langchain_core.runnables.retry",
            "langchain_core.runnables.fallbacks",
        ):
            self._patch(
                module_name, None, "patch_config", self._wrap_attempt_config
            )
        for method in ("invoke", "ainvoke"):
            self._patch(
                "langchain_core.language_models.chat_models",
                "BaseChatModel",
                method,
                self._wrap_input_config,
            )
        try:
            agents = import_module("langchain.agents")
        except ImportError as error:
            _logger.debug("Skipping optional model-call middleware: %s", error)
        else:
            if hasattr(agents, "create_agent"):
                for module_name, class_name in (
                    (
                        "langchain.agents.middleware.model_retry",
                        "ModelRetryMiddleware",
                    ),
                    (
                        "langchain.agents.middleware.model_fallback",
                        "ModelFallbackMiddleware",
                    ),
                ):
                    for method in ("wrap_model_call", "awrap_model_call"):
                        self._patch(
                            module_name,
                            class_name,
                            method,
                            self._wrap_model_attempts,
                        )

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

        # These start the chain run themselves, so the run id is chosen up
        # front for _started to match.
        for module_name, class_name, methods in (
            (
                "langchain_core.runnables.base",
                "Runnable",
                ("_call_with_config", "_acall_with_config"),
            ),
            # RunnableBranch dispatches the chosen branch straight from its
            # own run, so a branch without a boundary inherits it from here.
            (
                "langchain_core.runnables.branch",
                "RunnableBranch",
                ("invoke", "ainvoke"),
            ),
        ):
            for method, asynchronous in zip(methods, (False, True)):
                self._patch(
                    module_name,
                    class_name,
                    method,
                    partial(
                        _wrap_call,
                        invocations=self._invocations,
                        asynchronous=asynchronous,
                    ),
                )

        for module_name in (
            "langchain.chains.base",
            "langchain_classic.chains.base",
        ):
            try:
                import_module(module_name)
            except ImportError:
                continue
            self._patch(
                module_name,
                "Chain",
                "invoke",
                partial(
                    _wrap_call,
                    invocations=self._invocations,
                    asynchronous=False,
                ),
            )

        # The default batch runs a single input in the caller's frame.
        for method in ("batch", "abatch"):
            self._patch(
                "langchain_core.runnables.base",
                "Runnable",
                method,
                self._wrap_input_config,
            )

        for class_name, wrapper in (
            ("CallbackManager", _start_run),
            ("AsyncCallbackManager", _astart_run),
        ):
            for method in (
                "on_chat_model_start",
                "on_llm_start",
                "on_chain_start",
                "on_tool_start",
                "on_retriever_start",
            ):
                self._patch(
                    "langchain_core.callbacks.manager",
                    class_name,
                    method,
                    lambda _original, wrapper=wrapper: partial(
                        wrapper, refresh_handlers=self._refresh_handlers
                    ),
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
            (
                "langchain_core.runnables.branch",
                "RunnableBranch",
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
        # LangChain and LangGraph enter the step's copied context. Each caller
        # binds ``set_config_context`` by name at import, so the replacement
        # goes into the calling modules rather than ``runnables.config``; tools
        # bind it too, but ``BaseTool.run`` above already covers them.
        for module_name, fallback_modules in (
            ("langchain_core.runnables.base", ()),
            ("langchain_core.runnables.fallbacks", ()),
            # LangGraph moved the helper into _internal in 0.6.
            ("langgraph._internal._runnable", ("langgraph.utils.runnable",)),
        ):
            self._patch(
                module_name,
                None,
                "set_config_context",
                lambda _original: self._config_context,
                fallback_modules=fallback_modules,
            )

    def uninstrument(self) -> None:
        for owner, name in reversed(self._patched):
            unwrap(owner, name)
        self._patched.clear()
