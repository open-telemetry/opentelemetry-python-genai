# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import BoundArguments, Signature, signature
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks.manager import BaseRunManager

from opentelemetry.context import Context, attach, detach
from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)

__all__ = [
    "_astart_run",
    "_start_run",
    "_wrap_call",
    "_wrap_run",
    "_wrap_stream",
]

_logger = logging.getLogger(__name__)


class _RunScope:
    def __init__(
        self,
        run_id: UUID,
        invocations: _InvocationManager,
        on_error: Callable[..., None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.invocations = invocations
        self.context: Context | None = None
        self.on_error = on_error

    @contextmanager
    def activate(self) -> Iterator[None]:
        with ExitStack() as stack:
            token = _active_run.set(_RunActivation(self, stack))
            stack.callback(_active_run.reset, token)
            if self.context is not None:
                stack.callback(detach, attach(self.context))
            yield

    def finish(self, error: BaseException | None = None) -> None:
        invocation = self.invocations.get_invocation(self.run_id)
        try:
            if invocation is not None:
                if error is None:
                    invocation.stop()
                elif self.on_error is not None:
                    self.on_error(error, run_id=self.run_id)
                else:
                    invocation.fail(error)
        except Exception:
            _logger.exception("Failed to finalize LangChain run")
        finally:
            self.invocations.delete_invocation_state(self.run_id)


@dataclass
class _RunActivation:
    scope: _RunScope
    stack: ExitStack


_active_run: ContextVar[_RunActivation | None] = ContextVar(
    "otel_langchain_run_activation", default=None
)


def _started(result: BaseRunManager | list[BaseRunManager]) -> None:
    read = _active_run.get()
    if read is None:
        return
    managers = result if isinstance(result, list) else [result]
    if not any(manager.run_id == read.scope.run_id for manager in managers):
        return
    context = read.scope.invocations.get_parent_context(read.scope.run_id)
    if context is not None:
        # The manager returns in the runner's context, after handlers finish.
        # Its token belongs to this activation, never to a callback handler.
        read.scope.context = context
        read.stack.callback(detach, attach(context))


def _start_run(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    result = wrapped(*args, **kwargs)
    _started(result)
    return result


async def _astart_run(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    result = await wrapped(*args, **kwargs)
    _started(result)
    return result


class _SyncContextStream(SyncStreamWrapper[Any]):
    _self_scope: _RunScope

    def __init__(self, stream: Any, scope: _RunScope) -> None:
        super().__init__(stream)
        self._self_scope = scope

    def _execution_context(self) -> AbstractContextManager[None]:
        return self._self_scope.activate()

    def _process_chunk(self, chunk: Any) -> None:
        pass

    def _on_stream_end(self) -> None:
        self._self_scope.finish()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_scope.finish(error)


class _AsyncContextStream(AsyncStreamWrapper[Any]):
    _self_scope: _RunScope

    def __init__(self, stream: Any, scope: _RunScope) -> None:
        super().__init__(stream)
        self._self_scope = scope

    def _execution_context(self) -> AbstractContextManager[None]:
        return self._self_scope.activate()

    def _process_chunk(self, chunk: Any) -> None:
        pass

    def _on_stream_end(self) -> None:
        self._self_scope.finish()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_scope.finish(error)


def _bind_config_run(
    parameters: Signature,
    invocations: _InvocationManager,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[_RunScope, BoundArguments]:
    bound = parameters.bind(instance, *args, **kwargs)
    config = dict(bound.arguments.get("config") or {})
    run_id = config.get("run_id") or uuid4()
    config["run_id"] = run_id
    bound.arguments["config"] = config
    return _RunScope(run_id, invocations), bound


def _wrap_stream(
    original: Callable[..., Any],
    invocations: _InvocationManager,
    asynchronous: bool,
) -> Callable[..., Any]:
    parameters = signature(original)

    @wraps(original)
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        scope, bound = _bind_config_run(
            parameters, invocations, instance, args, kwargs
        )
        stream = wrapped(*bound.args[1:], **bound.kwargs)
        cls = _AsyncContextStream if asynchronous else _SyncContextStream
        return cls(stream, scope)

    return wrapper


def _wrap_call(
    original: Callable[..., Any],
    invocations: _InvocationManager,
    asynchronous: bool,
) -> Callable[..., Any]:
    parameters = signature(original)

    @wraps(original)
    def sync(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        scope, bound = _bind_config_run(
            parameters, invocations, instance, args, kwargs
        )
        with scope.activate():
            return wrapped(*bound.args[1:], **bound.kwargs)

    @wraps(original)
    async def asynchronous_call(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        scope, bound = _bind_config_run(
            parameters, invocations, instance, args, kwargs
        )
        with scope.activate():
            return await wrapped(*bound.args[1:], **bound.kwargs)

    return asynchronous_call if asynchronous else sync


def _wrap_run(
    original: Callable[..., Any],
    invocations: _InvocationManager,
    on_error: Callable[..., None],
    asynchronous: bool,
) -> Callable[..., Any]:
    def scope_for(kwargs: dict[str, Any]) -> _RunScope:
        run_id = kwargs.get("run_id") or uuid4()
        kwargs["run_id"] = run_id
        return _RunScope(run_id, invocations, on_error)

    @wraps(original)
    def sync(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        scope = scope_for(kwargs)
        with scope.activate():
            try:
                return wrapped(*args, **kwargs)
            except BaseException as error:
                # Some runners omit error callbacks for cancellation.
                scope.finish(error)
                raise

    @wraps(original)
    async def asynchronous_call(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        scope = scope_for(kwargs)
        with scope.activate():
            try:
                return await wrapped(*args, **kwargs)
            except BaseException as error:
                scope.finish(error)
                raise

    return asynchronous_call if asynchronous else sync
