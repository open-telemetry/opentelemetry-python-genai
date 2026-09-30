# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Generator,
    Iterator,
)
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import BoundArguments, signature
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks.manager import BaseRunManager

from opentelemetry.context import Context, attach, detach, get_current
from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)

__all__ = [
    "_RunScope",
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
        run_id: UUID | None,
        invocations: _InvocationManager,
        on_error: Callable[..., None] | None = None,
    ) -> None:
        # None until a deferred stream chooses its run id on the first read.
        self.run_id = run_id
        self.invocations = invocations
        self.context: Context | None = None
        self.on_error = on_error

    @contextmanager
    def activate(self) -> Iterator[None]:
        with ExitStack() as stack:
            token = _active_run.set(_RunActivation(self, stack))
            stack.callback(_active_run.reset, token)
            # A stream read inside an enclosing read of the same run tree
            # already has this context current.
            if self.context is not None and get_current() is not self.context:
                stack.callback(detach, attach(self.context))
            yield

    def finish(self, error: BaseException | None = None) -> None:
        # The callback handler ends the invocation from its end/error callback;
        # its state stays registered while a child is live, so this may find
        # an ended invocation, and finishing it again is a no-op in the util.
        # This is the end when no callback fires: a tool or retriever cancelled
        # by a BaseException the runner does not report, a chat model whose
        # agenerate gather child is cancelled or interrupted, or a stream
        # closed early by a runner that leaves its inner iterator open
        # (RunnableLambda.astream). on_error routes a tool or retriever error
        # through the handler's own error path so the attributes match.
        if self.run_id is None:
            return
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
        # An enclosing composite may have attached the same context already.
        read.scope.context = context
        if get_current() is not context:
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


def _config_run(bound: BoundArguments, scope: _RunScope) -> BoundArguments:
    # The run id is chosen here so _started can tell this frame's own run from
    # one started under the same activation before reaching its own boundary
    # (a chat model invoked from a lambda body): attaching that one here would
    # leave its ended span current for the rest of this frame.
    config = dict(bound.arguments.get("config") or {})
    scope.run_id = config.get("run_id") or uuid4()
    config["run_id"] = scope.run_id
    bound.arguments["config"] = config
    return bound


def _deferred(start: Callable[[], Iterator[Any]]) -> Generator[Any, Any, None]:
    yield from start()


async def _adeferred(
    start: Callable[[], AsyncIterator[Any]],
) -> AsyncGenerator[Any, None]:
    stream = start()
    try:
        async for chunk in stream:
            yield chunk
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


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
        scope = _RunScope(None, invocations)
        bound = parameters.bind(instance, *args, **kwargs)

        def start() -> Any:
            # A generator reads its config on the first advancement; keep
            # that so the caller may still edit the config until then.
            _config_run(bound, scope)
            return wrapped(*bound.args[1:], **bound.kwargs)

        stream: Any = _adeferred(start) if asynchronous else _deferred(start)
        # Keep the SDK's name on the generator the caller sees.
        stream.__qualname__ = original.__qualname__
        stream.__name__ = original.__name__
        if asynchronous:
            return _AsyncContextStream(stream, scope)
        return _SyncContextStream(stream, scope)

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
        scope = _RunScope(None, invocations)
        bound = _config_run(parameters.bind(instance, *args, **kwargs), scope)
        with scope.activate():
            return wrapped(*bound.args[1:], **bound.kwargs)

    @wraps(original)
    async def asynchronous_call(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        scope = _RunScope(None, invocations)
        bound = _config_run(parameters.bind(instance, *args, **kwargs), scope)
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
