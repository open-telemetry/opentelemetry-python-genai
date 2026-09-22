# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import Token
from types import TracebackType
from typing import Any, TypeAlias, cast

from typing_extensions import Self

from opentelemetry.context import Context, attach, detach
from opentelemetry.trace import Span
from opentelemetry.util.genai.types import (
    Error,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
)
from opentelemetry.util.types import AttributeValue

ContextToken: TypeAlias = Token[Context]


class GenAIInvocation(AbstractContextManager["GenAIInvocation"]):
    """
    Base class for all GenAI invocation types. Manages the lifecycle of a single
    GenAI operation (LLM call, embedding, tool execution, workflow, etc.).

    Use the factory methods on TelemetryHandler (inference, embedding,
    workflow, tool) rather than constructing invocations directly.
    """

    attributes: dict[str, AttributeValue]
    metric_attributes: dict[str, AttributeValue]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._context_token: ContextToken | None = None

    @property
    def span(self) -> Span:
        """The underlying span.

        .. deprecated:: 1.3b0
            Use :attr:`context` instead.
        """
        return cast(Any, super()).span

    @property
    def context(self) -> Context:
        """The OpenTelemetry Context containing this invocation's span."""
        return cast(Any, super()).context

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this invocation."""
        return cast(Any, super()).should_capture_content

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        span: Span = cast(Any, super()).start(
            name=name, context=context, start_time=start_time
        )
        self._context_token = attach(self.context)
        return span

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Apply finish telemetry. Finishes at most once."""
        if self._context_token is None:
            return
        context_token, self._context_token = self._context_token, None
        self._on_finish(context=context)
        try:
            cast(Any, super()).finish(
                error=error,
                duration_s=duration_s,
                end_time=end_time,
                content_capturing_mode=content_capturing_mode,
                emit_event=emit_event,
                context=context,
            )
        finally:
            try:
                detach(context_token)
            except Exception:  # pylint: disable=broad-except
                pass

    def _on_finish(self, context: Context | None = None) -> None:
        pass

    def stop(self) -> None:
        """Finalize the invocation successfully and end its span."""
        self.finish()

    def fail(self, error: Error | BaseException) -> None:
        """Fail the invocation and end its span with error status."""
        self.finish(error=error)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is not None:
            self.fail(exc_value)
        else:
            self.stop()
