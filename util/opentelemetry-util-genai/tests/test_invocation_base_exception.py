# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for how ``GenAIInvocation.__exit__`` finalizes on ``BaseException``.

``asyncio.CancelledError``, ``KeyboardInterrupt``, ``SystemExit`` and
``GeneratorExit`` derive from ``BaseException`` but not from ``Exception``. An
``__exit__`` that only recognizes ``Exception`` finalizes them through the
success path, so a cancelled operation is exported as a successful one.
"""

from __future__ import annotations

import asyncio
from unittest import TestCase

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler

# Bounds every await in the cancellation scenarios below. The scenarios park on
# ``asyncio.sleep(60)`` -- effectively forever -- so that a cancellation that
# failed to propagate would otherwise stall the run instead of failing it.
_TIMEOUT = 5


class _InvocationExitTestBase(TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=self.tracer_provider)

    def _only_span(self):
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        return spans[0]


class InvocationExitBaseExceptionTest(_InvocationExitTestBase):
    """A ``BaseException`` leaving the block must be recorded as a failure."""

    def test_cancelled_error_sets_error_status(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            with self.handler.workflow("wf"):
                raise asyncio.CancelledError()

        self.assertEqual(
            self._only_span().status.status_code, StatusCode.ERROR
        )

    def test_cancelled_error_sets_error_type(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            with self.handler.workflow("wf"):
                raise asyncio.CancelledError()

        self.assertEqual(
            self._only_span().attributes["error.type"],
            "asyncio.exceptions.CancelledError",
        )

    def test_keyboard_interrupt_sets_error_status(self) -> None:
        with pytest.raises(KeyboardInterrupt):
            with self.handler.workflow("wf"):
                raise KeyboardInterrupt()

        span = self._only_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "KeyboardInterrupt")

    def test_system_exit_sets_error_status(self) -> None:
        with pytest.raises(SystemExit):
            with self.handler.workflow("wf"):
                raise SystemExit(1)

        span = self._only_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "SystemExit")

    def test_generator_exit_sets_error_status(self) -> None:
        # Documents the deliberate consequence of guarding on BaseException:
        # an invocation abandoned via generator close is a failed invocation.
        # No `with <invocation>:` block in this repository contains a yield,
        # so this path is not reachable from the shipped instrumentations.
        with pytest.raises(GeneratorExit):
            with self.handler.workflow("wf"):
                raise GeneratorExit()

        span = self._only_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "GeneratorExit")

    def test_cancelled_task_sets_error_status(self) -> None:
        """The realistic path: a real awaited task cancelled by asyncio."""

        async def scenario() -> None:
            started = asyncio.Event()

            async def work() -> None:
                with self.handler.workflow("wf"):
                    started.set()
                    await asyncio.sleep(60)

            task = asyncio.ensure_future(work())
            await asyncio.wait_for(started.wait(), _TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, _TIMEOUT)

        asyncio.run(scenario())

        span = self._only_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(
            span.attributes["error.type"], "asyncio.exceptions.CancelledError"
        )


class InvocationExitMustNotSuppressTest(_InvocationExitTestBase):
    """``__exit__`` must keep returning ``None`` so nothing is suppressed.

    Suppressing ``CancelledError`` would break asyncio cancellation for every
    caller — a worse defect than the one being fixed.
    """

    def test_exit_returns_none_for_cancelled_error(self) -> None:
        invocation = self.handler.workflow("wf")
        invocation.__enter__()
        error = asyncio.CancelledError()
        self.assertIsNone(
            invocation.__exit__(type(error), error, error.__traceback__)
        )

    def test_exit_returns_none_on_success(self) -> None:
        invocation = self.handler.workflow("wf")
        invocation.__enter__()
        self.assertIsNone(invocation.__exit__(None, None, None))

    def test_cancelled_error_propagates_to_caller(self) -> None:
        with pytest.raises(asyncio.CancelledError):
            with self.handler.workflow("wf"):
                raise asyncio.CancelledError()

    def test_cancellation_of_a_task_still_cancels_it(self) -> None:
        """End-to-end: the task must still end up cancelled, not swallowed."""

        async def scenario() -> bool:
            started = asyncio.Event()

            async def work() -> None:
                with self.handler.workflow("wf"):
                    started.set()
                    await asyncio.sleep(60)

            task = asyncio.ensure_future(work())
            await asyncio.wait_for(started.wait(), _TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, _TIMEOUT)
            return task.cancelled()

        self.assertTrue(asyncio.run(scenario()))


class InvocationExitSuccessPathTest(_InvocationExitTestBase):
    """Regression guard: the success path must be untouched."""

    def test_clean_exit_leaves_status_unset(self) -> None:
        with self.handler.workflow("wf"):
            pass

        span = self._only_span()
        self.assertEqual(span.status.status_code, StatusCode.UNSET)
        self.assertNotIn("error.type", span.attributes or {})

    def test_ordinary_exception_still_sets_error_status(self) -> None:
        with pytest.raises(ValueError):
            with self.handler.workflow("wf"):
                raise ValueError("boom")

        span = self._only_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "ValueError")
        self.assertEqual(span.status.description, "boom")
