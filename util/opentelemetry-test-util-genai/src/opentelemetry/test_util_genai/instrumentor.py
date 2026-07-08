# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Context manager for the instrument/uninstrument cycle in tests.

Every instrumentation's ``tests/conftest.py`` carries a handful of fixtures
shaped like:

- reset ``_OpenTelemetrySemanticConventionStability._initialized``
- set ``OTEL_SEMCONV_STABILITY_OPT_IN`` and/or
  ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` (and sometimes
  ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT``)
- ``instrumentor.instrument(tracer_provider=..., logger_provider=..., meter_provider=...)``
- ``yield instrumentor``
- restore env vars; ``instrumentor.uninstrument()``; reset the stability flag

The body is identical across packages — only the instrumentor class and the
env values differ. This module hosts that body once so per-package
conftests collapse to a thin wrapper.

The stability-class reset is required because
``_OpenTelemetrySemanticConventionStability`` caches the first read of
``OTEL_SEMCONV_STABILITY_OPT_IN`` — without resetting ``_initialized``
mid-test, a later env change would not take effect on the next
``.instrument()`` call.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)


@contextmanager
def instrument(
    instrumentor: BaseInstrumentor,
    *,
    tracer_provider: Any,
    logger_provider: Any,
    meter_provider: Any,
    content_capture: str | None = None,
    emit_event: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> Iterator[BaseInstrumentor]:
    """Set semconv/content envs, instrument, yield, restore env + uninstrument.

    Use inside a fixture body::

        @pytest.fixture
        def instrument_with_content(
            tracer_provider, logger_provider, meter_provider
        ):
            with instrument(
                AnthropicInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ) as instrumentor:
                yield instrumentor

    ``content_capture`` is forwarded to
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` and
    ``emit_event=True`` sets ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` to
    ``"true"``; both default to leaving their variable untouched. Pass
    ``extra_env`` for anything else.

    Previous values are restored on exit so tests stay isolated.
    """
    overrides: dict[str, str] = {}
    if content_capture is not None:
        overrides[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
            content_capture
        )
    if emit_event:
        overrides[OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT] = "true"
    if extra_env:
        overrides.update(extra_env)
    previous = {k: os.environ.get(k) for k in overrides}

    _OpenTelemetrySemanticConventionStability._initialized = False
    os.environ.update(overrides)
    try:
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        try:
            yield instrumentor
        finally:
            instrumentor.uninstrument()
    finally:
        for key, prev in previous.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
        _OpenTelemetrySemanticConventionStability._initialized = False
