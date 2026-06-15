# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from base64 import b64encode
from functools import partial
from typing import Any

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.types import ContentCapturingMode

logger = logging.getLogger(__name__)


def get_content_capturing_mode() -> ContentCapturingMode:
    """Gets ContentCapturingMode from associated envvar, defaulting to NO_CONTENT if unset."""
    envvar = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, ""
    ).strip()
    if not envvar:
        return ContentCapturingMode.NO_CONTENT
    try:
        return ContentCapturingMode[envvar.upper()]
    except KeyError:
        logger.warning(
            "%s is not a valid option for `%s` environment variable. Must be one of %s. Defaulting to `NO_CONTENT`.",
            envvar,
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
            ", ".join(e.name for e in ContentCapturingMode),
        )
        return ContentCapturingMode.NO_CONTENT


def is_experimental_mode() -> bool:
    """
    Kept for backwards compatibility. The utils in this library only support the experimental mode sem convs now.
    Don't use this function always returns True.
    """
    return True


def should_emit_event() -> bool:
    """Check if event emission is enabled.

    Returns True if event emission is enabled, False otherwise.

    If the environment variable OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT is explicitly set,
    its value takes precedence. Otherwise, the default value is determined by
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT:
    - NO_CONTENT or SPAN_ONLY: defaults to False
    - EVENT_ONLY or SPAN_AND_EVENT: defaults to True
    """
    # If explicitly set (and not empty), use the user's value (highest priority)
    if (
        envvar := os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, "")
        .lower()
        .strip()
    ):
        if envvar == "true":
            return True
        if envvar == "false":
            return False
        logger.warning(
            "%s is not a valid option for `%s` environment variable. Must be one of true or false (case-insensitive). Defaulting based on content capturing mode.",
            envvar,
            OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
        )
    # EVENT_ONLY and SPAN_AND_EVENT require events, so default to True
    return get_content_capturing_mode() in (
        ContentCapturingMode.EVENT_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


def should_capture_content_on_spans() -> bool:
    """Returns whether capture content is enabled on spans."""
    return get_content_capturing_mode() in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


class _GenAiJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return super().default(o)


gen_ai_json_dump = partial(
    json.dump, separators=(",", ":"), cls=_GenAiJsonEncoder
)
"""Should be used by GenAI instrumentations when serializing objects that may contain
bytes, datetimes, etc. for GenAI observability."""

gen_ai_json_dumps = partial(
    json.dumps, separators=(",", ":"), cls=_GenAiJsonEncoder
)
"""Should be used by GenAI instrumentations when serializing objects that may contain
bytes, datetimes, etc. for GenAI observability."""
