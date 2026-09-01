# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import urllib.parse
from base64 import b64decode, b64encode
from functools import partial
from typing import Any

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    ContentCapturingMode,
    MessagePart,
    UriPart,
)

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


def decode_base64(data: str) -> bytes | None:
    """Decode a base64 string, returning ``None`` if it is malformed.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    try:
        return b64decode("".join(data.split()), validate=True)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def image_from_url(url: str, *, modality: str = "image") -> MessagePart | None:
    """Return a media part for an image ``url``.

    A ``data:<mime>;base64,<payload>`` URL is decoded into a
    :class:`~opentelemetry.util.genai.types.BlobPart`; a ``data:`` URL without
    base64 encoding has its percent-encoded payload decoded into bytes; any
    other URL becomes a :class:`~opentelemetry.util.genai.types.UriPart`. Shared
    by instrumentations that parse provider image blocks.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    if url.startswith("data:"):
        header, _, payload = url[len("data:") :].partition(",")
        mime_type = header.split(";", 1)[0] or None
        if ";base64" in header.lower():
            decoded = decode_base64(payload)
            if decoded is None:
                return None
            content = decoded
        else:
            # Non-base64 data URL payloads are percent-encoded (RFC 2397).
            content = urllib.parse.unquote_to_bytes(payload)
        return BlobPart(
            mime_type=mime_type,
            modality=modality,
            content=content,
        )
    return UriPart(mime_type=None, modality=modality, uri=url)


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


def fq_exception_type(exception: BaseException) -> str:
    """Return the fully qualified name of an exception's type.

    Matches the ``exception.type`` value the exception event records, so
    ``error.type`` and ``exception.type`` stay consistent. Builtins are returned
    unqualified (e.g. ``ValueError``, not ``builtins.ValueError``).
    """
    # Mirrors the SDK's Span.record_exception so error.type matches the
    # exception event's exception.type:
    # https://github.com/open-telemetry/opentelemetry-python/blob/main/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py
    exc_type = type(exception)
    module = exc_type.__module__
    qualname = exc_type.__qualname__
    if module and module != "builtins":
        return f"{module}.{qualname}"
    return qualname


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
