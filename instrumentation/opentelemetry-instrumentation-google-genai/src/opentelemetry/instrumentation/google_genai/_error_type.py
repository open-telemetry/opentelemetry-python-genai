# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from google.genai import errors as genai_errors


def resolve_error_type(exc: BaseException) -> str | None:
    """Derive error.type from a google.genai exception.

    google.genai collapses all 4xx into ClientError and 5xx into ServerError,
    so surface the HTTP status code (e.g. ``429``), which distinguishes them.

    The interactions API raises from its own error hierarchy, unrelated to
    ``google.genai.errors.APIError``; those exceptions carry the status as
    ``status_code`` instead.

    Returns None for other exceptions so the util layer falls back to the
    exception class name.
    """
    if isinstance(exc, genai_errors.APIError):
        return str(exc.code)
    if isinstance(status_code := getattr(exc, "status_code", None), int):
        return str(status_code)
    return None
