# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from typing import Final

EXCEPTION_MESSAGE: Final[str] = "exception.message"
"""The exception message.
> [!WARNING]
>
> This attribute may contain sensitive information."""

EXCEPTION_STACKTRACE: Final[str] = "exception.stacktrace"
"""A stacktrace as a string in the natural representation for the language runtime. The representation is to be determined and documented by each language SIG."""

EXCEPTION_TYPE: Final[str] = "exception.type"
"""The type of the exception (its fully-qualified class name, if applicable). The dynamic type of the exception should be preferred over the static type in languages that support it.
If the recorded exception type is a wrapper that is not meaningful for
failure classification, instrumentation MAY use the type of the inner
exception instead. For example, in Go, errors created with `fmt.Errorf`
using `%w` MAY be unwrapped when the wrapper type does not help
classify the failure."""


__all__ = [
    "EXCEPTION_MESSAGE",
    "EXCEPTION_STACKTRACE",
    "EXCEPTION_TYPE",
]
