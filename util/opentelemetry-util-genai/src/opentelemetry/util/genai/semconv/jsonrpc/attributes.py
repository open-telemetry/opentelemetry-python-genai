# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from typing import Final

JSONRPC_PROTOCOL_VERSION: Final[str] = "jsonrpc.protocol.version"
"""Protocol version, as specified in the `jsonrpc` property of the request and its corresponding response."""

JSONRPC_REQUEST_ID: Final[str] = "jsonrpc.request.id"
"""A string representation of the `id` property of the request and its corresponding response.
Under the [JSON-RPC specification](https://www.jsonrpc.org/specification), the `id` property may be a string, number, null, or omitted entirely. When omitted, the request is treated as a notification. Using `null` is not equivalent to omitting the `id`, but it is discouraged.
Instrumentations SHOULD NOT capture this attribute when the `id` is `null` or omitted."""


__all__ = [
    "JSONRPC_PROTOCOL_VERSION",
    "JSONRPC_REQUEST_ID",
]
