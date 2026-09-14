# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from typing import Final

SERVER_ADDRESS: Final[str] = "server.address"
"""Server domain name if available without reverse DNS lookup; otherwise, IP address or UNIX domain socket name.
When observed from the client side, and when communicating through an intermediary, `server.address` SHOULD represent the server address behind any intermediaries, for example proxies, if it's available."""

SERVER_PORT: Final[str] = "server.port"
"""Server port number.
When observed from the client side, and when communicating through an intermediary, `server.port` SHOULD represent the server port behind any intermediaries, for example proxies, if it's available."""


__all__ = [
    "SERVER_ADDRESS",
    "SERVER_PORT",
]
