# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from typing import Final

CLIENT_ADDRESS: Final[str] = "client.address"
"""Client address - domain name if available without reverse DNS lookup; otherwise, IP address or UNIX domain socket name.
When observed from the server side, and when communicating through an intermediary, `client.address` SHOULD represent the client address behind any intermediaries,  for example proxies, if it's available."""

CLIENT_PORT: Final[str] = "client.port"
"""Client port number.
When observed from the server side, and when communicating through an intermediary, `client.port` SHOULD represent the client port behind any intermediaries,  for example proxies, if it's available."""


__all__ = [
    "CLIENT_ADDRESS",
    "CLIENT_PORT",
]
