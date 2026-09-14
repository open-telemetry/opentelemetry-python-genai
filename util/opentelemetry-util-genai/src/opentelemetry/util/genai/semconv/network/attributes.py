# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from enum import Enum
from typing import Final

NETWORK_PROTOCOL_NAME: Final[str] = "network.protocol.name"
"""[OSI application layer](https://wikipedia.org/wiki/Application_layer) or non-OSI equivalent.
The value SHOULD be normalized to lowercase."""

NETWORK_PROTOCOL_VERSION: Final[str] = "network.protocol.version"
"""The actual version of the protocol used for network communication.
If protocol version is subject to negotiation (for example using [ALPN](https://www.rfc-editor.org/rfc/rfc7301.html)), this attribute SHOULD be set to the negotiated version. If the actual protocol version is not known, this attribute SHOULD NOT be set."""

NETWORK_TRANSPORT: Final[str] = "network.transport"
"""[OSI transport layer](https://wikipedia.org/wiki/Transport_layer) or [inter-process communication method](https://wikipedia.org/wiki/Inter-process_communication).
The value SHOULD be normalized to lowercase.

Consider always setting the transport when setting a port number, since
a port number is ambiguous without knowing the transport. For example
different processes could be listening on TCP port 12345 and UDP port 12345."""


class NetworkTransport(str, Enum):
    """Known values for `network.transport`."""

    TCP = "tcp"
    """TCP"""
    UDP = "udp"
    """UDP"""
    PIPE = "pipe"
    """Named or anonymous pipe."""
    UNIX = "unix"
    """UNIX domain socket"""
    QUIC = "quic"
    """QUIC"""


__all__ = [
    "NETWORK_PROTOCOL_NAME",
    "NETWORK_PROTOCOL_VERSION",
    "NETWORK_TRANSPORT",
    "NetworkTransport",
]
