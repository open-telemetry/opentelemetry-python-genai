# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""A local Bedrock endpoint and a recording request body.

``botocore.stub.Stubber`` answers a call before the request is serialized,
signed and sent, so a stubbed call never touches a file-like body. Tests that
need to see what the SDK does with such a body answer over a real socket
instead: the SDK builds, signs and sends the request as it does against AWS,
and the endpoint records the bytes that arrive.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import boto3

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"

INVOKE_MODEL = "invoke_model"
INVOKE_MODEL_STREAM = "invoke_model_with_response_stream"
OPERATIONS = (INVOKE_MODEL, INVOKE_MODEL_STREAM)

ANTHROPIC_REQUEST = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 50,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
    }
).encode()

ANTHROPIC_RESPONSE = json.dumps(
    {
        "id": "msg_12345",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }
).encode()

ERROR_RESPONSE = json.dumps({"message": "boom"}).encode()


def encode_event_stream(events: Sequence[dict[str, Any]]) -> bytes:
    """Encode *events* as the AWS event stream a streaming response uses."""
    return b"".join(_event_stream_message(event) for event in events)


def _event_stream_message(payload: dict[str, Any]) -> bytes:
    headers = {
        ":event-type": "chunk",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    encoded_headers = b"".join(
        bytes([len(name)])
        + name.encode()
        + b"\x07"
        + len(value).to_bytes(2, "big")
        + value.encode()
        for name, value in headers.items()
    )
    data = json.dumps(payload).encode()
    total_length = 16 + len(encoded_headers) + len(data)
    prelude = total_length.to_bytes(4, "big") + len(encoded_headers).to_bytes(
        4, "big"
    )
    prelude += (binascii.crc32(prelude) & 0xFFFFFFFF).to_bytes(4, "big")
    message = prelude + encoded_headers + data
    return message + (binascii.crc32(message) & 0xFFFFFFFF).to_bytes(4, "big")


def _chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """One Bedrock stream event, with its JSON payload base64-encoded."""
    return {"bytes": base64.b64encode(json.dumps(chunk).encode()).decode()}


ANTHROPIC_STREAM_RESPONSE = encode_event_stream(
    [
        _chunk(
            {
                "type": "message_start",
                "message": {
                    "role": "assistant",
                    "usage": {"input_tokens": 12},
                },
            }
        ),
        _chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi there!"},
            }
        ),
        _chunk({"type": "message_stop"}),
    ]
)


class RecordingBody(io.BytesIO):
    """A seekable file-like request body that records the calls made on it.

    It is a real ``io.IOBase``, as a body the SDKs accept has to be, and as the
    bodies callers pass in usually are (``io.BytesIO``, an open file).

    ``chunk_size`` limits how much a single read returns, which is how a body
    that is only read in parts is reproduced.
    """

    def __init__(
        self,
        data: bytes,
        *,
        position: int = 0,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__(data)
        self._chunk_size = chunk_size
        self._calls: list[str] = []
        self.seek(position)

    @property
    def calls(self) -> tuple[str, ...]:
        """Every call the SDK made on the body, in order."""
        return tuple(self._calls)

    @property
    def position(self) -> int:
        """Where the body is, as the SDK left it, without recording a call."""
        return io.BytesIO.tell(self)

    def read(self, size: int | None = -1) -> bytes:
        if self._chunk_size is not None and (
            size is None or size < 0 or size > self._chunk_size
        ):
            size = self._chunk_size
        data = super().read(size)
        self._calls.append(f"read({size})")
        return data

    def readinto(self, buffer: Any) -> int:
        written = super().readinto(buffer)
        self._calls.append("readinto")
        return written

    def seek(self, offset: int, whence: int = 0) -> int:
        position = super().seek(offset, whence)
        self._calls.append(f"seek({offset}, {whence})")
        return position

    def tell(self) -> int:
        position = super().tell()
        self._calls.append("tell")
        return position


@dataclass
class CallResult:
    """What one Bedrock call did to its request body."""

    calls: tuple[str, ...]
    position: int
    received: list[bytes]
    error: str | None


class LocalBedrockEndpoint:
    """Answers Bedrock calls over a real socket, recording what arrived.

    ``responses`` are ``(status, body)`` pairs served in order; the last one is
    reused for any further attempt, so a test can fail an attempt and then
    succeed.
    """

    def __init__(self, responses: Sequence[tuple[int, bytes]]) -> None:
        self._responses = list(responses)
        self._received: list[bytes] = []
        self._lock = threading.Lock()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                status, payload = endpoint._serve(body)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                """Keep the endpoint's request log out of the test output."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[0:2]
        return f"http://{host}:{port}"

    @property
    def received(self) -> list[bytes]:
        """The request bodies that arrived, one per attempt."""
        with self._lock:
            return list(self._received)

    def client(self) -> Any:
        """A Bedrock runtime client that talks to this endpoint."""
        return boto3.session.Session().client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            endpoint_url=self.url,
        )

    @asynccontextmanager
    async def async_client(self) -> AsyncIterator[Any]:
        """An aiobotocore Bedrock runtime client that talks to this endpoint.

        Imported here so the synchronous tests do not need aiobotocore.
        """
        import aiobotocore.session

        session = aiobotocore.session.get_session()
        async with session.create_client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            endpoint_url=self.url,
        ) as client:
            yield client

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _serve(self, body: bytes) -> tuple[int, bytes]:
        with self._lock:
            self._received.append(body)
            index = min(len(self._received) - 1, len(self._responses) - 1)
        return self._responses[index]


@contextmanager
def local_bedrock(
    responses: Sequence[tuple[int, bytes]],
) -> Iterator[LocalBedrockEndpoint]:
    """Serve canned Bedrock responses for the duration of a test."""
    endpoint = LocalBedrockEndpoint(responses)
    try:
        yield endpoint
    finally:
        endpoint.close()


def call_bedrock(
    endpoint: LocalBedrockEndpoint,
    operation: str,
    body: RecordingBody,
    model_id: str = MODEL_ID,
    *,
    drain: bool = False,
) -> CallResult:
    """Call Bedrock and report the body's I/O, the sent bytes and any error.

    A streaming response is left undrained unless *drain* is set: the request
    body's I/O is over by then, and the stream is only read once the caller
    iterates it.
    """
    client = endpoint.client()
    error: str | None = None
    try:
        if operation == INVOKE_MODEL:
            response = client.invoke_model(modelId=model_id, body=body)
            response["body"].read()
        else:
            response = client.invoke_model_with_response_stream(
                modelId=model_id, body=body
            )
            if drain:
                for _ in response["body"]:
                    pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return CallResult(body.calls, body.position, endpoint.received, error)


async def call_bedrock_async(
    endpoint: LocalBedrockEndpoint,
    operation: str,
    body: RecordingBody,
    model_id: str = MODEL_ID,
    *,
    drain: bool = False,
) -> CallResult:
    """Call Bedrock through aiobotocore, and report the same things."""
    error: str | None = None
    async with endpoint.async_client() as client:
        try:
            if operation == INVOKE_MODEL:
                response = await client.invoke_model(
                    modelId=model_id, body=body
                )
                await response["body"].read()
            else:
                response = await client.invoke_model_with_response_stream(
                    modelId=model_id, body=body
                )
                if drain:
                    async for _ in response["body"]:
                        pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return CallResult(body.calls, body.position, endpoint.received, error)
