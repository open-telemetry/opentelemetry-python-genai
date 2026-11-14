# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the capture that records the reads made on a request body."""

from __future__ import annotations

import io

from opentelemetry.instrumentation.genai.bedrock._body_capture import (
    BodyCapture,
    capture_body,
)

from .bedrock_endpoint import RecordingBody

DATA = b'{"max_tokens": 50, "temperature": 0.3}'


def read_pass(body: BodyCapture, start: int = 0) -> bytes:
    """Read a body the way botocore reads one: measure it, then read it out."""
    body.seek(0, 2)
    body.tell()
    body.seek(start)
    chunks: list[bytes] = []
    while chunk := body.read(1024 * 1024):
        chunks.append(chunk)
    body.seek(start)
    return b"".join(chunks)


def test_records_the_pass_the_sdk_makes() -> None:
    capture = capture_body(RecordingBody(DATA))
    assert capture is not None

    assert read_pass(capture) == DATA
    assert capture.snapshot() == DATA


def test_snapshot_reads_nothing() -> None:
    """The snapshot comes from what was recorded, never from the body."""
    body = RecordingBody(DATA)
    capture = capture_body(body)
    assert capture is not None
    read_pass(capture)
    calls_before = body.calls

    assert capture.snapshot() == DATA
    assert capture.snapshot() == DATA
    assert body.calls == calls_before


def test_capture_adds_no_calls_of_its_own() -> None:
    """Every call on the proxy is a call the test made, in the same order."""
    body = RecordingBody(DATA, chunk_size=8)
    capture = capture_body(body)
    assert capture is not None
    body.seek(0, 2)
    body.seek(0)
    while capture.read(8):
        pass
    capture.seek(0)

    # The constructor positions the body, which is the first call recorded.
    calls = body.calls[1:]
    assert calls[:2] == ("seek(0, 2)", "seek(0, 0)")
    assert set(calls[2:-1]) == {"read(8)"}
    assert calls[-1] == "seek(0, 0)"
    # The capture calls tell() only when the test does, here never.
    assert "tell" not in calls
    assert capture.snapshot() == DATA


def test_stops_after_the_first_complete_pass() -> None:
    """A retry rereads the body, and cannot duplicate what was recorded."""
    capture = capture_body(RecordingBody(DATA))
    assert capture is not None
    read_pass(capture)

    for _ in range(3):
        capture.seek(0)
        while capture.read(8):
            pass

    assert capture.snapshot() == DATA


def test_records_a_pass_read_in_parts() -> None:
    capture = capture_body(RecordingBody(DATA, chunk_size=7))
    assert capture is not None
    capture.seek(0, 2)
    capture.seek(0)
    chunks: list[bytes] = []
    while chunk := capture.read(7):
        chunks.append(chunk)

    assert b"".join(chunks) == DATA
    assert capture.snapshot() == DATA


def test_records_from_the_position_the_body_starts_at() -> None:
    """A body positioned part way in sends the rest, which is the payload."""
    capture = capture_body(RecordingBody(b"PAD!" + DATA, position=4))
    assert capture is not None

    assert read_pass(capture, start=4) == DATA
    assert capture.snapshot() == DATA


def test_a_body_larger_than_the_limit_is_left_uncaptured() -> None:
    capture = capture_body(RecordingBody(DATA), max_bytes=len(DATA) - 1)
    assert capture is not None
    read_pass(capture)

    assert capture.snapshot() is None


def test_a_body_of_exactly_the_limit_is_kept() -> None:
    capture = capture_body(RecordingBody(DATA), max_bytes=len(DATA))
    assert capture is not None
    read_pass(capture)

    assert capture.snapshot() == DATA


def test_a_text_body_is_not_captured() -> None:
    capture = capture_body(io.StringIO(DATA.decode()))
    assert capture is not None
    capture.seek(0)
    while capture.read(1024):
        pass

    assert capture.snapshot() is None


def test_a_partial_pass_yields_nothing() -> None:
    """Reading to the middle of the body is not a pass over it."""
    capture = capture_body(RecordingBody(DATA))
    assert capture is not None
    capture.seek(0)
    capture.read(8)

    assert capture.snapshot() is None


def test_a_body_read_without_a_seek_is_recorded() -> None:
    """Reads start where the body was positioned, so the pass is the payload."""
    capture = capture_body(RecordingBody(DATA))
    assert capture is not None
    while capture.read(1024):
        pass

    assert capture.snapshot() == DATA


def test_a_body_without_read_and_seek_is_not_wrapped() -> None:
    """Wrapping would give botocore methods the caller's object never had."""

    class OneShot:
        def read(self) -> bytes:
            return DATA

    assert capture_body(OneShot()) is None
    assert capture_body(None) is None
    assert capture_body(DATA) is None


def test_the_proxy_is_transparent() -> None:
    capture = capture_body(RecordingBody(DATA))
    assert capture is not None

    assert isinstance(capture, io.BytesIO)
    assert capture.read(4) == DATA[:4]
    assert capture.tell() == 4
    assert capture.seek(0) == 0
    assert hasattr(capture, "getvalue")
