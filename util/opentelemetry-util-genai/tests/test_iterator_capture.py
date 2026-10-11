# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Generator, Iterator
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pytest

from opentelemetry.util.genai.iterator_capture import (
    UNCONSUMED,
    IteratorCapture,
)


@pytest.mark.parametrize("on_disk", [False, True])
@pytest.mark.parametrize("binary", [False, True])
def test_file_objects_are_preserved(
    tmp_path: Path, on_disk: bool, binary: bool
) -> None:
    path = tmp_path / "content.txt"
    path.write_bytes(b"content")
    if on_disk:
        data = path.open("rb" if binary else "r")
    else:
        data = BytesIO(b"content") if binary else StringIO("content")
    with data:
        capture = IteratorCapture({"data": data})
        assert capture.wrapped["data"] is data
        assert data.tell() == 0
        assert capture.snapshot(unconsumed=()) == {"data": data}
        assert capture.wrapped["data"].read() == (
            b"content" if binary else "content"
        )
        assert capture.snapshot(unconsumed=()) == {"data": data}
        assert capture.complete


def test_capture_is_lazy_and_distinguishes_empty_from_unread() -> None:
    reads: list[str] = []

    def generate() -> Iterator[str]:
        reads.append("read")
        yield from ()

    capture = IteratorCapture(generate())
    assert reads == []
    assert capture.snapshot() is UNCONSUMED
    assert capture.snapshot(unconsumed=()) == ()
    assert not capture.complete
    assert list(capture.wrapped) == []
    assert reads == ["read"]
    assert capture.snapshot() == []
    assert capture.complete


def test_nested_capture_preserves_containers_and_partial_input() -> None:
    inner = iter(["first", "second"])
    source = {"arbitrary": [{"payload": (inner,)}]}
    capture = IteratorCapture(source)
    wrapped = capture.wrapped["arbitrary"][0]["payload"][0]
    assert isinstance(wrapped, type(inner))
    assert wrapped.__class__ is inner.__class__
    assert source["arbitrary"][0]["payload"][0] is inner
    assert capture.snapshot() == {"arbitrary": [{"payload": (UNCONSUMED,)}]}
    assert next(wrapped) == "first"
    assert capture.snapshot() == {"arbitrary": [{"payload": (["first"],)}]}
    assert not capture.complete
    assert list(wrapped) == ["second"]
    assert capture.complete
    assert capture.snapshot() == {
        "arbitrary": [{"payload": (["first", "second"],)}]
    }


def test_yielded_mutable_objects_are_snapshotted_per_item() -> None:
    shared: dict[str, object] = {"value": [0]}

    def generate() -> Iterator[dict[str, object]]:
        yield shared
        shared["value"] = [1]
        yield shared
        shared.clear()

    capture = IteratorCapture(generate())
    iterator = capture.wrapped
    assert next(iterator) is shared
    assert next(iterator) is shared
    assert list(iterator) == []
    assert capture.snapshot() == [{"value": [0]}, {"value": [1]}]


def test_nested_iterators_yielded_by_iterators_stay_lazy() -> None:
    reads: list[str] = []

    def inner() -> Iterator[str]:
        reads.append("inner")
        yield "value"

    capture = IteratorCapture(iter([{"data": inner()}]))
    item = next(capture.wrapped)
    assert reads == []
    assert capture.snapshot() == [{"data": UNCONSUMED}]
    assert list(item["data"]) == ["value"]
    assert capture.snapshot() == [{"data": ["value"]}]
    assert not capture.complete
    assert list(capture.wrapped) == []
    assert capture.complete


@pytest.mark.parametrize(
    "error", [ValueError("iterator failed"), asyncio.CancelledError()]
)
def test_iterator_errors_propagate_unchanged(error: BaseException) -> None:
    def generate() -> Iterator[str]:
        yield "first"
        raise error

    capture = IteratorCapture(generate())
    assert next(capture.wrapped) == "first"
    with pytest.raises(type(error)) as caught:
        next(capture.wrapped)
    assert caught.value is error
    assert capture.snapshot() == ["first"]
    assert not capture.complete


def test_generator_protocol_and_return_value() -> None:
    closed: list[bool] = []

    def generate() -> Generator[object, object, str]:
        try:
            received = yield "first"
            try:
                yield received
            except ValueError:
                yield "handled"
            return "finished"
        finally:
            closed.append(True)

    capture = IteratorCapture(generate())
    assert next(capture.wrapped) == "first"
    assert capture.wrapped.send("sent") == "sent"
    assert capture.wrapped.throw(ValueError()) == "handled"
    with pytest.raises(StopIteration) as caught:
        next(capture.wrapped)
    assert caught.value.value == "finished"
    assert capture.snapshot() == ["first", "sent", "handled"]
    assert capture.complete
    assert closed == [True]


def test_close_does_not_drain_input() -> None:
    def generate() -> Iterator[int]:
        yield 1
        pytest.fail("close must not advance the iterator")

    capture = IteratorCapture(generate())
    assert next(capture.wrapped) == 1
    capture.wrapped.close()
    assert capture.snapshot() == [1]
    assert not capture.complete


def test_repeated_iterator_reference_does_not_replay_items() -> None:
    iterator = iter([1, 2])
    capture = IteratorCapture([iterator, iterator])
    assert list(capture.wrapped[0]) == [1, 2]
    assert list(capture.wrapped[1]) == []
    assert capture.snapshot() == [[1, 2], []]


def test_plain_containers_are_not_replaced() -> None:
    source = {"values": [1, {"value": "text"}]}
    capture = IteratorCapture(source)
    assert capture.wrapped is source
    snapshot = capture.snapshot()
    source["values"].clear()
    assert snapshot == {"values": [1, {"value": "text"}]}
    assert capture.snapshot() == snapshot


def test_opaque_objects_are_not_inspected() -> None:
    class Opaque:
        def model_dump(self) -> None:
            pytest.fail("capture must not invoke model methods")

        def __iter__(self) -> Iterator[object]:
            pytest.fail("capture must not iterate unknown containers")

    value = Opaque()
    capture = IteratorCapture(value)
    assert capture.wrapped is value
    assert capture.snapshot() is value


def test_capture_failures_do_not_replace_sdk_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = {"value": "original"}
    capture = IteratorCapture(iter([value]))

    def fail_capture(value: object, ancestors: set[int]) -> None:
        raise ValueError("capture failed")

    monkeypatch.setattr(
        "opentelemetry.util.genai.iterator_capture._capture", fail_capture
    )
    assert next(capture.wrapped) is value
    assert capture.snapshot() == [UNCONSUMED]
    assert not capture.complete


def test_cycles_do_not_break_the_request() -> None:
    value: list[Any] = []
    value.append(value)
    capture = IteratorCapture(value)
    assert capture.wrapped is value
    assert capture.snapshot() == [UNCONSUMED]
    assert not capture.complete
