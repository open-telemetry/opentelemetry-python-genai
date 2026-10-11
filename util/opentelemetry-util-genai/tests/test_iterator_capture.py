# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import Generator, Iterable, Iterator, Sequence
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pytest

from opentelemetry.util.genai.iterator_capture import (
    UNCONSUMED,
    IteratorCapture,
)


@pytest.mark.parametrize(
    "state", ["disabled", "missing", "none", "unread", "empty"]
)
def test_extract_converts_snapshot_defaults(state: str) -> None:
    source = iter(["Hello"] if state != "empty" else [])
    kwargs: dict[str, Any] = {"messages": source}
    if state == "missing":
        kwargs.clear()
    elif state == "none":
        kwargs["messages"] = None
    capture = IteratorCapture(kwargs, "messages", enabled=state != "disabled")
    if state == "empty":
        assert list(kwargs["messages"]) == []

    def convert(values: Iterable[str] | None) -> str:
        return "absent" if values is None else ",".join(values)

    expected = "absent" if state in ("disabled", "missing", "none") else ""
    assert capture.extract(convert) == expected
    if state != "empty":
        assert next(source) == "Hello"


def test_extract_converts_partial_nested_input_after_failure() -> None:
    error = ValueError("request failed")
    unread = iter(["unread"])

    def generate() -> Iterator[dict[str, Iterator[str]]]:
        yield {"content": iter(["first", "second"]), "other": unread}
        raise error

    kwargs = {"messages": generate()}
    capture = IteratorCapture(kwargs, "messages")
    message = next(kwargs["messages"])
    assert next(message["content"]) == "first"
    with pytest.raises(ValueError) as caught:
        next(kwargs["messages"])
    assert caught.value is error

    def convert(
        messages: Iterable[dict[str, Sequence[str]]] | None,
    ) -> list[str]:
        return [
            text
            for message in messages or ()
            for content in message.values()
            for text in content
        ]

    assert capture.extract(convert) == ["first"]
    assert not capture.complete
    assert next(message["content"]) == "second"
    assert next(unread) == "unread"


@pytest.mark.parametrize("failure", ["snapshot", "converter"])
def test_extract_errors_return_none(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    capture = IteratorCapture({"messages": [1]}, "messages")

    def convert(value: object) -> str:
        raise ValueError("invalid content")

    if failure == "snapshot":

        def fail_snapshot(*, unconsumed: object) -> object:
            raise ValueError("invalid snapshot")

        monkeypatch.setattr(capture, "snapshot", fail_snapshot)

    with caplog.at_level(logging.DEBUG):
        assert capture.extract(convert) is None
    assert "Failed to extract request property messages" in caplog.text


def test_extract_does_not_swallow_cancellation() -> None:
    capture = IteratorCapture({}, "messages")
    error = asyncio.CancelledError()

    def convert(value: object) -> str:
        raise error

    with pytest.raises(asyncio.CancelledError) as caught:
        capture.extract(convert)
    assert caught.value is error


def test_disabled_capture_leaves_request_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_capture(value: object, ancestors: set[int]) -> None:
        pytest.fail("disabled capture must not inspect request values")

    monkeypatch.setattr(
        "opentelemetry.util.genai.iterator_capture._capture", fail_capture
    )
    messages = {"content": iter(["Hello"])}
    kwargs = {"messages": messages}
    capture = IteratorCapture(kwargs, "messages", enabled=False)
    assert kwargs["messages"] is messages
    assert capture.snapshot() is None
    assert not capture.complete


def test_missing_property_is_not_added() -> None:
    kwargs: dict[str, object] = {}
    capture = IteratorCapture(kwargs, "messages")
    assert kwargs == {}
    assert capture.snapshot() is None
    assert not capture.complete


def test_captures_are_independent_and_leave_other_properties_untouched() -> None:
    messages = iter(["Hello", "Goodbye"])
    tools = iter(["weather"])
    unrelated = iter(["stop"])
    kwargs = {"messages": messages, "tools": tools, "stop": unrelated}
    messages_capture = IteratorCapture(kwargs, "messages")
    tools_capture = IteratorCapture(kwargs, "tools")

    assert messages_capture.property_name == "messages"
    assert tools_capture.property_name == "tools"
    assert kwargs["stop"] is unrelated
    assert next(kwargs["messages"]) == "Hello"
    assert messages_capture.snapshot() == ["Hello"]
    assert tools_capture.snapshot() is UNCONSUMED
    assert list(kwargs["tools"]) == ["weather"]
    assert tools_capture.snapshot() == ["weather"]
    assert tools_capture.complete
    assert not messages_capture.complete


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
        kwargs = {"value": {"data": data}}
        capture = IteratorCapture(kwargs, "value")
        assert kwargs["value"]["data"] is data
        assert data.tell() == 0
        assert capture.snapshot(unconsumed=()) == {"data": data}
        assert kwargs["value"]["data"].read() == (
            b"content" if binary else "content"
        )
        assert capture.snapshot(unconsumed=()) == {"data": data}
        assert capture.complete


def test_capture_is_lazy_and_distinguishes_empty_from_unread() -> None:
    reads: list[str] = []

    def generate() -> Iterator[str]:
        reads.append("read")
        yield from ()

    kwargs = {"value": generate()}
    capture = IteratorCapture(kwargs, "value")
    assert reads == []
    assert capture.snapshot() is UNCONSUMED
    assert capture.snapshot(unconsumed=()) == ()
    assert not capture.complete
    assert list(kwargs["value"]) == []
    assert reads == ["read"]
    assert capture.snapshot() == []
    assert capture.complete


def test_nested_capture_preserves_containers_and_partial_input() -> None:
    inner = iter(["first", "second"])
    source = {"arbitrary": [{"payload": (inner,)}]}
    kwargs = {"value": source}
    capture = IteratorCapture(kwargs, "value")
    wrapped = kwargs["value"]["arbitrary"][0]["payload"][0]
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

    kwargs = {"value": generate()}
    capture = IteratorCapture(kwargs, "value")
    iterator = kwargs["value"]
    assert next(iterator) is shared
    assert next(iterator) is shared
    assert list(iterator) == []
    assert capture.snapshot() == [{"value": [0]}, {"value": [1]}]


def test_nested_iterators_yielded_by_iterators_stay_lazy() -> None:
    reads: list[str] = []

    def inner() -> Iterator[str]:
        reads.append("inner")
        yield "value"

    kwargs = {"value": iter([{"data": inner()}])}
    capture = IteratorCapture(kwargs, "value")
    item = next(kwargs["value"])
    assert reads == []
    assert capture.snapshot() == [{"data": UNCONSUMED}]
    assert list(item["data"]) == ["value"]
    assert capture.snapshot() == [{"data": ["value"]}]
    assert not capture.complete
    assert list(kwargs["value"]) == []
    assert capture.complete


@pytest.mark.parametrize(
    "error", [ValueError("iterator failed"), asyncio.CancelledError()]
)
def test_iterator_errors_propagate_unchanged(error: BaseException) -> None:
    def generate() -> Iterator[str]:
        yield "first"
        raise error

    kwargs = {"value": generate()}
    capture = IteratorCapture(kwargs, "value")
    assert next(kwargs["value"]) == "first"
    with pytest.raises(type(error)) as caught:
        next(kwargs["value"])
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

    kwargs = {"value": generate()}
    capture = IteratorCapture(kwargs, "value")
    assert next(kwargs["value"]) == "first"
    assert kwargs["value"].send("sent") == "sent"
    assert kwargs["value"].throw(ValueError()) == "handled"
    with pytest.raises(StopIteration) as caught:
        next(kwargs["value"])
    assert caught.value.value == "finished"
    assert capture.snapshot() == ["first", "sent", "handled"]
    assert capture.complete
    assert closed == [True]


def test_close_does_not_drain_input() -> None:
    def generate() -> Iterator[int]:
        yield 1
        pytest.fail("close must not advance the iterator")

    kwargs = {"value": generate()}
    capture = IteratorCapture(kwargs, "value")
    assert next(kwargs["value"]) == 1
    kwargs["value"].close()
    assert capture.snapshot() == [1]
    assert not capture.complete


def test_repeated_iterator_reference_does_not_replay_items() -> None:
    iterator = iter([1, 2])
    kwargs = {"value": [iterator, iterator]}
    capture = IteratorCapture(kwargs, "value")
    assert list(kwargs["value"][0]) == [1, 2]
    assert list(kwargs["value"][1]) == []
    assert capture.snapshot() == [[1, 2], []]


def test_plain_containers_are_not_replaced() -> None:
    source = {"values": [1, {"value": "text"}]}
    kwargs = {"value": source}
    capture = IteratorCapture(kwargs, "value")
    assert kwargs["value"] is source
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
    kwargs = {"value": value}
    capture = IteratorCapture(kwargs, "value")
    assert kwargs["value"] is value
    assert capture.snapshot() is value


def test_capture_failures_do_not_replace_sdk_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = {"value": "original"}
    kwargs = {"value": iter([value])}
    capture = IteratorCapture(kwargs, "value")

    def fail_capture(value: object, ancestors: set[int]) -> None:
        raise ValueError("capture failed")

    monkeypatch.setattr(
        "opentelemetry.util.genai.iterator_capture._capture", fail_capture
    )
    assert next(kwargs["value"]) is value
    assert capture.snapshot() == [UNCONSUMED]
    assert not capture.complete


def test_cycles_do_not_break_the_request() -> None:
    value: list[Any] = []
    value.append(value)
    kwargs = {"value": value}
    capture = IteratorCapture(kwargs, "value")
    assert kwargs["value"] is value
    assert capture.snapshot() == [UNCONSUMED]
    assert not capture.complete
