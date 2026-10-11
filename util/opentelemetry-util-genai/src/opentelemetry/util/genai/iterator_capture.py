# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe request iterators without advancing them ahead of the SDK."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from io import IOBase
from typing import TYPE_CHECKING, Generic, TypeVar, cast

if TYPE_CHECKING:

    class _ObjectProxy:
        __wrapped__: object

        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")
UNCONSUMED: object = object()
"""Snapshot placeholder for an iterator the SDK has not read."""


class _Node:
    def __init__(
        self,
        wrapped: object,
        children: dict[object, _Node] | list[_Node] | None = None,
        *,
        unavailable: bool = False,
    ) -> None:
        self.wrapped: object = wrapped
        self.children: dict[object, _Node] | list[_Node] | None = children
        self.unavailable: bool = unavailable

    @property
    def complete(self) -> bool:
        if self.unavailable:
            return False
        children = self.children
        if isinstance(children, dict):
            return all(child.complete for child in children.values())
        if isinstance(children, list):
            return all(child.complete for child in children)
        return True

    def snapshot(self, unconsumed: object) -> object:
        if self.unavailable:
            return unconsumed
        children = self.children
        if isinstance(children, dict):
            return {
                key: child.snapshot(unconsumed)
                for key, child in children.items()
            }
        if isinstance(children, list):
            items = [child.snapshot(unconsumed) for child in children]
            return tuple(items) if isinstance(self.wrapped, tuple) else items
        return self.wrapped


class _IteratorNode(_Node):
    def __init__(self, iterator: Iterator[object]) -> None:
        super().__init__(iterator)
        self.items: list[_Node] = []
        self.started: bool = False
        self.exhausted: bool = False
        self.wrapped = _RecordingIterator(iterator, self)

    @property
    def complete(self) -> bool:
        return self.exhausted and all(item.complete for item in self.items)

    def snapshot(self, unconsumed: object) -> object:
        if not self.started:
            return unconsumed
        return [item.snapshot(unconsumed) for item in self.items]


class _RecordingIterator(_ObjectProxy):
    _self_node: _IteratorNode

    def __init__(
        self, iterator: Iterator[object], node: _IteratorNode
    ) -> None:
        super().__init__(iterator)
        self._self_node = node

    def __iter__(self) -> _RecordingIterator:
        return self

    def __next__(self) -> object:
        iterator = cast("Iterator[object]", self.__wrapped__)
        return self._self_advance(lambda: next(iterator))

    def send(self, value: object) -> object:
        send = cast(
            "Callable[[object], object]", getattr(self.__wrapped__, "send")
        )
        return self._self_advance(lambda: send(value))

    def throw(self, *args: object) -> object:
        throw = cast(
            "Callable[..., object]", getattr(self.__wrapped__, "throw")
        )
        return self._self_advance(lambda: throw(*args))

    def _self_advance(self, advance: Callable[[], object]) -> object:
        node = self._self_node
        node.started = True
        try:
            item = advance()
        except StopIteration:
            node.exhausted = True
            raise
        # Snapshot before the next advance: generators can mutate and yield
        # the same dictionary repeatedly.
        captured = _safe_capture(item, set())
        node.items.append(captured)
        return captured.wrapped


def _safe_capture(value: object, ancestors: set[int]) -> _Node:
    try:
        return _capture(value, ancestors)
    except Exception:
        _logger.debug("Failed to capture request input", exc_info=True)
        return _Node(value, unavailable=True)


def _capture(value: object, ancestors: set[int]) -> _Node:
    if isinstance(value, IOBase):
        return _Node(value)
    if isinstance(value, Iterator):
        return _IteratorNode(cast("Iterator[object]", value))
    if type(value) not in (dict, list, tuple):
        return _Node(value)
    value_id = id(value)
    if value_id in ancestors:
        return _Node(value, unavailable=True)
    ancestors.add(value_id)
    try:
        if type(value) is dict:
            mapping = cast("dict[object, object]", value)
            children = {
                key: _safe_capture(item, ancestors)
                for key, item in mapping.items()
            }
            wrapped = {key: child.wrapped for key, child in children.items()}
            return _Node(
                mapping
                if all(wrapped[key] is item for key, item in mapping.items())
                else wrapped,
                children,
            )
        sequence = cast("Sequence[object]", value)
        items = [_safe_capture(item, ancestors) for item in sequence]
        wrapped_items = [item.wrapped for item in items]
        wrapped_sequence: object = value
        if any(new is not old for new, old in zip(wrapped_items, sequence)):
            wrapped_sequence = (
                tuple(wrapped_items) if type(value) is tuple else wrapped_items
            )
        return _Node(wrapped_sequence, items)
    finally:
        ancestors.remove(value_id)


class IteratorCapture(Generic[_T]):
    """Capture an object's iterators as the consumer reads them.

    Walks built-in dictionaries, lists, and tuples, wrapping nested synchronous
    iterators with transparent proxies. File objects pass through unchanged,
    as do other objects, without attribute inspection or conversion.
    Containers are copied only when a nested iterator needs replacing; the
    caller's containers are not modified. Snapshot containers are independent
    copies, including for each mutable item yielded by an iterator.

    Create only when content capture is enabled. Pass ``wrapped`` to the SDK,
    then extract telemetry from ``snapshot()`` after request serialization.
    Capture stores consumed input in memory; it neither drains nor replays
    iterators. The SDK's exceptions propagate unchanged.
    """

    def __init__(self, value: _T) -> None:
        self._node: _Node = _safe_capture(value, set())

    @property
    def wrapped(self) -> _T:
        """Input with lazy proxies in place of its iterators."""
        return cast("_T", self._node.wrapped)

    @property
    def complete(self) -> bool:
        """Whether all observed iterators are exhausted and capture succeeded."""
        return self._node.complete

    def snapshot(self, *, unconsumed: object = UNCONSUMED) -> object:
        """Return captured data without advancing iterators.

        Iterators become lists of consumed items. Unread iterators and failed
        captures use ``unconsumed``; cycles also use this placeholder. An empty
        iterator becomes ``[]`` only after the consumer reads it. Partially
        consumed input is available even when ``complete`` is false.
        """
        return self._node.snapshot(unconsumed)
