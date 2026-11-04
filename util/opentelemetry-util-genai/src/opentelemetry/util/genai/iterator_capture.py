# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Record request iterator values as the SDK consumes them."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from io import IOBase
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:

    class _ObjectProxy:
        __wrapped__: object

        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy

_logger = logging.getLogger(__name__)
_InputT = TypeVar("_InputT")
_ResultT = TypeVar("_ResultT")
UNCONSUMED: object = object()
"""Snapshot placeholder for an iterator the SDK has not read."""


class _Node:
    """A captured request value, with depth capped to avoid recursion errors."""

    _MAX_DEPTH = 64

    @staticmethod
    def capture(value: object, ancestors: set[int], depth: int = 0) -> _Node:
        if depth >= _Node._MAX_DEPTH:
            return _Node(value, unavailable=True)
        if isinstance(value, IOBase):
            return _Node(value)
        if isinstance(value, Iterator):
            return _IteratorNode(cast("Iterator[object]", value), depth)
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
                    key: _Node.capture(item, ancestors, depth + 1)
                    for key, item in mapping.items()
                }
                wrapped = {
                    key: child.wrapped for key, child in children.items()
                }
                return _Node(
                    mapping
                    if all(
                        wrapped[key] is item for key, item in mapping.items()
                    )
                    else wrapped,
                    children,
                )
            sequence = cast("Sequence[object]", value)
            items = [
                _Node.capture(item, ancestors, depth + 1) for item in sequence
            ]
            wrapped_items = [item.wrapped for item in items]
            wrapped_sequence: object = value
            if any(
                new is not old for new, old in zip(wrapped_items, sequence)
            ):
                wrapped_sequence = (
                    tuple(wrapped_items)
                    if type(value) is tuple
                    else wrapped_items
                )
            return _Node(wrapped_sequence, items)
        finally:
            ancestors.remove(value_id)

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
    def __init__(self, iterator: Iterator[object], depth: int) -> None:
        super().__init__(iterator)
        self.depth: int = depth
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
        captured = _Node.capture(item, set(), node.depth + 1)
        node.items.append(captured)
        return captured.wrapped


class IteratorCapture:
    """Capture input iterators as the SDK consumes them.

    Wrap synchronous iterators in the input, including those
    nested in built-in dicts, lists, and tuples. Copy containers that need
    wrappers; leave the originals, files, and other objects untouched.
    Store consumed values in memory, copying each yielded container.

    Read ``snapshot()`` or ``extract()`` after the SDK call, including on
    failure. Iterator exceptions propagate unchanged. Disabled captures pass
    the input through unchanged and return ``None`` snapshots.
    """

    def __init__(
        self,
        value: object,
        enabled: bool = True,
    ) -> None:
        self.property_name: str | None = None
        self._node: _Node | None = (
            _Node.capture(value, set()) if enabled else None
        )
        self.wrapped: object = self._node.wrapped if self._node else value

    @classmethod
    def from_property(
        cls,
        properties: MutableMapping[str, object],
        property_name: str,
        enabled: bool = True,
    ) -> IteratorCapture:
        """Replace one property's value; leave missing properties absent."""
        present = property_name in properties
        capture = cls(properties.get(property_name), enabled and present)
        capture.property_name = property_name
        if enabled and present:
            properties[property_name] = capture.wrapped
        return capture

    @property
    def complete(self) -> bool:
        """True if capture succeeded and all captured iterators are exhausted."""
        return self._node is not None and self._node.complete

    def snapshot(self, *, unconsumed: object = UNCONSUMED) -> object:
        """Return a snapshot without advancing iterators.

        Each iterator becomes a list of items consumed so far. Unread
        iterators, cycles, and values beyond the depth limit become
        ``unconsumed``.
        Empty iterators become ``[]`` after the SDK reads them.
        Disabled or missing properties return ``None``.
        """
        return self._node.snapshot(unconsumed) if self._node else None

    def extract(
        self, extractor: Callable[[_InputT | None], _ResultT]
    ) -> _ResultT | None:
        """Pass a snapshot to ``extractor`` and return its result.

        The extractor must accept lists for consumed iterators, ``()`` for
        unread or unavailable values, and ``None`` for disabled or missing
        properties. Log snapshot or conversion errors and return ``None``.
        """
        try:
            snapshot = cast("_InputT | None", self.snapshot(unconsumed=()))
            return extractor(snapshot)
        except Exception:
            # Extractors should handle user input defensively, but cannot guard
            # against every unexpected value or custom object. Extraction often
            # runs in finally blocks and its errors must not hide SDK failures.
            _logger.debug(
                "Failed to extract %s",
                self.property_name or "captured input",
                exc_info=True,
            )
            return None
