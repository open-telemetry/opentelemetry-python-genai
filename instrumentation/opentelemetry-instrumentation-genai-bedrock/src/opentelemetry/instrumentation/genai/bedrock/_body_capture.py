# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the reads the SDK makes of a file-like request body."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # ``wrapt.ObjectProxy`` is not subscriptable at runtime on the oldest
    # supported wrapt, so the generic form is only used for typing.
    class _ObjectProxy:
        __wrapped__: Any

        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy

#: Bytes kept at most; a larger body yields no snapshot.
_MAX_BYTES = 1024 * 1024

#: Methods the proxy shadows, and therefore the ones a body must already
#: expose to be wrapped.
_STREAM_METHODS = ("read", "seek")


def capture_body(
    body: Any, *, max_bytes: int = _MAX_BYTES
) -> BodyCapture | None:
    """Wrap *body* in a capture if it is a seekable file-like object.

    A body without ``read`` and ``seek`` is passed to the SDK as it is:
    wrapping it would give botocore methods the caller's object never had.
    """
    if not all(
        callable(getattr(body, name, None)) for name in _STREAM_METHODS
    ):
        return None
    return BodyCapture(body, max_bytes=max_bytes)


class BodyCapture(_ObjectProxy):
    """A proxy that records the first complete pass over a file-like body.

    Only ``read`` is recorded: ``readinto`` and the other read methods bypass
    the capture. A pass that does not run from its first position through end
    of stream, or one larger than ``max_bytes``, yields no snapshot rather
    than a partial one.
    """

    def __init__(self, wrapped: Any, *, max_bytes: int = _MAX_BYTES) -> None:
        super().__init__(wrapped)
        # ``_self_`` attributes stay on the proxy, never on the wrapped object.
        self._self_max_bytes = max_bytes
        self._self_buffer = bytearray()
        self._self_start: int | None = None
        self._self_complete = False
        self._self_usable = True

    def read(self, *args: Any, **kwargs: Any) -> Any:
        data = self.__wrapped__.read(*args, **kwargs)
        self._self_record_safely(data, _requested_size(args, kwargs))
        return data

    def seek(self, *args: Any, **kwargs: Any) -> Any:
        position = self.__wrapped__.seek(*args, **kwargs)
        self._self_track_seek_safely(position)
        return position

    def snapshot(self) -> bytes | None:
        """The recorded bytes, or ``None`` if the pass was not usable."""
        if self._self_complete and self._self_usable:
            return bytes(self._self_buffer)
        return None

    def _self_record_safely(self, data: Any, size: Any) -> None:
        try:
            self._self_record(data, size)
        except Exception:
            # Recording is telemetry; it must never break the SDK call.
            self._self_usable = False

    def _self_record(self, data: Any, size: Any) -> None:
        if self._self_complete or not self._self_usable:
            return

        if not isinstance(data, (bytes, bytearray)):
            # A text body, or a non-blocking stream that returned nothing.
            self._self_usable = False
            return

        if not data:
            # An empty read ends the pass, unless it asked for nothing:
            # read(0) returns nothing without reaching the end.
            if size != 0:
                self._self_complete = True
            return

        if len(self._self_buffer) + len(data) > self._self_max_bytes:
            self._self_usable = False
            self._self_buffer.clear()
            return

        self._self_buffer += data
        if size is None or (isinstance(size, int) and size < 0):
            # read() and read(-1) read through end of stream in one call.
            self._self_complete = True

    def _self_track_seek_safely(self, position: Any) -> None:
        try:
            self._self_track_seek(position)
        except Exception:
            self._self_usable = False

    def _self_track_seek(self, position: Any) -> None:
        if self._self_complete or not self._self_usable:
            return

        if not isinstance(position, int):
            # A stream that does not report a position cannot be placed.
            self._self_usable = False
            return

        if self._self_start is None or (
            position != self._self_start + len(self._self_buffer)
        ):
            # The stream moved away from the end of what was recorded, so what
            # the capture holds is not one contiguous pass over the body. Start
            # again from here.
            self._self_buffer.clear()
            self._self_start = position


def _requested_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The size a read asked for, as passed by the caller."""
    if args:
        return args[0]
    return kwargs.get("size")
