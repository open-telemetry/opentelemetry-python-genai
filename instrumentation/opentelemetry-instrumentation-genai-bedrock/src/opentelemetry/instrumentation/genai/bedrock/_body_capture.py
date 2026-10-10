# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the first full read botocore makes of a file-like request body.

botocore reads a file-like ``body`` itself: it seeks over the stream to measure
it, reads it through to sign and checksum the request, and reads it again to
send it, once per attempt. Those reads are the only chance to see the payload
without adding I/O of our own, so the body handed to the SDK is a transparent
proxy that keeps the bytes of the first read that ran to end of stream.
"""

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

#: Upper bound on the bytes a capture keeps. The recorded content exists to
#: populate telemetry, which needs the request parameters the body holds, not
#: a payload of arbitrary size: a larger body is left uncaptured.
_MAX_BYTES = 1024 * 1024

#: Methods the proxy shadows, and therefore the ones a body must already
#: expose to be wrapped.
_STREAM_METHODS = ("read", "seek")


def capture_body(
    body: Any, *, max_bytes: int = _MAX_BYTES
) -> BodyCapture | None:
    """Capture *body* if the SDK streams it, else return ``None``.

    A body without ``read`` and ``seek`` is passed to the SDK as it is:
    wrapping it would give botocore methods the caller's object never had, and
    turn a call that would have failed into one that succeeds.
    """
    if not all(
        callable(getattr(body, name, None)) for name in _STREAM_METHODS
    ):
        return None
    return BodyCapture(body, max_bytes=max_bytes)


class BodyCapture(_ObjectProxy):
    """A transparent proxy that records the first full pass over a body.

    Every call is forwarded to the wrapped object unchanged. ``read`` records
    the bytes returned and ``seek`` records where the stream was moved to; the
    first read that reaches end of stream ends the capture, so the retries and
    rereads botocore performs afterwards cost nothing and cannot duplicate
    what was already kept.

    Only ``read`` is recorded, which is the only read botocore performs on a
    body: ``readinto`` and the other read methods bypass the capture.

    Recording is best effort and never raises into the SDK call. A body read
    from an unknown position, a body larger than ``max_bytes`` and a body that
    is moved somewhere other than the end of what was recorded all yield no
    snapshot rather than a partial one.
    """

    def __init__(self, wrapped: Any, *, max_bytes: int = _MAX_BYTES) -> None:
        super().__init__(wrapped)
        # Attributes prefixed with ``_self_`` are stored on the proxy, so the
        # wrapped object never gains state it did not have.
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
        """The bytes of the first pass over the body, if it was read whole.

        Returns ``None`` when the SDK did not read the body from its first
        position through end of stream, so a partial pass is never mistaken
        for the payload. This performs no I/O: it never reads or repositions
        the wrapped object.
        """
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
