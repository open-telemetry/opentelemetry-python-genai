# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import urllib.parse
from base64 import b64decode, b64encode
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from functools import lru_cache, partial
from typing import Any, cast
from uuid import UUID

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    ContentCapturingMode,
    MessagePart,
    Modality,
    UriPart,
)
from opentelemetry.util.types import AnyValue

logger = logging.getLogger(__name__)


def get_content_capturing_mode() -> ContentCapturingMode:
    """Gets ContentCapturingMode from associated envvar, defaulting to NO_CONTENT if unset."""
    envvar = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, ""
    ).strip()
    if not envvar:
        return ContentCapturingMode.NO_CONTENT
    try:
        return ContentCapturingMode[envvar.upper()]
    except KeyError:
        logger.warning(
            "%s is not a valid option for `%s` environment variable. Must be one of %s. Defaulting to `NO_CONTENT`.",
            envvar,
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
            ", ".join(e.name for e in ContentCapturingMode),
        )
        return ContentCapturingMode.NO_CONTENT


def decode_base64(data: str) -> bytes | None:
    """Decode a base64 string, returning ``None`` if it is malformed.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    try:
        return b64decode("".join(data.split()), validate=True)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def image_from_url(
    url: str, *, modality: Modality | str = Modality.IMAGE
) -> MessagePart | None:
    """Return a media part for a ``url``, defaulting to the image modality.

    Override ``modality`` for other standard or provider-specific media,
    such as audio or documents.

    A ``data:<mime>;base64,<payload>`` URL is decoded into a
    :class:`~opentelemetry.util.genai.types.BlobPart`; a ``data:`` URL without
    base64 encoding has its percent-encoded payload decoded into bytes; any
    other URL becomes a :class:`~opentelemetry.util.genai.types.UriPart`. Shared
    by instrumentations that parse provider media blocks.

    Called only when content capture is enabled
    (``TelemetryHandler.should_capture_content()``).
    """
    if url.startswith("data:"):
        header, _, payload = url[len("data:") :].partition(",")
        mime_type = header.split(";", 1)[0] or None
        if ";base64" in header.lower():
            decoded = decode_base64(payload)
            if decoded is None:
                return None
            content = decoded
        else:
            # Non-base64 data URL payloads are percent-encoded (RFC 2397).
            content = urllib.parse.unquote_to_bytes(payload)
        return BlobPart(
            mime_type=mime_type,
            modality=modality,
            content=content,
        )
    return UriPart(mime_type=None, modality=modality, uri=url)


def is_experimental_mode() -> bool:
    """
    Kept for backwards compatibility. The utils in this library only support the experimental mode sem convs now.
    Don't use this function always returns True.
    """
    return True


def fq_exception_type(exception: BaseException) -> str:
    """Return the fully qualified name of an exception's type.

    Matches the ``exception.type`` value the exception event records, so
    ``error.type`` and ``exception.type`` stay consistent. Builtins are returned
    unqualified (e.g. ``ValueError``, not ``builtins.ValueError``).
    """
    # Mirrors the SDK's Span.record_exception so error.type matches the
    # exception event's exception.type:
    # https://github.com/open-telemetry/opentelemetry-python/blob/main/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py
    exc_type = type(exception)
    module = exc_type.__module__
    qualname = exc_type.__qualname__
    if module and module != "builtins":
        return f"{module}.{qualname}"
    return qualname


class _GenAiJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return super().default(o)


gen_ai_json_dump = partial(
    json.dump, separators=(",", ":"), cls=_GenAiJsonEncoder
)
"""Should be used by GenAI instrumentations when serializing objects that may contain
bytes, datetimes, etc. for GenAI observability."""

gen_ai_json_dumps = partial(
    json.dumps, separators=(",", ":"), cls=_GenAiJsonEncoder
)
"""Should be used by GenAI instrumentations when serializing objects that may contain
bytes, datetimes, etc. for GenAI observability."""


_MAX_DEPTH = 32
_OMIT = object()


def object_to_any_value(value: object) -> AnyValue | None:
    """Convert an object into an AnyValue, or None if it cannot be converted.

    Models, dataclasses and objects with public ``__dict__`` attributes become
    dicts; datetimes, UUIDs and enums become their primitive form. Other
    objects with a custom ``__str__`` (e.g. ``Decimal``, ``Path``) become
    strings. Values that cannot be converted (callables, types, non-finite
    floats, cycles, too deep structures, objects raising on access) are
    dropped from collections. Generators and other iterables are not consumed.
    """
    res = _sanitize_for_any_value(value, max_depth=_MAX_DEPTH)
    return None if res is _OMIT else cast(AnyValue | None, res)


def tool_arguments_to_any_value(arguments: object) -> AnyValue | None:
    """Convert tool call arguments to AnyValue, parsing JSON strings.

    Strings that are not valid JSON are returned as-is.
    """
    if isinstance(arguments, str):
        try:
            return object_to_any_value(json.loads(arguments))
        except ValueError:
            return arguments
    return object_to_any_value(arguments)


def _sanitize_for_any_value(
    value: object,
    *,
    max_depth: int,
    seen: frozenset[int] = frozenset(),
) -> object:
    if max_depth <= 0:
        return _OMIT

    if value is None or isinstance(value, (str, bool, int, bytes)):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else _OMIT
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _sanitize_for_any_value(
            value.value, max_depth=max_depth - 1, seen=seen
        )
    if callable(value) or isinstance(value, type) or id(value) in seen:
        return _OMIT

    # Arbitrary user objects can raise from model dumps, __getitem__, __dict__.
    try:
        return _sanitize_object(
            value, max_depth=max_depth - 1, seen=seen | {id(value)}
        )
    except Exception:
        return _OMIT


def _dump_model(value: object) -> object | None:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in fields(value)}
    for method_name in ("model_dump", "dict", "to_dict"):
        fn = getattr(value, method_name, None)
        if callable(fn):
            return cast(Callable[[], object], fn)()
    return None


def _sanitize_object(
    value: object, *, max_depth: int, seen: frozenset[int]
) -> object:
    dumped_model = _dump_model(value)
    if dumped_model is not None:
        return _sanitize_for_any_value(
            dumped_model, max_depth=max_depth, seen=seen
        )

    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(k, str) for k in mapping):
            return _OMIT
        return {
            k: cleaned
            for k, v in mapping.items()
            if (
                cleaned := _sanitize_for_any_value(
                    v, max_depth=max_depth, seen=seen
                )
            )
            is not _OMIT
        }

    # Arbitrary iterables (generators, streams) would be consumed or may never end.
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            cleaned
            for item in cast(Iterable[object], value)
            if (
                cleaned := _sanitize_for_any_value(
                    item, max_depth=max_depth, seen=seen
                )
            )
            is not _OMIT
        ]

    obj_dict: object = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        # Skip private state, e.g. pydantic SecretStr's raw _secret_value.
        public = {
            k: v
            for k, v in cast(dict[object, object], obj_dict).items()
            if isinstance(k, str) and not k.startswith("_")
        }
        if public:
            return _sanitize_for_any_value(
                public, max_depth=max_depth, seen=seen
            )

    # Default object.__str__ gives a repr with a memory address, not data.
    if type(value).__str__ is not object.__str__:
        return str(value)
    return _OMIT


_SIGNATURE_CACHE_MAX_SIZE = 1024
_inspect_signature = inspect.signature


@lru_cache(maxsize=_SIGNATURE_CACHE_MAX_SIZE)
def _cached_signature(
    fn: Callable[..., object], drop_first: bool
) -> inspect.Signature:
    sig = _inspect_signature(fn)
    if drop_first:
        params = list(sig.parameters.values())[1:]
        sig = sig.replace(parameters=params)
    return sig


def get_signature(func: Callable[..., object]) -> inspect.Signature:
    """Return the inspect.Signature for a callable, caching long-lived definitions.

    Bound methods are new objects on every attribute access, so key on the
    underlying function. Only long-lived definitions are cached; per-call
    closures and callable instances would otherwise be pinned in the cache.
    """
    try:
        underlying = getattr(func, "__func__", None)
        if (
            underlying is not None
            and getattr(func, "__self__", None) is not None
            and "<locals>" not in getattr(underlying, "__qualname__", "")
        ):
            return _cached_signature(underlying, True)
        if inspect.isfunction(func) and "<locals>" not in func.__qualname__:
            return _cached_signature(func, False)
    except TypeError:
        pass
    return _inspect_signature(func)


def bind_arguments(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    apply_defaults: bool = False,
) -> dict[str, object]:
    """Bind positional and keyword arguments to func's parameters by name."""
    try:
        sig = get_signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        if apply_defaults:
            bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def get_argument(
    name: str,
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    default: object = None,
    *,
    apply_defaults: bool = False,
) -> object:
    """Extract a named argument from kwargs or args via signature binding."""
    if name in kwargs:
        return kwargs[name]
    if not args and not apply_defaults:
        return default
    bound = bind_arguments(func, args, kwargs, apply_defaults=apply_defaults)
    return bound.get(name, default)
