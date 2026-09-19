# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, cast, overload

AttributeT = TypeVar("AttributeT")


class _Attribute(Generic[AttributeT]):
    def __init__(
        self,
        name: str | None = None,
        *,
        default_factory: Callable[[], AttributeT] | None = None,
    ) -> None:
        self._name = name
        self._default_factory = default_factory

    def __set_name__(self, owner: type[object], name: str) -> None:
        if self._name is None:
            self._name = name

    @overload
    def __get__(
        self, instance: None, owner: type[object]
    ) -> _Attribute[AttributeT]: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> AttributeT: ...

    def __get__(
        self, instance: object | None, owner: type[object]
    ) -> AttributeT | _Attribute[AttributeT]:
        if instance is None:
            return self
        name = cast("str", self._name)
        value = getattr(instance, name, None)
        if value is None and self._default_factory is not None:
            value = self._default_factory()
            setattr(instance, name, value)
        return cast("AttributeT", value)

    def __set__(self, instance: object, value: AttributeT) -> None:
        setattr(instance, cast("str", self._name), value)
