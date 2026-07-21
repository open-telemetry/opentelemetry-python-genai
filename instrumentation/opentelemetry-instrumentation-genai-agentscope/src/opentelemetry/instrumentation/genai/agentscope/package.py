# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_instruments_v1 = ("agentscope >= 1.0.0, < 2.0.0",)
_instruments_v2 = ("agentscope >= 2.0.0, < 3.0.0",)
_instruments = ("agentscope >= 1.0.0, < 3.0.0",)

_supports_metrics = False


def get_installed_instrumentation_dependencies() -> tuple[str, ...]:
    """Return the AgentScope dependency range matching the installed major."""
    # Imported lazily so tooling that execs this module (e.g. the instrumentation
    # README generator) does not require ``packaging`` to be installed.
    from packaging.requirements import Requirement  # noqa: PLC0415

    try:
        installed_version = version("agentscope")
    except PackageNotFoundError:
        return _instruments

    for requirement in (_instruments_v2[0], _instruments_v1[0]):
        if Requirement(requirement).specifier.contains(
            installed_version,
            prereleases=True,
        ):
            return (requirement,)

    return _instruments
