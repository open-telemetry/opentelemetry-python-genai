#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Guard the oldest-dependency testing invariant.

The oldest tox envs install the lowest versions of each package's *declared* deps straight from
pyproject.toml via ``UV_RESOLUTION=lowest-direct`` (see AGENTS.md). pyproject.toml is therefore the
single source of truth for those floors and running the env is what validates them. This script
covers the two gaps a passing oldest env can't see:

1. Re-introduced drift — a pyproject-declared dep hand-pinned again in tests/requirements.oldest.txt.
   Such a pin silently overrides the derived floor and can drift from the declared bound; remove it.
2. Missing coverage — a package with an oldest tox factor (a tests/requirements.latest.txt) but no
   tests/requirements.oldest.txt at all, so its declared floors are never exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def declared_dep_names(pyproject: dict) -> set[str]:
    """Canonical names of every dep declared in [project.dependencies] and optional-dependencies."""
    project = pyproject.get("project", {})
    names: set[str] = set()
    for dep in project.get("dependencies", []):
        names.add(canonicalize_name(Requirement(dep).name))
    for deps in project.get("optional-dependencies", {}).values():
        for dep in deps:
            names.add(canonicalize_name(Requirement(dep).name))
    return names


def pinned_names(oldest_req_path: Path) -> set[str]:
    """Canonical names of every pinned requirement in an oldest requirements file.

    Skips option lines (-e/-r/-c/--flag) and anything that isn't a parseable requirement.
    """
    names: set[str] = set()
    for raw in oldest_req_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            names.add(canonicalize_name(Requirement(line).name))
        except Exception:
            continue
    return names


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    errors: list[str] = []

    pyprojects = sorted(
        repo_root.glob("instrumentation/*/pyproject.toml"),
    ) + sorted(repo_root.glob("util/*/pyproject.toml"))

    for pyproject_path in pyprojects:
        pkg_dir = pyproject_path.parent
        oldest = pkg_dir / "tests" / "requirements.oldest.txt"
        latest = pkg_dir / "tests" / "requirements.latest.txt"

        if not oldest.exists():
            # Only a gap if the package has an oldest tox factor, signalled by a latest file.
            if latest.exists():
                errors.append(
                    f"{pkg_dir.name}: has tests/requirements.latest.txt but no "
                    f"tests/requirements.oldest.txt — declared floors are never tested."
                )
            continue

        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        redundant = declared_dep_names(pyproject) & pinned_names(oldest)
        for name in sorted(redundant):
            errors.append(
                f"{pkg_dir.name}: '{name}' is declared in pyproject.toml and also pinned in "
                f"tests/requirements.oldest.txt. Remove the pin — the oldest env derives it from "
                f"the pyproject.toml floor via UV_RESOLUTION=lowest-direct."
            )

    if errors:
        print("Oldest dependency check failed:\n", file=sys.stderr)
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        return 1

    print(
        "Oldest dependency checks passed: no declared deps re-pinned, no missing coverage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
